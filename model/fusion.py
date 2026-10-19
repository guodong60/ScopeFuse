# coding=utf-8
"""
跨层级注意力+门控融合(GranularityFusion)：方案里"自适应聚合语义信息"的核心实现。

这一版把原来写死"token对[语句,block,函数]三路"的CrossGranularityFusion泛化成了
通用的K路融合模块——原因是encoder.py改成"自底向上抽象+自顶向下回注"的级联结构后，
每一级refinement都需要同样的"用自己的query对若干个context做注意力打分+门控融合"逻辑，
只是K(context数量)和参与的对象不同：
  - block的refinement: K=1，context=[func]
  - stmt的refinement:  K=1，context=[refine过的block]
  - token的refinement: K=2，context=[refine过的stmt, refine过的block]
与其为每一级单独写一份，不如把"K路融合"这个操作本身抽成一个可复用模块。

区别于"简单相加"式的回注，这里让每个位置的query对若干context做真正的多头注意力打分，
权重由内容决定，再用门控(而不是直接相加)决定"保留多少自身信息、采纳多少上下文"。

调用方负责把各个context"广播/gather"成与own_repr相同的[B,N,D]形状再传进来
(比如token的语句级context，需要调用方先用token2stmt把stmt_repr gather成[B,L,D])，
本模块只管融合本身，不管索引换算——这样每一级怎么gather可以按自己的语义写，模块保持通用。
"""
import torch
import torch.nn as nn


class GranularityFusion(nn.Module):
    def __init__(self, d_model, n_heads, drop_rate=0.1, use_layer_norm=True, use_gate=True):
        super().__init__()
        self.use_gate = use_gate   # False：保留跨粒度注意力，但用残差相加 own+ctx 代替门控(消融门控用)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True,
                                                 dropout=drop_rate)
        self.gate = nn.Linear(2 * d_model, d_model) if use_gate else None
        self.dropout = nn.Dropout(drop_rate)
        self.norm = nn.LayerNorm(d_model) if use_layer_norm else None

    def forward(self, own_repr, own_pad_mask, context_list, need_weights=False):
        """
        own_repr:      [B, N, D]  要被融合/refine的表示(比如token_repr、stmt_repr、block_repr)
        own_pad_mask:  [B, N] bool
        context_list:  长度为K的list，每个元素形状都是[B, N, D]——即已经"对齐"到与own_repr
                       相同position数量的context(调用方通过gather/broadcast准备好)
        need_weights:  是否返回K路的注意力权重，用于消融分析/可视化

        返回: fused [B, N, D]，padding位置强制清零；need_weights=True时额外返回[B,N,K]
        """
        B, N, D = own_repr.shape
        K = len(context_list)

        # 把(B,N)展平成一个大batch，每个"样本"是own自己的query(长度1)对K个context做注意力
        memory = torch.stack(context_list, dim=2)  # [B,N,K,D]
        memory = memory.reshape(B * N, K, D)
        query = own_repr.reshape(B * N, 1, D)
        ctx, attn_weights = self.cross_attn(query, memory, memory, need_weights=need_weights)
        ctx = ctx.reshape(B, N, D)

        # 门控融合：g决定"保留多少own自身信息、采纳多少融合后的上下文ctx"，
        # 由[own_repr, ctx]拼接后线性+sigmoid算出，是内容相关的、逐维度的权重
        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([own_repr, ctx], dim=-1)))
            fused = g * own_repr + (1 - g) * self.dropout(ctx)
        else:
            fused = own_repr + self.dropout(ctx)   # 无门控：残差相加，之后同样过LayerNorm
        if self.norm is not None:
            fused = self.norm(fused)
        fused = fused * own_pad_mask.unsqueeze(-1).to(fused.dtype)

        if need_weights:
            return fused, attn_weights.reshape(B, N, K)
        return fused, None


if __name__ == '__main__':
    torch.manual_seed(0)
    B, L, S, K, D, n_heads = 2, 8, 3, 2, 16, 4

    # 用K=3模拟原来"token对[语句,block,函数]"的场景，验证泛化后行为和之前一致
    own_repr = torch.randn(B, L, D, requires_grad=True)
    own_pad_mask = torch.zeros(B, L, dtype=torch.bool)
    own_pad_mask[0, :5] = True
    own_pad_mask[1, :8] = True

    ctx1 = torch.randn(B, L, D, requires_grad=True)
    ctx2 = torch.randn(B, L, D, requires_grad=True)
    ctx3 = torch.randn(B, L, D, requires_grad=True)

    fusion = GranularityFusion(d_model=D, n_heads=n_heads)
    fusion.eval()  # 权重归一化检查必须在eval模式下做(训练模式dropout会让权重和不精确为1，
                   # 这是本项目反复踩到、反复记录过的同一个坑，这里直接绕开)
    fused, weights = fusion(own_repr, own_pad_mask, [ctx1, ctx2, ctx3], need_weights=True)

    print('fused shape:', fused.shape, '有无nan:', torch.isnan(fused).any().item())
    print('padding位置是否为0:', torch.allclose(fused[0, 5:], torch.zeros_like(fused[0, 5:])))
    print('attn_weights shape:', weights.shape, '每行K路权重和应为1:',
         torch.allclose(weights.sum(-1), torch.ones(B, L), atol=1e-5))

    # 再测K=1场景(对应block<-func、stmt<-block的用法)
    fusion1 = GranularityFusion(d_model=D, n_heads=n_heads)
    fusion1.eval()
    fused1, w1 = fusion1(own_repr, own_pad_mask, [ctx1], need_weights=True)
    print('K=1场景 fused shape:', fused1.shape, 'weights shape:', w1.shape,
         '权重应恒为1(只有一个context):', torch.allclose(w1[0, :5], torch.ones(5, 1), atol=1e-5))

    loss = fused.sum() + fused1.sum()
    loss.backward()
    grads_ok = all(t.grad is not None and not torch.isnan(t.grad).any().item()
                   for t in [own_repr, ctx1, ctx2, ctx3])
    print('own/context各路梯度都能正常回传:', grads_ok)