# coding=utf-8
"""
复制机制(pointer-generator风格)，对应HetCoS论文公式(6)-(9)的思路：
每个解码步骤的输出概率 = p_gen * 词表生成分布 + (1-p_gen) * 从源码复制的分布，
p_gen由内容决定(sigmoid门控)，不是固定权重。

和论文原代码(copy_attention.py的MultiCopyGenerator)相比，最大的实现差异是：
"复制分布"从源端token的注意力权重"分散"到词表空间这一步，原代码用meshgrid构造
三维索引再index_put(accumulate=True)手动实现，这里直接用PyTorch自带的
scatter_add_一步到位——语义完全等价(注意力权重按源token对应的词表id累加到该位置)，
但不用手写索引展开，不容易出维度错误。

能这样简洁实现还有个前提：咱们的词表是code和text共享的(vocab_builder.py默认
shared=True)，所以"每个源码token对应哪个词表id"直接就是code_ids本身，不需要
像论文原代码那样额外维护一份src_map_idx映射——这是当初特意把词表设计成共享的
原因，现在用上了。
"""
import torch
import torch.nn as nn


class CopyGenerator(nn.Module):
    def __init__(self, tgt_dims, vocab_size, src_dims=None, n_heads=8, drop_rate=0.1,
                pad_idx=0, tie_embedding=None):
        super().__init__()
        src_dims = src_dims or tgt_dims
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.out_fc = nn.Linear(tgt_dims, vocab_size)  # 生成词表分布用
        if tie_embedding is not None:
            # 和decoder的token embedding共享权重(标准的输出层权重共享技巧)。
            # use_copy=False那条路径(decoder.py里的out_proj)一直都有这个共享，
            # 之前接copy机制时漏掉了，这里补上——否则这层会凭空多出vocab_size*tgt_dims
            # 个参数，也丢掉了权重共享通常带来的正则化效果
            assert tie_embedding.weight.shape == (vocab_size, tgt_dims), \
                'tie_embedding的形状必须是[vocab_size, tgt_dims]才能共享'
            self.out_fc.weight = tie_embedding.weight
        # 复制注意力：query=解码器隐状态，key/value=encoder的token表示
        self.copy_attn = nn.MultiheadAttention(tgt_dims, n_heads, kdim=src_dims, vdim=src_dims,
                                               batch_first=True, dropout=drop_rate)
        # 门控p_gen：由[解码器隐状态, 复制注意力的上下文向量]拼接算出
        self.gate_fc = nn.Linear(tgt_dims + src_dims, 1)

    def forward(self, tgt_dec_out, tgt_pad_mask, memory, memory_pad_mask, memory_ids):
        """
        tgt_dec_out:    [B,T,D_tgt] 解码器最后一层的隐状态(还没过输出层)
        tgt_pad_mask:   [B,T] bool，True=真实数据
        memory:         [B,L,D_src] encoder输出(HierEncoder的token_repr)
        memory_pad_mask:[B,L] bool，True=真实数据
        memory_ids:     [B,L] long，每个源码token对应的词表id(即code_ids本身，
                        因为code/text共享词表)

        返回: probs [B,T,vocab_size]，融合后的概率分布(已经是概率、不是logits，
             用于配合NLLLoss而不是CrossEntropyLoss)；p_gen [B,T]供调试/分析用
        """
        B, T, _ = tgt_dec_out.shape
        L = memory.shape[1]

        vocab_logits = self.out_fc(tgt_dec_out)  # [B,T,V]
        vocab_probs = torch.softmax(vocab_logits, dim=-1)

        # need_weights=True时nn.MultiheadAttention默认对多头取平均，返回[B,T,L]的注意力权重，
        # 正好是我们要的"复制分布"(每个源token被复制的概率)
        context, attn_weights = self.copy_attn(
            tgt_dec_out, memory, memory,
            key_padding_mask=~memory_pad_mask, need_weights=True)  # context[B,T,D_tgt], attn_weights[B,T,L]

        copy_probs = torch.zeros_like(vocab_probs)  # [B,T,V]
        idx = memory_ids.unsqueeze(1).expand(B, T, L)  # [B,T,L]，每个源位置的词表id
        copy_probs.scatter_add_(dim=2, index=idx, src=attn_weights)

        p_gen = torch.sigmoid(self.gate_fc(torch.cat([tgt_dec_out, context], dim=-1)))  # [B,T,1]
        probs = p_gen * vocab_probs + (1 - p_gen) * copy_probs  # [B,T,V]
        probs = probs * tgt_pad_mask.unsqueeze(-1).to(probs.dtype)
        return probs, p_gen.squeeze(-1)


if __name__ == '__main__':
    torch.manual_seed(0)
    B, T, L, D, V, n_heads = 2, 5, 8, 16, 30, 4

    tgt_dec_out = torch.randn(B, T, D, requires_grad=True)
    tgt_pad_mask = torch.ones(B, T, dtype=torch.bool)
    tgt_pad_mask[1, 3:] = False

    memory = torch.randn(B, L, D, requires_grad=True)
    memory_pad_mask = torch.ones(B, L, dtype=torch.bool)
    memory_pad_mask[1, 6:] = False
    memory_ids = torch.randint(3, V, (B, L))

    gen = CopyGenerator(tgt_dims=D, vocab_size=V, n_heads=n_heads)
    gen.eval()  # 概率和检查必须在eval模式下做：训练模式下attention内部的dropout会缩放
                # 注意力权重，导致复制分布不再精确加和为1(属预期行为，不是bug)
    probs, p_gen = gen(tgt_dec_out, tgt_pad_mask, memory, memory_pad_mask, memory_ids)

    print('probs shape:', probs.shape, '有无nan:', torch.isnan(probs).any().item())
    print('padding位置是否为0:', torch.allclose(probs[1, 3:], torch.zeros_like(probs[1, 3:])))
    # 真实(非padding)位置的概率分布应该合法地加和为1(允许浮点误差)
    real_sum = probs[0, :].sum(dim=-1)
    print('真实位置概率和应为1:', torch.allclose(real_sum, torch.ones_like(real_sum), atol=1e-4))
    print('p_gen范围应在[0,1]:', (p_gen >= 0).all().item() and (p_gen <= 1).all().item())

    gen.train()  # 恢复训练模式后再测梯度

    loss = -torch.log(probs + 1e-10).sum()
    loss.backward()
    print('梯度能正常回传:', tgt_dec_out.grad is not None and memory.grad is not None and
         not torch.isnan(tgt_dec_out.grad).any().item())