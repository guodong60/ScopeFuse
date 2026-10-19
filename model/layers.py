# coding=utf-8
"""
层次Transformer的基础构件，设计成"可复用"——同一套模块在encoder.py里会被调用三次：
    token(同语句内) --ScopedSelfAttention--> ... --SegmentPooling--> 语句向量
    语句(同block内)  --ScopedSelfAttention--> ... --SegmentPooling--> block向量
    block(全体)       --ScopedSelfAttention(无scope限制)--> ... --GlobalAttnPool--> 函数向量

两个核心设计取舍:
  1. "局部attention"没有自己实现attention数学，而是复用nn.TransformerEncoderLayer，
     只是把attn_mask构造成"只允许同一segment内互相attend"——避免重复造轮子，
     且能直接享受TransformerEncoderLayer自带的FFN+残差+LayerNorm。
  2. pooling用"可学习query对同段内容做注意力加权"，而不是简单mean/max——
     不同token/语句对上级表示的贡献权重由内容决定，比如一条语句里的方法调用名
     应该比括号、逗号这类token贡献更大的权重，这是明确要写进方法论里的设计点。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScopedSelfAttention(nn.Module):
    """只在"同一个上级单位"内部做self-attention。用于:
      - 语句内local attention(token间, scope=token2stmt)
      - block内语句间attention(语句间, scope=stmt2block)
      - 也支持scope=None，退化成普通的全局self-attention(block间/函数级用这个)
    """

    def __init__(self, d_model, n_heads, ff_hid_dims=None, drop_rate=0.1):
        super().__init__()
        ff_hid_dims = ff_hid_dims or 4 * d_model
        self.n_heads = n_heads
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_hid_dims,
            dropout=drop_rate, batch_first=True)

    def forward(self, items, item_pad_mask, seg_ids=None):
        """
        items: [B, L, D]
        item_pad_mask: [B, L] bool，True=真实数据(不是padding)
        seg_ids: [B, L] long或None。给定时，只允许同一segment内的位置互相attend；
                 None时不做scope限制(普通全局self-attention，仍然遵守padding mask)
        返回: [B, L, D]，padding位置强制清零
        """
        B, L, _ = items.shape
        valid_pair = item_pad_mask.unsqueeze(1) & item_pad_mask.unsqueeze(2)  # [B,L,L]
        if seg_ids is not None:
            same_seg = seg_ids.unsqueeze(1) == seg_ids.unsqueeze(2)  # [B,L,L]
            allow = same_seg & valid_pair
        else:
            allow = valid_pair

        # 保证每一行至少能attend到自己，避免"整行全是-inf"导致softmax出nan
        # (padding行、或极端情况下某个非法segment行都可能出现这种情况)
        eye = torch.eye(L, device=items.device, dtype=torch.bool).unsqueeze(0)
        allow = allow | eye

        attn_mask = torch.zeros(B, L, L, device=items.device, dtype=items.dtype)
        attn_mask = attn_mask.masked_fill(~allow, float('-inf'))
        # nn.MultiheadAttention(batch_first=True)在传3D attn_mask时要求形状是
        # [B*n_heads, L, L]，且按"同一个batch样本的n_heads个头连续排列"展开
        attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)

        out = self.layer(items, src_mask=attn_mask)
        out = out * item_pad_mask.unsqueeze(-1).to(out.dtype)
        return out


class SegmentPooling(nn.Module):
    """把"下级单位"按segment分组，池化成"每个segment一个向量"。
    例: token按token2stmt分组 -> 每条语句一个向量；语句按stmt2block分组 -> 每个block一个向量。
    用可学习query(一个线性打分层)对同段内容做softmax加权，而不是简单mean/max。
    """

    def __init__(self, d_model):
        super().__init__()
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, items, item_pad_mask, seg_ids, num_segments, seg_pad_mask):
        """
        items: [B, L, D] 下级单位表示
        item_pad_mask: [B, L] bool
        seg_ids: [B, L] long，每个下级单位所属的上级segment编号(0..num_segments-1)
        num_segments: 上级segment的padding长度S(如stmt_max_num/block_max_num)
        seg_pad_mask: [B, S] bool，True=这个segment真实存在
        返回: pooled [B, S, D]，padding segment位置强制清零
        """
        B, L, D = items.shape
        S = num_segments
        scores = self.score_proj(items).squeeze(-1)  # [B, L]

        seg_range = torch.arange(S, device=items.device).view(1, S, 1)  # [1,S,1]
        assign = (seg_ids.unsqueeze(1) == seg_range) & item_pad_mask.unsqueeze(1)  # [B,S,L]

        masked_scores = scores.unsqueeze(1).masked_fill(~assign, float('-inf'))  # [B,S,L]
        # padding的segment槽位一个下级单位都分配不到，整行是-inf，softmax会出nan；
        # 这类"空行"先整体换成0再softmax，反正结果最后会被seg_pad_mask清零，不影响正确性
        empty_row = ~assign.any(dim=-1, keepdim=True)  # [B,S,1]
        masked_scores = torch.where(empty_row, torch.zeros_like(masked_scores), masked_scores)

        attn = torch.softmax(masked_scores, dim=-1)  # [B,S,L]
        pooled = torch.bmm(attn, items)  # [B,S,D]
        pooled = pooled * seg_pad_mask.unsqueeze(-1).to(pooled.dtype)
        return pooled


class GlobalAttnPool(nn.Module):
    """把一组item整体池化成"一个向量"(不分segment)，用于block级向量->函数级全局向量。
    同样用可学习query做注意力加权，而不是简单mean。"""

    def __init__(self, d_model):
        super().__init__()
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, items, item_pad_mask):
        """
        items: [B, L, D]
        item_pad_mask: [B, L] bool
        返回: [B, D]
        """
        scores = self.score_proj(items).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~item_pad_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)  # [B, L]
        pooled = torch.bmm(attn.unsqueeze(1), items).squeeze(1)  # [B, D]
        return pooled


if __name__ == '__main__':
    torch.manual_seed(0)
    B, L, D, n_heads = 3, 10, 16, 4
    items = torch.randn(B, L, D, requires_grad=True)
    item_pad_mask = torch.zeros(B, L, dtype=torch.bool)
    item_pad_mask[0, :7] = True
    item_pad_mask[1, :10] = True
    item_pad_mask[2, :3] = True
    seg_ids = torch.zeros(B, L, dtype=torch.long)
    seg_ids[0, :3] = 0; seg_ids[0, 3:7] = 1
    seg_ids[1, :4] = 0; seg_ids[1, 4:7] = 1; seg_ids[1, 7:10] = 2
    seg_ids[2, :3] = 0

    print('== ScopedSelfAttention ==')
    ssa = ScopedSelfAttention(d_model=D, n_heads=n_heads)
    out = ssa(items, item_pad_mask, seg_ids)
    print('输出shape:', out.shape, '有无nan:', torch.isnan(out).any().item())
    print('padding位置是否为0:', torch.allclose(out[0, 7:], torch.zeros_like(out[0, 7:])))
    out.sum().backward()
    print('梯度能正常回传:', items.grad is not None and not torch.isnan(items.grad).any().item())

    print('\n== SegmentPooling ==')
    items2 = torch.randn(B, L, D, requires_grad=True)
    num_segments = 3
    seg_pad_mask = torch.zeros(B, num_segments, dtype=torch.bool)
    seg_pad_mask[0, :2] = True
    seg_pad_mask[1, :3] = True
    seg_pad_mask[2, :1] = True
    pool = SegmentPooling(d_model=D)
    pooled = pool(items2, item_pad_mask, seg_ids, num_segments, seg_pad_mask)
    print('输出shape:', pooled.shape, '有无nan:', torch.isnan(pooled).any().item())
    print('padding segment是否为0:', torch.allclose(pooled[0, 2:], torch.zeros_like(pooled[0, 2:])))
    pooled.sum().backward()
    print('梯度能正常回传:', items2.grad is not None and not torch.isnan(items2.grad).any().item())

    print('\n== GlobalAttnPool ==')
    items3 = torch.randn(B, num_segments, D, requires_grad=True)
    gpool = GlobalAttnPool(d_model=D)
    g = gpool(items3, seg_pad_mask)
    print('输出shape:', g.shape, '有无nan:', torch.isnan(g).any().item())
    g.sum().backward()
    print('梯度能正常回传:', items3.grad is not None and not torch.isnan(items3.grad).any().item())