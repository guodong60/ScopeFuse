# coding=utf-8
"""
HierEncoder：token embedding到多粒度表示的完整编码流程。

【重要】这一版是"回退版"：之前尝试过"自底向上抽象+自顶向下逐级回注"的三步级联
(block先融合func、stmt再融合refine后的block、token最后融合refine后的[stmt,block])，
用真实数据做了完整的消融实验(hier_v2 vs hier_v3_h1，两者层数都是1层，唯一区别就是
要不要这套级联机制)，结果是：级联版本S-BLEU/ROUGE-L都明显低于一次性融合版本
(43.13→41.17，56.75→55.19)，且级联版本的train_loss/valid_loss差距远大于一次性版本，
说明级联引入的额外参数量和更深的信息传递链路，在本数据集规模下没有换来更好的泛化，
反而加剧了过拟合。因此改回更简单的设计：

    token embedding
      |
      v  (自底向上，逐级抽象，和之前一样)
    ① 语句内local attention -> pooling+type_emb -> 语句向量
      |
      v
    ② block内语句间attention -> pooling+type/depth_emb -> block向量
      |
      v
    ③ block间attention(函数级) -> pooling -> 函数级向量
      |
      v  (一次性回注，不再逐级级联refine)
    ④ token表示 一次性 对[语句向量,block向量,函数级向量]三路做注意力打分+门控融合

hier_layers参数保留(可以堆叠多层)，但默认改成1——已有证据表明堆叠更深层数在本任务上
收益为负，1层是目前验证过的最优配置。保留堆叠能力只是为了以后想做进一步消融时不用
重新写代码，不代表推荐使用。

【v2新增，三个独立开关，默认全关=和上面描述的旧版完全一致，旧checkpoint可直接加载】
  use_struct_pos   : 补齐结构位置信息。旧版只有token级绝对位置，语句/block向量只加了类型/深度
                     嵌入，语句间attention(mid_attn)和block间attention(global_attn)对输入顺序
                     基本不敏感。打开后额外加：
                       · 语句内位置嵌入(token是所在语句的第几个token)，加到token embedding上
                       · 语句序号嵌入(语句是全函数第几条)，加到语句向量上
                       · block序号嵌入(block是第几个创建的)，加到block向量上
                     block按先序创建，序号天然对应"在源码中出现的先后"
  use_parent_emb   : 给block向量加"父block类型"嵌入，区分结构位置(例如try里的if和catch里的if，
                     旧版两者类型、深度完全相同)。需要数据里有block_parent字段(重新预处理生成)
  use_token_global : 三路融合之后，再对token做一层不限scope的全局自注意力。旧版token只在
                     语句内互相可见，跨语句信息只能经3个池化向量传入；打开后解码器看到的
                     memory带有全局token交互。会改变"纯层次"的叙事，需要靠消融确定是否保留
"""
import torch
import torch.nn as nn

from layers import ScopedSelfAttention, SegmentPooling, GlobalAttnPool
from fusion import GranularityFusion


def make_pad_mask(lengths, max_len):
    """和data/dataset.py里的同名函数一致(避免encoder.py反向依赖data模块，这里独立放一份，
    两边实现必须保持一致——如果以后改动，两处都要同步改)"""
    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths)
    batch_size = lengths.shape[0]
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, max_len)
    return idx < lengths.unsqueeze(1)


def _gather_by_index(src, index):
    """src:[B,S,D], index:[B,N] -> [B,N,D]，按index把src的第1维gather到N个位置。
    用于把语句级/block级的表示"对齐"到token数量，供GranularityFusion当context用。
    """
    D = src.shape[-1]
    return torch.gather(src, 1, index.unsqueeze(-1).expand(-1, -1, D))


class HierBlock(nn.Module):
    """一层完整的"自底向上抽象 + token一次性三路回注"。"""

    def __init__(self, emb_dims, local_heads, mid_heads, global_heads, fusion_heads,
                drop_rate, use_fusion=True, use_token_global=False, drop_levels=(), fusion_gate=True):
        super().__init__()
        self.drop_levels = tuple(drop_levels)   # 消融：哪些级别不回注给token，取值stmt/block/func
        self.use_fusion = use_fusion
        self.use_token_global = use_token_global

        # 自底向上
        self.local_attn = ScopedSelfAttention(emb_dims, local_heads, drop_rate=drop_rate)
        self.stmt_pool = SegmentPooling(emb_dims)
        self.mid_attn = ScopedSelfAttention(emb_dims, mid_heads, drop_rate=drop_rate)
        self.block_pool = SegmentPooling(emb_dims)
        self.global_attn = ScopedSelfAttention(emb_dims, global_heads, drop_rate=drop_rate)
        self.func_pool = GlobalAttnPool(emb_dims)

        # 回注：token一次性对[语句,block,函数]三路(K=3)做注意力打分+门控融合
        if use_fusion:
            self.token_fusion = GranularityFusion(emb_dims, fusion_heads, drop_rate=drop_rate, use_gate=fusion_gate)
        else:
            # 消融基线：不做注意力+门控，直接简单相加，对应"不加跨粒度注意力"的设置
            self.token_norm = nn.LayerNorm(emb_dims)

        # 可选：融合之后再做一层全局token自注意力(seg_ids=None即不限scope)
        if use_token_global:
            self.token_global = ScopedSelfAttention(emb_dims, global_heads, drop_rate=drop_rate)

    def forward(self, token_repr, token_pad_mask, token2stmt, stmt_type_emb, stmt_pad_mask,
               stmt2block, block_type_nest_emb, block_pad_mask):
        """
        token_repr: [B,L,D] 本层的输入token表示(第一层是embedding，之后是上一层输出，
                    如果hier_layers>1的话)
        stmt_type_emb: [B,S,D] 语句类型embedding(每层重新加一次，结构信息不随层数变化)
        block_type_nest_emb: [B,K,D] block类型+嵌套深度embedding(同理每层重新加)
        其余参数含义同HierEncoder.forward

        返回: (token_repr_fused, stmt_repr, block_repr, func_vec)
        """
        S = stmt_type_emb.shape[1]
        K = block_type_nest_emb.shape[1]

        # ---- 自底向上 ----
        h1 = self.local_attn(token_repr, token_pad_mask, seg_ids=token2stmt)

        stmt_vec = self.stmt_pool(h1, token_pad_mask, token2stmt, S, stmt_pad_mask)
        stmt_vec = (stmt_vec + stmt_type_emb) * stmt_pad_mask.unsqueeze(-1).to(stmt_vec.dtype)
        stmt_vec1 = self.mid_attn(stmt_vec, stmt_pad_mask, seg_ids=stmt2block)

        block_vec = self.block_pool(stmt_vec1, stmt_pad_mask, stmt2block, K, block_pad_mask)
        block_vec = (block_vec + block_type_nest_emb) * block_pad_mask.unsqueeze(-1).to(block_vec.dtype)
        block_vec1 = self.global_attn(block_vec, block_pad_mask, seg_ids=None)

        func_vec = self.func_pool(block_vec1, block_pad_mask)  # [B,D]

        # ---- token一次性对三路做回注(不再逐级级联refine) ----
        stmt_of_token = _gather_by_index(stmt_vec1, token2stmt)  # [B,L,D]
        token2block = torch.gather(stmt2block, 1, token2stmt)  # [B,L]
        block_of_token = _gather_by_index(block_vec1, token2block)  # [B,L,D]
        func_broadcast = func_vec.unsqueeze(1).expand(-1, h1.shape[1], -1)  # [B,L,D]

        # 消融：drop_levels里的级别不回注给token(自底向上的计算链不变，只是该级表示不进融合)
        ctx = [t for name, t in (('stmt', stmt_of_token), ('block', block_of_token),
                                 ('func', func_broadcast)) if name not in self.drop_levels]
        if len(ctx) == 0:
            token_vec2 = h1          # 三级都不回注：只剩语句内局部注意力
        elif self.use_fusion:
            token_vec2, _ = self.token_fusion(h1, token_pad_mask, ctx)
        else:
            token_vec2 = self.token_norm(h1 + sum(ctx))
            token_vec2 = token_vec2 * token_pad_mask.unsqueeze(-1).to(token_vec2.dtype)

        if self.use_token_global:
            token_vec2 = self.token_global(token_vec2, token_pad_mask, seg_ids=None)

        return token_vec2, stmt_vec1, block_vec1, func_vec


class PlainEncoder(nn.Module):
    """普通Transformer编码器基线(论文对照用)：token嵌入 + 位置嵌入 -> N层标准Transformer编码层。

    和HierEncoder的区别只有一处：这里每个token对全部token做全局自注意力，没有语句/block/函数
    三级结构，也没有按语句限制可见范围。其余尽量保持一致：同样的嵌入初始化、同样的scale_emb
    选项、同样的dropout、同样输出token_repr/token_pad_mask给同一个解码器和复制机制。
    位置编码用学习的绝对位置(和HierEncoder的token级位置嵌入一致)。
    层数建议：4层(参数量约12.8M，最接近层次编码器的约11.3M)和6层(标准配置，约19M)各做一次。
    """
    def __init__(self, vocab_size, max_code_len, emb_dims=512, n_heads=8, n_layers=6,
                 ff_hid_dims=None, drop_rate=0.1, pad_idx=0, scale_emb=False):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_code_len = max_code_len
        self.emb_scale = emb_dims ** 0.5 if scale_emb else 1.0
        self.token_emb = nn.Embedding(vocab_size, emb_dims, padding_idx=pad_idx)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=emb_dims ** -0.5)
        with torch.no_grad():
            self.token_emb.weight[pad_idx].zero_()
        self.pos_emb = nn.Embedding(max_code_len, emb_dims)
        self.emb_dropout = nn.Dropout(drop_rate)
        layer = nn.TransformerEncoderLayer(d_model=emb_dims, nhead=n_heads,
                                           dim_feedforward=ff_hid_dims or emb_dims * 4,
                                           dropout=drop_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, batch):
        code_ids = batch['code_ids'].long()
        B, L = code_ids.shape
        token_pad_mask = make_pad_mask(batch['code_len'], L)           # True=真实token
        positions = torch.arange(L, device=code_ids.device).unsqueeze(0)
        h = self.emb_dropout(self.token_emb(code_ids) * self.emb_scale + self.pos_emb(positions))
        h = self.encoder(h, src_key_padding_mask=~token_pad_mask)
        h = h * token_pad_mask.unsqueeze(-1).to(h.dtype)
        return dict(token_repr=h, token_pad_mask=token_pad_mask)


class EmbedOnlyEncoder(nn.Module):
    """诊断用：不做任何编码，memory就是token嵌入本身(没有位置、没有结构)。

    对应\"HetCoS只保留解码器\"的对照：官方的解码器直接对节点嵌入做cross-attention，
    不经过GNN也能达到接近完整模型的分数。拿它在你自己的流水线里跑一遍，可以判断
    \"分数低\"到底是编码器的问题，还是解码器/词表/训练方式等共有部分的问题。
    输出字段和HierEncoder中seq2seq用到的两个字段保持一致：token_repr、token_pad_mask。
    """
    def __init__(self, vocab_size, emb_dims=512, drop_rate=0.1, pad_idx=0, scale_emb=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dims, padding_idx=pad_idx)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=emb_dims ** -0.5)
        with torch.no_grad():
            self.token_emb.weight[pad_idx].zero_()
        self.emb_scale = emb_dims ** 0.5 if scale_emb else 1.0
        self.emb_dropout = nn.Dropout(drop_rate)

    def forward(self, batch):
        code_ids = batch['code_ids'].long()
        L = code_ids.shape[1]
        token_pad_mask = make_pad_mask(batch['code_len'], L)
        h = self.emb_dropout(self.token_emb(code_ids) * self.emb_scale)
        h = h * token_pad_mask.unsqueeze(-1).to(h.dtype)
        return dict(token_repr=h, token_pad_mask=token_pad_mask)


class HierEncoder(nn.Module):
    def __init__(self, vocab_size, stmt_type_vocab_size, block_type_vocab_size,
                max_code_len, stmt_max_num, block_max_num, nest_depth_max,
                emb_dims=512, local_heads=8, mid_heads=8, global_heads=8, fusion_heads=8,
                drop_rate=0.1, pad_idx=0, use_fusion=True, hier_layers=1,
                use_struct_pos=False, use_parent_emb=False, use_token_global=False,
                intra_pos_max=64, scale_emb=False, drop_levels=(), fusion_gate=True):
        super().__init__()
        self.max_code_len = max_code_len
        self.stmt_max_num = stmt_max_num
        self.block_max_num = block_max_num
        self.nest_depth_max = nest_depth_max
        self.use_fusion = use_fusion
        self.hier_layers = hier_layers
        self.use_struct_pos = use_struct_pos
        self.use_parent_emb = use_parent_emb
        self.use_token_global = use_token_global
        self.intra_pos_max = intra_pos_max
        # scale_emb=True：前向里把token嵌入乘sqrt(d)(原版Transformer做法)。
        # 原因：token_emb为了和输出层共享权重初始化成std=d^-0.5(约0.044)，而pos_emb是PyTorch默认的std=1，
        # 两者初始尺度差约22.7倍，输入几乎被随机位置向量淹没；乘sqrt(d)后两者同量级。默认False=旧版行为
        self.emb_scale = emb_dims ** 0.5 if scale_emb else 1.0

        self.token_emb = nn.Embedding(vocab_size, emb_dims, padding_idx=pad_idx)
        # 和decoder.py的token_emb用同样的初始化策略，理由见那边的详细注释：
        # nn.Embedding默认std=1在emb_dims较大时会产生过大数值尺度，这里统一收敛到O(1)量级
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=emb_dims ** -0.5)
        with torch.no_grad():
            self.token_emb.weight[pad_idx].zero_()
        self.pos_emb = nn.Embedding(max_code_len, emb_dims)
        self.stmt_type_emb = nn.Embedding(stmt_type_vocab_size, emb_dims, padding_idx=pad_idx)
        self.block_type_emb = nn.Embedding(block_type_vocab_size, emb_dims, padding_idx=pad_idx)
        self.nest_depth_emb = nn.Embedding(nest_depth_max, emb_dims)
        self.emb_dropout = nn.Dropout(drop_rate)

        if use_struct_pos:
            self.intra_pos_emb = nn.Embedding(intra_pos_max, emb_dims)   # 语句内第几个token
            self.stmt_pos_emb = nn.Embedding(stmt_max_num, emb_dims)     # 第几条语句
            self.block_pos_emb = nn.Embedding(block_max_num, emb_dims)   # 第几个block
        if use_parent_emb:
            # 父block类型；padding_idx=pad_idx对应"没有父block"(顶层/签名block)，嵌入恒为0
            self.parent_type_emb = nn.Embedding(block_type_vocab_size, emb_dims, padding_idx=pad_idx)

        self.blocks = nn.ModuleList([
            HierBlock(emb_dims, local_heads, mid_heads, global_heads, fusion_heads,
                     drop_rate, use_fusion=use_fusion, use_token_global=use_token_global,
                     drop_levels=drop_levels, fusion_gate=fusion_gate)
            for _ in range(hier_layers)
        ])

    def forward(self, batch, need_fusion_weights=False):
        """
        batch: dict，至少包含以下字段(和dataset.py里CodeSummDataset吐出来的一致):
            code_ids [B,L], token2stmt [B,L], stmt_type_ids [B,S], stmt2block [B,S],
            block_nest_depth [B,K], block_type_ids [B,K],
            code_len [B], stmt_num [B], block_num [B]
        返回dict: token_repr(三路融合回注后的token表示)、stmt_repr、block_repr、func_repr
                 (语句/block向量是自底向上pooling出来的原始表示，供decoder做cross-attention
                 或消融分析用)，以及三级的pad_mask
        """
        code_ids = batch['code_ids'].long()
        token2stmt = batch['token2stmt'].long()
        stmt2block = batch['stmt2block'].long()
        stmt_type_ids = batch['stmt_type_ids'].long()
        block_type_ids = batch['block_type_ids'].long()
        block_nest_depth = batch['block_nest_depth'].long().clamp(max=self.nest_depth_max - 1)
        if self.use_parent_emb and 'block_parent' not in batch:
            raise KeyError('use_parent_emb=True需要数据里有block_parent字段，请用新版'
                           'build_dataset.py/run_preprocess.py重新生成数据')

        B, L = code_ids.shape
        device = code_ids.device
        token_pad_mask = make_pad_mask(batch['code_len'], L)
        stmt_pad_mask = make_pad_mask(batch['stmt_num'], self.stmt_max_num)
        block_pad_mask = make_pad_mask(batch['block_num'], self.block_max_num)

        # ---- token embedding(词+位置)，只算一次 ----
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        h = self.token_emb(code_ids) * self.emb_scale + self.pos_emb(positions)
        if self.use_struct_pos:
            # 语句内位置：token2stmt在一条语句内是连续递增的，语句起点="最近一次编号变化的位置"。
            # is_new标记每条语句的第一个token，cummax把起点位置向后传播，idx-start即语句内偏移
            is_new = torch.ones_like(token2stmt, dtype=torch.bool)
            is_new[:, 1:] = token2stmt[:, 1:] != token2stmt[:, :-1]
            start = torch.cummax(torch.where(is_new, positions, torch.zeros_like(positions)), dim=1).values
            intra = (positions - start).clamp(max=self.intra_pos_max - 1)
            h = h + self.intra_pos_emb(intra)
        h = self.emb_dropout(h)
        h = h * token_pad_mask.unsqueeze(-1).to(h.dtype)

        # 结构embedding也只算一次，每层复用同一份(结构信息不随层数变化)
        stmt_type_emb = self.stmt_type_emb(stmt_type_ids)
        block_type_nest_emb = self.block_type_emb(block_type_ids) + self.nest_depth_emb(block_nest_depth)
        if self.use_struct_pos:
            S, K = stmt_type_ids.shape[1], block_type_ids.shape[1]
            stmt_type_emb = stmt_type_emb + self.stmt_pos_emb(torch.arange(S, device=device)).unsqueeze(0)
            block_type_nest_emb = block_type_nest_emb + \
                self.block_pos_emb(torch.arange(K, device=device)).unsqueeze(0)
        if self.use_parent_emb:
            parent = batch['block_parent'].long()                      # [B,K]，-1=没有父block
            has_parent = parent >= 0
            parent_type = torch.gather(block_type_ids, 1, parent.clamp(min=0))  # 父block的类型id
            parent_type = torch.where(has_parent, parent_type, torch.zeros_like(parent_type))
            block_type_nest_emb = block_type_nest_emb + self.parent_type_emb(parent_type)

        stmt_repr = block_repr = func_vec = None
        for block in self.blocks:
            h, stmt_repr, block_repr, func_vec = block(
                h, token_pad_mask, token2stmt, stmt_type_emb, stmt_pad_mask,
                stmt2block, block_type_nest_emb, block_pad_mask)

        return dict(token_repr=h, stmt_repr=stmt_repr, block_repr=block_repr,
                   func_repr=func_vec, token_pad_mask=token_pad_mask,
                   stmt_pad_mask=stmt_pad_mask, block_pad_mask=block_pad_mask)


if __name__ == '__main__':
    torch.manual_seed(0)
    B = 4
    max_code_len, stmt_max_num, block_max_num, nest_depth_max = 20, 6, 4, 8
    vocab_size, stmt_type_vocab_size, block_type_vocab_size = 100, 12, 10

    code_len = torch.tensor([20, 15, 8, 20])
    stmt_num = torch.tensor([6, 4, 2, 6])
    block_num = torch.tensor([4, 3, 1, 4])

    code_ids = torch.randint(1, vocab_size, (B, max_code_len))
    token2stmt = torch.zeros(B, max_code_len, dtype=torch.long)
    for b in range(B):
        L_b, S_b = code_len[b].item(), stmt_num[b].item()
        token2stmt[b, :L_b] = torch.linspace(0, S_b - 1e-3, L_b).long()
    stmt2block = torch.zeros(B, stmt_max_num, dtype=torch.long)
    for b in range(B):
        S_b, K_b = stmt_num[b].item(), block_num[b].item()
        stmt2block[b, :S_b] = torch.linspace(0, K_b - 1e-3, S_b).long()
    stmt_type_ids = torch.randint(1, stmt_type_vocab_size, (B, stmt_max_num))
    block_type_ids = torch.randint(1, block_type_vocab_size, (B, block_max_num))
    block_nest_depth = torch.randint(0, nest_depth_max, (B, block_max_num))

    batch = dict(code_ids=code_ids, token2stmt=token2stmt, stmt2block=stmt2block,
                stmt_type_ids=stmt_type_ids, block_type_ids=block_type_ids,
                block_nest_depth=block_nest_depth,
                code_len=code_len, stmt_num=stmt_num, block_num=block_num)

    for hier_layers in [1, 3]:
        print('=' * 20, 'hier_layers=', hier_layers, '=' * 20)
        encoder = HierEncoder(vocab_size, stmt_type_vocab_size, block_type_vocab_size,
                              max_code_len, stmt_max_num, block_max_num, nest_depth_max,
                              emb_dims=32, local_heads=4, mid_heads=4, global_heads=4,
                              fusion_heads=4, hier_layers=hier_layers)
        out = encoder(batch)
        print('token_repr:', out['token_repr'].shape, '有无nan:', torch.isnan(out['token_repr']).any().item())
        print('stmt_repr:', out['stmt_repr'].shape)
        print('block_repr:', out['block_repr'].shape)
        print('func_repr:', out['func_repr'].shape)
        print('第2条样本padding token位置是否为0:',
             torch.allclose(out['token_repr'][2, 8:], torch.zeros_like(out['token_repr'][2, 8:])))

        loss = out['token_repr'].sum() + out['stmt_repr'].sum() + out['block_repr'].sum() + out['func_repr'].sum()
        loss.backward()
        n_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
        n_total = sum(1 for p in encoder.parameters())
        print('有梯度的参数 / 总参数:', n_grad, '/', n_total)
        print('总参数量:', sum(p.numel() for p in encoder.parameters()))

    print('\n' + '=' * 20, 'use_fusion=False(消融基线)', '=' * 20)
    encoder_nf = HierEncoder(vocab_size, stmt_type_vocab_size, block_type_vocab_size,
                             max_code_len, stmt_max_num, block_max_num, nest_depth_max,
                             emb_dims=32, local_heads=4, mid_heads=4, global_heads=4,
                             use_fusion=False, hier_layers=2)
    out_nf = encoder_nf(batch)
    print('token_repr:', out_nf['token_repr'].shape, '有无nan:', torch.isnan(out_nf['token_repr']).any().item())
    loss_nf = out_nf['token_repr'].sum() + out_nf['func_repr'].sum()
    loss_nf.backward()
    n_grad_nf = sum(1 for p in encoder_nf.parameters() if p.grad is not None)
    n_total_nf = sum(1 for p in encoder_nf.parameters())
    print('有梯度的参数 / 总参数:', n_grad_nf, '/', n_total_nf)

    print('\n' + '=' * 20, 'v2新开关: use_struct_pos / use_parent_emb / use_token_global', '=' * 20)
    # 构造block_parent：block0(sig)/block1(top)无父；之后的block父亲指向block1
    block_parent = torch.full((B, block_max_num), -1, dtype=torch.long)
    for b in range(B):
        for k in range(2, block_num[b].item()):
            block_parent[b, k] = 1
    batch_v2 = dict(batch, block_parent=block_parent)
    for flags in [dict(use_struct_pos=True), dict(use_parent_emb=True), dict(use_token_global=True),
                  dict(use_struct_pos=True, use_parent_emb=True, use_token_global=True)]:
        enc = HierEncoder(vocab_size, stmt_type_vocab_size, block_type_vocab_size,
                          max_code_len, stmt_max_num, block_max_num, nest_depth_max,
                          emb_dims=32, local_heads=4, mid_heads=4, global_heads=4,
                          fusion_heads=4, hier_layers=1, **flags)
        out = enc(batch_v2)
        loss = out['token_repr'].sum() + out['stmt_repr'].sum() + out['block_repr'].sum() + out['func_repr'].sum()
        loss.backward()
        n_grad = sum(1 for p in enc.parameters() if p.grad is not None)
        n_total = sum(1 for p in enc.parameters())
        pad_zero = torch.allclose(out['token_repr'][2, 8:], torch.zeros_like(out['token_repr'][2, 8:]))
        print(flags, '| nan:', torch.isnan(out['token_repr']).any().item(),
              '| padding为0:', pad_zero, '| 有梯度参数:', n_grad, '/', n_total)

    # 语句内位置的取值检查：手工构造一个token2stmt，核对intra偏移
    t2s = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2, 0]])
    pos = torch.arange(10).unsqueeze(0)
    is_new = torch.ones_like(t2s, dtype=torch.bool)
    is_new[:, 1:] = t2s[:, 1:] != t2s[:, :-1]
    start = torch.cummax(torch.where(is_new, pos, torch.zeros_like(pos)), dim=1).values
    print('语句内偏移(期望 0 1 2 0 1 0 1 2 3 0):', (pos - start)[0].tolist())

    # 旧开关全关时，缺少block_parent的batch仍能正常运行(向后兼容)
    enc_old = HierEncoder(vocab_size, stmt_type_vocab_size, block_type_vocab_size,
                          max_code_len, stmt_max_num, block_max_num, nest_depth_max,
                          emb_dims=32, local_heads=4, mid_heads=4, global_heads=4, fusion_heads=4)
    enc_old(batch)
    try:
        HierEncoder(vocab_size, stmt_type_vocab_size, block_type_vocab_size,
                    max_code_len, stmt_max_num, block_max_num, nest_depth_max,
                    emb_dims=32, local_heads=4, mid_heads=4, global_heads=4, fusion_heads=4,
                    use_parent_emb=True)(batch)
    except KeyError as e:
        print('use_parent_emb缺少block_parent时给出明确报错: OK')