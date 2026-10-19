# coding=utf-8
"""
把HierEncoder和HierDecoder打包成一个完整模型。这一层负责：
  1. 训练时的teacher forcing设置：text_ids是[<s>, w1,...,wk, </e>, pad,...]，
     解码器输入用text_ids[:,:-1]、监督目标用text_ids[:,1:]
  2. loss计算：use_copy=True时用NLLLoss(padding位置的target本来就是pad_idx，会被
     自动忽略)；use_copy=False时用CrossEntropyLoss。

【重要】decoder的cross-attention memory直接用encoder的token_repr(不再拼接语句/
block/函数级表示)——之前尝试过把四级表示拼接成"多粒度memory"喂给decoder，用真实
数据做了消融实验，发现明显拖累效果(详见encoder.py顶部的说明)，回退成更简单、
已验证有效的单一token级memory。这样copy机制也不再需要单独维护一份
copy_memory(直接和标准cross-attention共用同一份token_repr，decoder.py里
copy_memory参数不传时会自动退化成用memory本身，语义上完全一致)。
"""
import torch
import torch.nn as nn

from encoder import HierEncoder, EmbedOnlyEncoder, PlainEncoder, make_pad_mask
from decoder import HierDecoder


class HierSeq2Seq(nn.Module):
    def __init__(self, vocab_size, stmt_type_vocab_size, block_type_vocab_size,
                max_code_len, max_text_len, stmt_max_num, block_max_num, nest_depth_max,
                emb_dims=512, enc_local_heads=8, enc_mid_heads=8, enc_global_heads=8,
                enc_fusion_heads=8, dec_heads=8, dec_layers=6, drop_rate=0.1, pad_idx=0,
                bos_id=None, eos_id=None, tie_dec_weights=True, use_fusion=True,
                use_copy=False, hier_layers=1, use_struct_pos=False, use_parent_emb=False,
                use_token_global=False, dec_norm_first=False, scale_emb=False,
                encoder_type='hier', share_emb=False, dec_emb_ln=False,
                plain_layers=6, plain_heads=8, drop_levels=(), fusion_gate=True):
        super().__init__()
        self.pad_idx = pad_idx
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.use_copy = use_copy
        self.max_text_len = max_text_len  # 原始text_ids的padding长度(含<s>/</e>)

        if encoder_type == 'embed':
            # 诊断选项：编码器退化成纯嵌入(无位置、无结构)，用来判断问题出在编码器还是共有部分
            self.encoder = EmbedOnlyEncoder(vocab_size, emb_dims=emb_dims, drop_rate=drop_rate,
                                            pad_idx=pad_idx, scale_emb=scale_emb)
        elif encoder_type == 'plain':
            # 论文对照：普通Transformer编码器(全局自注意力，无层次结构)
            self.encoder = PlainEncoder(vocab_size, max_code_len, emb_dims=emb_dims, n_heads=plain_heads,
                                        n_layers=plain_layers, drop_rate=drop_rate, pad_idx=pad_idx,
                                        scale_emb=scale_emb)
        else:
            self.encoder = HierEncoder(
                vocab_size, stmt_type_vocab_size, block_type_vocab_size,
                max_code_len, stmt_max_num, block_max_num, nest_depth_max,
                emb_dims=emb_dims, local_heads=enc_local_heads, mid_heads=enc_mid_heads,
                global_heads=enc_global_heads, fusion_heads=enc_fusion_heads,
                drop_rate=drop_rate, pad_idx=pad_idx, use_fusion=use_fusion,
                hier_layers=hier_layers, use_struct_pos=use_struct_pos,
                use_parent_emb=use_parent_emb, use_token_global=use_token_global,
                scale_emb=scale_emb, drop_levels=drop_levels, fusion_gate=fusion_gate)

        # decoder自己的max_text_len是"输入序列的长度"，即原始max_text_len砍掉最后一位
        self.decoder = HierDecoder(
            vocab_size, max_text_len - 1, emb_dims=emb_dims, n_heads=dec_heads,
            n_layers=dec_layers, drop_rate=drop_rate, pad_idx=pad_idx,
            tie_weights=tie_dec_weights, use_copy=use_copy, norm_first=dec_norm_first,
            scale_emb=scale_emb, emb_layernorm=dec_emb_ln)

        # share_emb=True：编码器和解码器共用同一个词嵌入(HetCoS官方就是这样：io_token_emb_op同时
        # 传给编码器和解码器)。让编码器采用解码器的嵌入模块，而不是反过来，这样解码器里
        # "嵌入与输出层权重共享"的绑定关系保持不变。两边词表是同一个，维度也一样，所以可以直接共用。
        # 好处：同一个词在编码器侧和解码器侧的向量相同，cross-attention/复制一开始就有可用的对齐信号。
        if share_emb:
            self.encoder.token_emb = self.decoder.token_emb

        if use_copy:
            self.loss_fn = nn.NLLLoss(ignore_index=pad_idx)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, batch):
        """
        训练用。batch字段要求同HierEncoder + 额外的text_ids[B,max_text_len]、text_len[B]。
        返回dict: loss(标量), logits(use_copy=True时其实是概率分布)[B,max_text_len-1,vocab_size],
                 enc_out(HierEncoder的完整输出)
        """
        enc_out = self.encoder(batch)
        memory = enc_out['token_repr']
        memory_pad_mask = enc_out['token_pad_mask']

        text_ids = batch['text_ids'].long()
        text_len = batch['text_len']

        dec_input = text_ids[:, :-1]
        target = text_ids[:, 1:]
        T_dec = dec_input.shape[1]
        dec_pad_mask = make_pad_mask((text_len - 1).clamp(min=0), T_dec)

        memory_ids = batch['code_ids'].long() if self.use_copy else None
        out = self.decoder(dec_input, dec_pad_mask, memory, memory_pad_mask,
                          memory_ids=memory_ids)

        if self.use_copy:
            # 注意：这里必须用 log(out + eps)，不能用 log(out.clamp(min=eps))——
            # clamp在被裁剪的位置梯度直接是0(硬阈值截断)，而加法在任何取值下梯度都是
            # 1/(out+eps)，恒不为0。真实词表规模下(2万+)大量概率值会小于1e-10，
            # 如果用clamp，这些位置的训练信号会被完全掐断，这是实测复现过的真实bug，
            # 不是理论上的小概率事件。
            log_probs = torch.log(out + 1e-10)
            loss = self.loss_fn(log_probs.reshape(-1, log_probs.shape[-1]), target.reshape(-1))
        else:
            loss = self.loss_fn(out.reshape(-1, out.shape[-1]), target.reshape(-1))
        return dict(loss=loss, logits=out, enc_out=enc_out)

    @torch.no_grad()
    def generate(self, batch, beam_width=1, length_penalty=1.0):
        """
        推理用。beam_width=1时用贪心解码(快，适合训练过程中每个epoch的验证)，
        >1时用beam search(慢但通常质量更好，适合最终测试集评估)。
        返回: [B, <=max_text_len-1] 的token id序列
        """
        assert self.bos_id is not None and self.eos_id is not None, \
            'generate需要在构造HierSeq2Seq时提供bos_id/eos_id'
        enc_out = self.encoder(batch)
        memory = enc_out['token_repr']
        memory_pad_mask = enc_out['token_pad_mask']

        memory_ids = batch['code_ids'].long() if self.use_copy else None
        kwargs = dict(memory=memory, memory_pad_mask=memory_pad_mask,
                     bos_id=self.bos_id, eos_id=self.eos_id,
                     max_len=self.decoder.max_text_len, memory_ids=memory_ids)
        if beam_width > 1:
            return self.decoder.beam_search(beam_width=beam_width,
                                            length_penalty=length_penalty, **kwargs)
        return self.decoder.greedy_decode(**kwargs)


if __name__ == '__main__':
    torch.manual_seed(0)
    B = 4
    max_code_len, stmt_max_num, block_max_num, nest_depth_max = 20, 6, 4, 8
    max_text_len = 10
    vocab_size, stmt_type_vocab_size, block_type_vocab_size = 60, 12, 10
    PAD, BOS, EOS = 0, 1, 2

    code_len = torch.tensor([20, 15, 8, 20])
    stmt_num = torch.tensor([6, 4, 2, 6])
    block_num = torch.tensor([4, 3, 1, 4])
    text_len = torch.tensor([10, 6, 4, 8])

    code_ids = torch.randint(3, vocab_size, (B, max_code_len))
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

    text_ids = torch.zeros(B, max_text_len, dtype=torch.long)
    for b in range(B):
        L_t = text_len[b].item()
        text_ids[b, 0] = BOS
        if L_t > 2:
            text_ids[b, 1:L_t - 1] = torch.randint(3, vocab_size, (L_t - 2,))
        text_ids[b, L_t - 1] = EOS

    batch = dict(code_ids=code_ids, token2stmt=token2stmt, stmt2block=stmt2block,
                stmt_type_ids=stmt_type_ids, block_type_ids=block_type_ids,
                block_nest_depth=block_nest_depth, code_len=code_len, stmt_num=stmt_num,
                block_num=block_num, text_ids=text_ids, text_len=text_len)

    for use_fusion in [True, False]:
        for use_copy in [False, True]:
            print('=' * 20, f'use_fusion={use_fusion} use_copy={use_copy}', '=' * 20)
            model = HierSeq2Seq(vocab_size, stmt_type_vocab_size, block_type_vocab_size,
                                max_code_len, max_text_len, stmt_max_num, block_max_num,
                                nest_depth_max, emb_dims=32, enc_local_heads=4, enc_mid_heads=4,
                                enc_global_heads=4, enc_fusion_heads=4, dec_heads=4, dec_layers=2,
                                drop_rate=0.0, pad_idx=PAD, bos_id=BOS, eos_id=EOS,
                                use_fusion=use_fusion, use_copy=use_copy, hier_layers=1)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            losses = []
            for step in range(30):
                opt.zero_grad()
                out = model(batch)
                out['loss'].backward()
                opt.step()
                losses.append(out['loss'].item())
            print('  前3步loss:', [round(l, 3) for l in losses[:3]],
                 '后3步loss:', [round(l, 3) for l in losses[-3:]])
            print('  loss明显下降:', losses[-1] < losses[0] * 0.5)

            model.eval()
            g = model.generate(batch, beam_width=1)
            b = model.generate(batch, beam_width=3)
            print('  greedy shape:', g.shape, 'beam shape:', b.shape)
    print('\n' + '=' * 20, 'v2新开关组合(结构位置/父block/全局token/Pre-LN解码器)', '=' * 20)
    block_parent = torch.full((B, block_max_num), -1, dtype=torch.long)
    for b in range(B):
        for k in range(2, block_num[b].item()):
            block_parent[b, k] = 1
    batch_v2 = dict(batch, block_parent=block_parent)
    for use_copy in [False, True]:
        model = HierSeq2Seq(vocab_size, stmt_type_vocab_size, block_type_vocab_size,
                            max_code_len, max_text_len, stmt_max_num, block_max_num,
                            nest_depth_max, emb_dims=32, enc_local_heads=4, enc_mid_heads=4,
                            enc_global_heads=4, enc_fusion_heads=4, dec_heads=4, dec_layers=2,
                            drop_rate=0.0, pad_idx=PAD, bos_id=BOS, eos_id=EOS,
                            use_fusion=True, use_copy=use_copy, hier_layers=1,
                            use_struct_pos=True, use_parent_emb=True, use_token_global=True,
                            dec_norm_first=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for step in range(30):
            opt.zero_grad()
            out = model(batch_v2)
            out['loss'].backward()
            opt.step()
            losses.append(out['loss'].item())
        print('  use_copy={} 前3步loss:'.format(use_copy), [round(l, 3) for l in losses[:3]],
             '后3步loss:', [round(l, 3) for l in losses[-3:]], '明显下降:', losses[-1] < losses[0] * 0.5)
        model.eval()
        g = model.generate(batch_v2, beam_width=1)
        bm = model.generate(batch_v2, beam_width=3)
        print('  greedy shape:', g.shape, 'beam shape:', bm.shape)