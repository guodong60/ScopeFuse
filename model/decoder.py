"""
标准Transformer解码器，支持复制机制(use_copy)。

三个实现细节要留意：
  1. causal mask和padding mask统一用bool类型(True=不允许attend/是padding)，
     不要混用float(-inf)和bool——两者混用在新版PyTorch里会触发deprecation warning。
  2. 输出层的权重和输入词嵌入权重做了tie(共享同一份参数)，use_copy=True/False两条
     路径都做了(CopyGenerator的tie_embedding参数)——前提是decoder的词表和encoder
     共用同一套词表(vocab_builder.py默认shared=True)。
  3. **memory和copy_memory在接口上是两个东西**：接口保留了分别传入的能力(memory喂给
     cross-attention，copy_memory/copy_memory_pad_mask/memory_ids喂给复制机制)，因为复制
     机制只能从"真实的源码token"里复制词，语句/block/函数级的聚合向量不是词表里的词。
     但当前实际用法(见seq2seq.py)是：memory就是encoder的token_repr(纯token级)，不再拼接
     多粒度表示(消融实验显示拼接会拖累效果)，不传copy_memory，自动退化成直接用memory。
  4. norm_first=True时使用Pre-LN(LayerNorm放在子层之前)并在末尾加一个final LayerNorm。
     层数较深(默认8层)时Pre-LN的训练更稳定、对学习率不那么敏感；默认False保持旧版行为
     (Post-LN)，旧checkpoint可直接加载。
"""
import torch
import torch.nn as nn

from copy_generator import CopyGenerator


class HierDecoder(nn.Module):
    def __init__(self, vocab_size, max_text_len, emb_dims=512, n_heads=8, n_layers=6,
                ff_hid_dims=None, drop_rate=0.1, pad_idx=0, tie_weights=True, use_copy=False,
                norm_first=False, scale_emb=False, emb_layernorm=False):
        super().__init__()
        ff_hid_dims = ff_hid_dims or 4 * emb_dims
        self.max_text_len = max_text_len
        self.pad_idx = pad_idx
        self.use_copy = use_copy

        # scale_emb=True：前向里token嵌入乘sqrt(d)，理由见encoder.py(token嵌入std=d^-0.5而位置嵌入std=1，
        # 相差约22.7倍)。输出层共享的是未缩放的权重，所以不影响tie_weights。默认False=旧版行为
        self.emb_scale = emb_dims ** 0.5 if scale_emb else 1.0
        self.token_emb = nn.Embedding(vocab_size, emb_dims, padding_idx=pad_idx)
        # 重新初始化成更小的方差：这个embedding会被输出层权重共享(tie_weights)，
        # nn.Embedding默认初始化是N(0,1)(每个元素标准差=1)，两个标准差=1的emb_dims维
        # 向量做点积，方差≈emb_dims，emb_dims=512时logit标准差≈22.6——softmax在
        # 训练刚开始、模型还没学到任何东西的时候就会因为纯数值尺度问题变得接近one-hot
        # (不是学出来的，是巧合)，这是实测复现过的真实问题，不是理论假设。
        # 标准做法(BERT/GPT等)是把embedding初始化成std≈emb_dims^-0.5或固定小值(如0.02)，
        # 这里用emb_dims^-0.5，让tied输出层的logit标准差回到O(1)量级。
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=emb_dims ** -0.5)
        with torch.no_grad():
            self.token_emb.weight[pad_idx].zero_()  # 重新初始化会覆盖padding_idx行，需要手动清零
        self.pos_emb = nn.Embedding(max_text_len, emb_dims)
        # emb_layernorm=True：(token嵌入+位置嵌入)过dropout之后再接一层LayerNorm，和HetCoS官方解码器一致
        # (官方 Dec.forward: dropout(emb+pos) -> emb_layer_norm)。默认False=旧版
        self.emb_ln = nn.LayerNorm(emb_dims) if emb_layernorm else None
        self.emb_dropout = nn.Dropout(drop_rate)

        layer = nn.TransformerDecoderLayer(d_model=emb_dims, nhead=n_heads,
                                           dim_feedforward=ff_hid_dims, dropout=drop_rate,
                                           batch_first=True, norm_first=norm_first)
        # Pre-LN结构最后一层输出没有归一化，需要补一个final LayerNorm
        final_norm = nn.LayerNorm(emb_dims) if norm_first else None
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers, norm=final_norm)

        if use_copy:
            # 复制机制：输出的是"概率分布"而不是logits，训练时要配合NLLLoss(log(probs))，
            # 不能再用CrossEntropyLoss(它内部会再做一次softmax)
            self.copy_generator = CopyGenerator(
                tgt_dims=emb_dims, vocab_size=vocab_size, src_dims=emb_dims, n_heads=n_heads,
                drop_rate=drop_rate, pad_idx=pad_idx,
                tie_embedding=self.token_emb if tie_weights else None)
            self.out_proj = None
        else:
            self.copy_generator = None
            self.out_proj = nn.Linear(emb_dims, vocab_size, bias=False)
            if tie_weights:
                self.out_proj.weight = self.token_emb.weight

    def _embed(self, text_input_ids):
        B, T = text_input_ids.shape
        positions = torch.arange(T, device=text_input_ids.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(text_input_ids) * self.emb_scale + self.pos_emb(positions)
        h = self.emb_dropout(h)
        if self.emb_ln is not None:
            h = self.emb_ln(h)
        return h

    def forward(self, text_input_ids, text_input_pad_mask, memory, memory_pad_mask,
               copy_memory=None, copy_memory_pad_mask=None, memory_ids=None):
        """
        text_input_ids:      [B, T] 解码器输入
        text_input_pad_mask: [B, T] bool，True=真实数据
        memory:               [B, M, D] 喂给标准cross-attention的memory(可以是纯token级，
                              也可以是encoder拼好的多粒度[token;语句;block;函数]序列)
        memory_pad_mask:      [B, M] bool
        copy_memory:          [B, L, D] 复制机制专用的memory，必须是纯token级(每个位置
                              对应一个真实源码token)。不传则默认用memory(适合memory本来
                              就是纯token级的场景)
        copy_memory_pad_mask: [B, L] bool，不传则默认用memory_pad_mask
        memory_ids:           [B, L] long，copy_memory每个位置对应的词表id(即code_ids)。
                              use_copy=True时必须提供

        返回:
          use_copy=False: logits [B,T,vocab_size]（未softmax，配合CrossEntropyLoss）
          use_copy=True : probs  [B,T,vocab_size]（已是概率，配合NLLLoss(log(probs))）
        """
        T = text_input_ids.shape[1]
        device = text_input_ids.device
        h = self._embed(text_input_ids)

        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

        out = self.decoder(
            tgt=h, memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~text_input_pad_mask,
            memory_key_padding_mask=~memory_pad_mask,
        )

        if self.use_copy:
            assert memory_ids is not None, 'use_copy=True时forward必须传memory_ids(即code_ids)'
            cm = copy_memory if copy_memory is not None else memory
            cm_mask = copy_memory_pad_mask if copy_memory_pad_mask is not None else memory_pad_mask
            probs, _ = self.copy_generator(out, text_input_pad_mask, cm, cm_mask, memory_ids)
            return probs
        return self.out_proj(out)

    @torch.no_grad()
    def beam_search(self, memory, memory_pad_mask, bos_id, eos_id, beam_width=5,
                   length_penalty=1.0, max_len=None, copy_memory=None,
                   copy_memory_pad_mask=None, memory_ids=None):
        """
        Beam search解码。相比贪心解码每步只保留最优的一个词，这里每步保留beam_width条
        候选序列，最后返回累计对数概率(经长度惩罚归一化)最高的那条。

        实现要点:
          - 把[B, beam]两个维度合并成[B*beam]喂给forward，一次算完所有候选，不用循环
          - 已经生成过</s>的候选(finished)：强制只能继续接pad，且累计分数冻结不再变化
          - 长度惩罚: score = 累计logP / (长度^length_penalty)
          - 第一步特殊处理: 所有beam的初始状态都是[<s>]完全相同，只从单条序列展开出
            beam_width个不同的词，避免W条候选一开始就重复

        memory/memory_pad_mask/copy_memory/copy_memory_pad_mask/memory_ids: 同forward
        返回: [B, T] 最优序列(含开头的<s>)，遇到eos后的位置补pad_idx
        """
        max_len = max_len or self.max_text_len
        B, M, _ = memory.shape
        device = memory.device
        W = beam_width

        use_separate_copy_mem = copy_memory is not None
        cm_L = copy_memory.shape[1] if use_separate_copy_mem else None

        def _log_scores(scores):
            if self.use_copy:
                return torch.log(scores + 1e-10)  # 加法而非clamp，避免梯度死区(见seq2seq.py的说明)
            return torch.log_softmax(scores, dim=-1)

        def _expand_beam(t, extra_dim):
            """[B,*] -> [B*W,*]，把batch维展开成beam维"""
            shape = list(t.shape)
            t = t.unsqueeze(1).expand(B, W, *shape[1:]).reshape(B * W, *shape[1:])
            return t

        # ---- 第一步：从单条<s>展开出W个不同的候选 ----
        seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        pad_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        out = self.forward(seqs, pad_mask, memory, memory_pad_mask,
                           copy_memory=copy_memory, copy_memory_pad_mask=copy_memory_pad_mask,
                           memory_ids=memory_ids)
        log_probs = _log_scores(out[:, -1, :])  # [B, V]
        top_log_probs, top_ids = log_probs.topk(W, dim=-1)  # [B,W]

        seqs = torch.cat([
            seqs.unsqueeze(1).expand(B, W, 1),
            top_ids.unsqueeze(-1),
        ], dim=2).reshape(B * W, 2)  # [B*W, 2]
        cum_log_probs = top_log_probs.reshape(B * W)  # [B*W]
        finished = (top_ids.reshape(B * W) == eos_id)

        # memory等按beam展开: [B,...] -> [B*W,...]
        mem = _expand_beam(memory, 1)
        mem_pad = _expand_beam(memory_pad_mask, 1)
        cm = _expand_beam(copy_memory, 1) if use_separate_copy_mem else None
        cm_pad = _expand_beam(copy_memory_pad_mask, 1) if use_separate_copy_mem else None
        mem_ids = _expand_beam(memory_ids, 1) if memory_ids is not None else None

        # ---- 后续步骤 ----
        for step in range(2, max_len):
            if finished.all():
                break
            T = seqs.shape[1]
            pad_mask = torch.ones(B * W, T, dtype=torch.bool, device=device)
            out = self.forward(seqs, pad_mask, mem, mem_pad,
                               copy_memory=cm, copy_memory_pad_mask=cm_pad, memory_ids=mem_ids)
            log_probs = _log_scores(out[:, -1, :])  # [B*W, V]
            V = log_probs.shape[-1]

            if finished.any():
                frozen = torch.full_like(log_probs, float('-inf'))
                frozen[:, self.pad_idx] = 0.0
                log_probs = torch.where(finished.unsqueeze(-1), frozen, log_probs)

            cand_log_probs = cum_log_probs.unsqueeze(-1) + log_probs  # [B*W, V]
            cand_log_probs = cand_log_probs.reshape(B, W * V)

            seq_lens = (seqs != self.pad_idx).sum(dim=1).float()  # [B*W]
            new_lens = torch.where(finished, seq_lens, seq_lens + 1)  # [B*W]
            penalty = new_lens.pow(length_penalty).unsqueeze(-1).expand(B * W, V).reshape(B, W * V)
            scores = cand_log_probs / penalty.clamp(min=1e-6)

            top_scores, top_flat_ids = scores.topk(W, dim=-1)  # [B,W]
            beam_ids = top_flat_ids // V
            token_ids = top_flat_ids % V

            gather_idx = (torch.arange(B, device=device).unsqueeze(-1) * W + beam_ids).reshape(-1)
            seqs = seqs[gather_idx]
            cum_log_probs = cand_log_probs.reshape(B, W * V).gather(1, top_flat_ids).reshape(B * W)
            finished = finished[gather_idx]

            seqs = torch.cat([seqs, token_ids.reshape(B * W, 1)], dim=1)
            finished = finished | (token_ids.reshape(B * W) == eos_id)

        # ---- 收尾：每个样本选分数最高的那条beam ----
        seq_lens = (seqs != self.pad_idx).sum(dim=1).float().clamp(min=1)
        final_scores = (cum_log_probs / seq_lens.pow(length_penalty)).reshape(B, W)
        best_ids = final_scores.argmax(dim=-1)  # [B]
        best_idx = torch.arange(B, device=device) * W + best_ids
        return seqs[best_idx]  # [B, T]

    @torch.no_grad()
    def greedy_decode(self, memory, memory_pad_mask, bos_id, eos_id, max_len=None,
                     copy_memory=None, copy_memory_pad_mask=None, memory_ids=None):
        """
        贪心解码(每步选概率最高的词)。相比beam_search更快，适合训练过程中每个epoch的
        快速验证；最终测试集评估建议用beam_search取更好的结果。

        memory/memory_pad_mask/copy_memory/copy_memory_pad_mask/memory_ids: 同forward
        返回: generated_ids [B, T]，T<=max_len，遇到eos提前结束的位置后面补pad_idx
        """
        max_len = max_len or self.max_text_len
        B = memory.shape[0]
        device = memory.device

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            T = generated.shape[1]
            input_pad_mask = torch.ones(B, T, dtype=torch.bool, device=device)
            scores = self.forward(generated, input_pad_mask, memory, memory_pad_mask,
                                  copy_memory=copy_memory, copy_memory_pad_mask=copy_memory_pad_mask,
                                  memory_ids=memory_ids)
            next_token = scores[:, -1, :].argmax(dim=-1)  # [B]
            next_token = torch.where(finished, torch.full_like(next_token, self.pad_idx), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if finished.all():
                break
        return generated


if __name__ == '__main__':
    torch.manual_seed(0)
    B, L, T, D, vocab_size, max_text_len = 3, 10, 6, 32, 50, 12
    PAD, BOS, EOS = 0, 1, 2

    memory = torch.randn(B, L, D, requires_grad=True)
    memory_pad_mask = torch.ones(B, L, dtype=torch.bool)
    memory_pad_mask[2, 7:] = False  # 第3条样本只有7个真实token

    text_input_ids = torch.randint(3, vocab_size, (B, T))
    text_input_ids[:, 0] = BOS
    text_input_pad_mask = torch.ones(B, T, dtype=torch.bool)
    text_input_pad_mask[1, 4:] = False  # 第2条样本文本只有4个真实token

    decoder = HierDecoder(vocab_size, max_text_len, emb_dims=D, n_heads=4, n_layers=2)

    print('== forward(训练用，teacher forcing) ==')
    logits = decoder(text_input_ids, text_input_pad_mask, memory, memory_pad_mask)
    print('logits shape:', logits.shape, '有无nan:', torch.isnan(logits).any().item())

    print('\n== 权重共享检查 ==')
    print('out_proj.weight 和 token_emb.weight 是否共享同一份参数:',
         decoder.out_proj.weight is decoder.token_emb.weight)

    print('\n== causal mask有效性检查(改动未来token不应影响当前步的logits) ==')
    decoder.eval()
    logits1 = decoder(text_input_ids, text_input_pad_mask, memory, memory_pad_mask)
    text_input_ids2 = text_input_ids.clone()
    text_input_ids2[:, -1] = (text_input_ids2[:, -1] + 1) % vocab_size
    logits2 = decoder(text_input_ids2, text_input_pad_mask, memory, memory_pad_mask)
    unaffected_same = torch.allclose(logits1[:, :-1], logits2[:, :-1], atol=1e-5)
    print('前面位置的logits不受未来token影响:', unaffected_same)
    decoder.train()

    print('\n== 梯度回传检查 ==')
    loss = logits.sum()
    loss.backward()
    print('memory梯度正常:', memory.grad is not None and not torch.isnan(memory.grad).any().item())
    print('token_emb梯度正常:', decoder.token_emb.weight.grad is not None and
         not torch.isnan(decoder.token_emb.weight.grad).any().item())

    print('\n== greedy_decode(推理用) ==')
    memory_no_grad = memory.detach()
    generated = decoder.greedy_decode(memory_no_grad, memory_pad_mask, bos_id=BOS, eos_id=EOS, max_len=8)
    print('generated shape:', generated.shape)
    print('每条序列第一个token都是BOS:', (generated[:, 0] == BOS).all().item())

    print('\n== use_copy=True + 分离的copy_memory(多粒度memory场景) ==')
    dec2 = HierDecoder(vocab_size, max_text_len, emb_dims=D, n_heads=4, n_layers=2, use_copy=True)
    dec2.eval()
    # 模拟"多粒度memory"(比如L+S+K+1个位置)比copy_memory(纯token，L个位置)长的场景
    S, K = 3, 2
    multi_mem = torch.randn(B, L + S + K + 1, D)
    multi_mem_pad = torch.ones(B, L + S + K + 1, dtype=torch.bool)
    copy_mem = memory.detach()
    copy_mem_pad = memory_pad_mask
    mem_ids = torch.randint(3, vocab_size, (B, L))
    probs = dec2(text_input_ids, text_input_pad_mask, multi_mem, multi_mem_pad,
                copy_memory=copy_mem, copy_memory_pad_mask=copy_mem_pad, memory_ids=mem_ids)
    print('probs shape(应该是vocab_size，不受多粒度memory长度影响):', probs.shape)
    g2 = dec2.greedy_decode(multi_mem, multi_mem_pad, BOS, EOS, max_len=8,
                            copy_memory=copy_mem, copy_memory_pad_mask=copy_mem_pad, memory_ids=mem_ids)
    b2 = dec2.beam_search(multi_mem, multi_mem_pad, BOS, EOS, beam_width=3, max_len=8,
                          copy_memory=copy_mem, copy_memory_pad_mask=copy_mem_pad, memory_ids=mem_ids)
    print('greedy shape:', g2.shape, 'beam shape:', b2.shape)