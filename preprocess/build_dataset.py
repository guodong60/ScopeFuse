# # coding=utf-8
# """
# 把前面几步(stmt_block_builder + code_tokenizer + vocab_builder)串起来，
# 对train/valid/test各自产出一份可以直接喂给PyTorch Dataset的numpy数组文件(.npz)。

# 核心设计：截断/padding时不能沿用之前HETCOS版本那种"clip"做法(把超出上限的语句id
# 硬夹到最后一个槽位，导致几条不同语义的语句被拼成"一条")——那是之前排查出来的一个真实
# bug。这里改用"配额耗尽就整体截断"的做法：

#   按token顺序扫描，遇到新语句就检查是否还有语句名额(stmt_max_num)、遇到新block就检查
#   是否还有block名额(block_max_num)、token数是否超过max_code_len，三个条件任意一个先
#   触发就停止——保留的语句/block集合按"第一次出现的顺序"重新映射成紧凑的0..K-1编号，
#   不会出现"多条语句被挤压进同一个槽位"的情况，只是老老实实丢弃超出预算的尾部内容。

# 用法：
#     python3 build_dataset.py --raw_data_dir ../raw_data --vocab_path ../w2i2w/vocab.pkl \
#         --out_dir ../avail_data --max_code_len 300 --max_text_len 50 \
#         --stmt_max_num 64 --block_max_num 24 --nest_depth_max 8
# """
# import argparse
# import json
# import os
# import pickle

# import numpy as np

# from stmt_block_builder import iter_build_stmt_blocks
# from code_tokenizer import subtokenize_code, tokenize_text
# from vocab_builder import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


# def load_json(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         return json.load(f)


# def load_vocab(path):
#     with open(path, 'rb') as f:
#         return pickle.load(f)


# # ---------- 核心：按配额截断+紧凑重编号 ----------
# def compact_truncate(subtokens, sub2stmt, stmt_infos, block_infos,
#                      max_code_len, stmt_max_num, block_max_num):
#     """
#     输入: code_tokenizer.subtokenize_code的输出(subtokens, sub2stmt) +
#          stmt_block_builder的(stmt_infos, block_infos)(下标即原始stmt_id/block_id)
#     输出: (kept_tokens, kept_sub2stmt_compact, kept_stmt_infos_compact, kept_block_infos_compact)
#          语句/block都已经按"第一次出现顺序"重新映射成0..K-1的紧凑编号，
#          超出任一配额(token数/语句数/block数)时直接截断，不做任何"挤压合并"。
#     """
#     kept_tokens, kept_sub2stmt = [], []
#     stmt_order, stmt_id_map = [], {}     # old_stmt_id -> new_compact_id
#     block_order, block_id_map = [], {}   # old_block_id -> new_compact_id

#     for tok, old_sid in zip(subtokens, sub2stmt):
#         if len(kept_tokens) >= max_code_len:
#             break
#         if old_sid not in stmt_id_map:
#             old_bid = stmt_infos[old_sid]['block_id']
#             if old_bid not in block_id_map:
#                 if len(block_order) >= block_max_num:
#                     break
#                 block_id_map[old_bid] = len(block_order)
#                 block_order.append(old_bid)
#             if len(stmt_order) >= stmt_max_num:
#                 break
#             stmt_id_map[old_sid] = len(stmt_order)
#             stmt_order.append(old_sid)
#         kept_tokens.append(tok)
#         kept_sub2stmt.append(stmt_id_map[old_sid])

#     kept_stmt_infos = []
#     for old_sid in stmt_order:
#         info = stmt_infos[old_sid]
#         kept_stmt_infos.append(dict(block_id=block_id_map[info['block_id']],
#                                     stmt_type=info['stmt_type']))
#     kept_block_infos = []
#     for old_bid in block_order:
#         info = block_infos[old_bid]
#         # 父block一定在子block之前首次出现(控制结构的头部语句属于父block，且先于body里的语句)，
#         # 所以截断(只丢尾部)之后父block仍在block_id_map里；万一不在或没有父block，记-1
#         old_parent = info.get('parent')
#         new_parent = block_id_map.get(old_parent, -1) if old_parent is not None else -1
#         kept_block_infos.append(dict(nest_depth=info['nest_depth'], block_type=info['block_type'],
#                                      parent=new_parent))

#     return kept_tokens, kept_sub2stmt, kept_stmt_infos, kept_block_infos


# # ---------- 单条样本转id+padding ----------
# def encode_sample(item, code_tokens, token_types, token2stmt, stmt_infos, block_infos,
#                   vocab, max_code_len, max_text_len, stmt_max_num, block_max_num, nest_depth_max):
#     w2i = vocab['w2i'] if vocab.get('shared', True) else vocab['code_w2i']
#     text_w2i = vocab['w2i'] if vocab.get('shared', True) else vocab['text_w2i']
#     stmt_type_w2i = vocab['stmt_type_w2i']
#     block_type_w2i = vocab['block_type_w2i']
#     unk_id = w2i[UNK_TOKEN]
#     text_unk_id = text_w2i[UNK_TOKEN]

#     subtokens, sub2stmt = subtokenize_code(code_tokens, token_types, token2stmt)
#     kept_tokens, kept_sub2stmt, kept_stmt_infos, kept_block_infos = compact_truncate(
#         subtokens, sub2stmt, stmt_infos, block_infos,
#         max_code_len, stmt_max_num, block_max_num)

#     code_len = len(kept_tokens)
#     stmt_num = len(kept_stmt_infos)
#     block_num = len(kept_block_infos)

#     # 统计截断率时，block的"原始基准"不能直接用len(block_infos)：if(x){}/catch(e){}这类
#     # 空语句体会产生一个没有任何语句指向的"孤儿block"，即使完全不设截断上限也不会被
#     # 任何语句引用到，不该算作"被截断掉的"，否则会造成截断率虚高的假象
#     referenced_block_ids = {info['block_id'] for info in stmt_infos}
#     n_referenced_blocks = len(referenced_block_ids)

#     code_ids = np.zeros(max_code_len, dtype=np.int32)
#     code_ids[:code_len] = [w2i.get(t, unk_id) for t in kept_tokens]

#     token2stmt_arr = np.zeros(max_code_len, dtype=np.int32)
#     token2stmt_arr[:code_len] = kept_sub2stmt

#     stmt_type_ids = np.zeros(stmt_max_num, dtype=np.int32)
#     stmt2block_arr = np.zeros(stmt_max_num, dtype=np.int32)
#     for i, info in enumerate(kept_stmt_infos):
#         stmt_type_ids[i] = stmt_type_w2i.get(info['stmt_type'], stmt_type_w2i[UNK_TOKEN])
#         stmt2block_arr[i] = info['block_id']

#     block_nest_depth = np.zeros(block_max_num, dtype=np.int32)
#     block_type_ids = np.zeros(block_max_num, dtype=np.int32)
#     block_parent = np.full(block_max_num, -1, dtype=np.int32)  # -1=没有父block(或padding槽位)
#     for i, info in enumerate(kept_block_infos):
#         block_nest_depth[i] = min(info['nest_depth'], nest_depth_max - 1)
#         block_type_ids[i] = block_type_w2i.get(info['block_type'], block_type_w2i[UNK_TOKEN])
#         block_parent[i] = info['parent']

#     text_tokens = tokenize_text(item.get('text', ''))
#     text_tokens = [BOS_TOKEN] + text_tokens[:max_text_len - 2] + [EOS_TOKEN]
#     text_len = len(text_tokens)
#     text_ids = np.zeros(max_text_len, dtype=np.int32)
#     text_ids[:text_len] = [text_w2i.get(t, text_unk_id) for t in text_tokens]

#     truncated = dict(
#         code_truncated=(code_len < len(subtokens)),
#         stmt_truncated=(stmt_num < len(stmt_infos)),
#         block_truncated=(block_num < n_referenced_blocks),
#         text_truncated=(len(tokenize_text(item.get('text', ''))) > max_text_len - 2),
#     )

#     sample = dict(
#         id=item.get('id', -1),
#         code_ids=code_ids, token2stmt=token2stmt_arr,
#         stmt_type_ids=stmt_type_ids, stmt2block=stmt2block_arr,
#         block_nest_depth=block_nest_depth, block_type_ids=block_type_ids,
#         block_parent=block_parent,
#         text_ids=text_ids,
#         code_len=code_len, text_len=text_len, stmt_num=stmt_num, block_num=block_num,
#     )
#     return sample, truncated


# def build_split(data, vocab, max_code_len, max_text_len, stmt_max_num, block_max_num,
#                 nest_depth_max, desc, builder_opts=None):
#     samples = []
#     n_fail = 0
#     trunc_counts = dict(code_truncated=0, stmt_truncated=0, block_truncated=0, text_truncated=0)

#     builder_opts = builder_opts or {}
#     for item, result, error in iter_build_stmt_blocks(data, desc=desc, **builder_opts):
#         if error is not None:
#             n_fail += 1
#             continue
#         code_tokens, token_types, token2stmt, stmt_infos, block_infos = result
#         sample, truncated = encode_sample(
#             item, code_tokens, token_types, token2stmt, stmt_infos, block_infos,
#             vocab, max_code_len, max_text_len, stmt_max_num, block_max_num, nest_depth_max)
#         samples.append(sample)
#         for k, v in truncated.items():
#             trunc_counts[k] += int(v)

#     # 把list of dict"转置"成dict of array，方便后续Dataset按整块读取(而不是每条样本单独存取)
#     keys = samples[0].keys() if samples else []
#     stacked = {k: np.stack([s[k] for s in samples]) if isinstance(samples[0][k], np.ndarray)
#               else np.array([s[k] for s in samples]) for k in keys}

#     print('  样本数: {}  解析失败跳过: {}'.format(len(samples), n_fail))
#     if samples:
#         n = len(samples)
#         for k, v in trunc_counts.items():
#             print('  {}: {} ({:.2%})'.format(k, v, v / n))
#     return stacked


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--raw_data_dir', type=str, required=True)
#     parser.add_argument('--vocab_path', type=str, required=True)
#     parser.add_argument('--out_dir', type=str, required=True)
#     parser.add_argument('--max_code_len', type=int, default=300)
#     parser.add_argument('--max_text_len', type=int, default=50)
#     parser.add_argument('--stmt_max_num', type=int, default=64)
#     parser.add_argument('--block_max_num', type=int, default=24)
#     parser.add_argument('--nest_depth_max', type=int, default=8)
#     parser.add_argument('--include_modifiers', action='store_true',
#                         help='签名语句保留修饰符/注解/泛型参数。必须和建词表时用的一致')
#     parser.add_argument('--strip_braces', action='store_true',
#                         help='去掉语句首尾的结构性花括号。必须和建词表时用的一致')
#     args = parser.parse_args()
#     builder_opts = dict(include_modifiers=args.include_modifiers, strip_braces=args.strip_braces)

#     vocab = load_vocab(args.vocab_path)
#     os.makedirs(args.out_dir, exist_ok=True)

#     for split in ('train', 'valid', 'test'):
#         path = os.path.join(args.raw_data_dir, '{}_data.json'.format(split))
#         if not os.path.exists(path):
#             print('[跳过] 未找到 {}'.format(path))
#             continue
#         print('=' * 60)
#         print('处理: {}'.format(path))
#         data = load_json(path)
#         stacked = build_split(data, vocab, args.max_code_len, args.max_text_len,
#                               args.stmt_max_num, args.block_max_num, args.nest_depth_max,
#                               desc=split, builder_opts=builder_opts)
#         out_path = os.path.join(args.out_dir, '{}_data.pkl'.format(split))
#         with open(out_path, 'wb') as f:
#             pickle.dump(dict(data=stacked, params=dict(
#                 max_code_len=args.max_code_len, max_text_len=args.max_text_len,
#                 stmt_max_num=args.stmt_max_num, block_max_num=args.block_max_num,
#                 nest_depth_max=args.nest_depth_max, **builder_opts)), f)
#         print('已保存: {}'.format(out_path))


# if __name__ == '__main__':
#     main()

# coding=utf-8
"""
把前面几步(stmt_block_builder + code_tokenizer + vocab_builder)串起来，
对train/valid/test各自产出一份可以直接喂给PyTorch Dataset的numpy数组文件(.npz)。

核心设计：截断/padding时不能沿用之前HETCOS版本那种"clip"做法(把超出上限的语句id
硬夹到最后一个槽位，导致几条不同语义的语句被拼成"一条")——那是之前排查出来的一个真实
bug。这里改用"配额耗尽就整体截断"的做法：

  按token顺序扫描，遇到新语句就检查是否还有语句名额(stmt_max_num)、遇到新block就检查
  是否还有block名额(block_max_num)、token数是否超过max_code_len，三个条件任意一个先
  触发就停止——保留的语句/block集合按"第一次出现的顺序"重新映射成紧凑的0..K-1编号，
  不会出现"多条语句被挤压进同一个槽位"的情况，只是老老实实丢弃超出预算的尾部内容。

用法：
    python3 build_dataset.py --raw_data_dir ../raw_data --vocab_path ../w2i2w/vocab.pkl \
        --out_dir ../avail_data --max_code_len 300 --max_text_len 50 \
        --stmt_max_num 64 --block_max_num 24 --nest_depth_max 8
"""
import argparse
import json
import os
import pickle

import numpy as np

from stmt_block_builder import iter_build_stmt_blocks
from code_tokenizer import subtokenize_code, tokenize_text
from vocab_builder import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------- 核心：按配额截断+紧凑重编号 ----------
def compact_truncate(subtokens, sub2stmt, stmt_infos, block_infos,
                     max_code_len, stmt_max_num, block_max_num):
    """
    输入: code_tokenizer.subtokenize_code的输出(subtokens, sub2stmt) +
         stmt_block_builder的(stmt_infos, block_infos)(下标即原始stmt_id/block_id)
    输出: (kept_tokens, kept_sub2stmt_compact, kept_stmt_infos_compact, kept_block_infos_compact)
         语句/block都已经按"第一次出现顺序"重新映射成0..K-1的紧凑编号，
         超出任一配额(token数/语句数/block数)时直接截断，不做任何"挤压合并"。
    """
    kept_tokens, kept_sub2stmt = [], []
    stmt_order, stmt_id_map = [], {}     # old_stmt_id -> new_compact_id
    block_order, block_id_map = [], {}   # old_block_id -> new_compact_id

    for tok, old_sid in zip(subtokens, sub2stmt):
        if len(kept_tokens) >= max_code_len:
            break
        if old_sid not in stmt_id_map:
            old_bid = stmt_infos[old_sid]['block_id']
            if old_bid not in block_id_map:
                if len(block_order) >= block_max_num:
                    break
                block_id_map[old_bid] = len(block_order)
                block_order.append(old_bid)
            if len(stmt_order) >= stmt_max_num:
                break
            stmt_id_map[old_sid] = len(stmt_order)
            stmt_order.append(old_sid)
        kept_tokens.append(tok)
        kept_sub2stmt.append(stmt_id_map[old_sid])

    kept_stmt_infos = []
    for old_sid in stmt_order:
        info = stmt_infos[old_sid]
        kept_stmt_infos.append(dict(block_id=block_id_map[info['block_id']],
                                    stmt_type=info['stmt_type']))
    kept_block_infos = []
    for old_bid in block_order:
        info = block_infos[old_bid]
        # 父block一定在子block之前首次出现(控制结构的头部语句属于父block，且先于body里的语句)，
        # 所以截断(只丢尾部)之后父block仍在block_id_map里；万一不在或没有父block，记-1
        old_parent = info.get('parent')
        new_parent = block_id_map.get(old_parent, -1) if old_parent is not None else -1
        kept_block_infos.append(dict(nest_depth=info['nest_depth'], block_type=info['block_type'],
                                     parent=new_parent))

    return kept_tokens, kept_sub2stmt, kept_stmt_infos, kept_block_infos


# ---------- 单条样本转id+padding ----------
def encode_sample(item, code_tokens, token_types, token2stmt, stmt_infos, block_infos,
                  vocab, max_code_len, max_text_len, stmt_max_num, block_max_num, nest_depth_max):
    w2i = vocab['w2i'] if vocab.get('shared', True) else vocab['code_w2i']
    text_w2i = vocab['w2i'] if vocab.get('shared', True) else vocab['text_w2i']
    stmt_type_w2i = vocab['stmt_type_w2i']
    block_type_w2i = vocab['block_type_w2i']
    unk_id = w2i[UNK_TOKEN]
    text_unk_id = text_w2i[UNK_TOKEN]

    subtokens, sub2stmt = subtokenize_code(code_tokens, token_types, token2stmt)
    kept_tokens, kept_sub2stmt, kept_stmt_infos, kept_block_infos = compact_truncate(
        subtokens, sub2stmt, stmt_infos, block_infos,
        max_code_len, stmt_max_num, block_max_num)

    code_len = len(kept_tokens)
    stmt_num = len(kept_stmt_infos)
    block_num = len(kept_block_infos)

    # 统计截断率时，block的"原始基准"不能直接用len(block_infos)：if(x){}/catch(e){}这类
    # 空语句体会产生一个没有任何语句指向的"孤儿block"，即使完全不设截断上限也不会被
    # 任何语句引用到，不该算作"被截断掉的"，否则会造成截断率虚高的假象
    referenced_block_ids = {info['block_id'] for info in stmt_infos}
    n_referenced_blocks = len(referenced_block_ids)

    code_ids = np.zeros(max_code_len, dtype=np.int32)
    code_ids[:code_len] = [w2i.get(t, unk_id) for t in kept_tokens]

    token2stmt_arr = np.zeros(max_code_len, dtype=np.int32)
    token2stmt_arr[:code_len] = kept_sub2stmt

    stmt_type_ids = np.zeros(stmt_max_num, dtype=np.int32)
    stmt2block_arr = np.zeros(stmt_max_num, dtype=np.int32)
    for i, info in enumerate(kept_stmt_infos):
        stmt_type_ids[i] = stmt_type_w2i.get(info['stmt_type'], stmt_type_w2i[UNK_TOKEN])
        stmt2block_arr[i] = info['block_id']

    block_nest_depth = np.zeros(block_max_num, dtype=np.int32)
    block_type_ids = np.zeros(block_max_num, dtype=np.int32)
    block_parent = np.full(block_max_num, -1, dtype=np.int32)  # -1=没有父block(或padding槽位)
    for i, info in enumerate(kept_block_infos):
        block_nest_depth[i] = min(info['nest_depth'], nest_depth_max - 1)
        block_type_ids[i] = block_type_w2i.get(info['block_type'], block_type_w2i[UNK_TOKEN])
        block_parent[i] = info['parent']

    text_tokens = tokenize_text(item.get('text', ''))
    text_tokens = [BOS_TOKEN] + text_tokens[:max_text_len - 2] + [EOS_TOKEN]
    text_len = len(text_tokens)
    text_ids = np.zeros(max_text_len, dtype=np.int32)
    text_ids[:text_len] = [text_w2i.get(t, text_unk_id) for t in text_tokens]

    truncated = dict(
        code_truncated=(code_len < len(subtokens)),
        stmt_truncated=(stmt_num < len(stmt_infos)),
        block_truncated=(block_num < n_referenced_blocks),
        text_truncated=(len(tokenize_text(item.get('text', ''))) > max_text_len - 2),
    )

    sample = dict(
        id=item.get('id', -1),
        code_ids=code_ids, token2stmt=token2stmt_arr,
        stmt_type_ids=stmt_type_ids, stmt2block=stmt2block_arr,
        block_nest_depth=block_nest_depth, block_type_ids=block_type_ids,
        block_parent=block_parent,
        text_ids=text_ids,
        code_len=code_len, text_len=text_len, stmt_num=stmt_num, block_num=block_num,
    )
    return sample, truncated


def build_split(data, vocab, max_code_len, max_text_len, stmt_max_num, block_max_num,
                nest_depth_max, desc, builder_opts=None):
    samples = []
    n_fail = 0
    trunc_counts = dict(code_truncated=0, stmt_truncated=0, block_truncated=0, text_truncated=0)

    builder_opts = builder_opts or {}
    for item, result, error in iter_build_stmt_blocks(data, desc=desc, **builder_opts):
        if error is not None:
            n_fail += 1
            continue
        code_tokens, token_types, token2stmt, stmt_infos, block_infos = result
        sample, truncated = encode_sample(
            item, code_tokens, token_types, token2stmt, stmt_infos, block_infos,
            vocab, max_code_len, max_text_len, stmt_max_num, block_max_num, nest_depth_max)
        samples.append(sample)
        for k, v in truncated.items():
            trunc_counts[k] += int(v)

    # 把list of dict"转置"成dict of array，方便后续Dataset按整块读取(而不是每条样本单独存取)
    keys = samples[0].keys() if samples else []
    stacked = {k: np.stack([s[k] for s in samples]) if isinstance(samples[0][k], np.ndarray)
              else np.array([s[k] for s in samples]) for k in keys}

    print('  样本数: {}  解析失败跳过: {}'.format(len(samples), n_fail))
    if samples:
        n = len(samples)
        for k, v in trunc_counts.items():
            print('  {}: {} ({:.2%})'.format(k, v, v / n))
    return stacked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_code_len', type=int, default=300)
    parser.add_argument('--max_text_len', type=int, default=50)
    parser.add_argument('--stmt_max_num', type=int, default=64)
    parser.add_argument('--block_max_num', type=int, default=24)
    parser.add_argument('--nest_depth_max', type=int, default=8)
    parser.add_argument('--text_mode', type=str, default='raw', choices=['raw', 'pretokenized'],
                        help="摘要文本模式。raw=原始摘要(小写丢标点)；pretokenized=摘要已做词形还原且带'..'结束标记"
                             "(HetCoS的token_data)，直接按空格切。词表和数据必须用同一个模式")
    parser.add_argument('--include_modifiers', action='store_true',
                        help='签名语句保留修饰符/注解/泛型参数。必须和建词表时用的一致')
    parser.add_argument('--strip_braces', action='store_true',
                        help='去掉语句首尾的结构性花括号。必须和建词表时用的一致')
    args = parser.parse_args()
    builder_opts = dict(include_modifiers=args.include_modifiers, strip_braces=args.strip_braces)
    from code_tokenizer import set_text_mode
    set_text_mode(args.text_mode)

    vocab = load_vocab(args.vocab_path)
    os.makedirs(args.out_dir, exist_ok=True)

    for split in ('train', 'valid', 'test'):
        path = os.path.join(args.raw_data_dir, '{}_data.json'.format(split))
        if not os.path.exists(path):
            print('[跳过] 未找到 {}'.format(path))
            continue
        print('=' * 60)
        print('处理: {}'.format(path))
        data = load_json(path)
        stacked = build_split(data, vocab, args.max_code_len, args.max_text_len,
                              args.stmt_max_num, args.block_max_num, args.nest_depth_max,
                              desc=split, builder_opts=builder_opts)
        out_path = os.path.join(args.out_dir, '{}_data.pkl'.format(split))
        with open(out_path, 'wb') as f:
            pickle.dump(dict(data=stacked, params=dict(
                max_code_len=args.max_code_len, max_text_len=args.max_text_len,
                stmt_max_num=args.stmt_max_num, block_max_num=args.block_max_num,
                nest_depth_max=args.nest_depth_max, text_mode=args.text_mode, **builder_opts)), f)
        print('已保存: {}'.format(out_path))


if __name__ == '__main__':
    main()