# # coding=utf-8
# """
# 词表构建模块。只用train_data统计词频建词表(valid/test不参与，避免词表泄漏——
# 这是标准做法：模型部署后遇到的也是"没见过的新数据"，用valid/test帮着建词表
# 会让离线评估结果比真实泛化能力更乐观，掩盖问题)。

# 设计：code的子词(来自code_tokenizer.subtokenize_code)和text的词共用同一张词表
# (shared=True，默认)，原因：
#   1. 和之前config_v2.py里io_token_w2i的设计保持一致，减少心智负担
#   2. 如果以后要加copy机制(从源码里复制token到摘要里)，pointer network通常要求
#      源端和目标端共享同一个词表空间，现在就用共享词表，以后加copy不用推倒重来
#   如果之后发现共享词表让某一端(通常是text端，因为自然语言词和代码标识符词分布差异很大)
#   词表质量下降，可以把shared改成False，分别建两张词表，本文件两种方式都支持。

# 用法：
#     python3 vocab_builder.py --raw_data_dir ../raw_data --min_count 3 \
#         --max_vocab_size 50000 --out_path ../w2i2w/vocab.pkl
# """
# import argparse
# import json
# import os
# import pickle
# from collections import Counter

# from stmt_block_builder import iter_build_stmt_blocks
# from code_tokenizer import subtokenize_code, tokenize_text

# PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = '<pad>', '<unk>', '<s>', '</s>'
# RESERVED_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


# def load_json(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         return json.load(f)


# def build_vocab_from_counter(counter, min_count=3, max_vocab_size=None, reserved=RESERVED_TOKENS):
#     """频次>=min_count的词，按频次降序(频次相同按字母序，保证结果确定可复现)排列，
#     reserved token固定占据最前面几个id"""
#     items = [(tok, cnt) for tok, cnt in counter.items() if cnt >= min_count]
#     items.sort(key=lambda x: (-x[1], x[0]))
#     if max_vocab_size is not None:
#         items = items[:max(0, max_vocab_size - len(reserved))]

#     w2i = {tok: i for i, tok in enumerate(reserved)}
#     for tok, _ in items:
#         if tok not in w2i:
#             w2i[tok] = len(w2i)
#     i2w = {i: w for w, i in w2i.items()}
#     return w2i, i2w


# def collect_token_counters(train_data, show_progress=True, builder_opts=None):
#     """遍历训练集，统计code子词、text词、stmt_type、block_type各自的频次(Counter)。
#     复用stmt_block_builder.iter_build_stmt_blocks做遍历+进度条+异常收集，
#     解析失败的样本直接跳过(不计入统计)。
#     """
#     code_counter, text_counter = Counter(), Counter()
#     stmt_type_counter, block_type_counter = Counter(), Counter()
#     n_ok = 0
#     # builder_opts: dict(include_modifiers=..., strip_braces=...)，必须和build_dataset.py用同一组，
#     # 否则词表统计的token集合和后面实际编码的不一致(比如签名里的private/@Override会全变成<unk>)
#     builder_opts = builder_opts or {}
#     for item, result, error in iter_build_stmt_blocks(train_data, show_progress=show_progress,
#                                                         desc='统计词频', **builder_opts):
#         if error is not None:
#             continue
#         code_tokens, token_types, token2stmt, stmt_infos, block_infos = result
#         subtokens, _ = subtokenize_code(code_tokens, token_types, token2stmt)
#         code_counter.update(subtokens)
#         text_tokens = tokenize_text(item.get('text', ''))
#         text_counter.update(text_tokens)
#         stmt_type_counter.update(s['stmt_type'] for s in stmt_infos)
#         block_type_counter.update(b['block_type'] for b in block_infos)
#         n_ok += 1
#     return code_counter, text_counter, stmt_type_counter, block_type_counter, n_ok


# def coverage_rate(counter, w2i):
#     """给定词频统计和词表，算"词表能覆盖住的token出现次数占比"(1-OOV率)。
#     这个数字比"词表大小"更能说明词表设得好不好——词表很大但覆盖率低，
#     说明长尾词太多，min_count设太高了(或者数据里噪声词太多)。"""
#     total = sum(counter.values())
#     if total == 0:
#         return 0.0
#     covered = sum(cnt for tok, cnt in counter.items() if tok in w2i)
#     return covered / total


# TYPE_RESERVED_TOKENS = [PAD_TOKEN, UNK_TOKEN]  # 类型类词表不需要BOS/EOS，但需要PAD(padding位置占位)


# def build_vocabs(train_data, min_count=3, max_vocab_size=50000, shared=True, show_progress=True,
#                  builder_opts=None):
#     code_counter, text_counter, stmt_type_counter, block_type_counter, n_ok = \
#         collect_token_counters(train_data, show_progress=show_progress, builder_opts=builder_opts)

#     if shared:
#         combined = code_counter + text_counter
#         w2i, i2w = build_vocab_from_counter(combined, min_count=min_count, max_vocab_size=max_vocab_size)
#         result = dict(shared=True, w2i=w2i, i2w=i2w)
#     else:
#         code_w2i, code_i2w = build_vocab_from_counter(code_counter, min_count=min_count,
#                                                        max_vocab_size=max_vocab_size)
#         text_w2i, text_i2w = build_vocab_from_counter(text_counter, min_count=min_count,
#                                                        max_vocab_size=max_vocab_size)
#         result = dict(shared=False, code_w2i=code_w2i, code_i2w=code_i2w,
#                      text_w2i=text_w2i, text_i2w=text_i2w)

#     # stmt_type/block_type种类很少(几十种以内)，min_count=1全部保留，不设上限
#     stmt_type_w2i, stmt_type_i2w = build_vocab_from_counter(
#         stmt_type_counter, min_count=1, max_vocab_size=None, reserved=TYPE_RESERVED_TOKENS)
#     block_type_w2i, block_type_i2w = build_vocab_from_counter(
#         block_type_counter, min_count=1, max_vocab_size=None, reserved=TYPE_RESERVED_TOKENS)
#     result.update(stmt_type_w2i=stmt_type_w2i, stmt_type_i2w=stmt_type_i2w,
#                   block_type_w2i=block_type_w2i, block_type_i2w=block_type_i2w)

#     result.update(code_counter=code_counter, text_counter=text_counter,
#                   stmt_type_counter=stmt_type_counter, block_type_counter=block_type_counter,
#                   n_samples=n_ok)
#     return result


# def print_report(result, min_count, max_vocab_size):
#     code_counter, text_counter = result['code_counter'], result['text_counter']
#     print('参与建词表的样本数(解析成功的): {}'.format(result['n_samples']))
#     print('code子词种类数(去重): {}  出现总次数: {}'.format(len(code_counter), sum(code_counter.values())))
#     print('text词种类数(去重): {}  出现总次数: {}'.format(len(text_counter), sum(text_counter.values())))
#     print('min_count={}  max_vocab_size={}'.format(min_count, max_vocab_size))

#     if result['shared']:
#         w2i = result['w2i']
#         print('词表大小(共享): {}'.format(len(w2i)))
#         print('  code端覆盖率(1-OOV率): {:.2%}'.format(coverage_rate(code_counter, w2i)))
#         print('  text端覆盖率(1-OOV率): {:.2%}'.format(coverage_rate(text_counter, w2i)))
#     else:
#         print('code词表大小: {}  覆盖率: {:.2%}'.format(
#             len(result['code_w2i']), coverage_rate(code_counter, result['code_w2i'])))
#         print('text词表大小: {}  覆盖率: {:.2%}'.format(
#             len(result['text_w2i']), coverage_rate(text_counter, result['text_w2i'])))

#     print('code高频词Top20: {}'.format([t for t, _ in code_counter.most_common(20)]))
#     print('text高频词Top20: {}'.format([t for t, _ in text_counter.most_common(20)]))

#     stmt_type_counter, block_type_counter = result['stmt_type_counter'], result['block_type_counter']
#     print('stmt_type种类数: {}  词表大小(含PAD/UNK): {}'.format(
#         len(stmt_type_counter), len(result['stmt_type_w2i'])))
#     print('  全部stmt_type及出现次数: {}'.format(dict(stmt_type_counter.most_common())))
#     print('block_type种类数: {}  词表大小(含PAD/UNK): {}'.format(
#         len(block_type_counter), len(result['block_type_w2i'])))
#     print('  全部block_type及出现次数: {}'.format(dict(block_type_counter.most_common())))


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--raw_data_dir', type=str, required=True)
#     parser.add_argument('--min_count', type=int, default=3)
#     parser.add_argument('--max_vocab_size', type=int, default=50000)
#     parser.add_argument('--shared', action='store_true', default=True,
#                         help='code和text是否共用一张词表(默认True)')
#     parser.add_argument('--no_shared', dest='shared', action='store_false')
#     parser.add_argument('--out_path', type=str, default=None,
#                         help='词表保存路径(.pkl)，不指定则只打印报告不保存')
#     parser.add_argument('--include_modifiers', action='store_true',
#                         help='签名语句保留修饰符/注解/泛型参数(public/static/@Override等)')
#     parser.add_argument('--strip_braces', action='store_true',
#                         help='去掉语句首尾的结构性花括号')
#     args = parser.parse_args()

#     train_path = os.path.join(args.raw_data_dir, 'train_data.json')
#     train_data = load_json(train_path)
#     print('读取训练集: {} ({}条)'.format(train_path, len(train_data)))

#     result = build_vocabs(train_data, min_count=args.min_count,
#                           max_vocab_size=args.max_vocab_size, shared=args.shared,
#                           builder_opts=dict(include_modifiers=args.include_modifiers,
#                                             strip_braces=args.strip_braces))
#     print_report(result, args.min_count, args.max_vocab_size)

#     if args.out_path:
#         os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
#         # counter对象不需要存进最终词表文件，体积大且用不上，只保存w2i/i2w
#         save_obj = {k: v for k, v in result.items()
#                     if k not in ('code_counter', 'text_counter', 'stmt_type_counter', 'block_type_counter')}
#         with open(args.out_path, 'wb') as f:
#             pickle.dump(save_obj, f)
#         print('词表已保存到: {}'.format(args.out_path))


# if __name__ == '__main__':
#     main()

# coding=utf-8
"""
词表构建模块。只用train_data统计词频建词表(valid/test不参与，避免词表泄漏——
这是标准做法：模型部署后遇到的也是"没见过的新数据"，用valid/test帮着建词表
会让离线评估结果比真实泛化能力更乐观，掩盖问题)。

设计：code的子词(来自code_tokenizer.subtokenize_code)和text的词共用同一张词表
(shared=True，默认)，原因：
  1. 和之前config_v2.py里io_token_w2i的设计保持一致，减少心智负担
  2. 如果以后要加copy机制(从源码里复制token到摘要里)，pointer network通常要求
     源端和目标端共享同一个词表空间，现在就用共享词表，以后加copy不用推倒重来
  如果之后发现共享词表让某一端(通常是text端，因为自然语言词和代码标识符词分布差异很大)
  词表质量下降，可以把shared改成False，分别建两张词表，本文件两种方式都支持。

用法：
    python3 vocab_builder.py --raw_data_dir ../raw_data --min_count 3 \
        --max_vocab_size 50000 --out_path ../w2i2w/vocab.pkl
"""
import argparse
import json
import os
import pickle
from collections import Counter

from stmt_block_builder import iter_build_stmt_blocks
from code_tokenizer import subtokenize_code, tokenize_text

PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = '<pad>', '<unk>', '<s>', '</s>'
RESERVED_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_vocab_from_counter(counter, min_count=3, max_vocab_size=None, reserved=RESERVED_TOKENS):
    """频次>=min_count的词，按频次降序(频次相同按字母序，保证结果确定可复现)排列，
    reserved token固定占据最前面几个id"""
    items = [(tok, cnt) for tok, cnt in counter.items() if cnt >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_vocab_size is not None:
        items = items[:max(0, max_vocab_size - len(reserved))]

    w2i = {tok: i for i, tok in enumerate(reserved)}
    for tok, _ in items:
        if tok not in w2i:
            w2i[tok] = len(w2i)
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w


def collect_token_counters(train_data, show_progress=True, builder_opts=None):
    """遍历训练集，统计code子词、text词、stmt_type、block_type各自的频次(Counter)。
    复用stmt_block_builder.iter_build_stmt_blocks做遍历+进度条+异常收集，
    解析失败的样本直接跳过(不计入统计)。
    """
    code_counter, text_counter = Counter(), Counter()
    stmt_type_counter, block_type_counter = Counter(), Counter()
    n_ok = 0
    # builder_opts: dict(include_modifiers=..., strip_braces=...)，必须和build_dataset.py用同一组，
    # 否则词表统计的token集合和后面实际编码的不一致(比如签名里的private/@Override会全变成<unk>)
    builder_opts = builder_opts or {}
    for item, result, error in iter_build_stmt_blocks(train_data, show_progress=show_progress,
                                                        desc='统计词频', **builder_opts):
        if error is not None:
            continue
        code_tokens, token_types, token2stmt, stmt_infos, block_infos = result
        subtokens, _ = subtokenize_code(code_tokens, token_types, token2stmt)
        code_counter.update(subtokens)
        text_tokens = tokenize_text(item.get('text', ''))
        text_counter.update(text_tokens)
        stmt_type_counter.update(s['stmt_type'] for s in stmt_infos)
        block_type_counter.update(b['block_type'] for b in block_infos)
        n_ok += 1
    return code_counter, text_counter, stmt_type_counter, block_type_counter, n_ok


def coverage_rate(counter, w2i):
    """给定词频统计和词表，算"词表能覆盖住的token出现次数占比"(1-OOV率)。
    这个数字比"词表大小"更能说明词表设得好不好——词表很大但覆盖率低，
    说明长尾词太多，min_count设太高了(或者数据里噪声词太多)。"""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    covered = sum(cnt for tok, cnt in counter.items() if tok in w2i)
    return covered / total


TYPE_RESERVED_TOKENS = [PAD_TOKEN, UNK_TOKEN]  # 类型类词表不需要BOS/EOS，但需要PAD(padding位置占位)


def build_vocabs(train_data, min_count=3, max_vocab_size=50000, shared=True, show_progress=True,
                 builder_opts=None):
    code_counter, text_counter, stmt_type_counter, block_type_counter, n_ok = \
        collect_token_counters(train_data, show_progress=show_progress, builder_opts=builder_opts)

    if shared:
        combined = code_counter + text_counter
        w2i, i2w = build_vocab_from_counter(combined, min_count=min_count, max_vocab_size=max_vocab_size)
        result = dict(shared=True, w2i=w2i, i2w=i2w)
    else:
        code_w2i, code_i2w = build_vocab_from_counter(code_counter, min_count=min_count,
                                                       max_vocab_size=max_vocab_size)
        text_w2i, text_i2w = build_vocab_from_counter(text_counter, min_count=min_count,
                                                       max_vocab_size=max_vocab_size)
        result = dict(shared=False, code_w2i=code_w2i, code_i2w=code_i2w,
                     text_w2i=text_w2i, text_i2w=text_i2w)

    # stmt_type/block_type种类很少(几十种以内)，min_count=1全部保留，不设上限
    stmt_type_w2i, stmt_type_i2w = build_vocab_from_counter(
        stmt_type_counter, min_count=1, max_vocab_size=None, reserved=TYPE_RESERVED_TOKENS)
    block_type_w2i, block_type_i2w = build_vocab_from_counter(
        block_type_counter, min_count=1, max_vocab_size=None, reserved=TYPE_RESERVED_TOKENS)
    result.update(stmt_type_w2i=stmt_type_w2i, stmt_type_i2w=stmt_type_i2w,
                  block_type_w2i=block_type_w2i, block_type_i2w=block_type_i2w)

    result.update(code_counter=code_counter, text_counter=text_counter,
                  stmt_type_counter=stmt_type_counter, block_type_counter=block_type_counter,
                  n_samples=n_ok)
    return result


def print_report(result, min_count, max_vocab_size):
    code_counter, text_counter = result['code_counter'], result['text_counter']
    print('参与建词表的样本数(解析成功的): {}'.format(result['n_samples']))
    print('code子词种类数(去重): {}  出现总次数: {}'.format(len(code_counter), sum(code_counter.values())))
    print('text词种类数(去重): {}  出现总次数: {}'.format(len(text_counter), sum(text_counter.values())))
    print('min_count={}  max_vocab_size={}'.format(min_count, max_vocab_size))

    if result['shared']:
        w2i = result['w2i']
        print('词表大小(共享): {}'.format(len(w2i)))
        print('  code端覆盖率(1-OOV率): {:.2%}'.format(coverage_rate(code_counter, w2i)))
        print('  text端覆盖率(1-OOV率): {:.2%}'.format(coverage_rate(text_counter, w2i)))
    else:
        print('code词表大小: {}  覆盖率: {:.2%}'.format(
            len(result['code_w2i']), coverage_rate(code_counter, result['code_w2i'])))
        print('text词表大小: {}  覆盖率: {:.2%}'.format(
            len(result['text_w2i']), coverage_rate(text_counter, result['text_w2i'])))

    print('code高频词Top20: {}'.format([t for t, _ in code_counter.most_common(20)]))
    print('text高频词Top20: {}'.format([t for t, _ in text_counter.most_common(20)]))

    stmt_type_counter, block_type_counter = result['stmt_type_counter'], result['block_type_counter']
    print('stmt_type种类数: {}  词表大小(含PAD/UNK): {}'.format(
        len(stmt_type_counter), len(result['stmt_type_w2i'])))
    print('  全部stmt_type及出现次数: {}'.format(dict(stmt_type_counter.most_common())))
    print('block_type种类数: {}  词表大小(含PAD/UNK): {}'.format(
        len(block_type_counter), len(result['block_type_w2i'])))
    print('  全部block_type及出现次数: {}'.format(dict(block_type_counter.most_common())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--min_count', type=int, default=3)
    parser.add_argument('--max_vocab_size', type=int, default=50000)
    parser.add_argument('--shared', action='store_true', default=True,
                        help='code和text是否共用一张词表(默认True)')
    parser.add_argument('--no_shared', dest='shared', action='store_false')
    parser.add_argument('--out_path', type=str, default=None,
                        help='词表保存路径(.pkl)，不指定则只打印报告不保存')
    parser.add_argument('--text_mode', type=str, default='raw', choices=['raw', 'pretokenized'],
                        help="摘要文本模式。raw=原始摘要(小写丢标点)；pretokenized=摘要已做词形还原且带'..'结束标记"
                             "(HetCoS的token_data)，直接按空格切。词表和数据必须用同一个模式")
    parser.add_argument('--include_modifiers', action='store_true',
                        help='签名语句保留修饰符/注解/泛型参数(public/static/@Override等)')
    parser.add_argument('--strip_braces', action='store_true',
                        help='去掉语句首尾的结构性花括号')
    args = parser.parse_args()

    train_path = os.path.join(args.raw_data_dir, 'train_data.json')
    train_data = load_json(train_path)
    print('读取训练集: {} ({}条)'.format(train_path, len(train_data)))

    from code_tokenizer import set_text_mode
    set_text_mode(args.text_mode)
    result = build_vocabs(train_data, min_count=args.min_count,
                          max_vocab_size=args.max_vocab_size, shared=args.shared,
                          builder_opts=dict(include_modifiers=args.include_modifiers,
                                            strip_braces=args.strip_braces))
    print_report(result, args.min_count, args.max_vocab_size)

    if args.out_path:
        os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
        # counter对象不需要存进最终词表文件，体积大且用不上，只保存w2i/i2w
        save_obj = {k: v for k, v in result.items()
                    if k not in ('code_counter', 'text_counter', 'stmt_type_counter', 'block_type_counter')}
        with open(args.out_path, 'wb') as f:
            pickle.dump(save_obj, f)
        print('词表已保存到: {}'.format(args.out_path))


if __name__ == '__main__':
    main()