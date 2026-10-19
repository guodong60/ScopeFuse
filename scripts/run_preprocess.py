# # coding=utf-8
# """
# 一条命令跑完整个预处理流程：建词表(vocab_builder.py) -> 构建训练数据(build_dataset.py)。
# 不用手动分两条命令跑、还要自己把词表路径从第一步的输出填到第二步的参数里。

# 直接复用vocab_builder.py和build_dataset.py里已经写好、测过的函数(build_vocabs/
# build_split等)，不是简单地用subprocess去调两个脚本——这样能省掉"训练集json重复读取
# 两次"之外的额外开销，也能让两步共用同一个进程里的中间结果。

# 用法:
#     python3 run_preprocess.py --raw_data_dir ../raw_data \
#         --vocab_out_path ../w2i2w/vocab.pkl \
#         --avail_data_out_dir ../avail_data
#     (其余超参数如min_count/max_code_len等都有和之前讨论过的默认值，按需用命令行参数覆盖)

# 注意: stmt_block_builder.py的语法覆盖率验证(validate_stmt_blocks.py)不包含在这条流水线里
# ——那是开发阶段用来排查stmt_block_builder.py本身有没有bug的诊断工具，不是每次重新生成
# 数据都必须跑的步骤。正常迭代(换个min_count/max_code_len重新生成avail_data)时不需要重新
# 跑一遍语法覆盖率验证。
# """
# import argparse
# import os
# import pickle
# import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocess'))

# from vocab_builder import build_vocabs, print_report, load_json  # noqa: E402
# from build_dataset import build_split, load_vocab  # noqa: E402


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--raw_data_dir', type=str, required=True)
#     parser.add_argument('--vocab_out_path', type=str, required=True)
#     parser.add_argument('--avail_data_out_dir', type=str, required=True)

#     # 词表相关(对应vocab_builder.py)
#     parser.add_argument('--min_count', type=int, default=3)
#     parser.add_argument('--max_vocab_size', type=int, default=50000)
#     parser.add_argument('--no_shared_vocab', action='store_true',
#                         help='默认code和text共用一张词表；加这个参数改成各自独立建词表')

#     # 截断/padding相关(对应build_dataset.py)
#     parser.add_argument('--max_code_len', type=int, default=300)
#     parser.add_argument('--max_text_len', type=int, default=50)
#     parser.add_argument('--stmt_max_num', type=int, default=64)
#     parser.add_argument('--block_max_num', type=int, default=24)
#     parser.add_argument('--nest_depth_max', type=int, default=8)

#     # 预处理改进开关(词表和数据两步共用同一组，保证一致)
#     parser.add_argument('--include_modifiers', action='store_true',
#                         help='签名语句保留修饰符/注解/泛型参数(public/static/@Override等)')
#     parser.add_argument('--strip_braces', action='store_true',
#                         help='去掉语句首尾的结构性花括号')
#     args = parser.parse_args()
#     builder_opts = dict(include_modifiers=args.include_modifiers, strip_braces=args.strip_braces)

#     print('=' * 60)
#     print('第1步: 建词表')
#     print('=' * 60)
#     train_path = os.path.join(args.raw_data_dir, 'train_data.json')
#     train_data = load_json(train_path)
#     vocab_result = build_vocabs(train_data, min_count=args.min_count,
#                                 max_vocab_size=args.max_vocab_size,
#                                 shared=not args.no_shared_vocab, builder_opts=builder_opts)
#     print_report(vocab_result, args.min_count, args.max_vocab_size)

#     os.makedirs(os.path.dirname(args.vocab_out_path) or '.', exist_ok=True)
#     save_obj = {k: v for k, v in vocab_result.items()
#                if k not in ('code_counter', 'text_counter', 'stmt_type_counter', 'block_type_counter')}
#     with open(args.vocab_out_path, 'wb') as f:
#         pickle.dump(save_obj, f)
#     print('词表已保存: {}'.format(args.vocab_out_path))

#     print('\n' + '=' * 60)
#     print('第2步: 构建训练数据(截断+紧凑重编号+padding)')
#     print('=' * 60)
#     vocab = load_vocab(args.vocab_out_path)
#     os.makedirs(args.avail_data_out_dir, exist_ok=True)

#     for split in ('train', 'valid', 'test'):
#         path = os.path.join(args.raw_data_dir, '{}_data.json'.format(split))
#         if not os.path.exists(path):
#             print('[跳过] 未找到 {}'.format(path))
#             continue
#         print('-' * 60)
#         print('处理: {}'.format(path))
#         data = load_json(path)
#         stacked = build_split(data, vocab, args.max_code_len, args.max_text_len,
#                               args.stmt_max_num, args.block_max_num, args.nest_depth_max,
#                               desc=split, builder_opts=builder_opts)
#         out_path = os.path.join(args.avail_data_out_dir, '{}_data.pkl'.format(split))
#         with open(out_path, 'wb') as f:
#             pickle.dump(dict(data=stacked, params=dict(
#                 max_code_len=args.max_code_len, max_text_len=args.max_text_len,
#                 stmt_max_num=args.stmt_max_num, block_max_num=args.block_max_num,
#                 nest_depth_max=args.nest_depth_max, **builder_opts)), f)
#         print('已保存: {}'.format(out_path))

#     print('\n预处理全部完成。')
#     print('词表: {}'.format(args.vocab_out_path))
#     print('训练数据目录: {}'.format(args.avail_data_out_dir))


# if __name__ == '__main__':
#     main()

# coding=utf-8
"""
一条命令跑完整个预处理流程：建词表(vocab_builder.py) -> 构建训练数据(build_dataset.py)。
不用手动分两条命令跑、还要自己把词表路径从第一步的输出填到第二步的参数里。

直接复用vocab_builder.py和build_dataset.py里已经写好、测过的函数(build_vocabs/
build_split等)，不是简单地用subprocess去调两个脚本——这样能省掉"训练集json重复读取
两次"之外的额外开销，也能让两步共用同一个进程里的中间结果。

用法:
    python3 run_preprocess.py --raw_data_dir ../raw_data \
        --vocab_out_path ../w2i2w/vocab.pkl \
        --avail_data_out_dir ../avail_data
    (其余超参数如min_count/max_code_len等都有和之前讨论过的默认值，按需用命令行参数覆盖)

注意: stmt_block_builder.py的语法覆盖率验证(validate_stmt_blocks.py)不包含在这条流水线里
——那是开发阶段用来排查stmt_block_builder.py本身有没有bug的诊断工具，不是每次重新生成
数据都必须跑的步骤。正常迭代(换个min_count/max_code_len重新生成avail_data)时不需要重新
跑一遍语法覆盖率验证。
"""
import argparse
import os
import pickle
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocess'))

from vocab_builder import build_vocabs, print_report, load_json  # noqa: E402
from build_dataset import build_split, load_vocab  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--vocab_out_path', type=str, required=True)
    parser.add_argument('--avail_data_out_dir', type=str, required=True)

    # 词表相关(对应vocab_builder.py)
    parser.add_argument('--min_count', type=int, default=3)
    parser.add_argument('--max_vocab_size', type=int, default=50000)
    parser.add_argument('--no_shared_vocab', action='store_true',
                        help='默认code和text共用一张词表；加这个参数改成各自独立建词表')

    # 截断/padding相关(对应build_dataset.py)
    parser.add_argument('--max_code_len', type=int, default=300)
    parser.add_argument('--max_text_len', type=int, default=50)
    parser.add_argument('--stmt_max_num', type=int, default=64)
    parser.add_argument('--block_max_num', type=int, default=24)
    parser.add_argument('--nest_depth_max', type=int, default=8)

    # 预处理改进开关(词表和数据两步共用同一组，保证一致)
    parser.add_argument('--text_mode', type=str, default='raw', choices=['raw', 'pretokenized'],
                        help="摘要文本模式。raw=原始摘要(小写丢标点)；pretokenized=摘要已做词形还原且带'..'结束标记"
                             "(HetCoS的token_data)，直接按空格切。词表和数据必须用同一个模式")
    parser.add_argument('--include_modifiers', action='store_true',
                        help='签名语句保留修饰符/注解/泛型参数(public/static/@Override等)')
    parser.add_argument('--strip_braces', action='store_true',
                        help='去掉语句首尾的结构性花括号')
    args = parser.parse_args()
    builder_opts = dict(include_modifiers=args.include_modifiers, strip_braces=args.strip_braces)
    from code_tokenizer import set_text_mode
    set_text_mode(args.text_mode)
    print('摘要文本模式:', args.text_mode)

    print('=' * 60)
    print('第1步: 建词表')
    print('=' * 60)
    train_path = os.path.join(args.raw_data_dir, 'train_data.json')
    train_data = load_json(train_path)
    vocab_result = build_vocabs(train_data, min_count=args.min_count,
                                max_vocab_size=args.max_vocab_size,
                                shared=not args.no_shared_vocab, builder_opts=builder_opts)
    print_report(vocab_result, args.min_count, args.max_vocab_size)

    os.makedirs(os.path.dirname(args.vocab_out_path) or '.', exist_ok=True)
    save_obj = {k: v for k, v in vocab_result.items()
               if k not in ('code_counter', 'text_counter', 'stmt_type_counter', 'block_type_counter')}
    with open(args.vocab_out_path, 'wb') as f:
        pickle.dump(save_obj, f)
    print('词表已保存: {}'.format(args.vocab_out_path))

    print('\n' + '=' * 60)
    print('第2步: 构建训练数据(截断+紧凑重编号+padding)')
    print('=' * 60)
    vocab = load_vocab(args.vocab_out_path)
    os.makedirs(args.avail_data_out_dir, exist_ok=True)

    for split in ('train', 'valid', 'test'):
        path = os.path.join(args.raw_data_dir, '{}_data.json'.format(split))
        if not os.path.exists(path):
            print('[跳过] 未找到 {}'.format(path))
            continue
        print('-' * 60)
        print('处理: {}'.format(path))
        data = load_json(path)
        stacked = build_split(data, vocab, args.max_code_len, args.max_text_len,
                              args.stmt_max_num, args.block_max_num, args.nest_depth_max,
                              desc=split, builder_opts=builder_opts)
        out_path = os.path.join(args.avail_data_out_dir, '{}_data.pkl'.format(split))
        with open(out_path, 'wb') as f:
            pickle.dump(dict(data=stacked, params=dict(
                max_code_len=args.max_code_len, max_text_len=args.max_text_len,
                stmt_max_num=args.stmt_max_num, block_max_num=args.block_max_num,
                nest_depth_max=args.nest_depth_max, text_mode=args.text_mode, **builder_opts)), f)
        print('已保存: {}'.format(out_path))

    print('\n预处理全部完成。')
    print('词表: {}'.format(args.vocab_out_path))
    print('训练数据目录: {}'.format(args.avail_data_out_dir))


if __name__ == '__main__':
    main()