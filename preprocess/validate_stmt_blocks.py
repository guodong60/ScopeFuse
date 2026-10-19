# coding=utf-8
"""
对raw_data下的train/valid/test.json全量跑一遍stmt_block_builder + code_tokenizer，输出：
  1. 解析成功率
  2. 语句数/block数/最大嵌套深度的分布(用于判断stmt_max_num等超参数该设多大，
     也顺便能看出后面搭Dataset时padding长度大概设多少合理)
  3. 子词化后的token数分布、子词膨胀比(subtoken/原始token)、摘要文本分词后长度分布
     (用于判断max_code_len/max_text_len该设多大)
  4. 失败样本抽样(默认最多20条)，包含错误信息和该方法code的前200字符，
     方便定位是哪类语法结构触发的解析失败

用法：
    python3 validate_stmt_blocks.py --raw_data_dir /path/to/raw_data --max_fail_show 20

会依次处理 train_data.json / valid_data.json / test_data.json（只要目录下存在就处理，
不强制三个文件都在）。
"""
import argparse
import json
import os
import sys
from collections import Counter

from stmt_block_builder import iter_build_stmt_blocks
from code_tokenizer import subtokenize_code, tokenize_text


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_one_file(path, max_fail_show=20):
    data = load_json(path)
    total = len(data)
    success = 0
    fail_samples = []  # [(id, error_str, code_snippet)]
    stmt_counts, block_counts, max_depths = [], [], []
    subtoken_counts, expansion_ratios, text_len_counts = [], [], []

    # 循环+进度条+异常收集这套逻辑统一由stmt_block_builder.py里的iter_build_stmt_blocks维护，
    # 这里只负责消费结果、做统计——以后build_dataset.py也应该这样用，不再各自重写一遍
    for item, result, error in iter_build_stmt_blocks(data, desc=os.path.basename(path)):
        if error is None:
            code_tokens, token_types, token2stmt, stmt_infos, block_infos = result
            success += 1
            stmt_counts.append(len(stmt_infos))
            block_counts.append(len(block_infos))
            max_depths.append(max(s['nest_depth'] for s in stmt_infos))

            subtokens, sub2stmt = subtokenize_code(code_tokens, token_types, token2stmt)
            subtoken_counts.append(len(subtokens))
            expansion_ratios.append(len(subtokens) / len(code_tokens))

            text_tokens = tokenize_text(item.get('text', ''))
            text_len_counts.append(len(text_tokens))
        else:
            if len(fail_samples) < max_fail_show:
                item_id = item.get('id', '?')
                err_str = '{}: {}'.format(type(error).__name__, error)
                code_snippet = item.get('code', '')[:200].replace('\n', ' ')
                fail_samples.append((item_id, err_str, code_snippet))

    return dict(
        path=path,
        total=total,
        success=success,
        fail=total - success,
        success_rate=success / total if total else 0.0,
        stmt_counts=stmt_counts,
        block_counts=block_counts,
        max_depths=max_depths,
        subtoken_counts=subtoken_counts,
        expansion_ratios=expansion_ratios,
        text_len_counts=text_len_counts,
        fail_samples=fail_samples,
    )


def print_distribution(name, values):
    if not values:
        print('  {}: 无数据(全部失败？)'.format(name))
        return
    values_sorted = sorted(values)
    n = len(values_sorted)

    def pct(p):
        idx = min(n - 1, int(n * p))
        return values_sorted[idx]

    print('  {}: 均值={:.1f}  中位数={}  p90={}  p99={}  max={}'.format(
        name, sum(values) / n, pct(0.5), pct(0.9), pct(0.99), values_sorted[-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True,
                        help='包含train_data.json/valid_data.json/test_data.json的目录')
    parser.add_argument('--max_fail_show', type=int, default=20)
    args = parser.parse_args()

    file_names = ['train_data.json', 'valid_data.json', 'test_data.json']
    any_found = False

    for fname in file_names:
        path = os.path.join(args.raw_data_dir, fname)
        if not os.path.exists(path):
            print('[跳过] 未找到 {}'.format(path))
            continue
        any_found = True
        print('=' * 60)
        print('处理: {}'.format(path))
        result = validate_one_file(path, max_fail_show=args.max_fail_show)

        print('总样本数: {}'.format(result['total']))
        print('解析成功: {} ({:.2%})'.format(result['success'], result['success_rate']))
        print('解析失败: {}'.format(result['fail']))
        print('-- 结构分布(仅统计解析成功的样本) --')
        print_distribution('每方法语句数(stmt count)', result['stmt_counts'])
        print_distribution('每方法block数', result['block_counts'])
        print_distribution('最大嵌套深度', result['max_depths'])
        print('-- 子词化/文本长度分布(用于确定max_code_len/max_text_len) --')
        print_distribution('子词化后code长度(subtoken count)', result['subtoken_counts'])
        if result['expansion_ratios']:
            avg_ratio = sum(result['expansion_ratios']) / len(result['expansion_ratios'])
            print('  子词/原始token 平均膨胀比: {:.2f}x'.format(avg_ratio))
        print_distribution('摘要文本分词后长度(text token count)', result['text_len_counts'])

        if result['fail_samples']:
            print('-- 失败样本抽样(最多{}条) --'.format(args.max_fail_show))
            for item_id, err, snippet in result['fail_samples']:
                print('  id={} | {} | code前200字符: {}'.format(item_id, err, snippet))
        print()

    if not any_found:
        print('警告: 在 {} 下没有找到任何 train/valid/test_data.json，请检查路径是否正确'.format(
            args.raw_data_dir))
        sys.exit(1)


if __name__ == '__main__':
    main()