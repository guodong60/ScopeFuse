# """
# 诊断：参考摘要里有多少词不在词表里(会变成<unk>)，其中有多少其实出现在源码里(本来可以靠复制得到)。
# 这决定了"为词表外的词做复制(扩展词表)"值不值得做。只读数据，不训练，不需要GPU。

# 用法(在项目根目录)：
#   python3 diagnose/diag_oov.py --vocab_path w2i2w/vocab.pkl --raw_path raw_data/test_data.json
# """
# import argparse, json, pickle, sys, os
# HERE = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(HERE, '..', 'preprocess'))
# from stmt_block_builder import iter_build_stmt_blocks   # noqa: E402
# from code_tokenizer import subtokenize_code, tokenize_text  # noqa: E402


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--vocab_path', required=True)
#     ap.add_argument('--raw_path', required=True)
#     ap.add_argument('--include_modifiers', action='store_true')
#     ap.add_argument('--strip_braces', action='store_true')
#     ap.add_argument('--max_samples', type=int, default=0, help='0=全部')
#     args = ap.parse_args()

#     vocab = pickle.load(open(args.vocab_path, 'rb'))
#     w2i = vocab['w2i'] if vocab.get('shared') else vocab['text_w2i']
#     data = json.load(open(args.raw_path, encoding='utf-8'))
#     if args.max_samples:
#         data = data[:args.max_samples]

#     n_samples = n_tok = n_oov = n_oov_in_code = 0
#     n_samples_with_copyable = 0
#     n_sent_all_in_vocab = 0
#     for item, result, err in iter_build_stmt_blocks(
#             data, desc='诊断', include_modifiers=args.include_modifiers, strip_braces=args.strip_braces):
#         if err is not None or result is None:
#             continue
#         code_tokens, token_types, token2stmt, _, _ = result
#         subs, _ = subtokenize_code(code_tokens, token_types, token2stmt)
#         code_set = set(subs)
#         text = tokenize_text(item.get('text', ''))
#         n_samples += 1
#         oov = [t for t in text if t not in w2i]
#         n_tok += len(text)
#         n_oov += len(oov)
#         copyable = [t for t in oov if t in code_set]
#         n_oov_in_code += len(copyable)
#         n_samples_with_copyable += bool(copyable)
#         n_sent_all_in_vocab += (len(oov) == 0)

#     pct = lambda a, b: 100.0 * a / max(1, b)
#     print('\n解析成功样本数: %d' % n_samples)
#     print('参考摘要词总数: %d' % n_tok)
#     print('  词表外(会变成<unk>)的词: %d  (%.2f%% 的参考词)' % (n_oov, pct(n_oov, n_tok)))
#     print('  其中出现在源码里、本可复制的词: %d  (词表外词的 %.1f%%，占全部参考词 %.2f%%)'
#           % (n_oov_in_code, pct(n_oov_in_code, n_oov), pct(n_oov_in_code, n_tok)))
#     print('含有"本可复制的词表外词"的样本: %d / %d  (%.1f%%)' % (n_samples_with_copyable, n_samples,
#                                                           pct(n_samples_with_copyable, n_samples)))
#     print('参考摘要完全在词表内的样本: %.1f%%' % pct(n_sent_all_in_vocab, n_samples))
#     print('\n解读：这些样本在当前设置下，模型最多只能输出<unk>，一定拿不到满分。')


# if __name__ == '__main__':
#     main()

"""
诊断：参考摘要里有多少词不在词表里(会变成<unk>)，其中有多少其实出现在源码里(本来可以靠复制得到)。
这决定了"为词表外的词做复制(扩展词表)"值不值得做。只读数据，不训练，不需要GPU。

用法(在项目根目录)：
  python3 diagnose/diag_oov.py --vocab_path w2i2w/vocab.pkl --raw_path raw_data/test_data.json
"""
import argparse, json, pickle, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'preprocess'))
from stmt_block_builder import iter_build_stmt_blocks   # noqa: E402
from code_tokenizer import subtokenize_code, tokenize_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab_path', required=True)
    ap.add_argument('--raw_path', required=True)
    ap.add_argument('--include_modifiers', action='store_true')
    ap.add_argument('--strip_braces', action='store_true')
    ap.add_argument('--max_samples', type=int, default=0, help='0=全部')
    ap.add_argument('--text_mode', default='raw', choices=['raw', 'pretokenized'])
    args = ap.parse_args()
    from code_tokenizer import set_text_mode
    set_text_mode(args.text_mode)

    vocab = pickle.load(open(args.vocab_path, 'rb'))
    w2i = vocab['w2i'] if vocab.get('shared') else vocab['text_w2i']
    data = json.load(open(args.raw_path, encoding='utf-8'))
    if args.max_samples:
        data = data[:args.max_samples]

    n_samples = n_tok = n_oov = n_oov_in_code = 0
    n_samples_with_copyable = 0
    n_sent_all_in_vocab = 0
    for item, result, err in iter_build_stmt_blocks(
            data, desc='诊断', include_modifiers=args.include_modifiers, strip_braces=args.strip_braces):
        if err is not None or result is None:
            continue
        code_tokens, token_types, token2stmt, _, _ = result
        subs, _ = subtokenize_code(code_tokens, token_types, token2stmt)
        code_set = set(subs)
        text = tokenize_text(item.get('text', ''))
        n_samples += 1
        oov = [t for t in text if t not in w2i]
        n_tok += len(text)
        n_oov += len(oov)
        copyable = [t for t in oov if t in code_set]
        n_oov_in_code += len(copyable)
        n_samples_with_copyable += bool(copyable)
        n_sent_all_in_vocab += (len(oov) == 0)

    pct = lambda a, b: 100.0 * a / max(1, b)
    print('\n解析成功样本数: %d' % n_samples)
    print('参考摘要词总数: %d' % n_tok)
    print('  词表外(会变成<unk>)的词: %d  (%.2f%% 的参考词)' % (n_oov, pct(n_oov, n_tok)))
    print('  其中出现在源码里、本可复制的词: %d  (词表外词的 %.1f%%，占全部参考词 %.2f%%)'
          % (n_oov_in_code, pct(n_oov_in_code, n_oov), pct(n_oov_in_code, n_tok)))
    print('含有"本可复制的词表外词"的样本: %d / %d  (%.1f%%)' % (n_samples_with_copyable, n_samples,
                                                          pct(n_samples_with_copyable, n_samples)))
    print('参考摘要完全在词表内的样本: %.1f%%' % pct(n_sent_all_in_vocab, n_samples))
    print('\n解读：这些样本在当前设置下，模型最多只能输出<unk>，一定拿不到满分。')


if __name__ == '__main__':
    main()