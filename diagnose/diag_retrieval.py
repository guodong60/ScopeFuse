# """
# 诊断：最近邻检索基线 + 训练/测试近似重复统计。不训练模型，不需要GPU。

# 做法：对每个测试样本，在训练集里按代码子词的TF-IDF余弦相似度找最像的一条，
# 直接把那条的摘要当作预测，用和 evaluate.py 相同的 metrics.py 打分。
# 它回答两个问题：
#   1) "零模型"的检索基线能拿多少分？你的模型是否比它高、高多少？
#   2) 测试集里有多少样本在训练集里有几乎相同的代码(近似重复)？
# 可选：给出 --result_path(evaluate.py 生成的json)，会按相似度分组，对比检索基线和你的模型。
# 也可以用来评估论文作者发布的预测(HETCOS仓库 data/java/result/*.json)：加
#   --hyp_key pred_text --ref_key gold_text
# 脚本会按 id 对齐，并用 raw_code 校验对齐是否正确(对不上会给出警告，此时分组结果不可信)。
# 注意论文作者的参考摘要是词形还原后的，所以它和你的模型的分数只能看"相对分组趋势"。

# 用法(在项目根目录)：
#   pip install scikit-learn
#   python3 diagnose/diag_retrieval.py \
#       --train_raw raw_data/train_data.json --test_raw raw_data/test_data.json \
#       --result_path result/exp1_structpos_test.json
# """
# import argparse, json, os, sys
# import numpy as np

# HERE = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(HERE, '..', 'preprocess'))
# sys.path.insert(0, os.path.join(HERE, '..', 'train'))
# from stmt_block_builder import iter_build_stmt_blocks   # noqa: E402
# from code_tokenizer import subtokenize_code, tokenize_text  # noqa: E402
# from metrics import compute_metrics                        # noqa: E402


# def load(path, desc):
#     data = json.load(open(path, encoding='utf-8'))
#     out = []
#     for item, result, err in iter_build_stmt_blocks(data, desc=desc):
#         if err is not None or result is None:
#             continue
#         code_tokens, token_types, token2stmt, _, _ = result
#         subs, _ = subtokenize_code(code_tokens, token_types, token2stmt)
#         out.append(dict(id=item.get('id'), raw=item.get('code', ''), code=subs, text=' '.join(tokenize_text(item.get('text', '')))))
#     return out


# def score(hyps, refs):
#     m = compute_metrics(hyps, refs, full=True)
#     return m['s_bleu'], m['rougeL']


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--train_raw', required=True)
#     ap.add_argument('--test_raw', required=True)
#     ap.add_argument('--result_path', default=None)
#     ap.add_argument('--chunk', type=int, default=500)
#     ap.add_argument('--hyp_key', default='pred')
#     ap.add_argument('--ref_key', default='ref')
#     ap.add_argument('--code_key', default='raw_code', help='结果json里的原始代码字段(有则用于校验id对齐)')
#     args = ap.parse_args()

#     from sklearn.feature_extraction.text import TfidfVectorizer
#     train = load(args.train_raw, 'train')
#     test = load(args.test_raw, 'test')
#     print('train: %d  test: %d' % (len(train), len(test)))

#     vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True)
#     Xtr = vec.fit_transform([s['code'] for s in train])
#     Xte = vec.transform([s['code'] for s in test])

#     best_idx = np.zeros(len(test), dtype=int)
#     best_sim = np.zeros(len(test))
#     for i in range(0, len(test), args.chunk):
#         sims = (Xte[i:i + args.chunk] @ Xtr.T).toarray()
#         best_idx[i:i + args.chunk] = sims.argmax(1)
#         best_sim[i:i + args.chunk] = sims.max(1)

#     hyps = [train[j]['text'] if train[j]['text'].strip() else '<empty>' for j in best_idx]
#     refs = [s['text'] for s in test]
#     sb, rl = score(hyps, refs)

#     train_codes = set(tuple(s['code']) for s in train)
#     train_texts = set(s['text'] for s in train)
#     dup_code = np.mean([tuple(s['code']) in train_codes for s in test])
#     dup_text = np.mean([s['text'] in train_texts for s in test])
#     top1_exact = np.mean([h == r for h, r in zip(hyps, refs)])

#     print('\n===== 近似重复统计 =====')
#     print('测试代码(子词序列)与训练集完全相同的比例: %.1f%%' % (100 * dup_code))
#     print('测试摘要与训练集某条摘要完全相同的比例  : %.1f%%' % (100 * dup_text))
#     print('检索top1的摘要与参考完全一致的比例      : %.1f%%' % (100 * top1_exact))
#     for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0, 0.5)):
#         print('  代码相似度 [%.1f, %.1f): %.1f%% 的测试样本' % (lo, min(hi, 1.0),
#                                                        100 * np.mean((best_sim >= lo) & (best_sim < hi))))

#     print('\n===== 检索基线(你的 metrics.py 口径，可直接和 evaluate.py 的数字比) =====')
#     print('S-BLEU = %.2f   ROUGE-L = %.2f' % (sb, rl))

#     if args.result_path:
#         data = json.load(open(args.result_path, encoding='utf-8'))
#         recs = data['predictions'] if isinstance(data, dict) else data
#         by_id = {r['id']: r for r in recs}
#         rows = [(k, test[k]['id']) for k in range(len(test)) if test[k]['id'] in by_id]
#         has_code = [k for k, _ in rows if args.code_key in by_id[test[k]['id']]]
#         if has_code:
#             squash = lambda t: ''.join(t.split())
#             ok = np.mean([squash(test[k]['raw']) == squash(by_id[test[k]['id']][args.code_key]) for k in has_code])
#             print('\nid对齐校验：%d 条带原始代码，其中代码完全一致的比例 = %.1f%%' % (len(has_code), 100 * ok))
#             if ok < 0.95:
#                 print('!! 警告：对齐比例偏低，id 可能不是同一套编号，下面的分组结果不可信 !!')
#         print('\n===== 按相似度分组：检索基线 vs 你的模型(%d 条可对齐) =====' % len(rows))
#         print('%-16s %6s | %-22s | %-22s' % ('代码相似度', '样本数', '检索 S-BLEU/ROUGE-L', '模型 S-BLEU/ROUGE-L'))
#         for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0, 0.5)):
#             sel = [k for k, _ in rows if lo <= best_sim[k] < hi]
#             if len(sel) < 5:
#                 continue
#             h1 = [hyps[k] for k in sel]
#             r1 = [refs[k] for k in sel]
#             h2 = [by_id[test[k]['id']][args.hyp_key] or '<empty>' for k in sel]
#             r2 = [by_id[test[k]['id']][args.ref_key] for k in sel]
#             a, b = score(h1, r1)
#             c, d = score(h2, r2)
#             print('[%.1f, %.1f)       %6d | %9.2f / %-9.2f   | %9.2f / %-9.2f' % (lo, min(hi, 1.0), len(sel), a, b, c, d))
#     print('\n注：检索基线的参考/预测都是原词(未<unk>化)，模型结果里的参考若含<unk>会有极小差异，不影响量级判断。')


# if __name__ == '__main__':
#     main()

"""
诊断：最近邻检索基线 + 训练/测试近似重复统计。不训练模型，不需要GPU。

做法：对每个测试样本，在训练集里按代码子词的TF-IDF余弦相似度找最像的一条，
直接把那条的摘要当作预测，用和 evaluate.py 相同的 metrics.py 打分。
它回答两个问题：
  1) "零模型"的检索基线能拿多少分？你的模型是否比它高、高多少？
  2) 测试集里有多少样本在训练集里有几乎相同的代码(近似重复)？
可选：给出 --result_path(evaluate.py 生成的json)，会按相似度分组，对比检索基线和你的模型。
也可以用来评估论文作者发布的预测(HETCOS仓库 data/java/result/*.json)：加
  --hyp_key pred_text --ref_key gold_text
脚本会按 id 对齐，并用 raw_code 校验对齐是否正确(对不上会给出警告，此时分组结果不可信)。
注意论文作者的参考摘要是词形还原后的，所以它和你的模型的分数只能看"相对分组趋势"。

用法(在项目根目录)：
  pip install scikit-learn
  python3 diagnose/diag_retrieval.py \
      --train_raw raw_data/train_data.json --test_raw raw_data/test_data.json \
      --result_path result/exp1_structpos_test.json
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'preprocess'))
sys.path.insert(0, os.path.join(HERE, '..', 'train'))
from stmt_block_builder import iter_build_stmt_blocks   # noqa: E402
from code_tokenizer import subtokenize_code, tokenize_text  # noqa: E402
from metrics import compute_metrics                        # noqa: E402


def load(path, desc):
    data = json.load(open(path, encoding='utf-8'))
    out = []
    for item, result, err in iter_build_stmt_blocks(data, desc=desc):
        if err is not None or result is None:
            continue
        code_tokens, token_types, token2stmt, _, _ = result
        subs, _ = subtokenize_code(code_tokens, token_types, token2stmt)
        out.append(dict(id=item.get('id'), raw=item.get('code', ''), code=subs, text=' '.join(tokenize_text(item.get('text', '')))))
    return out


def score(hyps, refs):
    m = compute_metrics(hyps, refs, full=True)
    return m['s_bleu'], m['rougeL']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_raw', required=True)
    ap.add_argument('--test_raw', required=True)
    ap.add_argument('--result_path', default=None)
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--text_mode', default='raw', choices=['raw', 'pretokenized'])
    ap.add_argument('--hyp_key', default='pred')
    ap.add_argument('--ref_key', default='ref')
    ap.add_argument('--code_key', default='raw_code', help='结果json里的原始代码字段(有则用于校验id对齐)')
    args = ap.parse_args()
    from code_tokenizer import set_text_mode
    set_text_mode(args.text_mode)

    from sklearn.feature_extraction.text import TfidfVectorizer
    train = load(args.train_raw, 'train')
    test = load(args.test_raw, 'test')
    print('train: %d  test: %d' % (len(train), len(test)))

    vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True)
    Xtr = vec.fit_transform([s['code'] for s in train])
    Xte = vec.transform([s['code'] for s in test])

    best_idx = np.zeros(len(test), dtype=int)
    best_sim = np.zeros(len(test))
    for i in range(0, len(test), args.chunk):
        sims = (Xte[i:i + args.chunk] @ Xtr.T).toarray()
        best_idx[i:i + args.chunk] = sims.argmax(1)
        best_sim[i:i + args.chunk] = sims.max(1)

    hyps = [train[j]['text'] if train[j]['text'].strip() else '<empty>' for j in best_idx]
    refs = [s['text'] for s in test]
    sb, rl = score(hyps, refs)

    train_codes = set(tuple(s['code']) for s in train)
    train_texts = set(s['text'] for s in train)
    dup_code = np.mean([tuple(s['code']) in train_codes for s in test])
    dup_text = np.mean([s['text'] in train_texts for s in test])
    top1_exact = np.mean([h == r for h, r in zip(hyps, refs)])

    print('\n===== 近似重复统计 =====')
    print('测试代码(子词序列)与训练集完全相同的比例: %.1f%%' % (100 * dup_code))
    print('测试摘要与训练集某条摘要完全相同的比例  : %.1f%%' % (100 * dup_text))
    print('检索top1的摘要与参考完全一致的比例      : %.1f%%' % (100 * top1_exact))
    for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0, 0.5)):
        print('  代码相似度 [%.1f, %.1f): %.1f%% 的测试样本' % (lo, min(hi, 1.0),
                                                       100 * np.mean((best_sim >= lo) & (best_sim < hi))))

    print('\n===== 检索基线(你的 metrics.py 口径，可直接和 evaluate.py 的数字比) =====')
    print('S-BLEU = %.2f   ROUGE-L = %.2f' % (sb, rl))

    if args.result_path:
        data = json.load(open(args.result_path, encoding='utf-8'))
        recs = data['predictions'] if isinstance(data, dict) else data
        by_id = {r['id']: r for r in recs}
        rows = [(k, test[k]['id']) for k in range(len(test)) if test[k]['id'] in by_id]
        has_code = [k for k, _ in rows if args.code_key in by_id[test[k]['id']]]
        if has_code:
            squash = lambda t: ''.join(t.split())
            ok = np.mean([squash(test[k]['raw']) == squash(by_id[test[k]['id']][args.code_key]) for k in has_code])
            print('\nid对齐校验：%d 条带原始代码，其中代码完全一致的比例 = %.1f%%' % (len(has_code), 100 * ok))
            if ok < 0.95:
                print('!! 警告：对齐比例偏低，id 可能不是同一套编号，下面的分组结果不可信 !!')
        print('\n===== 按相似度分组：检索基线 vs 你的模型(%d 条可对齐) =====' % len(rows))
        print('%-16s %6s | %-22s | %-22s' % ('代码相似度', '样本数', '检索 S-BLEU/ROUGE-L', '模型 S-BLEU/ROUGE-L'))
        for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0, 0.5)):
            sel = [k for k, _ in rows if lo <= best_sim[k] < hi]
            if len(sel) < 5:
                continue
            h1 = [hyps[k] for k in sel]
            r1 = [refs[k] for k in sel]
            h2 = [by_id[test[k]['id']][args.hyp_key] or '<empty>' for k in sel]
            r2 = [by_id[test[k]['id']][args.ref_key] for k in sel]
            a, b = score(h1, r1)
            c, d = score(h2, r2)
            print('[%.1f, %.1f)       %6d | %9.2f / %-9.2f   | %9.2f / %-9.2f' % (lo, min(hi, 1.0), len(sel), a, b, c, d))
    print('\n注：检索基线的参考/预测都是原词(未<unk>化)，模型结果里的参考若含<unk>会有极小差异，不影响量级判断。')


if __name__ == '__main__':
    main()