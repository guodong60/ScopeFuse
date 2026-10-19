"""
分桶对比：按代码长度/语句数/最大嵌套深度把测试样本分桶，对比多个模型在每个桶里的分数，
并对"相对基线的差"做配对bootstrap(对测试样本重采样)给出95%区间。
用来回答：层次编码器是否在长代码/深嵌套的样本上优于普通Transformer。不训练，不需要GPU。

注意：
  - 分数用你的 metrics.py 口径(sacrebleu句子级S-BLEU、rouge_score ROUGE-L)，只用于模型之间的相对比较，
    绝对数值不能和论文比；
  - bootstrap只反映"测试样本抽样"的不确定性，不包含不同随机种子之间的波动。

用法(在项目根目录 java/ 下)：
  T=/data/home/wls_gd/my_model/hier_claude/data/java/token_data
  python3 diagnose/bucket_eval.py --test_json $T/test_data.json --text_mode pretokenized \
      --results exp9=result/exp9_tk_base_test.json exp20=result/exp20_tk_plain4_test.json \
                exp21=result/exp21_tk_plain6_test.json exp10=result/exp10_tk_embedonly_test.json
第一个结果是基线，其余模型都和它比。
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'preprocess'))
sys.path.insert(0, os.path.join(HERE, '..', 'train'))
from stmt_block_builder import iter_build_stmt_blocks   # noqa: E402
from code_tokenizer import subtokenize_code, set_text_mode   # noqa: E402
from metrics import compute_metrics                        # noqa: E402
import sacrebleu                                           # noqa: E402


def code_features(items):
    feats = {}
    for item, result, err in iter_build_stmt_blocks(items, desc='统计代码特征'):
        if err is not None or result is None:
            continue
        code_tokens, token_types, token2stmt, stmt_infos, block_infos = result
        subs, _ = subtokenize_code(code_tokens, token_types, token2stmt)
        feats[item['id']] = dict(n_sub=len(subs), n_stmt=len(stmt_infos),
                                 depth=max([s['nest_depth'] for s in stmt_infos] + [0]))
    return feats


def make_buckets(values, n=4):
    """按分位数分桶，重复边界会合并；返回每个样本的桶编号和桶名"""
    edges = sorted(set(np.quantile(values, [i / n for i in range(1, n)]).tolist()))
    idx = np.searchsorted(edges, values, side='left')
    names, lo = [], None
    for b in range(len(edges) + 1):
        hi = edges[b] if b < len(edges) else None
        names.append(('<=%g' % hi) if b == 0 else (('>%g' % lo) if hi is None else '(%g,%g]' % (lo, hi)))
        lo = hi
    return idx, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_json', required=True)
    ap.add_argument('--text_mode', default='pretokenized', choices=['raw', 'pretokenized'])
    ap.add_argument('--results', nargs='+', required=True, help='名字=结果json路径，第一个是基线')
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()
    set_text_mode(args.text_mode)

    items = json.load(open(args.test_json, encoding='utf-8'))
    feats = code_features(items)
    models = []
    for spec in args.results:
        name, path = spec.split('=', 1)
        d = json.load(open(path, encoding='utf-8'))
        recs = {r['id']: r for r in (d['predictions'] if isinstance(d, dict) else d)}
        models.append((name, recs))
    ids = [i for i in feats if all(i in m[1] for m in models)]
    print('参与分析的样本数: %d' % len(ids))
    hyps = {n: [(m[i]['pred'] or '<empty>') for i in ids] for n, m in models}
    refs = [models[0][1][i]['ref'] for i in ids]
    sent = {n: np.array([sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps[n], refs)]) for n, _ in models}
    base = models[0][0]
    rng = np.random.default_rng(0)

    def boot_diff(a, b, sel):
        x = (a - b)[sel]
        if len(x) < 5:
            return float('nan'), float('nan')
        m = rng.integers(0, len(x), size=(args.boot, len(x)))
        means = x[m].mean(1)
        return np.percentile(means, 2.5), np.percentile(means, 97.5)

    all_sel = np.ones(len(ids), dtype=bool)
    print('\n===== 全体 (S-BLEU 均值；括号内是相对基线 %s 的差及95%%区间) =====' % base)
    for n, _ in models:
        d = '' if n == base else '  差 %+.2f [%+.2f, %+.2f]' % ((sent[n] - sent[base]).mean(), *boot_diff(sent[n], sent[base], all_sel))
        print('  %-8s S-BLEU %.2f%s' % (n, sent[n].mean(), d))

    for fname, key in (('代码子词数', 'n_sub'), ('语句数', 'n_stmt'), ('最大嵌套深度', 'depth')):
        vals = np.array([feats[i][key] for i in ids])
        bidx, bnames = make_buckets(vals)
        print('\n===== 按 %s 分桶 =====' % fname)
        print('%-12s %6s | ' % ('桶', '样本数') + ' | '.join('%-30s' % (n if n == base else n + ' (对基线差,95%区间)') for n, _ in models))
        for b, bn in enumerate(bnames):
            sel = bidx == b
            if sel.sum() < 5:
                continue
            cells = []
            for n, _ in models:
                m = compute_metrics([hyps[n][k] for k in np.where(sel)[0]], [refs[k] for k in np.where(sel)[0]], full=True)
                if n == base:
                    cells.append('%-30s' % ('S-BLEU %.2f ROUGE-L %.2f' % (sent[n][sel].mean(), m['rougeL'])))
                else:
                    lo, hi = boot_diff(sent[n], sent[base], sel)
                    cells.append('%-30s' % ('%.2f(%+.2f[%+.2f,%+.2f]) R%.1f' % (sent[n][sel].mean(), (sent[n][sel] - sent[base][sel]).mean(), lo, hi, m['rougeL'])))
            print('%-12s %6d | ' % (bn, sel.sum()) + ' | '.join(cells))
    print('\n读法：区间不包含0才说明该桶里的差异在测试样本抽样意义下可信；区间跨0就是看不出差别。')


if __name__ == '__main__':
    main()