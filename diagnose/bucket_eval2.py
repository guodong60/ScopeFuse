"""
分桶对比 v2：回答"层次结构在哪些代码上有用"。不训练，不需要GPU。

相比 bucket_eval.py 多了：
  1. 第四个分组维度：测试代码与训练集最相似一条的相似度(TF-IDF余弦，和 diag_retrieval.py 同一做法)。
     近似重复样本几乎不需要泛化，层次结构的作用更可能出现在低相似度样本上。
  2. 每个分组都分两个范围报告：全部样本、低相似度子集(相似度 < --sim_thresh，默认0.9)。
     分桶边界在全部样本上定下，两个范围共用，方便对照。
  3. 每个维度给一个趋势检验：逐样本的差值(对比模型 - 基线)与该特征的 Spearman 相关系数及 bootstrap 区间。
     它直接回答"优势是否随长度/语句数/深度增加"，一个维度只做一次检验，比盯着四个桶各自的区间更不容易被噪声骗。
  4. ROUGE-L 也给逐样本差值的 bootstrap 区间(原来只给S-BLEU)。
  5. 汇总"区间排除0的格子数 / 总格子数"，并给出纯随机情况下的期望个数，防止从许多格子里挑出个别显著的当结论。
  6. 输出长表 CSV，方便做表格和图。

分数口径：S-BLEU 用 metrics.sentence_bleu(sacrebleu)，ROUGE-L 用 metrics 里的 rouge_score，
逐样本计算后取均值(ROUGE-L 均值与 compute_metrics 一致)。只用于模型之间的相对比较，
绝对数值不能和论文的官方口径比。bootstrap 只反映测试样本抽样的不确定性，不含不同种子之间的波动，
所以建议把同一结构的另一个种子也作为一个"对比模型"放进来，作为种子噪声的参照。

用法(在项目根目录 java/ 下；第一个结果是基线)：
  T=<token_data目录>
  python3 diagnose/bucket_eval2.py \
      --train_json $T/train_data.json --test_json $T/test_data.json --text_mode pretokenized \
      --results main=result/exp9_tk_base_test.json \
                seed1=result/exp9s1_tk_base_test.json \
                plain4=result/exp20_tk_plain4_test.json \
                noinject=result/exp25_localonly_test.json \
      --out_csv result/bucket_eval2.csv
"""
import argparse, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'preprocess'))
sys.path.insert(0, os.path.join(HERE, '..', 'train'))
from stmt_block_builder import iter_build_stmt_blocks   # noqa: E402
from code_tokenizer import subtokenize_code, set_text_mode   # noqa: E402
from metrics import sentence_bleu, _rouge_scorer          # noqa: E402


def load_features(items, desc):
    """每个样本：id、子词序列、子词数、语句数、最大嵌套深度"""
    out = {}
    for item, result, err in iter_build_stmt_blocks(items, desc=desc):
        if err is not None or result is None:
            continue
        code_tokens, token_types, token2stmt, stmt_infos, block_infos = result
        subs, _ = subtokenize_code(code_tokens, token_types, token2stmt)
        out[item['id']] = dict(subs=subs, n_sub=len(subs), n_stmt=len(stmt_infos),
                               depth=max([s['nest_depth'] for s in stmt_infos] + [0]))
    return out


def best_similarity(train_feats, test_feats, test_ids, chunk=500):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True)
    xtr = vec.fit_transform([f['subs'] for f in train_feats.values()])
    xte = vec.transform([test_feats[i]['subs'] for i in test_ids])
    sims = np.zeros(len(test_ids))
    for s in range(0, len(test_ids), chunk):
        sims[s:s + chunk] = (xte[s:s + chunk] @ xtr.T).toarray().max(1)
    return sims


def quantile_edges(values, n=4):
    return sorted(set(np.quantile(values, [i / n for i in range(1, n)]).tolist()))


def bucketize(values, edges):
    idx = np.searchsorted(edges, values, side='left')
    names, lo = [], None
    for b in range(len(edges) + 1):
        hi = edges[b] if b < len(edges) else None
        names.append(('<=%g' % hi) if b == 0 else (('>%g' % lo) if hi is None else '(%g,%g]' % (lo, hi)))
        lo = hi
    return idx, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_json', required=True)
    ap.add_argument('--test_json', required=True)
    ap.add_argument('--text_mode', default='pretokenized', choices=['raw', 'pretokenized'])
    ap.add_argument('--results', nargs='+', required=True, help='名字=结果json路径，第一个是基线')
    ap.add_argument('--sim_thresh', type=float, default=0.9, help='低相似度子集：相似度小于该值')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()
    set_text_mode(args.text_mode)

    train_feats = load_features(json.load(open(args.train_json, encoding='utf-8')), '训练集特征')
    test_feats = load_features(json.load(open(args.test_json, encoding='utf-8')), '测试集特征')

    models = []
    for spec in args.results:
        name, path = spec.split('=', 1)
        d = json.load(open(path, encoding='utf-8'))
        recs = {r['id']: r for r in (d['predictions'] if isinstance(d, dict) else d)}
        models.append((name, recs))
    ids = [i for i in test_feats if all(i in m[1] for m in models)]
    print('参与分析的样本数: %d' % len(ids))
    base = models[0][0]
    refs = [models[0][1][i]['ref'] or '<empty>' for i in ids]

    sims = best_similarity(train_feats, test_feats, ids)
    feats = {
        '代码子词数': np.array([test_feats[i]['n_sub'] for i in ids], dtype=float),
        '语句数': np.array([test_feats[i]['n_stmt'] for i in ids], dtype=float),
        '最大嵌套深度': np.array([test_feats[i]['depth'] for i in ids], dtype=float),
        '与训练集最相似代码的相似度': sims,
    }
    sim_edges = [0.5, 0.7, 0.9]            # 相似度用固定边界，和 diag_retrieval 的分组一致
    edges = {k: (sim_edges if k.startswith('与训练集') else quantile_edges(v)) for k, v in feats.items()}

    print('逐样本打分 ...')
    score = {}
    for name, recs in models:
        hyps = [(recs[i]['pred'] or '<empty>') for i in ids]
        sb = np.array([sentence_bleu(h, r) for h, r in zip(hyps, refs)])
        rl = np.array([100.0 * _rouge_scorer.score(r, h)['rougeL'].fmeasure for h, r in zip(hyps, refs)])
        score[name] = dict(sb=sb, rl=rl)

    rng = np.random.default_rng(0)

    def boot_mean(x):
        if len(x) < 10:
            return float('nan'), float('nan')
        m = rng.integers(0, len(x), size=(args.boot, len(x)))
        mean = x[m].mean(1)
        return float(np.percentile(mean, 2.5)), float(np.percentile(mean, 97.5))

    def spearman(x, y):
        from scipy.stats import spearmanr
        r = spearmanr(x, y).correlation
        return 0.0 if r != r else float(r)

    def boot_spearman(x, y):
        if len(x) < 30:
            return float('nan'), float('nan'), float('nan')
        rho = spearman(x, y)
        bs = []
        for _ in range(300):
            k = rng.integers(0, len(x), size=len(x))
            bs.append(spearman(x[k], y[k]))
        return rho, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    low = sims < args.sim_thresh
    scopes = [('全部样本', np.ones(len(ids), dtype=bool)), ('低相似度(<%g)' % args.sim_thresh, low)]
    print('低相似度子集样本数: %d (%.1f%%)' % (low.sum(), 100.0 * low.mean()))

    rows = []
    n_cells = n_sig = 0
    others = [n for n, _ in models if n != base]
    for scope_name, scope in scopes:
        print('\n' + '=' * 78)
        print('范围：%s  (%d 条)' % (scope_name, scope.sum()))
        print('=' * 78)
        # 全体
        print('  全体 %-9s ' % base + 'S-BLEU %.2f  ROUGE-L %.2f' % (score[base]['sb'][scope].mean(), score[base]['rl'][scope].mean()))
        for n in others:
            dsb, drl = score[n]['sb'] - score[base]['sb'], score[n]['rl'] - score[base]['rl']
            lo, hi = boot_mean(dsb[scope]); lo2, hi2 = boot_mean(drl[scope])
            print('  全体 %-9s 对基线差  S-BLEU %+.2f [%+.2f,%+.2f]  ROUGE-L %+.2f [%+.2f,%+.2f]'
                  % (n, dsb[scope].mean(), lo, hi, drl[scope].mean(), lo2, hi2))
            rows.append([scope_name, '全体', 'all', int(scope.sum()), n, score[n]['sb'][scope].mean(), score[n]['rl'][scope].mean(),
                         dsb[scope].mean(), lo, hi, drl[scope].mean(), lo2, hi2])
        for fname, vals in feats.items():
            bidx, bnames = bucketize(vals, edges[fname])
            print('\n  --- 按 %s 分桶 ---' % fname)
            for b, bn in enumerate(bnames):
                sel = (bidx == b) & scope
                if sel.sum() < 10:
                    continue
                line = '  %-12s n=%-5d %s: S-BLEU %.1f R-L %.1f' % (bn, sel.sum(), base, score[base]['sb'][sel].mean(), score[base]['rl'][sel].mean())
                print(line)
                for n in others:
                    dsb, drl = score[n]['sb'] - score[base]['sb'], score[n]['rl'] - score[base]['rl']
                    lo, hi = boot_mean(dsb[sel]); lo2, hi2 = boot_mean(drl[sel])
                    sig = '*' if (lo > 0 or hi < 0) else ' '
                    n_cells += 1; n_sig += int(lo > 0 or hi < 0)
                    print('        %-9s S-BLEU差 %+.2f [%+.2f,%+.2f]%s  ROUGE-L差 %+.2f [%+.2f,%+.2f]'
                          % (n, dsb[sel].mean(), lo, hi, sig, drl[sel].mean(), lo2, hi2))
                    rows.append([scope_name, fname, bn, int(sel.sum()), n, score[n]['sb'][sel].mean(), score[n]['rl'][sel].mean(),
                                 dsb[sel].mean(), lo, hi, drl[sel].mean(), lo2, hi2])
            if not fname.startswith('与训练集'):
                print('  趋势检验(逐样本差值与特征的 Spearman 相关；区间含0=没有"随该特征增大而变化"的证据)：')
                for n in others:
                    d = (score[n]['sb'] - score[base]['sb'])[scope]
                    rho, lo, hi = boot_spearman(vals[scope], d)
                    print('        %-9s S-BLEU差 vs %s: rho=%+.3f [%+.3f,%+.3f]' % (n, fname, rho, lo, hi))
                    rows.append([scope_name, fname, '趋势(Spearman rho,对S-BLEU差)', int(scope.sum()), n, '', '', rho, lo, hi, '', '', ''])

    print('\n' + '=' * 78)
    print('汇总：S-BLEU 差值 95%% 区间排除 0 的格子 %d / %d；若所有差异都只是噪声，期望约 %.1f 个。' % (n_sig, n_cells, 0.05 * n_cells))
    print('读法：带*的格子要和这个期望比较；零星几个带*且方向不一致，不能当作证据。')
    print('      对比模型里放进"同结构另一个种子"，它的差值就是种子噪声的参照。')

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        with open(args.out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['范围', '维度', '桶', '样本数', '对比模型', '该模型S-BLEU', '该模型ROUGE-L',
                        'S-BLEU差(对基线)', '区间下限', '区间上限', 'ROUGE-L差', '区间下限', '区间上限'])
            w.writerows(rows)
        print('已写出: %s' % args.out_csv)


if __name__ == '__main__':
    main()