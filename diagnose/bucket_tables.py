"""
生成三张分组对比表(S-BLEU / ROUGE-L / METEOR)，用于画图。不训练，不占GPU。

每张表的行：{代码长度, 语句数, 嵌套深度} 三种分组依据 x {短, 中, 长} 三个区间；
列：分组依据 | 区间 | 区间范围 | 样本数 | 完整模型 | Transformer-6 | 无回注 | 完整-Transformer | 完整-无回注
分数用 HetCoS 官方指标函数(my_lib)，在每个区间的样本子集上整体计算，和你论文里的主表同一口径。
区间：按全部测试样本的三分位数划分(短/中/长各约三分之一；嵌套深度取值少，三分位会有重复边界，
      此时区间数可能少于3，范围列会写明)。

必须在能 import my_lib 的目录运行(和 official_eval.py 一样)，另外用 --java_root 指向项目 java/ 目录。
  cd ~/my_model/hier_claude/src_code/java/code_sum_41
  J=<本项目java目录>
  T=<token_data目录>
  python $J/diagnose/bucket_tables.py --java_root $J --test_json $T/test_data.json \
      --full $J/result/exp9_tk_base_test.json \
      --transformer $J/result/exp21_tk_plain6_test.json \
      --noinject $J/result/exp25_localonly_test.json \
      --seed1 $J/result/exp9s1_tk_base_test.json \
      --out_prefix $J/result/bucket_table

输出：<out_prefix>_sbleu.csv / _rougel.csv / _meteor.csv(画图直接读这三个)，以及终端打印。
--seed1 可选：同一完整模型的另一个种子，会多一列"种子差"作为噪声参照
(完整模型两个种子之间的差，画图时可当作"差异小于这个就看不出"的参考线)。
说明：这里只给点估计；是否可信看"差值是否明显大于种子差"，以及之前 bucket_eval2.py 的 bootstrap 区间。
"""
import argparse, csv, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--java_root', required=True)
    ap.add_argument('--test_json', required=True)
    ap.add_argument('--text_mode', default='pretokenized')
    ap.add_argument('--full', required=True)
    ap.add_argument('--transformer', required=True)
    ap.add_argument('--noinject', required=True)
    ap.add_argument('--seed1', default=None)
    ap.add_argument('--out_prefix', default='bucket_table')
    ap.add_argument('--my_lib_parent', default=None)
    args = ap.parse_args()
    if args.my_lib_parent:
        sys.path.insert(0, args.my_lib_parent)
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(args.java_root, 'preprocess'))
    from stmt_block_builder import iter_build_stmt_blocks
    from code_tokenizer import subtokenize_code, set_text_mode
    try:
        from my_lib.util.eval.translate_metric import get_nltk33_sent_bleu, get_meteor, get_rouge
    except ImportError as e:
        sys.exit('找不到 my_lib(%s)，请在官方脚本所在目录运行或用 --my_lib_parent。' % e)
    set_text_mode(args.text_mode)

    feats = {}
    for item, res, err in iter_build_stmt_blocks(json.load(open(args.test_json, encoding='utf-8')), desc='统计代码特征'):
        if err is not None or res is None:
            continue
        toks, ttypes, t2s, stmts, blocks = res
        subs, _ = subtokenize_code(toks, ttypes, t2s)
        feats[item['id']] = (len(subs), len(stmts), max([s['nest_depth'] for s in stmts] + [0]))

    def load(p):
        d = json.load(open(p, encoding='utf-8'))
        return {r['id']: r for r in (d['predictions'] if isinstance(d, dict) else d)}
    names = [('full', args.full), ('tf', args.transformer), ('noinj', args.noinject)]
    if args.seed1:
        names.append(('seed1', args.seed1))
    runs = {k: load(p) for k, p in names}
    ids = [i for i in feats if all(i in r for r in runs.values())]
    print('样本数 %d' % len(ids))
    F = np.array([feats[i] for i in ids], dtype=float)

    def score(key, sel):
        pred = [(runs[key][ids[k]]['pred'] or '<empty>').split() for k in sel]
        gold = [[runs[key][ids[k]]['ref'].split()] for k in sel]
        return dict(rougel=get_rouge(pred, gold), sbleu=get_nltk33_sent_bleu(pred, gold), meteor=get_meteor(pred, gold))

    dims = [('代码长度(子词数)', 0), ('语句数', 1), ('嵌套深度', 2)]
    labels = ['短', '中', '长']
    rows = {m: [] for m in ('sbleu', 'rougel', 'meteor')}
    for dname, col in dims:
        v = F[:, col]
        edges = sorted(set(int(np.floor(x)) for x in np.quantile(v, [1 / 3, 2 / 3])))
        idx = np.searchsorted(edges, v, side='left')
        nb = len(edges) + 1
        rng, lo = [], None
        for b in range(nb):
            hi = edges[b] if b < nb - 1 else None
            rng.append(('<=%g' % hi) if b == 0 else (('>%g' % lo) if hi is None else '(%g,%g]' % (lo, hi)))
            lo = hi
        lab = labels if nb == 3 else (['短', '长'] if nb == 2 else ['全部'])
        for b in range(nb):
            sel = np.where(idx == b)[0]
            print('%s %s %s n=%d ...' % (dname, lab[b], rng[b], len(sel)), flush=True)
            sc = {k: score(k, sel) for k in runs}
            for m in rows:
                r = [dname, lab[b], rng[b], len(sel), sc['full'][m], sc['tf'][m], sc['noinj'][m],
                     sc['full'][m] - sc['tf'][m], sc['full'][m] - sc['noinj'][m]]
                if args.seed1:
                    r.append(sc['seed1'][m] - sc['full'][m])
                rows[m].append(r)

    head = ['分组依据', '区间', '区间范围', '样本数', '完整模型', 'Transformer-6', '无回注', '完整-Transformer', '完整-无回注']
    if args.seed1:
        head.append('种子差(seed1-完整,噪声参照)')
    title = {'sbleu': 'S-BLEU', 'rougel': 'ROUGE-L', 'meteor': 'METEOR'}
    for m, rs in rows.items():
        path = '%s_%s.csv' % (args.out_prefix, m)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f); w.writerow(head)
            w.writerows([[x if not isinstance(x, float) else round(x, 2) for x in r] for r in rs])
        print('\n===== %s =====' % title[m])
        print(' | '.join(head))
        for r in rs:
            print(' | '.join(('%+.2f' % x if i >= 7 else '%.2f' % x) if isinstance(x, float) else str(x) for i, x in enumerate(r)))
        print('已写出', path)


if __name__ == '__main__':
    main()