"""
导出真实生成案例：把完整模型、Transformer-6、无回注模型的预测并排放在一起，附代码、参考摘要、样本ID。
不训练，不占GPU。用来整理成功/失败案例，解释分数提升具体体现在哪里。

输出两个CSV(utf-8-sig，Excel可直接打开)：
  <out_prefix>_all.csv         全部测试样本一行：id、代码、参考、三个模型预测、各自逐样本S-BLEU/ROUGE-L、
                               代码子词数/语句数/嵌套深度、案例标签
  <out_prefix>_candidates.csv  按标签各挑 --per_tag 条候选(固定随机种子，不按你的喜好挑)，方便从中整理案例
案例标签(只描述现象，不含主观判断)：
  full_better   完整模型比 Transformer-6 和无回注都高 >= --delta 个S-BLEU
  full_worse    完整模型比 Transformer-6 和无回注都低 >= --delta
  all_exact     三个模型都和参考完全一致
  all_poor      三个模型 S-BLEU 都 < --poor
  models_differ 三个预测互不相同(其余标签之外的随机样本，用于看差异形态)
注意：写论文时案例要"成功和失败都放"，并用 full_better / full_worse 的数量对比说明总体是否有倾向，
不要只挑完整模型占优的例子。逐样本分数用 metrics.py 口径，仅用于挑案例。

用法(在 java/ 目录下)：
  T=<token_data目录>
  python3 diagnose/export_cases.py --test_json $T/test_data.json --text_mode pretokenized \
      --full result/exp9_tk_base_test.json --transformer result/exp21_tk_plain6_test.json \
      --noinject result/exp25_localonly_test.json --out_prefix result/cases
"""
import argparse, csv, json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'preprocess'))
sys.path.insert(0, os.path.join(HERE, '..', 'train'))
from stmt_block_builder import iter_build_stmt_blocks   # noqa: E402
from code_tokenizer import subtokenize_code, set_text_mode   # noqa: E402
from metrics import sentence_bleu, _rouge_scorer          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_json', required=True)
    ap.add_argument('--text_mode', default='pretokenized')
    ap.add_argument('--full', required=True)
    ap.add_argument('--transformer', required=True)
    ap.add_argument('--noinject', required=True)
    ap.add_argument('--out_prefix', default='result/cases')
    ap.add_argument('--delta', type=float, default=20.0, help='full_better/full_worse 的S-BLEU差阈值')
    ap.add_argument('--poor', type=float, default=10.0)
    ap.add_argument('--per_tag', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    set_text_mode(args.text_mode)

    items = json.load(open(args.test_json, encoding='utf-8'))
    code_of = {it['id']: it.get('code', '') for it in items}
    feats = {}
    for item, res, err in iter_build_stmt_blocks(items, desc='统计代码特征'):
        if err is not None or res is None:
            continue
        toks, tt, t2s, stmts, blocks = res
        subs, _ = subtokenize_code(toks, tt, t2s)
        feats[item['id']] = (len(subs), len(stmts), max([s['nest_depth'] for s in stmts] + [0]))

    def load(p):
        d = json.load(open(p, encoding='utf-8'))
        return {r['id']: r for r in (d['predictions'] if isinstance(d, dict) else d)}
    runs = [('full', load(args.full)), ('tf', load(args.transformer)), ('noinj', load(args.noinject))]
    ids = [it['id'] for it in items if it['id'] in feats and all(it['id'] in r for _, r in runs)]
    print('样本数', len(ids))

    rows = []
    for i in ids:
        ref = runs[0][1][i]['ref']
        row = dict(id=i, code=code_of[i], ref=ref)
        for k, r in runs:
            pred = r[i]['pred'] or '<empty>'
            row['pred_' + k] = pred
            row['sbleu_' + k] = sentence_bleu(pred, ref or '<empty>')
            row['rougel_' + k] = 100.0 * _rouge_scorer.score(ref, pred)['rougeL'].fmeasure
        row['n_subtokens'], row['n_stmts'], row['max_depth'] = feats[i]
        s = (row['sbleu_full'], row['sbleu_tf'], row['sbleu_noinj'])
        tags = []
        if row['sbleu_full'] - max(s[1], s[2]) >= args.delta:
            tags.append('full_better')
        if min(s[1], s[2]) - row['sbleu_full'] >= args.delta:
            tags.append('full_worse')
        if all(row['pred_' + k] == ref for k, _ in runs):
            tags.append('all_exact')
        if max(s) < args.poor:
            tags.append('all_poor')
        if len({row['pred_full'], row['pred_tf'], row['pred_noinj']}) == 3:
            tags.append('models_differ')
        row['tags'] = '|'.join(tags)
        rows.append(row)

    cols = ['id', 'tags', 'n_subtokens', 'n_stmts', 'max_depth', 'code', 'ref', 'pred_full', 'pred_tf', 'pred_noinj',
            'sbleu_full', 'sbleu_tf', 'sbleu_noinj', 'rougel_full', 'rougel_tf', 'rougel_noinj']

    def write(path, rs):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rs:
                w.writerow([round(r[c], 2) if isinstance(r[c], float) else r[c] for c in cols])
        print('已写出', path, len(rs), '行')
    write(args.out_prefix + '_all.csv', rows)

    rng = random.Random(args.seed)
    cand, seen = [], set()
    print('\n各标签样本数(全部 %d)：' % len(rows))
    for tag in ('full_better', 'full_worse', 'all_exact', 'all_poor', 'models_differ'):
        pool = [r for r in rows if tag in r['tags'].split('|')]
        print('  %-14s %d' % (tag, len(pool)))
        for r in rng.sample(pool, min(args.per_tag, len(pool))):
            if (tag, r['id']) not in seen:
                seen.add((tag, r['id']))
                cand.append(dict(r, tags=tag))
    write(args.out_prefix + '_candidates.csv', cand)
    print('\nfull_better 与 full_worse 的数量对比反映"完整模型是否整体占优"：两者接近说明分数差异只是互相抵消的个例。')


if __name__ == '__main__':
    main()