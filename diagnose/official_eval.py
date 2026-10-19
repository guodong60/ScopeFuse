"""
用 HetCoS 官方的指标函数(my_lib)一次评多个结果文件，输出一张对比表。
等价于对每个文件把 s3_eval_whole_test_set.py 的 res_path 改一遍，但不用手改。

必须在能 import my_lib 的环境里运行，最简单：在官方脚本所在目录执行，例如
  cd ~/my_model/hier_claude/src_code/java/code_sum_41
  python /绝对路径/java/diagnose/official_eval.py --result_paths \
      /绝对路径/java/result/exp9_tk_base_test.json /绝对路径/java/result/exp10_tk_embedonly_test.json
如果 my_lib 不在当前目录，用 --my_lib_parent 指定 my_lib 文件夹的上一级目录。

输入：evaluate.py 生成的json({"metrics":..., "predictions":[{"pred":..,"ref":..,...}]})，
      或者官方格式(列表，含 pred_text/gold_text)。
"""
import argparse, json, os, sys


def load_pairs(path):
    data = json.load(open(path, encoding='utf-8'))
    recs = data['predictions'] if isinstance(data, dict) else data
    gold, pred = [], []
    for r in recs:
        g = r['gold_text'] if 'gold_text' in r else r['ref']
        p = r['pred_text'] if 'pred_text' in r else r['pred']
        gold.append([g.split()])
        pred.append(p.split())
    return pred, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_paths', nargs='+', required=True)
    ap.add_argument('--my_lib_parent', default=None, help='my_lib文件夹所在的目录')
    ap.add_argument('--csv', default=None, help='可选：把结果表另存为csv')
    args = ap.parse_args()
    if args.my_lib_parent:
        sys.path.insert(0, args.my_lib_parent)
    sys.path.insert(0, os.getcwd())
    try:
        from my_lib.util.eval.translate_metric import (
            get_nltk33_sent_bleu1, get_nltk33_sent_bleu2, get_nltk33_sent_bleu3,
            get_nltk33_sent_bleu4, get_nltk33_sent_bleu, get_meteor, get_rouge)
    except ImportError as e:
        sys.exit('找不到 my_lib(%s)。请在官方脚本所在目录运行，或用 --my_lib_parent 指定 my_lib 的上一级目录。' % e)

    rows = []
    for p in args.result_paths:
        pred, gold = load_pairs(p)
        name = os.path.basename(p).replace('_test.json', '').replace('.json', '')
        print('评估 %s (%d 条) ...' % (name, len(pred)), flush=True)
        rows.append((name, len(pred), get_rouge(pred, gold), get_nltk33_sent_bleu(pred, gold), get_meteor(pred, gold),
                     get_nltk33_sent_bleu1(pred, gold), get_nltk33_sent_bleu2(pred, gold),
                     get_nltk33_sent_bleu3(pred, gold), get_nltk33_sent_bleu4(pred, gold)))

    print('\n%-26s %6s | %8s %8s %8s | %7s %7s %7s %7s' % ('实验', '条数', 'ROUGE-L', 'S-BLEU', 'METEOR', 'B-1', 'B-2', 'B-3', 'B-4'))
    for r in rows:
        print('%-26s %6d | %8.2f %8.2f %8.2f | %7.2f %7.2f %7.2f %7.2f' % r)
    # 多个种子(名字形如 exp9_xxx / exp9s1_xxx / exp9s2_xxx)按实验分组，给均值±标准差
    import re, statistics, collections
    groups = collections.OrderedDict()
    for r in rows:
        m = re.match(r'^(exp\d+)(?:s\d+)?_(.*)$', r[0])
        groups.setdefault('%s_%s' % (m.group(1), m.group(2)) if m else r[0], []).append(r)
    if any(len(v) > 1 for v in groups.values()):
        print('\n===== 按实验分组：均值 ± 标准差 (n=种子数) =====')
        print('%-26s %3s | %-15s %-15s %-15s' % ('实验', 'n', 'ROUGE-L', 'S-BLEU', 'METEOR'))
        for k, v in groups.items():
            def ms(i):
                x = [t[i] for t in v]
                return '%.2f ± %.2f' % (statistics.mean(x), statistics.stdev(x) if len(x) > 1 else float('nan'))
            print('%-26s %3d | %-15s %-15s %-15s' % (k, len(v), ms(2), ms(3), ms(4)))
        print('注：n<3 时标准差不可靠。')
    print('%-26s %6s | %8.2f %8.2f %8.2f |' % ('HetCoS 论文(Java)', '8714', 63.10, 50.70, 34.35))
    if args.csv:
        with open(args.csv, 'w') as f:
            f.write('name,n,rouge_l,s_bleu,meteor,b1,b2,b3,b4\n')
            for r in rows:
                f.write(','.join(str(x) for x in r) + '\n')
        print('已保存', args.csv)


if __name__ == '__main__':
    main()