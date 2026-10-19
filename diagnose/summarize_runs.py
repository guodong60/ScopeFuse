"""
汇总多个训练日志：对比表 + 按验证集自动选 best.pt/avg.pt + 生成测试集评估脚本。
只读日志，不碰测试集。选checkpoint只看验证集S-BLEU。

用法(在项目根目录 java/ 下)：
  python3 diagnose/summarize_runs.py --ref_json <token_data/test_data.json 的路径> --gpus 0
  # 不加 --map 时使用下面的 DEFAULT_MAP(作者实验 exp9~21 的日志与目录名，仅作示例)；
  # 自己的实验请用 --map 指定。
  # 日志和目录对应关系可以覆盖，格式 日志文件=checkpoint目录名：
  python3 diagnose/summarize_runs.py --ref_json <路径> --map logs_exp9.txt=exp9_tk_base logs_exp10.txt=exp10_tk_embedonly
生成 run_eval_all.sh，里面是每个实验的评估命令(用环境变量 GPU 指定显卡)。
"""
import argparse, os, re

DEFAULT_MAP = [
    ('logs_exp9.txt', 'exp9_tk_base'), ('logs_exp9s1.txt', 'exp9s1_tk_base'), ('logs_exp9s2.txt', 'exp9s2_tk_base'),
    ('logs_exp10.txt', 'exp10_tk_embedonly'), ('logs_exp10s1.txt', 'exp10s1_tk_embedonly'), ('logs_exp10s2.txt', 'exp10s2_tk_embedonly'),
    ('logs_exp11.txt', 'exp11_tk_shareemb'), ('logs_exp11s1.txt', 'exp11s1_tk_shareemb'), ('logs_exp11s2.txt', 'exp11s2_tk_shareemb'),
    ('logs_exp12.txt', 'exp12_tk_embln'),
    ('logs_exp13.txt', 'exp13_tk_lrdecay'),
    ('logs_exp15.txt', 'exp15_tk_official'),
    ('logs_exp16.txt', 'exp16_tk_tokglobal'),
    ('logs_exp17.txt', 'exp17_tk_hier2'),
    # 下面这些使用了不同的预处理，所以词表和测试数据路径不同(第3、4项)
    ('logs_exp18.txt', 'exp18_tk_mod', '../w2i2w_tk_mod/vocab.pkl', '../avail_data_tk_mod/test_data.pkl'),
    ('logs_exp19.txt', 'exp19_tk_len', '../w2i2w_tk_len/vocab.pkl', '../avail_data_tk_len/test_data.pkl'),
    ('logs_exp20.txt', 'exp20_tk_plain4'),
    ('logs_exp21.txt', 'exp21_tk_plain6'),
]
VALID = re.compile(r'epoch (\d+) valid_loss=([\d.]+)\s+BLEU\(corpus\)=([\d.]+)\s+S-BLEU=([\d.]+)\s+ROUGE-L=([\d.]+)')
AVG = re.compile(r'平均后 valid_loss=([\d.]+)\s+BLEU\(corpus\)=([\d.]+)\s+S-BLEU=([\d.]+)\s+ROUGE-L=([\d.]+)')
BEST = re.compile(r'训练结束，最佳验证s_bleu: ([\d.]+)')


def parse(path):
    text = open(path, encoding='utf-8', errors='ignore').read().replace('\r', '\n')
    rows = {}
    for m in VALID.finditer(text):
        rows[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])   # loss, bleu, sbleu, rougeL
    avg = AVG.search(text)
    best = BEST.search(text)
    return rows, (tuple(float(x) for x in avg.groups()) if avg else None), (float(best.group(1)) if best else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', nargs='*', default=None, help='日志文件=checkpoint目录名')
    ap.add_argument('--ckpt_root', default='../model_ckpt')
    ap.add_argument('--vocab', default='../w2i2w_tk/vocab.pkl')
    ap.add_argument('--test_pkl', default='../avail_data_tk/test_data.pkl')
    ap.add_argument('--ref_json', required=True, help='测试集原始json(token_data/test_data.json)，评估时用它取真实参考摘要')
    ap.add_argument('--out_sh', default='run_eval_all.sh')
    ap.add_argument('--gpus', default='0', help='评估用的显卡编号，逗号分隔，如 0,1,2,3,4,6；命令会自动分配并分批并行')
    ap.add_argument('--result_dir', default='result', help='已有评估结果所在目录(相对当前目录)，已存在的实验会被跳过')
    ap.add_argument('--force', action='store_true', help='即使结果已存在也重新评估')
    args = ap.parse_args()
    raw_pairs = [tuple(x.split('=', 1)) for x in args.map] if args.map else DEFAULT_MAP
    pairs = [(p[0], p[1], p[2] if len(p) > 2 else args.vocab, p[3] if len(p) > 3 else args.test_pkl) for p in raw_pairs]

    res = []
    for log, d, vocab_p, test_p in pairs:
        if not os.path.exists(log):
            if args.map:
                print('!! 找不到', log)
            continue
        rows, avg, best = parse(log)
        if not rows:
            print('!! %s 里没有解析到验证行' % log); continue
        last = max(rows)
        best_ep = max(rows, key=lambda e: rows[e][2])
        best_s = rows[best_ep][2]
        use_avg = avg is not None and avg[2] > best_s
        res.append(dict(log=log, dir=d, vocab=vocab_p, test=test_p, rows=rows, last=last, best_ep=best_ep, best_s=best_s,
                        avg=avg, ckpt='avg.pt' if use_avg else 'best.pt',
                        chosen_s=avg[2] if use_avg else best_s, done=best is not None))

    print('%-10s %6s | %-24s | %-9s | %-12s | %-13s | %s' % ('日志', '轮数', '验证S-BLEU @10/@20/@50', '最佳(轮)', '平均后S-BLEU', '选用ckpt', '完成?'))
    for r in res:
        g = lambda e: ('%.2f' % r['rows'][e][2]) if e in r['rows'] else '  -  '
        print('%-10s %6d | %-24s | %-9s | %-12s | %-13s | %s' % (
            r['log'].replace('logs_', '').replace('.txt', ''), r['last'], '/'.join(g(e) for e in (10, 20, 50)),
            '%.2f(%d)' % (r['best_s'], r['best_ep']), ('%.2f' % r['avg'][2]) if r['avg'] else '-',
            '%s %.2f' % (r['ckpt'], r['chosen_s']), '是' if r['done'] else '否(仍在跑/中断)'))

    base = next((r for r in res if r['log'].endswith('exp9.txt')), None)
    seed = next((r for r in res if r['log'].endswith('exp9s1.txt')), None)
    if base:
        noise = abs(base['chosen_s'] - seed['chosen_s']) if seed else None
        print('\n相对基线 exp9 的验证S-BLEU差(用选定ckpt)，两个种子之差=%s (只有一对种子，估计很粗，仅作参考)' % ('%.2f' % noise if noise is not None else '未知(缺exp9s1)'))
        for r in res:
            if r is base:
                continue
            d = r['chosen_s'] - base['chosen_s']
            # 两个种子只能给出一个很粗的噪声估计(可能碰巧很小)，所以设下限0.3、并要求超过2倍
            thr = max(2 * noise, 0.3) if noise is not None else None
            flag = '' if thr is None else ('  <- 超出噪声(门槛%.2f)' % thr if abs(d) > thr else '  (在噪声范围内，门槛%.2f)' % thr)
            print('  %-10s %+6.2f%s' % (r['log'].replace('logs_', '').replace('.txt', ''), d, flag))


    # ---- 按实验分组(exp9 / exp9s1 / exp9s2 同属一组)，多个种子给均值±标准差 ----
    import statistics, collections
    groups = collections.OrderedDict()
    for r in res:
        m = re.match(r'^(exp\d+)(?:s\d+)?_(.*)$', r['dir'])
        key = '%s_%s' % (m.group(1), m.group(2)) if m else r['dir']
        groups.setdefault(key, []).append(r['chosen_s'])
    if any(len(v) > 1 for v in groups.values()):
        print('\n===== 按实验分组：验证S-BLEU 均值 ± 标准差 (种子数) =====')
        base_key = next((k for k in groups if k.startswith('exp9_')), None)
        base_mean = statistics.mean(groups[base_key]) if base_key else None
        for k, v in groups.items():
            sd = statistics.stdev(v) if len(v) > 1 else float('nan')
            diff = '' if base_mean is None or k == base_key else '  对基线 %+.2f' % (statistics.mean(v) - base_mean)
            print('  %-24s %.2f ± %.2f  (n=%d)%s' % (k, statistics.mean(v), sd, len(v), diff))
        print('  注：n<3 时标准差不可靠；论文里的关键对比建议至少3个种子。')

    gpus = [g.strip() for g in args.gpus.split(',') if g.strip()]
    cmds, skipped = [], []
    for r in res:
        if not r['done']:
            skipped.append('%s(训练未完成)' % r['dir']); continue
        if not args.force and os.path.exists(os.path.join(args.result_dir, '%s_test.json' % r['dir'])):
            skipped.append('%s(结果已存在)' % r['dir']); continue
        cmds.append((r['dir'], '%s/%s/%s' % (args.ckpt_root, r['dir'], r['ckpt']), r['vocab'], r['test']))
    with open(args.out_sh, 'w') as f:
        f.write('#!/bin/bash\n# 在 java/train 目录下运行: bash ../%s\nmkdir -p ../result\n' % args.out_sh)
        for b in range(0, len(cmds), len(gpus)):
            for k, (d, ck, vp, tp) in enumerate(cmds[b:b + len(gpus)]):
                f.write('python3 evaluate.py --vocab_path %s --test_path %s --ckpt_path %s '
                        '--out_path ../result/%s_test.json --ref_json %s --text_mode pretokenized '
                        '--beam_width 5 --length_penalty 1.0 --device cuda:%s > ../logs_eval_%s.txt 2>&1 &\n'
                        % (vp, tp, ck, d, args.ref_json, gpus[k], d))
            f.write('wait\n')
    print('\n待评估 %d 个实验，分 %d 批在显卡 %s 上并行' % (len(cmds), -(-len(cmds) // len(gpus)) if cmds else 0, ','.join(gpus)))
    if skipped:
        print('跳过: ' + ', '.join(skipped))
    print('已生成 %s' % args.out_sh)


if __name__ == '__main__':
    main()