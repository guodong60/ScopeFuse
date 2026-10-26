# coding=utf-8
"""
FunCom 结果的最终评估：统计 <unk>，并用官方指标(diagnose/official_eval.py)给出两种摘要末尾约定下的结果。

  A. 原样：预测和参考都不带结束标记（我们转换 FunCom 时的格式）。
  B. 预测和参考末尾都追加 ".."（TL-CodeSum/HetCoS 的约定；只是敏感性分析）。

两个都打印、都标明，不替你选。写论文时用哪一个，只能由对比对象的约定决定：
去看 HierarchyNet 等基线的数据处理代码，它们的参考摘要末尾有没有句号/结束符，
然后选与之相同的那个，并在论文里写明。不能因为 B 分数更高而选 B。
"""
import argparse, json, os, subprocess, sys


def load(path):
    d = json.load(open(path, encoding='utf-8'))
    return d, (d['predictions'] if isinstance(d, dict) else d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True, help='evaluate.py 生成的 json')
    ap.add_argument('--my_lib_parent', default=None)
    ap.add_argument('--skip_official', action='store_true', help='只做转换和 <unk> 统计，不跑官方指标')
    a = ap.parse_args()

    d, recs = load(a.result)
    n_unk = sum('<unk>' in (r['pred'] or '') for r in recs)
    print('预测含 <unk> 的比例: %d / %d = %.1f%%' % (n_unk, len(recs), 100.0 * n_unk / len(recs)))

    out = a.result.replace('.json', '_dots.json')
    for r in recs:
        r['pred'] = ((r['pred'] or '') + ' ..').strip()
        r['ref'] = ((r['ref'] or '') + ' ..').strip()
    json.dump(d, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    print('已写出变体B: %s' % out)
    if a.skip_official:
        return

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, '..', 'diagnose', 'official_eval.py')
    cmd = [sys.executable, script, '--result_paths', a.result, out]
    if a.my_lib_parent:
        cmd += ['--my_lib_parent', a.my_lib_parent]
    print('\n A=原样(文件名无后缀)  B=两边都加 ".."(文件名带 _dots)；论文用哪个由基线约定决定\n')
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()