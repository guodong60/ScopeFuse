# coding=utf-8
"""
把 FunCom-50 (funcom_train.jsonl / funcom_test.jsonl) 转成本项目的原始数据格式：
    <out_dir>/train_data.json, valid_data.json, test_data.json   每条 {"id", "code", "text"}
之后就和 TL-CodeSum 一样走 vocab_builder -> build_dataset -> train -> eval。

转换做了四件事：
  1. 去掉 FunCom 给每个方法套的外壳 "public class A09360549 { ... }"，只留方法本身
     (我们的 java_parser 会自己再套 class _dummy_，外壳不去掉会变成嵌套类，语句/块结构就错了)。
  2. text = " ".join(summary_seq) 并转小写，去掉末尾的 "."（和 --text_mode pretokenized 配合）。
  3. FunCom 没有验证集：从训练集里按固定随机种子切出 valid_ratio 作为验证集，测试集原样保留。
  4. 打印检查：训练/测试代码完全相同的比例、空摘要数，防止泄漏被忽略。

用法：
  python3 funcom/convert_funcom.py --in_dir raw_data/funcom --out_dir raw_data/funcom_conv_full \
      --valid_ratio 0.05 --seed 0 --train_size 300000 --valid_size 2000
测试集始终使用全部样本，不抽样。
"""
import argparse, json, os, random, re

WRAP = re.compile(r'^\s*public\s+class\s+\w+\s*\{')


def unwrap(code):
    m = WRAP.match(code)
    if not m:
        return None
    body = code[m.end():]
    k = body.rstrip().rfind('}')
    if k < 0:
        return None
    return body[:k].strip()


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert(path, start_id):
    out, bad = [], 0
    for r in read_jsonl(path):
        code = unwrap(r['code'])
        toks = [t for t in r['summary_seq']]
        while toks and toks[-1] == '.':
            toks.pop()
        text = ' '.join(toks).lower().strip()
        if not code or not text:
            bad += 1
            continue
        out.append(dict(id=start_id + len(out), code=code, text=text))
    return out, bad


def norm(c):
    return re.sub(r'\s+', '', c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--valid_ratio', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_size', type=int, default=0, help='训练集随机取这么多条，0=全部')
    ap.add_argument('--valid_size', type=int, default=0, help='验证集随机取这么多条，0=全部(验证集每个epoch都要解码，建议2000)')
    a = ap.parse_args()

    train, bad_tr = convert(os.path.join(a.in_dir, 'funcom_train.jsonl'), 0)
    test, bad_te = convert(os.path.join(a.in_dir, 'funcom_test.jsonl'), 10 ** 7)
    print('丢弃(去壳失败/空摘要): train %d, test %d' % (bad_tr, bad_te))

    rng = random.Random(a.seed)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    nv = int(len(train) * a.valid_ratio)
    vset = set(idx[:nv])
    valid = [train[i] for i in sorted(vset)]
    train = [t for i, t in enumerate(train) if i not in vset]

    if a.valid_size and len(valid) > a.valid_size:
        valid = sorted(rng.sample(valid, a.valid_size), key=lambda t: t['id'])
    if a.train_size and len(train) > a.train_size:
        train = sorted(rng.sample(train, a.train_size), key=lambda t: t['id'])
    tr_codes = {norm(t['code']) for t in train}
    dup = sum(norm(t['code']) in tr_codes for t in test)
    print('train %d / valid %d / test %d' % (len(train), len(valid), len(test)))
    print('测试代码与训练集完全相同: %d (%.1f%%)' % (dup, 100.0 * dup / max(1, len(test))))

    os.makedirs(a.out_dir, exist_ok=True)
    for name, d in (('train', train), ('valid', valid), ('test', test)):
        with open(os.path.join(a.out_dir, name + '_data.json'), 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False)
    print('已写出到', a.out_dir)


if __name__ == '__main__':
    main()