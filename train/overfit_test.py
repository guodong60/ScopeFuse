# coding=utf-8
"""
用真实词表(vocab.pkl)和真实预处理数据(avail_data/train_data.pkl)切一个小子集(默认50条)，
跑过拟合测试——不追求好效果，只验证"模型在这份真实规模的数据/词表上，梯度链路是否
真的通、有没有实现bug"。方法论和之前排查HETCOS版本代码时完全一致：小样本、关掉
dropout/正则化、不用学习率衰减(参考之前踩过的坑：lr_decay太快会让后半程训练在
接近0的学习率上空转，误判成"模型学不动")、loss应该能持续降到接近0。

同时用--use_fusion参数控制对比:
  both  (默认) 用同一批50条样本，分别跑一次use_fusion=True和False，对比两条loss曲线——
        这是能拿到的第一个关于"跨粒度融合到底有没有用"的粗略信号(虽然只是过拟合小样本，
        不代表真实泛化能力，但如果连这都学不动或者两者毫无差别，说明模块本身有问题，
        值得先查清楚再上全量训练)
  true / false  只跑其中一种

用法:
    python3 overfit_test.py --vocab_path ../w2i2w/vocab.pkl \
        --data_path ../avail_data/train_data.pkl \
        --n_samples 50 --steps 300 --lr 1e-3 --emb_dims 128
"""
import argparse
import pickle
import sys
import os

import torch
from torch.utils.data import Subset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from dataset import CodeSummDataset  # noqa: E402
from seq2seq import HierSeq2Seq  # noqa: E402


def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_one(vocab, params, subset_loader, device, steps, lr, emb_dims, n_heads, n_layers,
           use_fusion, use_copy, hier_layers, log_every):
    pad_id = vocab['w2i']['<pad>']
    bos_id = vocab['w2i']['<s>']
    eos_id = vocab['w2i']['</s>']

    model = HierSeq2Seq(
        vocab_size=len(vocab['w2i']),
        stmt_type_vocab_size=len(vocab['stmt_type_w2i']),
        block_type_vocab_size=len(vocab['block_type_w2i']),
        max_code_len=params['max_code_len'], max_text_len=params['max_text_len'],
        stmt_max_num=params['stmt_max_num'], block_max_num=params['block_max_num'],
        nest_depth_max=params['nest_depth_max'],
        emb_dims=emb_dims, enc_local_heads=n_heads, enc_mid_heads=n_heads,
        enc_global_heads=n_heads, enc_fusion_heads=n_heads, dec_heads=n_heads,
        dec_layers=n_layers, drop_rate=0.0, pad_idx=pad_id, bos_id=bos_id, eos_id=eos_id,
        use_fusion=use_fusion, use_copy=use_copy, hier_layers=hier_layers,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)  # 不用学习率衰减，避免重蹈之前的坑
    n_params = sum(p.numel() for p in model.parameters())
    print('  参数量: {:,}'.format(n_params))

    losses = []
    step = 0
    model.train()
    while step < steps:
        for batch in subset_loader:
            if step >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(batch)
            out['loss'].backward()
            opt.step()
            losses.append(out['loss'].item())
            if step % log_every == 0 or step == steps - 1:
                print('  step {:4d}  loss {:.4f}'.format(step, losses[-1]))
            step += 1

    return model, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True,
                        help='avail_data下的xxx_data.pkl，通常用train_data.pkl')
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--emb_dims', type=int, default=128,
                        help='过拟合测试用较小的emb_dims换取更快的迭代速度，'
                             '不代表最终训练要用这个规模')
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hier_layers', type=int, default=2,
                        help='编码器堆叠层数。过拟合测试图快，默认给2(不是正式训练的默认值6)，'
                             '只是为了验证梯度链路通不通，不需要用最终规模')
    parser.add_argument('--use_copy', type=lambda s: s.lower() == 'true', default=True,
                        help='是否启用复制机制。默认true，因为这是目前最值得优先验证的新功能')
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--use_fusion', type=str, default='both',
                        choices=['both', 'true', 'false'])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    vocab = load_vocab(args.vocab_path)
    full_dataset = CodeSummDataset(args.data_path)
    params = full_dataset.params
    n = min(args.n_samples, len(full_dataset))
    subset = Subset(full_dataset, list(range(n)))
    # batch_size不能超过样本数，否则又会撞上之前那个drop_last/空batch的坑
    # (虽然dataset.py的get_dataloader默认drop_last=False已经不会空转了，这里额外保险一下)
    batch_size = min(args.batch_size, n)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)
    print('过拟合测试样本数: {}  batch_size: {}'.format(n, batch_size))
    print('数据参数:', params)
    print('hier_layers={}  use_copy={}'.format(args.hier_layers, args.use_copy))

    results = {}
    settings = {'both': [True, False], 'true': [True], 'false': [False]}[args.use_fusion]
    for use_fusion in settings:
        print('\n{} use_fusion={} {}'.format('=' * 20, use_fusion, '=' * 20))
        _, losses = run_one(vocab, params, loader, device, args.steps, args.lr,
                            args.emb_dims, args.n_heads, args.n_layers, use_fusion,
                            args.use_copy, args.hier_layers, args.log_every)
        results[use_fusion] = losses

    print('\n{} 汇总 {}'.format('=' * 20, '=' * 20))
    for use_fusion, losses in results.items():
        first_avg = sum(losses[:10]) / min(10, len(losses))
        last_avg = sum(losses[-10:]) / min(10, len(losses))
        print('use_fusion={}: 前10步平均loss={:.4f}  后10步平均loss={:.4f}  下降比例={:.1%}'.format(
            use_fusion, first_avg, last_avg, 1 - last_avg / first_avg))
    if len(results) == 2:
        final_true = sum(results[True][-10:]) / 10
        final_false = sum(results[False][-10:]) / 10
        print('\n两者收敛后的loss差异(use_fusion=False - use_fusion=True): {:.4f}'.format(
            final_false - final_true))
        print('(正值说明加了跨粒度融合确实更容易拟合；注意这只是50条样本的过拟合能力对比，'
             '不代表真实泛化性能，只用来确认两条链路本身都是通的、且融合模块在起作用)')


if __name__ == '__main__':
    main()