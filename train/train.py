import argparse
import os
import pickle
import sys

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from dataset import get_dataloader  # noqa: E402
from seq2seq import HierSeq2Seq  # noqa: E402
from metrics import ids_to_words, compute_metrics  # noqa: E402


def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


@torch.no_grad()
def evaluate(model, loader, vocab, device, max_examples_print=3, beam_width=1,
            length_penalty=1.0):
    """跑一遍验证集：算teacher-forcing loss(监控用)，再解码生成摘要算BLEU/ROUGE-L
    (具体指标计算逻辑在metrics.py里，train.py/evaluate.py共用同一份)。
    beam_width=1(默认)用贪心解码，训练过程中每个epoch都要跑，快是第一位的；
    最终测试集评估时(evaluate.py)才用beam search换取更好的结果。
    返回dict(loss, bleu, s_bleu, rouge_l)，并顺带打印几条预测/参考样例。
    """
    model.eval()
    i2w = vocab['i2w']
    pad_id = vocab['w2i']['<pad>']
    bos_id = vocab['w2i']['<s>']
    eos_id = vocab['w2i']['</s>']

    hyps, refs = [], []
    total_loss, total_count = 0.0, 0
    printed = 0

    for batch in tqdm(loader, desc='validating', leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        bsz = batch['code_ids'].shape[0]
        total_loss += out['loss'].item() * bsz
        total_count += bsz

        gen_ids = model.generate(batch, beam_width=beam_width, length_penalty=length_penalty)
        for i in range(bsz):
            hyp_words = ids_to_words(gen_ids[i].tolist(), i2w, bos_id, eos_id, pad_id)
            ref_words = ids_to_words(batch['text_ids'][i].tolist(), i2w, bos_id, eos_id, pad_id)
            hyp_text = ' '.join(hyp_words)
            ref_text = ' '.join(ref_words)
            hyps.append(hyp_text)
            refs.append(ref_text)
            if printed < max_examples_print:
                print('    [示例] 预测: {}'.format(hyp_text if hyp_text else '<empty>'))
                print('           参考: {}'.format(ref_text if ref_text else '<empty>'))
                printed += 1

    avg_loss = total_loss / max(1, total_count)
    m = compute_metrics(hyps, refs, full=False)
    model.train()
    return dict(loss=avg_loss, bleu=m['bleu'], s_bleu=m['s_bleu'], rouge_l=m['rouge_l'])


def average_state_dicts(state_dicts):
    """对多份state_dict按元素取平均(checkpoint平均)。浮点参数取平均，非浮点(本模型里没有，
    只是兜底)取第一份。模型里没有BatchNorm等带统计量的层，直接平均权重是安全的。"""
    avg = {}
    n = len(state_dicts)
    for k, v0 in state_dicts[0].items():
        if v0.is_floating_point():
            avg[k] = sum(sd[k] for sd in state_dicts) / n
        else:
            avg[k] = v0.clone()
    return avg


def build_model(vocab, params, args, device):
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
        emb_dims=args.emb_dims, enc_local_heads=args.n_heads, enc_mid_heads=args.n_heads,
        enc_global_heads=args.n_heads, enc_fusion_heads=args.n_heads, dec_heads=args.n_heads,
        dec_layers=args.dec_layers, drop_rate=args.drop_rate, pad_idx=pad_id,
        bos_id=bos_id, eos_id=eos_id, use_fusion=args.use_fusion,
        use_copy=getattr(args, 'use_copy', False),
        hier_layers=getattr(args, 'hier_layers', 1),  # getattr兜底：旧checkpoint里的args
                                                       # 可能没有hier_layers字段(最早的版本固定是1层)
        # v2新增开关：同样用getattr兜底，旧checkpoint没有这些字段时一律按"关"处理，
        # 结构和旧版完全一致，权重可直接加载
        use_struct_pos=getattr(args, 'use_struct_pos', False),
        use_parent_emb=getattr(args, 'use_parent_emb', False),
        use_token_global=getattr(args, 'use_token_global', False),
        dec_norm_first=getattr(args, 'dec_norm_first', False),
        scale_emb=getattr(args, 'scale_emb', False),
        encoder_type=getattr(args, 'encoder_type', 'hier'),
        share_emb=getattr(args, 'share_emb', False),
        dec_emb_ln=getattr(args, 'dec_emb_ln', False),
        plain_layers=getattr(args, 'plain_layers', 6), plain_heads=args.n_heads,
        drop_levels=tuple(x for x in str(getattr(args, 'drop_levels', '') or '').split(',') if x),
        fusion_gate=getattr(args, 'fusion_gate', True),   # 旧checkpoint没有该字段时按有门控处理
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)

    parser.add_argument('--emb_dims', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--hier_layers', type=int, default=1,
                        help='编码器"自底向上抽象+token三路回注"这一整块的堆叠层数。'
                             '默认1——真实数据消融实验显示堆叠到6层反而让S-BLEU/ROUGE-L'
                             '明显下降(过拟合更严重)，1层是目前验证过的最优配置。'
                             '保留这个参数是为了以后想做进一步消融时不用改代码')
    parser.add_argument('--use_struct_pos', type=lambda s: s.lower() == 'true', default=True,
                        help='[v2] 补齐结构位置信息：语句内token位置、语句序号、block序号嵌入。'
                             '旧版语句/block级没有顺序信息，建议打开；设false可复现旧版结构')
    parser.add_argument('--fusion_gate', type=lambda s: s.lower() == 'true', default=True,
                        help='融合是否用门控。false=保留跨粒度注意力，只把门控换成残差相加(own+ctx)再LayerNorm；仅在use_fusion=true时有效')
    parser.add_argument('--drop_levels', type=str, default='',
                        help='[消融] 逗号分隔，取值 stmt/block/func：这些级别的表示不回注给token。'
                             '例 --drop_levels func ；--drop_levels stmt,block,func=只剩语句内局部注意力。'
                             '默认空=完整模型')
    parser.add_argument('--use_parent_emb', type=lambda s: s.lower() == 'true', default=False,
                        help='[v2] block向量加父block类型嵌入。需要用新版预处理重新生成数据'
                             '(数据里要有block_parent字段)。收益不确定，建议先做消融再决定')
    parser.add_argument('--use_token_global', type=lambda s: s.lower() == 'true', default=False,
                        help='[v2] 三路融合之后再对token做一层全局自注意力，让memory带跨语句的'
                             'token交互。会改变\"纯层次\"叙事，建议先做消融再决定')
    parser.add_argument('--scale_emb', type=lambda s: s.lower() == 'true', default=False,
                        help='[v2] 编码器/解码器前向里token嵌入乘sqrt(d)。旧版token嵌入(std~0.044)和位置嵌入'
                             '(std=1)初始尺度差约22.7倍；打开后两者同量级。默认false=旧版，旧checkpoint兼容')
    parser.add_argument('--encoder_type', type=str, default='hier', choices=['hier', 'embed', 'plain'],
                        help="[v2] hier=层次编码器(默认)；embed=诊断用，编码器退化成纯token嵌入(无位置无结构)，"
                             "对应\"HetCoS只保留解码器\"的对照实验；plain=普通Transformer编码器基线(层数由--plain_layers指定)")
    parser.add_argument('--plain_layers', type=int, default=6,
                        help='encoder_type=plain时的编码层数。4层参数量最接近层次编码器，6层是标准配置')
    parser.add_argument('--share_emb', type=lambda s: s.lower() == 'true', default=False,
                        help='[v2] 编码器和解码器共用同一个词嵌入(HetCoS官方做法)。默认false=各用各的')
    parser.add_argument('--dec_emb_ln', type=lambda s: s.lower() == 'true', default=False,
                        help='[v2] 解码器的(嵌入+位置)经dropout后再接一层LayerNorm(HetCoS官方做法)。默认false')
    parser.add_argument('--dec_norm_first', type=lambda s: s.lower() == 'true', default=False,
                        help='[v2] 解码器改用Pre-LN(更适合较深的解码器，训练更稳)。默认false=旧版Post-LN')
    parser.add_argument('--dec_layers', type=int, default=8)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--use_fusion', type=lambda s: s.lower() == 'true', default=True)
    parser.add_argument('--use_copy', type=lambda s: s.lower() == 'true', default=True,
                        help='是否启用复制机制(从源码复制token到摘要)。代码摘要任务里方法名/'
                             '变量名经常同时出现在代码和摘要中，复制机制通常能带来明显提升')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--start_valid_epoch', type=int, default=1)
    parser.add_argument('--valid_every', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--select_metric', type=str, default='s_bleu',
                        choices=['bleu', 's_bleu', 'rouge_l'],
                        help='用哪个验证指标挑选最佳checkpoint/做early stop判断。'
                             '默认s_bleu(句子级BLEU)，这是大多数代码摘要论文报告的口径，'
                             '如果只关心语料级整体质量可以改成bleu(corpus BLEU)')

    parser.add_argument('--lr_base', type=float, default=2e-4,
                        help='原HetCoS config.py用的是5e-4，但那是针对他们自己的架构调的；'
                             '实测发现咱们这个更深的堆叠encoder+copy机制组合在5e-4峰值学习率下'
                             '会在warmup结束后训练崩溃(预测坍缩成单词复读)，2e-4实测稳定')
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='验证BLEU连续多少次不提升就降学习率(ReduceLROnPlateau的patience)')
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--min_lr_rate', type=float, default=0.05,
                        help='学习率下限=lr_base*这个比例，避免像之前那样衰减到几乎为0')
    parser.add_argument('--lr_schedule', type=str, default='plateau', choices=['plateau', 'exp'],
                        help='学习率调度。plateau=旧版，验证指标连续lr_patience次不提升才降(指标缓慢上涨时'
                             '学习率会一直不衰减)；exp=和HetCoS官方一致，warmup结束后每个epoch乘lr_decay，'
                             '直到lr_base*min_lr_rate，训练后期自然退火')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help='lr_schedule=exp时每个epoch的衰减系数(官方0.95)')
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--regular_rate', type=float, default=1e-5,
                        help='L2权重衰减(weight decay)，对齐HetCoS原config.py的设置。'
                             '之前train.py一直没配置这个，是个疏漏——尤其是模型变大'
                             '(比如hier_layers堆叠更深)之后，缺少权重衰减会让过拟合'
                             '风险明显上升')
    parser.add_argument('--valid_beam_width', type=int, default=1,
                        help='验证集解码的beam宽度。默认1(贪心)，因为每个epoch都要跑一次验证，'
                             '用beam会显著拖慢训练；最终测试集评估(evaluate.py)默认用beam=5')
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--avg_topk', type=int, default=0,
                        help='[v2] checkpoint平均：训练结束后，把验证指标最好的K个epoch的权重取平均，'
                             '在验证集上评估并保存为avg.pt(不覆盖best.pt)。0=关闭，推荐3~5。'
                             '内存里会额外保留K份模型权重(CPU)')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    vocab = load_vocab(args.vocab_path)
    train_loader = get_dataloader(args.train_path, batch_size=args.batch_size, shuffle=True)
    valid_loader = get_dataloader(args.valid_path, batch_size=args.eval_batch_size, shuffle=False)
    params = train_loader.dataset.params
    print('train样本数: {}  valid样本数: {}'.format(len(train_loader.dataset), len(valid_loader.dataset)))
    print('数据参数:', params)

    model = build_model(vocab, params, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print('模型参数量: {:,}  use_fusion={}'.format(n_params, args.use_fusion))
    print('scale_emb={} share_emb={} dec_emb_ln={} encoder_type={}'.format(
        args.scale_emb, args.share_emb, args.dec_emb_ln, args.encoder_type))
    print('lr_schedule={} lr_base={} lr_decay={} min_lr_rate={}'.format(
        args.lr_schedule, args.lr_base, args.lr_decay, args.min_lr_rate))
    print('v2开关: use_struct_pos={} use_parent_emb={} use_token_global={} dec_norm_first={} avg_topk={}'.format(
        args.use_struct_pos, args.use_parent_emb, args.use_token_global, args.dec_norm_first, args.avg_topk))

    # weight_decay不能对所有参数一视同仁：LayerNorm的gamma/beta和所有bias项不应该参与衰减
    # (标准实践)，把LayerNorm的缩放系数往0拉会让归一化变得病态，这次实测复现过：加了
    # weight_decay之后，训练在warmup刚结束、学习率冲到峰值的时候直接NaN崩溃，去掉/排除
    # 掉LayerNorm和bias的衰减后应该能避免这个问题。
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'norm' in name.lower() or name.endswith('.bias') or name.endswith('_bias'):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    opt = torch.optim.Adam([
        dict(params=decay_params, weight_decay=args.regular_rate),
        dict(params=no_decay_params, weight_decay=0.0),
    ], lr=args.lr_base)
    print('weight_decay应用于{}个参数张量，跳过{}个(LayerNorm/bias)'.format(
        len(decay_params), len(no_decay_params)))
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=args.lr_factor,
                                  patience=args.lr_patience,
                                  min_lr=args.lr_base * args.min_lr_rate)

    os.makedirs(args.model_dir, exist_ok=True)
    best_metric = -1.0
    epochs_no_improve = 0
    global_step = 0
    topk = []  # [(指标, epoch, state_dict(cpu))]，按指标降序，只保留最好的avg_topk份

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc='epoch {}'.format(epoch))
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            if global_step < args.warmup_steps:
                lr = args.lr_base * (global_step + 1) / args.warmup_steps
                for g in opt.param_groups:
                    g['lr'] = lr

            opt.zero_grad()
            out = model(batch)
            out['loss'].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            global_step += 1
            epoch_losses.append(out['loss'].item())
            recent = epoch_losses[-50:]
            pbar.set_postfix(loss='{:.4f}'.format(sum(recent) / len(recent)),
                            lr='{:.2e}'.format(opt.param_groups[0]['lr']))

        train_loss = sum(epoch_losses) / len(epoch_losses)
        print('epoch {} 结束  train_loss={:.4f}  lr={:.2e}'.format(epoch, train_loss, opt.param_groups[0]['lr']))

        if args.lr_schedule == 'exp' and global_step >= args.warmup_steps:
            min_lr = args.lr_base * args.min_lr_rate
            for g in opt.param_groups:
                g['lr'] = max(g['lr'] * args.lr_decay, min_lr)

        if epoch >= args.start_valid_epoch and epoch % args.valid_every == 0:
            metrics = evaluate(model, valid_loader, vocab, device,
                               beam_width=args.valid_beam_width,
                               length_penalty=args.length_penalty)
            print('epoch {} valid_loss={:.4f}  BLEU(corpus)={:.2f}  S-BLEU={:.2f}  ROUGE-L={:.2f}'.format(
                epoch, metrics['loss'], metrics['bleu'], metrics['s_bleu'], metrics['rouge_l']))

            cur_metric = metrics[args.select_metric]
            if args.lr_schedule == 'plateau' and global_step >= args.warmup_steps:
                scheduler.step(cur_metric)

            if args.avg_topk > 1 and (len(topk) < args.avg_topk or cur_metric > topk[-1][0]):
                topk.append((cur_metric, epoch,
                             {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
                topk.sort(key=lambda x: x[0], reverse=True)
                del topk[args.avg_topk:]

            if cur_metric > best_metric:
                best_metric = cur_metric
                epochs_no_improve = 0
                torch.save(dict(model=model.state_dict(), epoch=epoch, bleu=metrics['bleu'],
                               s_bleu=metrics['s_bleu'], rouge_l=metrics['rouge_l'],
                               select_metric=args.select_metric, best_metric=best_metric,
                               args=vars(args)),
                          os.path.join(args.model_dir, 'best.pt'))
                print('  -> 新的最佳{}({:.2f})，已保存到 best.pt'.format(args.select_metric, best_metric))
            else:
                epochs_no_improve += 1
                print('  -> 未提升 ({}/{})'.format(epochs_no_improve, args.early_stop))
                if epochs_no_improve >= args.early_stop:
                    print('触发early stop，停止训练')
                    break

        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch),
                  os.path.join(args.model_dir, 'last.pt'))

    print('训练结束，最佳验证{}: {:.2f}'.format(args.select_metric, best_metric))

    if args.avg_topk > 1 and len(topk) > 1:
        used = [(round(m, 2), e) for m, e, _ in topk]
        print('checkpoint平均: 取验证{}最好的{}个epoch (指标, epoch)={}'.format(
            args.select_metric, len(topk), used))
        model.load_state_dict(average_state_dicts([sd for _, _, sd in topk]))
        m = evaluate(model, valid_loader, vocab, device, beam_width=args.valid_beam_width,
                     length_penalty=args.length_penalty)
        print('平均后 valid_loss={:.4f}  BLEU(corpus)={:.2f}  S-BLEU={:.2f}  ROUGE-L={:.2f}  '
              '(单个最佳 {}={:.2f})'.format(m['loss'], m['bleu'], m['s_bleu'], m['rouge_l'],
                                          args.select_metric, best_metric))
        torch.save(dict(model=model.state_dict(), epoch=-1, bleu=m['bleu'], s_bleu=m['s_bleu'],
                       rouge_l=m['rouge_l'], select_metric=args.select_metric,
                       best_metric=m[args.select_metric], args=vars(args),
                       averaged_epochs=[e for _, e, _ in topk]),
                  os.path.join(args.model_dir, 'avg.pt'))
        print('  -> 已保存到 avg.pt (评估时用 --ckpt_name avg.pt 或 evaluate.py --ckpt_path .../avg.pt)')


if __name__ == '__main__':
    main()