# # coding=utf-8
# """
# 最终测试集评估脚本。和train.py里validate用的实时评估不同，这里是训练完成后、
# 拿best checkpoint在test集上做一次完整评估：
#   1. 算更全面的一套指标(BLEU-1~4、corpus BLEU、ROUGE-1/2/L)
#   2. 把每一条样本的(id, 预测, 参考)都存成json文件，供后续人工做错误案例分析——
#      之前讨论过，光看聚合分数容易被指标本身的问题掩盖真实效果(比如之前发现过自定义
#      BLEU-1实现的bug)，肉眼抽查预测结果这一步不能省。

# 模型结构参数不用重新手动指定——直接从checkpoint里train.py当时存的args字段读取，
# 避免因为忘记改某个超参数、导致加载的模型结构和checkpoint权重对不上这种低级错误。

# 用法:
#     python3 evaluate.py --vocab_path ../w2i2w/vocab.pkl \
#         --test_path ../avail_data/test_data.pkl \
#         --ckpt_path ../model_ckpt/hier_v1/best.pt \
#         --out_path ../result/hier_v1_test_predictions.json
# """
# import argparse
# import json
# import os
# import pickle
# import sys

# import torch
# from tqdm import tqdm

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

# from dataset import get_dataloader  # noqa: E402
# from metrics import ids_to_words, compute_metrics  # noqa: E402
# from train import build_model  # noqa: E402  复用train.py里已经写好、测过的建模逻辑(内部会构造HierSeq2Seq)


# def load_vocab(path):
#     with open(path, 'rb') as f:
#         return pickle.load(f)


# class _Args:
#     """把checkpoint里存的args字典包装成一个有属性访问的对象，
#     这样可以直接复用train.py的build_model(vocab, params, args, device)，不用重写一遍。"""
#     def __init__(self, d):
#         self.__dict__.update(d)


# @torch.no_grad()
# def run_inference(model, loader, vocab, device, beam_width=5, length_penalty=1.0):
#     model.eval()
#     i2w = vocab['i2w']
#     pad_id = vocab['w2i']['<pad>']
#     bos_id = vocab['w2i']['<s>']
#     eos_id = vocab['w2i']['</s>']

#     records = []  # [{id, pred, ref}]
#     for batch in tqdm(loader, desc='inference'):
#         ids = batch['id'].tolist()
#         batch_gpu = {k: v.to(device) for k, v in batch.items()}
#         gen_ids = model.generate(batch_gpu, beam_width=beam_width, length_penalty=length_penalty)
#         for i in range(len(ids)):
#             hyp_words = ids_to_words(gen_ids[i].tolist(), i2w, bos_id, eos_id, pad_id)
#             ref_words = ids_to_words(batch['text_ids'][i].tolist(), i2w, bos_id, eos_id, pad_id)
#             records.append(dict(id=ids[i], pred=' '.join(hyp_words), ref=' '.join(ref_words)))
#     return records


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--vocab_path', type=str, required=True)
#     parser.add_argument('--test_path', type=str, required=True)
#     parser.add_argument('--ckpt_path', type=str, required=True)
#     parser.add_argument('--out_path', type=str, required=True,
#                         help='逐条预测结果保存路径(.json)，供后续错误案例分析用')
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--beam_width', type=int, default=5,
#                         help='beam搜索宽度，默认5(和HetCoS论文一致)。设为1则退化成贪心解码')
#     parser.add_argument('--length_penalty', type=float, default=1.0,
#                         help='长度惩罚指数: score=累计logP/(长度^该值)。1.0相当于按长度取平均'
#                              '对数概率(避免模型偏爱短句)，0则是纯累计概率')
#     parser.add_argument('--device', type=str, default='cuda')
#     args = parser.parse_args()

#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     print('device:', device)

#     vocab = load_vocab(args.vocab_path)
#     test_loader = get_dataloader(args.test_path, batch_size=args.batch_size, shuffle=False)
#     params = test_loader.dataset.params
#     print('test样本数: {}'.format(len(test_loader.dataset)))

#     ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
#     model_args = _Args(ckpt['args'])
#     model = build_model(vocab, params, model_args, device)
#     model.load_state_dict(ckpt['model'])
#     print('已加载checkpoint: {}  (来自epoch {}, 当时验证BLEU={:.2f})'.format(
#         args.ckpt_path, ckpt.get('epoch', '?'), ckpt.get('bleu', float('nan'))))

#     print('解码配置: beam_width={} length_penalty={}'.format(args.beam_width, args.length_penalty))
#     records = run_inference(model, test_loader, vocab, device,
#                             beam_width=args.beam_width, length_penalty=args.length_penalty)
#     hyps = [r['pred'] for r in records]
#     refs = [r['ref'] for r in records]
#     metrics = compute_metrics(hyps, refs, full=True)

#     print('\n===== test集指标 =====')
#     print('corpus BLEU: {:.2f}'.format(metrics['corpus_bleu']))
#     print('S-BLEU(句子级，和多数论文报告口径一致): {:.2f}'.format(metrics['s_bleu']))
#     print('BLEU-1/2/3/4: {:.2f} / {:.2f} / {:.2f} / {:.2f}'.format(
#         metrics['bleu1'], metrics['bleu2'], metrics['bleu3'], metrics['bleu4']))
#     print('ROUGE-1/2/L: {:.2f} / {:.2f} / {:.2f}'.format(
#         metrics['rouge1'], metrics['rouge2'], metrics['rougeL']))

#     os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
#     with open(args.out_path, 'w', encoding='utf-8') as f:
#         json.dump(dict(metrics=metrics, predictions=records), f, ensure_ascii=False, indent=2)
#     print('\n逐条预测结果+指标已保存到: {}'.format(args.out_path))


# if __name__ == '__main__':
#     main()

# coding=utf-8
"""
最终测试集评估脚本。和train.py里validate用的实时评估不同，这里是训练完成后、
拿best checkpoint在test集上做一次完整评估：
  1. 算更全面的一套指标(BLEU-1~4、corpus BLEU、ROUGE-1/2/L)
  2. 把每一条样本的(id, 预测, 参考)都存成json文件，供后续人工做错误案例分析——
     之前讨论过，光看聚合分数容易被指标本身的问题掩盖真实效果(比如之前发现过自定义
     BLEU-1实现的bug)，肉眼抽查预测结果这一步不能省。

模型结构参数不用重新手动指定——直接从checkpoint里train.py当时存的args字段读取，
避免因为忘记改某个超参数、导致加载的模型结构和checkpoint权重对不上这种低级错误。

用法:
    python3 evaluate.py --vocab_path ../w2i2w/vocab.pkl \
        --test_path ../avail_data/test_data.pkl \
        --ckpt_path ../model_ckpt/hier_v1/best.pt \
        --out_path ../result/hier_v1_test_predictions.json
"""
import argparse
import json
import os
import pickle
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from dataset import get_dataloader  # noqa: E402
from metrics import ids_to_words, compute_metrics  # noqa: E402
from train import build_model  # noqa: E402  复用train.py里已经写好、测过的建模逻辑(内部会构造HierSeq2Seq)


def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class _Args:
    """把checkpoint里存的args字典包装成一个有属性访问的对象，
    这样可以直接复用train.py的build_model(vocab, params, args, device)，不用重写一遍。"""
    def __init__(self, d):
        self.__dict__.update(d)


@torch.no_grad()
def run_inference(model, loader, vocab, device, beam_width=5, length_penalty=1.0):
    model.eval()
    i2w = vocab['i2w']
    pad_id = vocab['w2i']['<pad>']
    bos_id = vocab['w2i']['<s>']
    eos_id = vocab['w2i']['</s>']

    records = []  # [{id, pred, ref}]
    for batch in tqdm(loader, desc='inference'):
        ids = batch['id'].tolist()
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        gen_ids = model.generate(batch_gpu, beam_width=beam_width, length_penalty=length_penalty)
        for i in range(len(ids)):
            hyp_words = ids_to_words(gen_ids[i].tolist(), i2w, bos_id, eos_id, pad_id)
            ref_words = ids_to_words(batch['text_ids'][i].tolist(), i2w, bos_id, eos_id, pad_id)
            records.append(dict(id=ids[i], pred=' '.join(hyp_words), ref=' '.join(ref_words)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True,
                        help='逐条预测结果保存路径(.json)，供后续错误案例分析用')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beam_width', type=int, default=5,
                        help='beam搜索宽度，默认5(和HetCoS论文一致)。设为1则退化成贪心解码')
    parser.add_argument('--length_penalty', type=float, default=1.0,
                        help='长度惩罚指数: score=累计logP/(长度^该值)。1.0相当于按长度取平均'
                             '对数概率(避免模型偏爱短句)，0则是纯累计概率')
    parser.add_argument('--ref_json', type=str, default=None,
                        help='可选：包含真实参考摘要的json(含id和text字段，如token_data/test_data.json)。'
                             '给了就用它按id取参考摘要来打分；不给则从test_data.pkl的id还原参考摘要，'
                             '此时词表外的词在参考里是<unk>，模型预测<unk>也会算匹配(略偏宽松)。'
                             '建议始终提供，和官方评分用真实token一致')
    parser.add_argument('--text_mode', type=str, default='raw', choices=['raw', 'pretokenized'],
                        help='--ref_json里text字段的格式，必须和预处理时一致：raw=原始摘要；pretokenized=已分词/词形还原')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    vocab = load_vocab(args.vocab_path)
    test_loader = get_dataloader(args.test_path, batch_size=args.batch_size, shuffle=False)
    params = test_loader.dataset.params
    print('test样本数: {}'.format(len(test_loader.dataset)))

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model_args = _Args(ckpt['args'])
    model = build_model(vocab, params, model_args, device)
    model.load_state_dict(ckpt['model'])
    print('已加载checkpoint: {}  (来自epoch {}, 当时验证BLEU={:.2f})'.format(
        args.ckpt_path, ckpt.get('epoch', '?'), ckpt.get('bleu', float('nan'))))

    print('解码配置: beam_width={} length_penalty={}'.format(args.beam_width, args.length_penalty))
    records = run_inference(model, test_loader, vocab, device,
                            beam_width=args.beam_width, length_penalty=args.length_penalty)
    if args.ref_json:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preprocess'))
        from code_tokenizer import set_text_mode, tokenize_text
        set_text_mode(args.text_mode)
        with open(args.ref_json, encoding='utf-8') as f:
            ref_items = {it['id']: it for it in json.load(f)}
        n_unk = n_hit = 0
        for r in records:
            it = ref_items.get(r['id'])
            if it is None:
                continue
            n_hit += 1
            n_unk += ('<unk>' in r['ref'].split())
            r['ref_unk'] = r['ref']                      # 保留旧的(含<unk>)版本备查
            r['ref'] = ' '.join(tokenize_text(it.get('text', '')))
        print('参考摘要改用真实token: 按id匹配到 {}/{} 条(原来含<unk>的参考有 {} 条)'.format(
            n_hit, len(records), n_unk))
        if n_hit < len(records):
            print('!! 警告：有 {} 条没匹配到id，这些样本仍沿用含<unk>的参考'.format(len(records) - n_hit))
    hyps = [r['pred'] for r in records]
    refs = [r['ref'] for r in records]
    metrics = compute_metrics(hyps, refs, full=True)

    print('\n===== test集指标 =====')
    print('corpus BLEU: {:.2f}'.format(metrics['corpus_bleu']))
    print('S-BLEU(句子级，和多数论文报告口径一致): {:.2f}'.format(metrics['s_bleu']))
    print('BLEU-1/2/3/4: {:.2f} / {:.2f} / {:.2f} / {:.2f}'.format(
        metrics['bleu1'], metrics['bleu2'], metrics['bleu3'], metrics['bleu4']))
    print('ROUGE-1/2/L: {:.2f} / {:.2f} / {:.2f}'.format(
        metrics['rouge1'], metrics['rouge2'], metrics['rougeL']))

    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    with open(args.out_path, 'w', encoding='utf-8') as f:
        json.dump(dict(metrics=metrics, predictions=records), f, ensure_ascii=False, indent=2)
    print('\n逐条预测结果+指标已保存到: {}'.format(args.out_path))


if __name__ == '__main__':
    main()