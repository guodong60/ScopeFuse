"""
用论文口径重新给你的预测结果打分(不需要重新推理，不需要GPU)。

背景(已用官方仓库的预测结果验证)：同一份预测，用不同的指标实现打分，数字差很多：
  - ROUGE-L：论文用 pycocoevalcap 的 Rouge(beta=1.2)，在官方预测上精确复现论文的 63.10；
             我们 metrics.py 用的 rouge_score(F1)在同一份预测上只有 60.04
  - S-BLEU ：论文用其私有库 my_lib 里的 NLTK3.3 句子级BLEU(带平滑)，官方预测上是 50.70；
             我们用 sacrebleu 句子级是 47.22；本脚本的 NLTK method2 近似是 49.33
             (拿不到 my_lib，无法逐位复现，所以这里的S-BLEU只是"更接近论文"的近似)

用法(在项目根目录)：
  pip install nltk pycocoevalcap
  python3 diagnose/rescore_paper_style.py --result_path result/exp1_structpos_test.json
  # 再加 --lemmatize：对预测和参考两边做同样的近似词形还原后再打分，
  # 用来估计"论文参考摘要做了词形还原"这件事对分数的影响(首次运行会联网下载nltk数据)
  python3 diagnose/rescore_paper_style.py --result_path result/exp1_structpos_test.json --lemmatize

注意：--lemmatize 是近似(官方的 my_lib 拿不到)，在官方数据上整句完全一致率只有约30%，
所以它给出的是"量级估计"，不是论文数字的精确复现，也不能把它的结果当成最终报告的数字。
"""
import argparse, json, warnings
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.rouge.rouge import Rouge

warnings.filterwarnings('ignore')


def make_lemmatizer():
    """近似官方的词形还原：NLTK POS标注 + WordNet lemmatize(returns->return, serialized->serialize, being->be)"""
    import nltk
    for pkg in ('wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng', 'punkt_tab'):
        nltk.download(pkg, quiet=True)
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    lem = WordNetLemmatizer()
    to_wn = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

    def norm(text):
        toks = text.split()
        if not toks:
            return text
        tagged = nltk.pos_tag(toks)
        return ' '.join(lem.lemmatize(w, to_wn.get(t[0], wordnet.NOUN)) for w, t in tagged)
    return norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_path', required=True, help='evaluate.py 的 --out_path 生成的json')
    ap.add_argument('--hyp_key', default='pred')
    ap.add_argument('--ref_key', default='ref')
    ap.add_argument('--lemmatize', action='store_true', help='两边都做近似词形还原后再打分')
    args = ap.parse_args()

    data = json.load(open(args.result_path, encoding='utf-8'))
    recs = data['predictions'] if isinstance(data, dict) else data
    hyps = [r[args.hyp_key] if r[args.hyp_key].strip() else '<empty>' for r in recs]
    refs = [r[args.ref_key] for r in recs]
    print('样本数:', len(recs))
    if args.lemmatize:
        norm = make_lemmatizer()
        hyps = [norm(h) for h in hyps]
        refs = [norm(r) for r in refs]
        print('已对预测和参考两边做近似词形还原')

    rouge = Rouge()
    rl = 100 * np.mean([rouge.calc_score([h], [r]) for h, r in zip(hyps, refs)])
    sm = SmoothingFunction()
    sb = 100 * np.mean([sentence_bleu([r.split()], h.split(), smoothing_function=sm.method2)
                        for h, r in zip(hyps, refs)])

    print('\n===== 论文口径(近似)%s =====' % ('，已词形还原' if args.lemmatize else ''))
    print('ROUGE-L (pycocoevalcap, beta=1.2, 与论文一致): %.2f' % rl)
    print('S-BLEU  (NLTK method2, 近似论文)              : %.2f' % sb)
    if 'metrics' in data:
        m = data['metrics']
        print('\n(对照) 你原来的口径: S-BLEU=%.2f  ROUGE-L=%.2f' % (m.get('s_bleu', float('nan')),
                                                                  m.get('rougeL', float('nan'))))
    print('\n提醒：即使指标实现对齐，论文的参考摘要经过了词形还原(returns->return, serialized->serialize)'
          '并在句末加了 ".." 标记，你现在的参考摘要没有，这部分差异无法靠换指标消除。')


if __name__ == '__main__':
    main()