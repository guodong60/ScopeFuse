# coding=utf-8
"""
统一的指标计算模块，train.py(训练中监控用)和evaluate.py(最终测试集评估用)都从这里导入，
不再各自维护一份几乎一样的BLEU/ROUGE计算代码——之前发现过自定义指标实现出bug的教训，
指标计算逻辑只能有一份来源，散落在多个文件里迟早会出现两边口径不一致又没人发现的情况。
"""
import sacrebleu
from rouge_score import rouge_scorer

_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def sentence_bleu(hyp, ref):
    """句子级BLEU：单条样本自己算一个BLEU分数(0~100)。
    用sacrebleu.sentence_bleu(自带平滑，避免n-gram命中0次导致分数直接归零)。
    注意：不同论文/代码库对"S-BLEU"的平滑方法选择不完全一致，这里只能做到
    "标准、有据可查的实现"，不保证和某篇论文的具体代码逐位对齐——但至少
    比之前混用corpus BLEU去对比论文里的S-BLEU数字要公平得多。
    """
    return sacrebleu.sentence_bleu(hyp, [ref]).score


def corpus_level_and_sentence_level_bleu(hyps, refs):
    """同时返回corpus BLEU(整个语料统一算n-gram)和S-BLEU(逐句算完再平均)。
    这两个数字对同一批预测结果可以差出5~15个点(S-BLEU通常明显偏高，
    尤其短句多的时候)，混着比较毫无意义——报告/论文对比时必须先确认
    对方用的是哪一种，再用同一种口径去比。
    """
    corpus_bleu_score = sacrebleu.corpus_bleu(hyps, [refs], tokenize='none').score
    s_bleu_score = sum(sentence_bleu(h, r) for h, r in zip(hyps, refs)) / max(1, len(hyps))
    return corpus_bleu_score, s_bleu_score


def ids_to_words(ids, i2w, bos_id, eos_id, pad_id, unk_token='<unk>'):
    """把id序列还原成词序列：跳过<s>/<pad>，遇到</s>就截断(后面的忽略)"""
    words = []
    for tid in ids:
        tid = int(tid)
        if tid == bos_id or tid == pad_id:
            continue
        if tid == eos_id:
            break
        words.append(i2w.get(tid, unk_token))
    return words


def compute_metrics(hyps, refs, full=False):
    """
    hyps/refs: 字符串列表(每条样本一个，词之间空格分隔，和code_tokenizer.tokenize_text的
              分词风格一致)，等长。
    full=False(训练时验证用，图快): 算corpus BLEU、S-BLEU(句子级)和ROUGE-L
    full=True (最终测试集评估用): 额外算BLEU-1~4分项精确率、ROUGE-1/2

    返回dict，字段见下面两个分支。s_bleu字段专门用于和论文里常见的"S-BLEU"口径对比，
    不要拿corpus_bleu/bleu字段去跟论文的S-BLEU数字比，口径不同没有可比性。
    """
    hyps = [h if h else '<empty>' for h in hyps]
    refs = [r if r else '<empty>' for r in refs]

    corpus_bleu_score, s_bleu_score = corpus_level_and_sentence_level_bleu(hyps, refs)

    n = max(1, len(refs))
    if not full:
        rl_sum = sum(_rouge_scorer.score(r, h)['rougeL'].fmeasure for r, h in zip(refs, hyps))
        return dict(bleu=corpus_bleu_score, s_bleu=s_bleu_score, rouge_l=100.0 * rl_sum / n)

    # tokenize='none': hyps/refs已经是我们自己分好词的结果，不需要sacrebleu再用它自己的
    # 规则重新分词，否则标点/标识符切分规则不一致，BLEU会算得不准
    corpus_bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
    bleu1, bleu2, bleu3, bleu4 = corpus_bleu.precisions
    r1_sum = r2_sum = rl_sum = 0.0
    for ref, hyp in zip(refs, hyps):
        scores = _rouge_scorer.score(ref, hyp)
        r1_sum += scores['rouge1'].fmeasure
        r2_sum += scores['rouge2'].fmeasure
        rl_sum += scores['rougeL'].fmeasure
    return dict(
        corpus_bleu=corpus_bleu_score, s_bleu=s_bleu_score,
        bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4,
        rouge1=100.0 * r1_sum / n, rouge2=100.0 * r2_sum / n, rougeL=100.0 * rl_sum / n,
    )


if __name__ == '__main__':
    i2w = {0: '<pad>', 1: '<s>', 2: '</s>', 3: 'runs', 4: 'a', 5: 'command'}
    ids = [1, 3, 4, 5, 2, 0, 0]  # <s> runs a command </s> <pad> <pad>
    words = ids_to_words(ids, i2w, bos_id=1, eos_id=2, pad_id=0)
    print('ids_to_words:', words)
    assert words == ['runs', 'a', 'command']

    m = compute_metrics(['runs a command', 'find price'], ['runs a command', 'find price list'], full=True)
    print('compute_metrics(full=True):', m)
    assert 'corpus_bleu' in m and 's_bleu' in m
    m2 = compute_metrics(['runs a command'], ['runs a command'], full=False)
    print('compute_metrics(full=False):', m2)
    assert 'bleu' in m2 and 's_bleu' in m2
    print('自测通过')