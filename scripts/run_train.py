# coding=utf-8
"""
一条命令跑完：训练(train.py) -> 用训练出的best checkpoint在test集上做最终评估(evaluate.py)。

这里用subprocess分别调用train.py/evaluate.py(而不是像run_preprocess.py那样直接import
函数复用)，是因为train.py的main()是一次性、有副作用的长流程(整个训练循环)，不像
vocab_builder/build_dataset那样是"调一下函数拿到结果"这种干净的调用方式，subprocess
能保留train.py/evaluate.py各自独立可运行、方便单独调试的特性，也能让训练和评估分处
两个独立进程(训练显存能在评估开始前正常释放，避免同一进程里模型/优化器状态残留导致
显存不够用)。

用法:
    python3 run_train.py --vocab_path ../w2i2w/vocab.pkl \
        --train_path ../avail_data/train_data.pkl \
        --valid_path ../avail_data/valid_data.pkl \
        --test_path ../avail_data/test_data.pkl \
        --model_dir ../model_ckpt/hier_v1 \
        --result_path ../result/hier_v1_test_predictions.json \
        --use_fusion true --emb_dims 512 --batch_size 64
    其余train.py支持的超参数(--n_heads/--dec_layers/--max_epochs/--lr_base/...)
    都可以直接加在后面，会原样透传给train.py。
"""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True,
                        help='最终test集逐条预测结果+指标的保存路径(.json)')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='evaluate.py用的batch_size，不指定则和--batch_size一致')
    parser.add_argument('--eval_beam_width', type=int, default=5,
                        help='最终test集评估的beam宽度(默认5，和HetCoS论文一致)。'
                             '注意训练过程中的验证默认仍用贪心(见train.py的--valid_beam_width)，'
                             '这样每个epoch的验证不会被beam拖慢')
    parser.add_argument('--eval_length_penalty', type=float, default=1.0)
    parser.add_argument('--ckpt_name', type=str, default='best.pt',
                        help='最终test评估使用model_dir下哪个checkpoint，默认best.pt；'
                             '训练时设了--avg_topk的话可以改成avg.pt')
    parser.add_argument('--skip_train', action='store_true',
                        help='跳过训练，只用model_dir下已有的best.pt做评估(方便重新跑评估而不用重新训练)')

    args, extra_train_args = parser.parse_known_args()
    # extra_train_args里是所有run_train.py没有显式定义、但train.py认识的参数，
    # 比如--emb_dims/--n_heads/--use_fusion/--max_epochs这些，原样透传给train.py

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_py = os.path.join(script_dir, '..', 'train', 'train.py')
    evaluate_py = os.path.join(script_dir, '..', 'train', 'evaluate.py')

    if not args.skip_train:
        print('=' * 60)
        print('第1步: 训练')
        print('=' * 60)
        train_cmd = [sys.executable, train_py,
                    '--vocab_path', args.vocab_path,
                    '--train_path', args.train_path,
                    '--valid_path', args.valid_path,
                    '--model_dir', args.model_dir] + extra_train_args
        print('执行命令:', ' '.join(train_cmd))
        ret = subprocess.run(train_cmd)
        if ret.returncode != 0:
            print('训练失败(退出码{})，不再继续做test集评估'.format(ret.returncode))
            sys.exit(ret.returncode)
    else:
        print('--skip_train已指定，跳过训练，直接用已有的{}做评估'.format(args.ckpt_name))

    best_ckpt = os.path.join(args.model_dir, args.ckpt_name)
    if not os.path.exists(best_ckpt):
        print('找不到{}，无法做test集评估'.format(best_ckpt))
        sys.exit(1)

    print('\n' + '=' * 60)
    print('第2步: 用{}在test集上做最终评估'.format(args.ckpt_name))
    print('=' * 60)
    eval_cmd = [sys.executable, evaluate_py,
               '--vocab_path', args.vocab_path,
               '--test_path', args.test_path,
               '--ckpt_path', best_ckpt,
               '--out_path', args.result_path,
               '--beam_width', str(args.eval_beam_width),
               '--length_penalty', str(args.eval_length_penalty)]
    if args.eval_batch_size is not None:
        eval_cmd += ['--batch_size', str(args.eval_batch_size)]
    # evaluate.py的--device没有透传参数入口，从train.py的透传参数里找一下(如果指定了--device)
    if '--device' in extra_train_args:
        idx = extra_train_args.index('--device')
        eval_cmd += ['--device', extra_train_args[idx + 1]]
    print('执行命令:', ' '.join(eval_cmd))
    ret = subprocess.run(eval_cmd)
    sys.exit(ret.returncode)


if __name__ == '__main__':
    main()