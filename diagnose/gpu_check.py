"""
10秒检测某张显卡能否承受训练时的大张量，用来区分"环境/驱动问题"和"批大小问题"。
模拟的是复制生成器里最大的那类张量：[批大小, 目标长度, 词表大小]，做softmax+前向+反向。
不依赖我们的模型和数据。

用法：python3 diagnose/gpu_check.py --device cuda:3
"""
import argparse, ctypes, os
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--vocab', type=int, default=16905)
    ap.add_argument('--tlen', type=int, default=49)
    ap.add_argument('--batches', default='64,100,128,200')
    args = ap.parse_args()

    print('torch %s | cuda %s | cuda可用: %s' % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
    print('PYTORCH_CUDA_ALLOC_CONF =', repr(os.environ.get('PYTORCH_CUDA_ALLOC_CONF')))
    print('LD_LIBRARY_PATH         =', repr(os.environ.get('LD_LIBRARY_PATH')))
    try:
        ctypes.CDLL('libcuda.so.1')
        print('libcuda.so.1: 可以加载')
    except OSError as e:
        print('libcuda.so.1: 加载失败 ->', e)

    dev = torch.device(args.device)
    if dev.type == 'cuda':
        p = torch.cuda.get_device_properties(dev)
        free, total = torch.cuda.mem_get_info(dev)
        print('显卡: %s | 总显存 %.1f GB | 当前空闲 %.1f GB' % (p.name, total / 2**30, free / 2**30))
    for b in [int(x) for x in args.batches.split(',')]:
        try:
            x = torch.randn(b, args.tlen, args.vocab, device=dev, requires_grad=True)
            y = (0.5 * x.softmax(-1) + 0.5 * x.softmax(-1)).clamp_min(1e-10).log().sum()
            y.backward()
            if dev.type == 'cuda':
                torch.cuda.synchronize(dev)
                peak = torch.cuda.max_memory_allocated(dev) / 2**30
                print('批大小=%-4d [B,T,V]前向+反向: 正常 (峰值 %.2f GB)' % (b, peak))
            else:
                print('批大小=%-4d [B,T,V]前向+反向: 正常 (CPU)' % b)
        except Exception as e:
            print('批大小=%-4d 失败: %s' % (b, str(e).strip().splitlines()[0][:110]))
        finally:
            x = y = None
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)


if __name__ == '__main__':
    main()