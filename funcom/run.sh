#!/usr/bin/env bash
# FunCom-50 专用流程。只调用现有程序，不修改任何现有文件；所有产物都带 funcom 后缀，和 TL-CodeSum 的互不覆盖。
# 把 convert_funcom.py 从 preprocess/ 移到 funcom/ 即可（两份内容相同）。
# 在任意目录运行都可以（脚本会先 cd 到 java/）：
#   bash funcom/run.sh convert       # jsonl -> train/valid/test_data.json
#   bash funcom/run.sh preprocess    # 建词表 + 生成 pkl
#   bash funcom/run.sh train         # 训练（约15小时，建议在 screen/tmux 里跑）
#   bash funcom/run.sh eval          # 测试集解码(beam 5)
#   bash funcom/run.sh official      # 官方指标，同时给出两种摘要末尾约定的结果
# 可用环境变量覆盖：NAME DEVICE MAX_EPOCHS MIN_COUNT MAX_VOCAB TRAIN_SIZE MY_LIB_PARENT SEED
set -e
cd "$(dirname "$0")/.."

NAME=${NAME:-fc_full_all}
DEVICE=${DEVICE:-cuda:4}
SEED=${SEED:-0}
MAX_EPOCHS=${MAX_EPOCHS:-10}
MIN_COUNT=${MIN_COUNT:-3}
MAX_VOCAB=${MAX_VOCAB:-50000}
TRAIN_SIZE=${TRAIN_SIZE:-0}                  # 0=全部训练样本
MY_LIB_PARENT=${MY_LIB_PARENT:-$(cd .. && pwd)}
RAW=raw_data/funcom
CONV=raw_data/funcom_conv_full
VOC=w2i2w_funcom_full/vocab.pkl
AVAIL=avail_data_funcom_full

case "$1" in
  convert)
    python3 funcom/convert_funcom.py --in_dir $RAW --out_dir $CONV --valid_ratio 0.05 --seed 0 \
        --train_size $TRAIN_SIZE --valid_size 2000 ;;
  preprocess)
    python3 scripts/run_preprocess.py --raw_data_dir $CONV --vocab_out_path $VOC \
        --avail_data_out_dir $AVAIL --text_mode pretokenized \
        --min_count $MIN_COUNT --max_vocab_size $MAX_VOCAB ;;
  train)
    python3 train/train.py --vocab_path $VOC --train_path $AVAIL/train_data.pkl \
        --valid_path $AVAIL/valid_data.pkl --use_struct_pos true --scale_emb true \
        --avg_topk 3 --max_epochs $MAX_EPOCHS --lr_patience 1 --early_stop 3 \
        --seed $SEED --model_dir model_ckpt/$NAME --device $DEVICE 2>&1 | tee logs_$NAME.txt ;;
  eval)
    CKPT=${CKPT:-model_ckpt/$NAME/avg.pt}      # 按验证集结果决定用 avg.pt 还是 best.pt
    python3 train/evaluate.py --vocab_path $VOC --test_path $AVAIL/test_data.pkl \
        --ckpt_path $CKPT --out_path result/${NAME}_test.json \
        --ref_json $CONV/test_data.json --text_mode pretokenized --device $DEVICE ;;
  official)
    python3 funcom/finalize_eval.py --result result/${NAME}_test.json --my_lib_parent $MY_LIB_PARENT ;;
  *) echo "用法: bash funcom/run.sh {convert|preprocess|train|eval|official}"; exit 1 ;;
esac