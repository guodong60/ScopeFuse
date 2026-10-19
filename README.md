# ScopeFuse: Hierarchical Encoding with Token-Aligned Context Fusion for Java Code Summarization

Method-level Java code summarization. The model combines a hierarchical encoder
(token → statement → block → function), an 8-layer Transformer decoder, and a pointer-generator copy mechanism.
This repository contains the preprocessing, model, training and evaluation code in the paper.

## 1. Environment

- OS: Linux (developed on Ubuntu)
- Python: Python 3.10.21
- GPU: NVIDIA GPU, driver 570.133.07 / CUDA 12.8; PyTorch built for CUDA 11.8 (`torch==2.1.0+cu118`). GPU model: NVIDIA GeForce RTX 4090
- Java runtime (needed by the official METEOR scorer): openjdk version "24.0.2-internal" 
- Python packages: see `requirements.txt`

```bash
conda create -n hgcodesum python=3.10.21    
conda activate hgcodesum
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The official metrics (ROUGE-L / S-BLEU / METEOR) use `my_lib` from the HetCoS repository. Please obtain it from that
repository and follow its license; it is not included here. Below, `MYLIB_PARENT` is the directory that contains the
`my_lib` folder.

## 2. Data and released artifacts

We use the Java method–comment dataset of Hu et al. (IJCAI 2018, TL-CodeSum), with the lemmatized summary version
(`token_data`) released by HetCoS (summaries end with `..`; preprocessing uses `--text_mode pretokenized`).
Dataset source: https://drive.google.com/open?id=108AjmO2-uer3tTk2yJ_LENLdrehIz81A&usp=drive_fs.

Place the data as follows (test set: 8,714 samples):

```
raw_data/train_data.json
raw_data/valid_data.json
raw_data/test_data.json
```

For convenience, `HierSeq2Seq_dataset.zip` bundles the data, the test-set predictions and the main-model checkpoint:
[download](https://drive.google.com/file/d/108AjmO2-uer3tTk2yJ_LENLdrehIz81A/view). Unzip it at the repository root to get
`raw_data/`, `result/` and `model_ckpt/`. 

## 3. Repository layout

```
preprocess/   parsing and preprocessing: java_parser, stmt_block_builder, code_tokenizer, vocab_builder, build_dataset
model/        hierarchical encoder, fusion module, decoder, copy generator
train/        train.py, evaluate.py, metrics.py
data/         dataset.py
scripts/      run_preprocess.py (vocabulary + dataset in one command)
diagnose/     evaluation and analysis: official_eval, summarize_runs, bucket_eval2, bucket_tables, export_cases, diag_retrieval
```

## 4. Preprocessing

```bash
cd scripts
python3 run_preprocess.py --raw_data_dir ../raw_data \
    --vocab_out_path ../w2i2w_tk/vocab.pkl \
    --avail_data_out_dir ../avail_data_tk \
    --text_mode pretokenized
```

Defaults: shared code/summary vocabulary, `min_count=3`, max vocabulary 50,000 (16,905 in practice). Length limits:
300 code sub-tokens, 64 statements, 24 blocks, nesting depth 8, 50 summary words.

## 5. Training the main model

Run from `train/`:

```bash
cd train
COMMON="--vocab_path ../w2i2w_tk/vocab.pkl --train_path ../avail_data_tk/train_data.pkl \
        --valid_path ../avail_data_tk/valid_data.pkl --use_struct_pos true --scale_emb true --avg_topk 5"
python3 train.py $COMMON --seed 0 --model_dir ../model_ckpt/exp9_tk_base --device cuda:0 \
    2>&1 | tee ../logs_exp9.txt
```

Main settings: batch size 64, dropout 0.2, base learning rate 2e-4, 2,000 warm-up steps, ReduceLROnPlateau,
gradient clipping 5.0, at most 100 epochs, early stopping patience 10; one hierarchical block, 8 decoder layers,
512-dim embeddings, 8 heads, tied output weights, copy mechanism on.
Checkpoint rule: each epoch is scored by greedy-decoding S-BLEU on the validation set to pick `best.pt`; the weights of the
5 best epochs are averaged into `avg.pt`; whichever has the higher validation S-BLEU is used. Training takes about
15 hours on a single GPU (excluding validation).

## 6. Evaluation

```bash
# (1) from the repository root: read logs, pick best.pt/avg.pt on validation, generate the evaluation script
python3 diagnose/summarize_runs.py --ref_json raw_data/test_data.json --map logs_exp9.txt=exp9_tk_base --gpus 0
# (2) from train/: run the generated script -> ../result/exp9_tk_base_test.json (beam 5, length penalty 1.0)
cd train && bash ../run_eval_all.sh
# (3) official metrics: run where my_lib can be imported
cd <HetCoS official script directory>
python <this repo>/diagnose/official_eval.py --my_lib_parent MYLIB_PARENT \
    --result_paths <this repo>/result/exp9_tk_base_test.json
```
