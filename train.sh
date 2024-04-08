#!/bin/bash

# 実行:
#chmod +x train.sh
#bash trian.sh
# 引数を設定
projects_name="BERTsentiment"
runs_name="training_imdb_001"
wandb=true
seed=0
bert_model_name="bert-base-uncased"
hf_dataset_name="stanfordnlp/imdb"
max_seq_length=512
testdata_split_rate=10
class_num=2
batch_size=64
epochs=50
is_DataParallel=true

# スクリプトの実行
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --projects_name "$projects_name" \
    --runs_name "$runs_name" \
    --wandb "$wandb" \
    --seed "$seed" \
    --bert_model_name "$bert_model_name" \
    --hf_dataset_name "$hf_dataset_name" \
    --max_seq_length "$max_seq_length" \
    --testdata_split_rate "$testdata_split_rate" \
    --class_num "$class_num" \
    --batch_size "$batch_size" \
    --epochs "$epochs" \
    --is_DataParallel "$is_DataParallel"