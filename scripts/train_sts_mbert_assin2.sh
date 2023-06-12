#!/bin/bash

datasets_dir="processed-data"
output_dir="output"
# "bert-base-multilingual-cased"
model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
dataset="assin2"
output_file=$model_name-$dataset-sts


python sts/train.py \
    --dataset $dataset \
    --model_name $model_name \
    --epochs 10 \
    --warmup_steps 100