#!/bin/bash

datasets_dir="processed-data"
model_dir="output"
output_dir=$model_dir
model_name="facebook/mbart-large-50"
lr="1e-4"
dataset="assin2"
name=$model_name-ospt-lr$lr-standard
checkpoint="checkpoint-15000"


python para_gen/generate_predictions.py \
    --validation_file $datasets_dir/$dataset/test.jsonl \
    --model_name_or_path $model_dir/${name}/${checkpoint} \
    --output_dir $output_dir/$name \
    --source_lang pt_XX \
    --target_lang pt_XX \
    --forced_bos_token pt_XX \
    --source_column source \
    --target_column target \
    --per_device_eval_batch_size 16 \
