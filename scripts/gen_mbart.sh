#!/bin/bash

datasets_dir="processed-data"
model_dir="output"
output_dir=$model_dir
model_name="facebook/mbart-large-50"
lr="1e-4"
dataset="opensubtitles"
name=$model_name-$dataset-lr$lr-standard
checkpoint="checkpoint-7264"


python para_gen/generate_predictions.py \
    --validation_file $datasets_dir/$dataset/test.jsonl \
    --model_name_or_path $model_dir/${name}/${checkpoint} \
    --output_dir $output_dir/$name \
    --source_lang pt \
    --target_lang pt \
    --source_column source \
    --target_column target \
    --per_device_eval_batch_size 16 \
