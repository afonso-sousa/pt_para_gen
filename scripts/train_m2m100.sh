#!/bin/bash

datasets_dir="processed-data"
output_dir="output"
model_name="facebook/m2m100_418M"
lr="1e-4"
dataset="opensubtitles"
output_file=$model_name-$dataset-lr$lr-standard

python para_gen/train.py \
    --model_name_or_path $model_name \
    --do_train \
    --do_eval \
    --source_lang pt \
    --target_lang pt \
    --forced_bos_token pt \
    --source_column source \
    --target_column target \
    --output_dir $output_dir/$output_file \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --overwrite_output_dir \
    --train_file $datasets_dir/$dataset/train.jsonl \
    --validation_file $datasets_dir/$dataset/validation.jsonl \
    --warmup_steps 100 \
    --logging_step 100 \
    --learning_rate $lr \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 1 \
    --max_source_length 64 \
    --max_target_length 64 \
    --load_best_model_at_end \
    --logging_strategy steps \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2
