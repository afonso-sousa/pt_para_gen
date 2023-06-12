#!/bin/bash

datasets_dir="processed-data"
output_dir="output"
model_name="facebook/mbart-large-50"
lr="1e-4"
dataset="ospt"
output_file=$model_name-$dataset-lr$lr-standard
validation_option=""
validation_file="$datasets_dir/$dataset/validation.jsonl"
if [ -f "$validation_file" ]; then
  validation_option="--validation_file $validation_file"
  do_eval_option="--do_eval"
fi

evaluation_strategy_option="--evaluation_strategy epoch"
save_strategy_option="--save_strategy epoch"
if [ -z "$validation_option" ]; then
  evaluation_strategy_option="--evaluation_strategy no"
  save_strategy_option="--save_strategy no"
fi

CUDA_VISIBLE_DEVICES=1 python para_gen/train.py \
    --model_name_or_path $model_name \
    --do_train \
    $do_eval_option \
    --source_lang pt_XX \
    --target_lang pt_XX \
    --forced_bos_token pt_XX \
    --source_column source \
    --target_column target \
    --output_dir $output_dir/$output_file \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --overwrite_output_dir \
    --train_file $datasets_dir/$dataset/train.jsonl \
    $validation_option \
    --warmup_steps 100 \
    --logging_step 100 \
    --learning_rate $lr \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 1 \
    --max_source_length 64 \
    --max_target_length 64 \
    --load_best_model_at_end \
    --logging_strategy steps \
    $evaluation_strategy_option \
    $save_strategy_option \
    --save_total_limit 2
