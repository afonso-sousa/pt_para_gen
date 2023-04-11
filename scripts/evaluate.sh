#!/bin/bash

seed=1
predictions_dir="output"
output_dir=$predictions_dir
model_name="facebook/m2m100_418M"
dataset="opensubtitles"
lr="1e-4"
name=$model_name-$dataset-lr$lr-standard
metrics="my_metric"
output_file="$output_dir/$name/$metrics.csv"

if [ ! -f "$output_file" ]; then
    job="python compute_metrics.py \
            --source_column source \
            --target_column target \
            --predictions_column prediction \
            --train_file $predictions_dir/$name/generated_predictions.csv \
            --dataset_split train \
            --metric metrics/$metrics \
            --output_path $output_file \
        "
    eval $job
fi            
