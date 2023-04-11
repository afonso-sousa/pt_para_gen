raw_datasets_dir="raw-data"
processed_datasets_dir="processed-data"
dataset="opensubtitles"

output_dir=$processed_datasets_dir/$dataset
command="python make_splits.py \
        --data_file $raw_datasets_dir/$dataset/full_data.jsonl \
        --predictions_column source \
        --references_column target \
        --output_dir $output_dir"

if [ ! -d "$output_dir" ]; then
    eval $command
fi
