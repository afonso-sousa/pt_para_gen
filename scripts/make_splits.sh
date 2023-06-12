processed_datasets_dir="processed-data"
dataset="ospt"

output_dir=$processed_datasets_dir/$dataset
command="python processed-data/make_splits.py \
        --data_file $processed_datasets_dir/$dataset/full.jsonl \
        --output_dir $output_dir"

if [ ! -f "$output_dir/train.jsonl" ]; then
    eval $command
else
    echo "Splits already exists at '$output_dir'"
fi
