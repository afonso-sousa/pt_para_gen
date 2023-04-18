raw_datasets_dir="raw-data"
processed_datasets_dir="processed-data"
dataset="opensubtitles"

output_dir=$processed_datasets_dir/$dataset
command="python processed-data/make_splits.py \
        --data_file $raw_datasets_dir/$dataset/full.jsonl \
        --output_dir $output_dir"

if [ ! -d "$output_dir" ]; then
    eval $command
else
    echo "Splits already exists at '$output_dir'"
fi
