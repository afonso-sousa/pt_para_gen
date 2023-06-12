input_dir="raw-data"
output_dir="processed-data"

python processed-data/filter.py \
        --input_path $input_dir/opensubtitles/full.jsonl \
        --output_path $output_dir/ospt/full.jsonl \
        --prune_threshold 0.5

