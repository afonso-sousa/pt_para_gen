datasets_dir="raw-data"
dataset="opensubtitles"
input_dir="raw-data"
output_file=$datasets_dir/$dataset/full.jsonl

command="python raw-data/translate.py \
        --input_dir $datasets_dir/$dataset \
        --output_path $output_file \
        --num_examples 2000000"

if [ ! -f "$output_file" ]; then
    eval $command
else
    echo "Translations already exists at '$output_file'"
fi
