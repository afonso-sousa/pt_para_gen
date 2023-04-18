import argparse
import codecs
import re

import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm

tqdm.pandas()


def get_num_words(sample):
    return len(word_tokenize(sample, language="portuguese"))


def process_data(row):
    match_begin_parentheses = "^[\(\[].*?[\)\]]"
    for k, v in row.items():
        # filter closed captions
        v = re.sub(f"^{match_begin_parentheses}", "", v)
        # normalize spaces and characters
        chars_to_quotes = [
            "``",
            "` `",
            "‘‘",
            "‘ ‘",
            "’’",
            "’ ’",
            "’",
            "“",
            "”",
            "''",
            "' '",
            '""',
        ]
        for ch in chars_to_quotes:
            v.replace(ch, '"')
        v = v.strip()
        # filter hyphen stuff
        v = re.sub("^-", "", v)
        # covers spaces, tabs and newline chars
        v = re.sub(r"\s+", " ", v)
        # remove leading and trailing spaces
        v = v.strip()

        row[k] = v

    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Filter input dataset")
    parser.add_argument(
        "--input_path",
        default="raw-data/full_data.jsonl",
        help="The path to load the dataset.",
    )
    parser.add_argument(
        "--output_path",
        default="processed-data/full_data.jsonl",
        help="The path to save the dataset.",
    )
    parser.add_argument(
        "--prune_threshold",
        type=float,
        default=None,
        help="Word count threshold.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    print("Loading data...")
    df = pd.read_json(
        codecs.open(args.input_path, "r", "utf-8"), lines=True, orient="records"
    )
    print("Normalizing and removing closed captions...")
    # normalize chars and remove closed captions
    df[["source", "target"]] = df[["source", "target"]].apply(process_data, axis=1)
    print("Computing word counts...")
    # new word count columns
    df["src_len"] = df["source"].progress_apply(get_num_words)
    df["tgt_len"] = df["target"].progress_apply(get_num_words)
    # remove high difference in word count
    df = df[
        abs(df["src_len"] - df["tgt_len"]) / df[["src_len", "tgt_len"]].max(axis=1)
        < args.prune_threshold
    ]
    # remove equal
    df = df[df["source"] != df["target"]]

    # prepare to save
    df = df.reset_index(drop=True)
    df = df.drop(["src_len", "tgt_len"], axis=1)
    df.to_json(
        args.output_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )
