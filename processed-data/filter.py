import argparse
import codecs
import os
import re

import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import evaluate

tqdm.pandas()


def get_num_words(sample):
    return len(word_tokenize(sample, language="portuguese"))


def normalize_quotes(text):
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
        text = text.replace(ch, '"')

    start_index = text.find('"')
    if start_index != -1:  # if the first quote was found
        end_index = text.find('"', start_index + 1)
        if end_index == -1:
            text = text[:start_index] + text[start_index + 1 :]
        if (
            start_index == 0 and end_index == len(text) - 1
        ):  # leading and trailing quotes
            text = text[1 : len(text) - 1]

    return text


def process_data(row):
    match_begin_parentheses = "^[\(\[].*?[\)\]]"
    for k, v in row.items():
        # replace non-breaking spaces
        v = v.replace(chr(160), " ")
        # filter closed captions
        v = re.sub(f"^{match_begin_parentheses}", "", v)
        # normalize spaces and characters
        v = normalize_quotes(v)
        # covers spaces, tabs and newline chars
        v = re.sub(r"\s+", " ", v)
        v = v.strip()
        # filter leading hyphen
        v = re.sub("^-", "", v)
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
        default=0.5,
        help="Word count threshold.",
    )
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=0.8,
        help="Semantic threshold.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

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

    # remove based on semantic similarity
    sbert = evaluate.load(
        "metrics/mSBERT",
    )

    df["sem_sim"] = sbert.compute(
        predictions=df["target"].tolist(), references=df["source"].tolist()
    )["scores"]

    df = df[df["sem_sim"] > args.semantic_threshold]

    # prepare to save
    df = df.reset_index(drop=True)
    # df = df.drop(["src_len", "tgt_len"], axis=1)
    df.to_json(
        args.output_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )
