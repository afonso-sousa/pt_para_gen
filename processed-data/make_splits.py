import argparse
import os

import pandas as pd
import codecs


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset with translation")
    parser.add_argument(
        "--data_file",
        type=str,
        default="raw-data",
        help="The input training data file (a jsonlines or csv file).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=300_000,
        help="Number of samples. Defaults to 300000 samples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)

    df = pd.read_json(
        codecs.open(args.data_file, "r", "utf-8"), lines=True, orient="records"
    )

    df = df.sample(n=args.num_samples if args.num_samples else len(df), random_state=42)

    train = df.sample(frac=0.8, random_state=42)
    aux = df.drop(train.index)
    val = aux.sample(frac=0.5, random_state=42)
    test = aux.drop(val.index)

    train.to_json(
        os.path.join(args.output_dir, "train.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    val.to_json(
        os.path.join(args.output_dir, "validation.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    test.to_json(
        os.path.join(args.output_dir, "test.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print(f"New train/val/test splits at '{args.output_dir}'")


if __name__ == "__main__":
    main()
