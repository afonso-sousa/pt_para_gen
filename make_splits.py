import sys
import os

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from transformers import HfArgumentParser
import codecs


@dataclass
class IOArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    data_file: Optional[str] = field(
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    predictions_column: Optional[str] = field(metadata={"help": "The source column."})
    references_column: Optional[str] = field(
        default=None, metadata={"help": "The target column."}
    )
    output_dir: Optional[str] = field(
        default=".", metadata={"help": "Output directory."}
    )


def main():
    from datasets import disable_caching

    disable_caching()

    parser = HfArgumentParser(IOArguments)
    io_args = parser.parse_args_into_dataclasses()[0]

    os.makedirs(os.path.abspath(io_args.output_dir), exist_ok=True)

    df = pd.read_json(
        codecs.open(io_args.data_file, "r", "utf-8"), lines=True, orient="records"
    )

    train = df.sample(frac=0.8, random_state=42)
    aux = df.drop(train.index)
    val = aux.sample(frac=0.5, random_state=42)
    test = aux.drop(val.index)

    train.to_json(
        os.path.join(io_args.output_dir, "train.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    val.to_json(
        os.path.join(io_args.output_dir, "validation.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    test.to_json(
        os.path.join(io_args.output_dir, "test.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print(f"New train/val/test splits at '{io_args.output_dir}'")


if __name__ == "__main__":
    main()
