import sys
import os

from dataclasses import dataclass, field
from typing import Optional
from data import DatasetArguments, prepare_dataset
from transformers import HfArgumentParser
import evaluate
import pandas as pd
import numpy as np


@dataclass
class EvalArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    source_column: Optional[str] = field(metadata={"help": "the source column"})
    target_column: Optional[str] = field(metadata={"help": "the target column"})
    predictions_column: Optional[str] = field(
        metadata={"help": "the prediction column"}
    )
    lower_case: bool = field(
        default=False,
        metadata={"help": "Whether to convert source and targets to lowercase."},
    )
    metric_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "metric you wish to apply."}
    )
    output_path: Optional[str] = field(
        default="pairs_evals.csv", metadata={"help": "File to output the results."}
    )


def main():
    import datasets

    datasets.disable_caching()

    parser = HfArgumentParser((DatasetArguments, EvalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        dataset_args, eval_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        dataset_args, eval_args = parser.parse_args_into_dataclasses()

    os.makedirs(os.path.abspath(os.path.dirname(eval_args.output_path)), exist_ok=True)

    dataset_args.csv_delimiter = "\t"
    dataset = prepare_dataset(dataset_args)
    # breakpoint()
    column_names = dataset.column_names

    sources = (
        dataset[column_names[0]]
        if eval_args.source_column is None
        else dataset[eval_args.source_column]
    )
    references = (
        dataset[column_names[1]]
        if eval_args.target_column is None
        else dataset[eval_args.target_column]
    )
    predictions = (
        dataset[column_names[2]]
        if eval_args.predictions_column is None
        else dataset[eval_args.predictions_column]
    )

    if eval_args.lower_case:
        sources = list(map(str.lower, sources))
        references = list(map(str.lower, references))

    my_metric = evaluate.load(eval_args.metric_name_or_path, experiment_id=os.getpid())

    print("Computing metric...")
    result = my_metric.compute(
        sources=sources, predictions=predictions, references=references
    )

    result = [
        print(f"{name}: {round(scores * 100, 3)}") for name, scores in result.items()
    ]


if __name__ == "__main__":
    main()
