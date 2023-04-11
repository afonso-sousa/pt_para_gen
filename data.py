from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
import sys
import unicodedata
import re
from functools import partial


def extract_suffix(file_name):
    extension = None
    parts = file_name.split(".")
    if parts[-1] in ["gz"]:
        parts.pop()
    if parts[-1] in ["csv", "json"]:
        extension = parts.pop()
    if parts[-1] == "jsonl":
        extension = "json"
    return extension


nonprintable = (
    ord(c)
    for c in (chr(i) for i in range(sys.maxunicode))
    if "C" in unicodedata.category(c)
)
nonprintable_dict = {character: None for character in nonprintable}


def remove_non_printables(text):
    return text.translate(nonprintable_dict)


def bad_chars_filter(dictionary):
    result = {
        k: remove_non_printables(v) if isinstance(v, str) else v
        for k, v in dictionary.items()
    }
    return result


def remove_whitespaces(example):
    source = example.pop("source")
    target = example.pop("target")
    # remove leading and trailing spaces
    source = source.strip()
    # remove double spaces
    source = re.sub(" +", " ", source)
    target = target.strip()
    target = re.sub(" +", " ", target)
    return {"source": source, "target": target, **example}


def normalize_special_chars(example):
    source = example.pop("source")
    target = example.pop("target")
    output = {"source": source, "target": target}
    for k, v in output.items():
        output[k] = (
            v.replace("``", '"')
            .replace("` `", '"')
            .replace("‘‘", '"')
            .replace("‘ ‘", '"')
            .replace("’’", '"')
            .replace("’ ’", '"')
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("''", '"')
            .replace("' '", '"')
        )
    return {**output, **example}


def clean_data(example, remove_bad_chars):
    if remove_bad_chars:
        example = bad_chars_filter(example)
    example = remove_whitespaces(example)
    example = normalize_special_chars(example)
    return example


@dataclass
class DatasetArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_split: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset split json in the datasets split dictionary format."
        },
    )
    dataset_filter: Optional[str] = field(
        default=None,
        metadata={
            "help": "python string for filtering by dataset fields, for example 'len(sentence) > 3 and sentiment < 0.5'."
        },
    )
    dataset_map: Optional[str] = field(
        default=None,
        metadata={
            "help": "python string for mapping dataset fields, for example 'length = len(sentence); t = 10)'."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory of the dataset cache."}
    )
    dataset_keep_in_memory: Optional[bool] = field(
        default=False, metadata={"help": "Directory of the dataset cache."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    remove_bad_chars: Optional[bool] = field(
        default=False,
        metadata={
            "help": "remove any non printable chars from any string value. In practice remove the 'Other' category from unicodedata library."
        },
    )
    delimiter: Optional[str] = field(
        default=None,
        metadata={"help": "Delimiter for csv-formatted dataset."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = extract_suffix(self.train_file)
                assert (
                    extension is not None
                ), "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = extract_suffix(self.validation_file)
                assert (
                    extension is not None
                ), "`validation_file` should be a csv or a json file."


def prepare_dataset(dataset_args, logger=None):
    if dataset_args.dataset_split is not None:
        try:
            dataset_split = eval(dataset_args.dataset_split)
        except:
            dataset_split = str(dataset_args.dataset_split)
            if logger is not None:
                logger.warning(
                    f"Dataset split name: '{dataset_split}' treated as string. if you want to use json make sure it can be parsed properly."
                )

        if logger is not None:
            logger.info(f"Dataset is splitted by '{dataset_split}'")
    else:
        dataset_split = None

    if dataset_args.dataset_name is not None:
        datasets = load_dataset(
            dataset_args.dataset_name,
            dataset_args.dataset_config_name,
            split=dataset_split,
            keep_in_memory=dataset_args.dataset_keep_in_memory,
            cache_dir=dataset_args.dataset_cache_dir,
        )
    else:
        data_files = {}
        if dataset_args.train_file is not None:
            data_files["train"] = dataset_args.train_file
            extension = extract_suffix(dataset_args.train_file)
        if dataset_args.validation_file is not None:
            data_files["validation"] = dataset_args.validation_file
            extension = extract_suffix(dataset_args.validation_file)
        if dataset_args.test_file is not None:
            data_files["test"] = dataset_args.test_file
            extension = extract_suffix(dataset_args.test_file)

    load_dataset_partial = partial(
        load_dataset,
        extension,
        data_files=data_files,
        split=dataset_split,
        keep_in_memory=dataset_args.dataset_keep_in_memory,
        cache_dir=dataset_args.dataset_cache_dir,
    )
    if extension == "csv":
        datasets = load_dataset_partial(
            sep=dataset_args.delimiter,
        )
    else:
        datasets = load_dataset_partial()

    datasets = datasets.map(
        clean_data,
        fn_kwargs={
            "remove_bad_chars": dataset_args.remove_bad_chars,
        },
    )

    if dataset_args.max_train_samples is not None and "train" in datasets:
        datasets["train"] = datasets["train"].select(
            range(dataset_args.max_train_samples)
        )

    if dataset_args.max_eval_samples is not None and "validation" in datasets:
        datasets["validation"] = datasets["validation"].select(
            range(dataset_args.max_eval_samples)
        )

    if dataset_args.max_predict_samples is not None and "test" in datasets:
        datasets["test"] = datasets["test"].select(
            range(dataset_args.max_predict_samples)
        )

    if dataset_args.dataset_filter:
        datasets = datasets.filter(
            lambda x: eval(dataset_args.dataset_filter, None, x),
            keep_in_memory=dataset_args.dataset_keep_in_memory,
            load_from_cache=False,
        )
        if logger is not None:
            logger.info(f"Dataset was filtered by '{dataset_args.dataset_filter}'")

    if dataset_args.dataset_map:
        datasets = datasets.map(
            lambda x: eval(
                f"""locals() if not exec('{dataset_args.dataset_map}') else None""",
                None,
                x,
            ),
            keep_in_memory=dataset_args.dataset_keep_in_memory,
            load_from_cache_file=False,
        )
        if logger is not None:
            logger.info(f"Dataset was mapped with '{dataset_args.dataset_filter}'")

    return datasets
