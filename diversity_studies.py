import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from accelerate import Accelerator
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
import evaluate
from transformers.trainer_utils import is_main_process
from tqdm.auto import tqdm

from data import DatasetArguments, prepare_dataset
from train import (
    DataTrainingArguments,
    ModelArguments,
    normalize_condition,
    processing_function_wrapper,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationArguments:
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "Number of sequences to generate."},
    )
    beam_width: int = field(
        default=3,
        metadata={"help": "Number of beam groups."},
    )
    num_beam_groups: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beam groups."},
    )
    repetition_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "The parameter for repetition penalty."},
    )
    diversity_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "The parameter for diversity penalty."},
    )
    early_stopping: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to stop the beam search when at least \
                num_beams sentences are finished per batch or not."
            )
        },
    )
    penalty_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "The degeneration penalty for contrastive search."},
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "The size of the candidate set that is used to re-rank for contrastive search."
        },
    )


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DatasetArguments,
            DataTrainingArguments,
            Seq2SeqTrainingArguments,
            GenerationArguments,
        )
    )
    (
        model_args,
        dataset_args,
        data_args,
        training_args,
        generate_args,
    ) = parser.parse_args_into_dataclasses()

    training_args.do_train = False
    training_args.do_evaluate = False
    training_args.do_predict = True
    training_args.predict_with_generate = True

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    accelerator = Accelerator()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    processed_datasets = prepare_dataset(dataset_args, logger)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.config_name is not None:
        model = AutoModelForSeq2SeqLM.from_config(config)
    elif model_args.model_name_or_path is not None:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError("You must specify model_name_or_path or config_name")

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = processed_datasets["validation"].column_names

    # Get the column names for input/target.
    if data_args.source_column is None:
        source_column = column_names[0]
    else:
        source_column = data_args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"--source_column' value '{data_args.source_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.target_column is None:
        target_column = column_names[1]
    else:
        target_column = data_args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"--target_column' value '{data_args.target_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    preprocess_function = processing_function_wrapper(
        data_args,
        source_column,
        target_column,
        prefix,
        tokenizer,
        padding,
        max_target_length,
    )

    # breakpoint()

    input_dataset = processed_datasets.map(
        preprocess_function,
        batched=not (
            data_args.add_graph or data_args.add_dp
        ),  # if graph, we need to process entry by entry,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    dataloader = DataLoader(
        input_dataset["validation"],
        # input_dataset["validation"].select(range(10)),
        collate_fn=data_collator,
        batch_size=training_args.per_device_eval_batch_size,
    )

    logger.info("*** Predict ***")

    metric = evaluate.load("metrics/my_metric")

    model.eval()

    gen_kwargs = {
        "max_length": data_args.val_max_target_length,
        "num_beams": generate_args.num_return_sequences * generate_args.beam_width,
        "num_return_sequences": generate_args.num_return_sequences,
        "num_beam_groups": generate_args.num_beam_groups,
        "repetition_penalty": generate_args.repetition_penalty,
        "diversity_penalty": generate_args.diversity_penalty,  # higher the penalty, the more diverse are the outputs
        "early_stopping": generate_args.early_stopping,
    }
    src_texts, tgt_texts, preds_texts = [], [], []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            generated_tokens, labels = accelerator.gather_for_metrics(
                (generated_tokens, labels)
            )
            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_sources = tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=True
            )
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]
            decoded_sources = [source.strip() for source in decoded_sources]

            # multiply sources and targets according to num_return_sequences
            decoded_sources = list(
                np.repeat(decoded_sources, generate_args.num_return_sequences)
            )
            decoded_labels = list(
                np.repeat(decoded_labels, generate_args.num_return_sequences)
            )

            metric.add_batch(
                sources=decoded_sources,
                predictions=decoded_preds,
                references=decoded_labels,
            )
            src_texts += decoded_sources
            tgt_texts += decoded_labels
            preds_texts += decoded_preds

    result = metric.compute()
    result = {
        name: list(np.around(np.array(scores) * 100, 3))
        for name, scores in result.items()
    }

    # logger.info(result)

    output_prediction_file = os.path.join(
        training_args.output_dir, data_args.output_file_name
    )
    result["source"] = src_texts
    result["target"] = tgt_texts
    result["prediction"] = preds_texts

    df = pd.DataFrame(result)
    df.loc["mean"] = df.mean(numeric_only=True).round(3)
    df.to_csv(output_prediction_file, index=False, sep="\t")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
