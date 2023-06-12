import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from data import DatasetArguments, prepare_dataset
from train import ModelArguments, DataTrainingArguments, processing_function_wrapper


logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    M2M100Tokenizer,
]


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
    from datasets import disable_caching

    disable_caching()

    parser = HfArgumentParser(
        (
            ModelArguments,
            DatasetArguments,
            DataTrainingArguments,
            Seq2SeqTrainingArguments,
            GenerationArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (
            model_args,
            dataset_args,
            data_args,
            training_args,
            generate_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
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

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.target_lang is not None and data_args.source_lang is not None
        ), (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )

        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token]
            if data_args.forced_bos_token is not None
            else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

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

    input_dataset = processed_datasets.map(
        preprocess_function,
        batched=True,
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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("*** Predict ***")

    # from datasets import Dataset

    predict_results = trainer.predict(
        # Dataset.from_dict(input_dataset["validation"][10:20]),
        input_dataset["validation"],
        max_length=data_args.val_max_target_length,
        penalty_alpha=generate_args.penalty_alpha,  # 0.6,
        top_k=generate_args.top_k,  # 6,
    )

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                predict_results.predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(
                training_args.output_dir, data_args.output_file_name
            )
            result = pd.DataFrame(
                {
                    "source": processed_datasets["validation"][source_column],
                    "target": processed_datasets["validation"][target_column],
                    "prediction": predictions,
                }
            )
            result.to_csv(output_prediction_file, index=False, sep="\t")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
