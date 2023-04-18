import argparse
import json
import math
import os

import torch
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from optimum.bettertransformer import BetterTransformer


def _get_translation(text, src, tgt):
    tokenizer.src_lang = src
    encoded = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding=True
    ).to(device)
    output = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt],
        max_new_tokens=512,
    )
    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    return decoded


def _to_batches(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset with translation")
    parser.add_argument(
        "--input_dir",
        default="raw-data",
        help="The folder with the raw data.",
    )
    parser.add_argument(
        "--output_path",
        default="raw-data/full_data.jsonl",
        help="The path to save the dataset.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to translate.",
    )
    parser.add_argument(
        "--trim_word_count",
        type=int,
        default=None,
        help="Threshold of minimum number of words from which to discard sentence.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Uses GPU if available

    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model_hf = MBartForConditionalGeneration.from_pretrained(
        model_name, device_map="auto"
    )  # .to(device)
    model = BetterTransformer.transform(model_hf, keep_original_model=True)
    tokenizer = MBart50TokenizerFast.from_pretrained(
        model_name
    )  # loading the tokenizer

    print("Loading data...")
    with open(os.path.join(args.input_dir, "OpenSubtitles.en-pt.en")) as f:
        en_sentences = f.readlines()
    en_sentences = [sent.strip() for sent in en_sentences]

    with open(os.path.join(args.input_dir, "OpenSubtitles.en-pt.pt")) as f:
        pt_sentences = f.readlines()
    pt_sentences = [sent.strip() for sent in pt_sentences]

    pt_sentences = pt_sentences[:-1]  # last line is noise
    en_sentences = en_sentences[:-1]  # last line is noise

    sent_pairs = zip(pt_sentences, en_sentences)

    if args.num_examples:
        from random import sample

        num_examples = args.num_examples  # 1_000_000
        print(f"Randomly sampling {num_examples} entries...")
        sent_pairs = sample(list(sent_pairs), num_examples)

    if args.trim_word_count:
        threshold = args.trim_word_count
        print(f"Removing entries with less than {threshold} words...")
        sent_pairs = [
            (pt, en)
            for pt, en in tqdm(
                sent_pairs, total=len(sent_pairs), desc="Removing small sentences"
            )
            if len(word_tokenize(pt, language="portuguese")) > threshold
        ]

    pt_sentences, en_sentences = list(zip(*sent_pairs))

    batch_size = 32
    pt_batches = _to_batches(pt_sentences, batch_size)
    en_batches = _to_batches(en_sentences, batch_size)

    print("Translating...")
    data = []
    for pt_batch, en_batch in tqdm(
        zip(pt_batches, en_batches),
        total=int(math.ceil(len(en_sentences) / batch_size)),
        desc="Translating",
    ):
        translated_batch = _get_translation(en_batch, "en_XX", "pt_XX")
        new_batch = list(zip(translated_batch, pt_batch))
        new_batch_dict = [
            dict(zip(["source", "target"], values)) for values in new_batch
        ]
        data.extend(new_batch_dict)

    print(f"Saving data to {args.output_path}...")
    with open(args.output_path, "w", encoding="utf8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print("Done!")
