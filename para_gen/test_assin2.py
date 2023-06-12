# %%
from datasets import load_dataset

dataset = load_dataset("processed-data/assin2")

# %%
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

model_name = "output/facebook/mbart-large-50-ospt-lr1e-4-standard/checkpoint-7500"
# tapaco_model_name = (
#     "output/facebook/mbart-large-50-tapaco-lr1e-4-standard/checkpoint-912"
# )

model = MBartForConditionalGeneration.from_pretrained(model_name).to("cuda")
tokenizer = MBart50Tokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.src_lang = "pt_XX"
tokenizer.tgt_lang = "pt_XX"


# %%
def to_batches(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


batch_size = 64
inputs = to_batches(dataset["test"]["source"], batch_size)
references = to_batches(dataset["test"]["target"], batch_size)

# %%
predictions = []
for batch in inputs:
    # predictions +=
    encoded_pt = tokenizer(
        batch, padding="max_length", truncation=True, return_tensors="pt"
    ).to("cuda")

    generated_tokens = model.generate(
        **encoded_pt, forced_bos_token_id=tokenizer.lang_code_to_id["pt_XX"]
    )

    gens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    predictions += gens
# %%
