# %%
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

sample_text = "Rapaces, era como um circo lá fora."

model_name = "output/facebook/m2m100_418M-opensubtitles-lr1e-4-standard/checkpoint-5625"

model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.src_lang = "pt"

# %%
encoded_pt = tokenizer(sample_text, return_tensors="pt")

generated_tokens = model.generate(
    **encoded_pt, forced_bos_token_id=tokenizer.get_lang_id("pt")
)

out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(out[0])
# %%
