# Process tapaco

# %%
from datasets import load_dataset

tapaco = load_dataset("tapaco")
# %%
pt_set = tapaco["train"].filter(lambda example: example["language"] == "pt")
# %%
pt_set = pt_set.remove_columns(["sentence_id", "lists", "tags", "language"])
# %%
from collections import defaultdict

pt_set_dict = defaultdict(list)
for example in pt_set:
    pt_set_dict[example["paraphrase_set_id"]].append(example["paraphrase"])

# %%
import random

pt_pairs = []
for cluster in pt_set_dict.values():
    random.shuffle(cluster)
    pt_pairs += list(zip(cluster[::2], cluster[1::2]))

# %%
pairs_dict = [dict(zip(["source", "target"], values)) for values in pt_pairs]
# %%
import json

with open("raw-data/tapaco.jsonl", "w", encoding="utf8") as f:
    for entry in pairs_dict:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

# %%
import pandas as pd
import codecs

df = pd.read_json(
    codecs.open("raw-data/tapaco.jsonl", "r", "utf-8"), lines=True, orient="records"
)

# %%
import os

df.to_json(
    os.path.join("processed-data/tapaco", "train.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False,
)

# %%
