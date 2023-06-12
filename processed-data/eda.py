# %%
import pandas as pd
import codecs
from nltk.tokenize import word_tokenize

# %%
data_file = "processed-data/full.jsonl"
df = pd.read_json(codecs.open(data_file, "r", "utf-8"), lines=True, orient="records")

# %%
# it is better to sample as the dataset is big
df = df.sample(n=800_000)

# %%
from matplotlib import pyplot as plt
from tqdm import tqdm

tqdm.pandas()


def get_num_words(sample):
    return len(word_tokenize(sample, language="portuguese"))


df["src_len"] = df["source"].progress_apply(get_num_words)
df["tgt_len"] = df["target"].progress_apply(get_num_words)


plt.hist(
    df["src_len"],
    bins=range(min(df["src_len"]), max(df["src_len"]) + 1, 1),
    alpha=0.4,
    color="red",
    density=True,
)
plt.hist(
    df["tgt_len"],
    bins=range(min(df["tgt_len"]), max(df["tgt_len"]) + 1, 1),
    alpha=0.4,
    color="blue",
    density=True,
)
plt.axvline(df["src_len"].mean(), color="blue", linestyle="dashed", linewidth=1)
plt.axvline(df["tgt_len"].mean(), color="red", linestyle="dashed", linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(
    df["src_len"].mean() * 1.1,
    max_ylim * 0.9,
    "Mean: {:.2f}".format(df["src_len"].mean()),
    color="blue",
)
plt.text(
    df["tgt_len"].mean() * 1.1,
    max_ylim * 0.8,
    "Mean: {:.2f}".format(df["src_len"].mean()),
    color="red",
)
labels = ["source sentences", "target sentences"]
plt.legend(labels)
plt.xlabel("length of sentence")
plt.ylabel("proportion")

print(f"Max source sentence token length: {max(df['src_len'])}")
print(f"Max target sentence token length: {max(df['tgt_len'])}")


# %%
import evaluate
import numpy as np

sbert = evaluate.load(
    "metrics/mSBERT",
)

df["sem_sim"] = sbert.compute(
    predictions=df["target"].tolist(), references=df["source"].tolist()
)["scores"]

hist, bins = np.histogram(df["sem_sim"], bins=100)
bin_centers = (bins[1:] + bins[:-1]) * 0.5
plt.xlabel("semantic similarity")
plt.ylabel("proportion")
plt.plot(bin_centers, hist / len(df["sem_sim"]))

# %%
# normalize
norm_data = (df["sem_sim"] - np.min(df["sem_sim"])) / (
    np.max(df["sem_sim"]) - np.min(df["sem_sim"])
)

# %%
# General statistics from other datasets
from datasets import load_dataset

paws = load_dataset("paws", "labeled_final")
qqp = load_dataset("SetFit/qqp")
tapaco = load_dataset("processed-data/tapaco")

# %%
paws_pos = paws.filter(lambda example: example["label"] == 1)
qqp_pos = qqp.filter(lambda example: example["label"] == 1)

# %%
from tqdm import tqdm
from nltk.tokenize import word_tokenize

tqdm.pandas()


def get_num_words(sample, language):
    return len(word_tokenize(sample, language=language))


paws_df = pd.DataFrame(
    {
        "source": paws_pos["validation"]["sentence1"],
        "target": paws_pos["validation"]["sentence2"],
    }
)
paws_df["src_len"] = paws_df["source"].progress_apply(get_num_words, language="english")
paws_df["tgt_len"] = paws_df["target"].progress_apply(get_num_words, language="english")
paws_df.mean(numeric_only=True).round(3)

# %%
qqp_df = pd.DataFrame(
    {"source": qqp_pos["validation"]["text1"], "target": qqp_pos["validation"]["text2"]}
)
qqp_df["src_len"] = qqp_df["source"].progress_apply(get_num_words, language="english")
qqp_df["tgt_len"] = qqp_df["target"].progress_apply(get_num_words, language="english")
qqp_df.mean(numeric_only=True).round(3)

# %%
tapaco_df = pd.DataFrame(
    {"source": tapaco["train"]["source"], "target": tapaco["train"]["target"]}
)
tapaco_df["src_len"] = tapaco_df["source"].progress_apply(
    get_num_words, language="portuguese"
)
tapaco_df["tgt_len"] = tapaco_df["target"].progress_apply(
    get_num_words, language="portuguese"
)
tapaco_df.mean(numeric_only=True).round(3)

# %%
df["src_len"] = df["source"].progress_apply(get_num_words, language="portuguese")
df["tgt_len"] = df["target"].progress_apply(get_num_words, language="portuguese")
df.mean(numeric_only=True).round(3)

# %%
