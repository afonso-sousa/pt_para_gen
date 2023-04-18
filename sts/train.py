# %%
# TO PRETRAIN UNCOMMENT FOLLOWING CODE
# from sentence_transformers import SentenceTransformer

# model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# model = SentenceTransformer(model_name)

from sentence_transformers import SentenceTransformer, models

model_name = "distilroberta-base"
word_embedding_model = models.Transformer(model_name)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# %%
from datasets import load_dataset

datasets = load_dataset(
    "processed-data/opensubtitles",
    data_files={
        "train": "train.jsonl",
        # "validation": "validation.jsonl",
        # "test": "test.jsonl",
    },
)
# %%
from sentence_transformers import InputExample

train_examples = []
train_data = datasets["train"]
n_examples = train_data.num_rows

for i in range(n_examples):
    example = train_data[i]
    train_examples.append(InputExample(texts=[example["source"], example["target"]]))

# %%
from torch.utils.data import DataLoader
from sentence_transformers import losses

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=128)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
num_epochs = 10
warmup_steps = 100

# %%
from datetime import datetime

model_save_path = (
    "output/training_paraphrases_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    checkpoint_path=model_save_path,
    checkpoint_save_steps=len(train_dataloader),
    checkpoint_save_total_limit=2,
)
