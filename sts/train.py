import argparse
from datetime import datetime

from datasets import load_dataset
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train STS")
    parser.add_argument(
        "--dataset",
        default="processed-data/ospt",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Name of the Huggingface model to start with.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # model_name = "distilroberta-base"
    # model_name = "bert-base-multilingual-cased"
    # word_embedding_model = models.Transformer(args.model_name)
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model = SentenceTransformer(args.model_name)

    datasets = load_dataset(
        args.dataset,
    )

    train_data = datasets["train"]

    if not all(key in train_data.column_names for key in ["source", "target"]):
        # Specific for ASSIN2
        cols_to_remove = list(
            set(train_data.column_names) - set(["premise", "hypothesis"])
        )
        train_data = train_data.rename_columns(
            {"premise": "source", "hypothesis": "target"}
        )
        train_data = train_data.remove_columns(cols_to_remove)

    train_examples = []

    for i in range(train_data.num_rows):
        example = train_data[i]
        train_examples.append(
            InputExample(texts=[example["source"], example["target"]])
        )

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    num_epochs = args.epochs
    warmup_steps = args.warmup_steps

    model_save_path = f'output/{args.dataset.split("/")[-1]}_{args.model_name.replace("/", "-")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        checkpoint_path=model_save_path,
        checkpoint_save_steps=len(train_dataloader),
        checkpoint_save_total_limit=2,
    )
