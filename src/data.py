from datasets import load_dataset
from transformers import AutoTokenizer

from src.config import DATASET_NAME, MODEL_NAME, MAX_LENGTH


def load_imdb_dataset():
    """
    Load the IMDB dataset from Hugging Face.

    Only the train and test splits are required
    for this project.
    """

    dataset = load_dataset(DATASET_NAME)

    dataset = {
        "train": dataset["train"],
        "test": dataset["test"],
    }

    print("\nDataset loaded successfully.")
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")

    print("\nFeatures:")
    print(dataset["train"].features)

    return dataset


def tokenize_dataset(dataset):
    """
    Tokenize IMDB reviews using DistilBERT tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    tokenized_dataset = {}

    for split in ["train", "test"]:

        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            desc=f"Tokenizing {split} dataset",
        )

        # The raw text is no longer required by the model.
        tokenized_dataset[split] = tokenized_dataset[split].remove_columns(
            ["text"]
        )

    return tokenized_dataset, tokenizer


if __name__ == "__main__":

    dataset = load_imdb_dataset()

    tokenized_dataset, tokenizer = tokenize_dataset(dataset)

    print("\nTokenization completed.")

    print("\nTokenized columns:")
    print(tokenized_dataset["train"].column_names)

    print("\nFirst example:")
    print(tokenized_dataset["train"][0])