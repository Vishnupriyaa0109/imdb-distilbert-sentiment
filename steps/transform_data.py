from datasets import DatasetDict
from transformers import AutoTokenizer
from zenml import step

from src.config import MODEL_NAME, MAX_LENGTH


@step
def transform_data(
    dataset: DatasetDict,
) -> DatasetDict:
    """
    Tokenize the IMDB dataset using the DistilBERT tokenizer.

    Args:
        dataset: Validated IMDB DatasetDict.

    Returns:
        Tokenized DatasetDict.
    """

    print("\n" + "=" * 60)
    print("TRANSFORMING IMDB DATASET")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        desc="Tokenizing IMDB reviews",
    )

    print("\nTokenization completed.")

    print(
        "Tokenized columns:"
    )

    print(
        tokenized_dataset["train"].column_names
    )

    return tokenized_dataset