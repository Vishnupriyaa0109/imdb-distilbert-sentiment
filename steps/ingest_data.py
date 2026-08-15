from typing import Tuple

from datasets import Dataset, DatasetDict, load_dataset
from zenml import step


@step
def ingest_data() -> DatasetDict:
    """
    Load the IMDB sentiment dataset from Hugging Face.

    Returns:
        DatasetDict containing train, test, and unsupervised splits.
    """

    dataset = load_dataset("stanfordnlp/imdb")

    if "train" not in dataset:
        raise ValueError("IMDB dataset is missing the train split.")

    if "test" not in dataset:
        raise ValueError("IMDB dataset is missing the test split.")

    print("IMDB dataset loaded successfully.")
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")

    return dataset