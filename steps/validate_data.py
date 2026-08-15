from datasets import DatasetDict
from zenml import step


@step
def validate_data(dataset: DatasetDict) -> DatasetDict:
    """
    Validate the structure and quality of the IMDB dataset.

    The pipeline fails immediately if any critical validation
    check does not pass.
    """

    print("\n" + "=" * 60)
    print("VALIDATING IMDB DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Required splits
    # --------------------------------------------------------

    required_splits = {"train", "test"}

    missing_splits = required_splits - set(dataset.keys())

    if missing_splits:
        raise ValueError(
            f"Missing required dataset splits: {missing_splits}"
        )

    # --------------------------------------------------------
    # 2. Required columns
    # --------------------------------------------------------

    required_columns = {"text", "label"}

    for split_name in required_splits:

        columns = set(dataset[split_name].column_names)

        missing_columns = required_columns - columns

        if missing_columns:
            raise ValueError(
                f"{split_name} split is missing columns: "
                f"{missing_columns}"
            )

    # --------------------------------------------------------
    # 3. Dataset size
    # --------------------------------------------------------

    train_size = len(dataset["train"])
    test_size = len(dataset["test"])

    if train_size == 0:
        raise ValueError("Training dataset is empty.")

    if test_size == 0:
        raise ValueError("Test dataset is empty.")

    print(f"Training examples: {train_size}")
    print(f"Test examples: {test_size}")

    # --------------------------------------------------------
    # 4. Missing / empty text validation
    # --------------------------------------------------------

    for split_name in required_splits:

        split = dataset[split_name]

        null_text_count = sum(
            text is None
            for text in split["text"]
        )

        empty_text_count = sum(
            isinstance(text, str) and not text.strip()
            for text in split["text"]
        )

        if null_text_count > 0:
            raise ValueError(
                f"{split_name} contains "
                f"{null_text_count} null text values."
            )

        if empty_text_count > 0:
            raise ValueError(
                f"{split_name} contains "
                f"{empty_text_count} empty text values."
            )

    # --------------------------------------------------------
    # 5. Label validation
    # --------------------------------------------------------

    allowed_labels = {0, 1}

    for split_name in required_splits:

        labels = set(dataset[split_name]["label"])

        invalid_labels = labels - allowed_labels

        if invalid_labels:
            raise ValueError(
                f"{split_name} contains invalid labels: "
                f"{invalid_labels}"
            )

        if labels != allowed_labels:
            raise ValueError(
                f"{split_name} must contain both sentiment classes. "
                f"Found labels: {labels}"
            )

    # --------------------------------------------------------
    # 6. Class distribution
    # --------------------------------------------------------

    for split_name in required_splits:

        labels = dataset[split_name]["label"]

        negative_count = labels.count(0)
        positive_count = labels.count(1)

        total = len(labels)

        negative_ratio = negative_count / total
        positive_ratio = positive_count / total

        print(
            f"\n{split_name.upper()} CLASS DISTRIBUTION"
        )

        print(
            f"Negative: {negative_count} "
            f"({negative_ratio:.2%})"
        )

        print(
            f"Positive: {positive_count} "
            f"({positive_ratio:.2%})"
        )

        # Reject extremely imbalanced data.
        if negative_ratio < 0.20 or positive_ratio < 0.20:
            raise ValueError(
                f"{split_name} has severe class imbalance."
            )

    # --------------------------------------------------------
    # Validation successful
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA VALIDATION PASSED")
    print("=" * 60)

    return dataset