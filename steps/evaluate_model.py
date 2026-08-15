import numpy as np

from datasets import DatasetDict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from zenml import step


@step(enable_cache=False)
def evaluate_model(
    model_path: str,
    tokenized_dataset: DatasetDict,
    eval_samples: int = 200,
) -> dict:
    """
    Evaluate the fine-tuned DistilBERT model.

    During development, only eval_samples examples are used
    to keep CPU execution practical.
    """

    print("\n" + "=" * 60)
    print("EVALUATING DISTILBERT")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    # --------------------------------------------------------
    # Select evaluation samples
    # --------------------------------------------------------

    full_test_dataset = tokenized_dataset["test"]

    sample_count = min(
        eval_samples,
        len(full_test_dataset),
    )

    test_dataset = (
        full_test_dataset
        .shuffle(seed=42)
        .select(range(sample_count))
    )

    print(
        f"Evaluation examples: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Dynamic padding
    # --------------------------------------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # Evaluation arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir="./models/evaluation",

        per_device_eval_batch_size=8,

        report_to="none",

        use_cpu=True,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,

        args=training_args,

        eval_dataset=test_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...\n")

    predictions = trainer.predict(
        test_dataset
    )

    logits = predictions.predictions
    labels = predictions.label_ids

    predicted_labels = np.argmax(
        logits,
        axis=-1,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predicted_labels,
            average="binary",
            zero_division=0,
        )
    )

    accuracy = accuracy_score(
        labels,
        predicted_labels,
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nEvaluation Results")
    print("-" * 40)

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    print("=" * 60)

    return metrics