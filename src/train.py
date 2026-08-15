import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from src.config import (
    MODEL_NAME,
    MODEL_DIR,
    LEARNING_RATE,
    BATCH_SIZE,
    WEIGHT_DECAY,
    SEED,
)

from src.data import (
    load_imdb_dataset,
    tokenize_dataset,
)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):
    """
    Calculate accuracy, precision, recall, and F1 score.
    """

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Model
# ============================================================

def build_model():
    """
    Load pretrained DistilBERT for binary sentiment classification.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "negative",
            1: "positive",
        },
        label2id={
            "negative": 0,
            "positive": 1,
        },
    )

    return model


# ============================================================
# Main Training Pipeline
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS - DISTILBERT")
    print("=" * 60)

    # --------------------------------------------------------
    # Load IMDB dataset
    # --------------------------------------------------------

    dataset = load_imdb_dataset()

    # --------------------------------------------------------
    # Tokenize dataset
    # --------------------------------------------------------

    tokenized_dataset, tokenizer = tokenize_dataset(dataset)

    # --------------------------------------------------------
    # Smoke Test Dataset
    # --------------------------------------------------------
    # The IMDB dataset is ordered by sentiment.
    # Therefore, shuffle before selecting a small subset.
    #
    # We use a small subset because the current machine
    # is CPU-only.
    # --------------------------------------------------------

    train_dataset = (
        tokenized_dataset["train"]
        .shuffle(seed=SEED)
        .select(range(500))
    )

    eval_dataset = (
        tokenized_dataset["test"]
        .shuffle(seed=SEED)
        .select(range(200))
    )

    print(
        f"\nSmoke-test training examples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Smoke-test evaluation examples: "
        f"{len(eval_dataset)}"
    )

    # --------------------------------------------------------
    # Build DistilBERT model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Dynamic Padding
    # --------------------------------------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # Training Arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(
            MODEL_DIR / "smoke_test"
        ),

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=1,

        # Warmup is disabled for this smoke test.
        # The proper warmup-ratio calculation will be
        # implemented in the Optuna training pipeline.
        warmup_steps=0,

        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",

        save_strategy="no",

        logging_strategy="steps",

        logging_steps=10,

        report_to="none",

        seed=SEED,

        # Current machine is CPU-only.
        use_cpu=True,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\nStarting smoke-test training...\n"
    )

    trainer.train()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating model...\n"
    )

    metrics = trainer.evaluate()

    # --------------------------------------------------------
    # Display Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)

    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()