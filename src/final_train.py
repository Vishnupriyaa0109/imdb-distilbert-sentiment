from pathlib import Path

import mlflow
import mlflow.transformers
import numpy as np

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

from src.config import (
    MODEL_NAME,
    MODEL_DIR,
    SEED,
)

from src.data import (
    load_imdb_dataset,
    tokenize_dataset,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

mlflow.set_experiment(
    "IMDB-DistilBERT-Sentiment"
)


# ============================================================
# Optuna-selected hyperparameters
# ============================================================

LEARNING_RATE = 1.827226177606625e-05
BATCH_SIZE = 4
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.015599452033620266
WARMUP_RATIO = 0.011616722433639893


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("FINAL OPTIMIZED DISTILBERT TRAINING")
    print("=" * 60)

    print("\nOptuna-selected parameters:")
    print(f"Learning rate : {LEARNING_RATE}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Epochs        : {NUM_EPOCHS}")
    print(f"Weight decay  : {WEIGHT_DECAY}")
    print(f"Warmup ratio  : {WARMUP_RATIO}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_imdb_dataset()

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    tokenized_dataset, tokenizer = tokenize_dataset(
        dataset
    )

    # --------------------------------------------------------
    # Final training data
    # --------------------------------------------------------
    #
    # Use the complete IMDB training split.
    #
    # Evaluation uses the complete IMDB test split.
    # --------------------------------------------------------

    train_dataset = tokenized_dataset["train"]

    eval_dataset = tokenized_dataset["test"]

    print(
        f"\nTraining examples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Evaluation examples: "
        f"{len(eval_dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
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
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        MODEL_DIR / "final_model"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dynamic padding
    # --------------------------------------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=str(
            output_dir
        ),

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,

        warmup_ratio=WARMUP_RATIO,

        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",

        save_strategy="no",

        logging_strategy="steps",

        logging_steps=100,

        report_to="none",

        seed=SEED,

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
    # MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="final-optimized-model"
    ) as run:

        print(
            f"\nMLflow Run ID: "
            f"{run.info.run_id}"
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params({

            "model_name": MODEL_NAME,

            "learning_rate": LEARNING_RATE,

            "batch_size": BATCH_SIZE,

            "num_epochs": NUM_EPOCHS,

            "weight_decay": WEIGHT_DECAY,

            "warmup_ratio": WARMUP_RATIO,

            "train_samples": len(
                train_dataset
            ),

            "eval_samples": len(
                eval_dataset
            ),

            "optimized_by": "Optuna",

            "optuna_best_trial": 0,

            "optuna_best_f1": 0.8260869565217391,

            "device": "cpu",
        })

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print(
            "\nStarting final training...\n"
        )

        train_result = trainer.train()

        # ----------------------------------------------------
        # Log training metrics
        # ----------------------------------------------------

        if train_result.metrics:

            for key, value in (
                train_result.metrics.items()
            ):

                if isinstance(
                    value,
                    (int, float),
                ):

                    mlflow.log_metric(
                        key,
                        float(value),
                    )

        # ----------------------------------------------------
        # Final evaluation
        # ----------------------------------------------------

        print(
            "\nEvaluating final model...\n"
        )

        evaluation_metrics = (
            trainer.evaluate()
        )

        # ----------------------------------------------------
        # Log evaluation metrics
        # ----------------------------------------------------

        final_metrics = {}

        for key, value in (
            evaluation_metrics.items()
        ):

            if isinstance(
                value,
                (int, float),
            ):

                clean_key = key.replace(
                    "eval_",
                    "",
                )

                final_metrics[
                    clean_key
                ] = float(value)

        mlflow.log_metrics(
            final_metrics
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        trainer.save_model(
            str(output_dir)
        )

        tokenizer.save_pretrained(
            str(output_dir)
        )

        print(
            f"\nFinal model saved to:\n"
            f"{output_dir}"
        )

        # ----------------------------------------------------
        # Log model artifact to MLflow
        # ----------------------------------------------------

        mlflow.transformers.log_model(
            transformers_model={
                "model": trainer.model,
                "tokenizer": tokenizer,
            },
            name="sentiment_model",
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("FINAL MODEL RESULTS")
        print("=" * 60)

        print(
            f"MLflow Run ID: "
            f"{run.info.run_id}"
        )

        for key, value in (
            final_metrics.items()
        ):

            print(
                f"{key}: {value:.4f}"
            )

        print("=" * 60)


if __name__ == "__main__":
    main()
    