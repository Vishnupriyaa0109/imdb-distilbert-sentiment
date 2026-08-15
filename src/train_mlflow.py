from pathlib import Path

import mlflow
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

from zenml import step

from src.config import (
    MODEL_NAME,
    MODEL_DIR,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    SEED,
)


# ============================================================
# MLflow Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MLFLOW_URI = f"sqlite:///{MLFLOW_DB}"

mlflow.set_tracking_uri(
    MLFLOW_URI
)

mlflow.set_experiment(
    "IMDB-DistilBERT-Sentiment"
)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):
    """
    Calculate accuracy, precision, recall, and F1 score.
    """

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
# ZenML Training Step
# ============================================================

@step
def train_model(
    tokenized_dataset,
    train_samples: int = 500,
    eval_samples: int = 200,
) -> str:
    """
    Fine-tune DistilBERT on the selected IMDB training subset.

    MLflow tracks:
        - Hyperparameters
        - Training metrics
        - Evaluation metrics
        - Model metadata

    The trained model is saved locally and the path is
    returned to the next ZenML step.
    """

    print("\n" + "=" * 60)
    print("TRAINING DISTILBERT + MLFLOW")
    print("=" * 60)

    # ========================================================
    # Prepare datasets
    # ========================================================

    train_dataset = (
        tokenized_dataset["train"]
        .shuffle(seed=SEED)
        .select(
            range(train_samples)
        )
    )

    eval_dataset = (
        tokenized_dataset["test"]
        .shuffle(seed=SEED)
        .select(
            range(eval_samples)
        )
    )

    print(
        f"Training examples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Evaluation examples: "
        f"{len(eval_dataset)}"
    )

    # ========================================================
    # Model
    # ========================================================

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

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ========================================================
    # Dynamic Padding
    # ========================================================

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # ========================================================
    # Output Directory
    # ========================================================

    output_dir = (
        MODEL_DIR / "zenml_model"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Training Arguments
    # ========================================================

    training_args = TrainingArguments(
        output_dir=str(
            output_dir
        ),

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,

        warmup_steps=0,

        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",

        save_strategy="no",

        logging_strategy="steps",

        logging_steps=25,

        report_to="none",

        seed=SEED,

        use_cpu=True,
    )

    # ========================================================
    # Trainer
    # ========================================================

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    # ========================================================
    # MLflow Run
    # ========================================================

    with mlflow.start_run(
        run_name="zenml-distilbert-training"
    ) as run:

        print(
            f"\nMLflow Run ID: "
            f"{run.info.run_id}"
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": 0,
                "train_samples": len(
                    train_dataset
                ),
                "eval_samples": len(
                    eval_dataset
                ),
                "max_length": 256,
                "device": "cpu",
                "framework": "HuggingFace Transformers",
                "orchestrator": "ZenML",
            }
        )

        # ----------------------------------------------------
        # Log tags
        # ----------------------------------------------------

        mlflow.set_tags(
            {
                "model_type": "DistilBERT",
                "task": "binary_sentiment_classification",
                "dataset": "IMDB",
                "pipeline": "sentiment_pipeline",
                "training_type": "fine_tuning",
            }
        )

        # ====================================================
        # Train
        # ====================================================

        print(
            "\nStarting model training...\n"
        )

        train_result = trainer.train()

        print(
            "\nTraining completed successfully."
        )

        # ----------------------------------------------------
        # Log training metrics
        # ----------------------------------------------------

        if train_result.metrics:

            train_loss = (
                train_result.metrics.get(
                    "train_loss"
                )
            )

            train_runtime = (
                train_result.metrics.get(
                    "train_runtime"
                )
            )

            train_samples_per_second = (
                train_result.metrics.get(
                    "train_samples_per_second"
                )
            )

            if train_loss is not None:

                mlflow.log_metric(
                    "train_loss",
                    float(train_loss),
                )

            if train_runtime is not None:

                mlflow.log_metric(
                    "train_runtime",
                    float(train_runtime),
                )

            if (
                train_samples_per_second
                is not None
            ):

                mlflow.log_metric(
                    "train_samples_per_second",
                    float(
                        train_samples_per_second
                    ),
                )

        # ====================================================
        # Evaluation
        # ====================================================

        print(
            "\nEvaluating training run...\n"
        )

        evaluation_metrics = (
            trainer.evaluate()
        )

        # ----------------------------------------------------
        # Log evaluation metrics
        # ----------------------------------------------------

        metrics_to_log = {}

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

                metrics_to_log[
                    clean_key
                ] = float(value)

        if metrics_to_log:

            mlflow.log_metrics(
                metrics_to_log
            )

        # ====================================================
        # Save model locally
        # ====================================================

        print(
            "\nSaving trained model..."
        )

        trainer.save_model(
            str(output_dir)
        )

        tokenizer.save_pretrained(
            str(output_dir)
        )

        print(
            f"Model saved to: "
            f"{output_dir}"
        )

        # ====================================================
        # Log lightweight metadata
        # ====================================================
        #
        # IMPORTANT:
        # We intentionally DO NOT use:
        #
        # mlflow.transformers.log_model()
        #
        # because that would create another copy of the
        # ~268 MB model weights.
        #
        # The model already exists in:
        #
        # models/zenml_model
        # ====================================================

        config_file = (
            output_dir / "config.json"
        )

        tokenizer_config_file = (
            output_dir /
            "tokenizer_config.json"
        )

        if config_file.exists():

            mlflow.log_artifact(
                str(config_file),
                artifact_path="model_metadata",
            )

        if tokenizer_config_file.exists():

            mlflow.log_artifact(
                str(tokenizer_config_file),
                artifact_path="model_metadata",
            )

        # ====================================================
        # Print MLflow Results
        # ====================================================

        print("\n" + "=" * 60)
        print("MLFLOW TRAINING RESULTS")
        print("=" * 60)

        print(
            f"Run ID: "
            f"{run.info.run_id}"
        )

        for key, value in (
            metrics_to_log.items()
        ):

            print(
                f"{key}: {value:.4f}"
            )

        print("=" * 60)

    # ========================================================
    # Return Model Path
    # ========================================================

    return str(
        output_dir
    )