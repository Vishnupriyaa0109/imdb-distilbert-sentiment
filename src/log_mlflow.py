from pathlib import Path

import mlflow


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MODEL_DIR = PROJECT_ROOT / "models" / "zenml_model"


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    "IMDB-DistilBERT-Sentiment"
)


# ============================================================
# BASELINE INFORMATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

LEARNING_RATE = 2e-5

BATCH_SIZE = 8

NUM_EPOCHS = 2

WEIGHT_DECAY = 0.01

MAX_LENGTH = 256

TRAIN_SAMPLES = 500

EVAL_SAMPLES = 25000

DEVICE = "cpu"


# ============================================================
# BASELINE METRICS
# ============================================================

# These are the metrics from the completed full-test evaluation.

ACCURACY = 0.8436

PRECISION = 0.8369

RECALL = 0.8534

F1 = 0.8451


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("MLFLOW - LOG EXISTING DISTILBERT BASELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Verify model exists
    # --------------------------------------------------------

    if not MODEL_DIR.exists():

        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}"
        )

    model_file = MODEL_DIR / "model.safetensors"

    if not model_file.exists():

        raise FileNotFoundError(
            f"Model weights not found: {model_file}"
        )

    print(
        f"\nModel found:\n{MODEL_DIR}"
    )

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="zenml-distilbert-baseline"
    ) as run:

        print(
            f"\nMLflow Run ID: {run.info.run_id}"
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
                "max_length": MAX_LENGTH,
                "train_samples": TRAIN_SAMPLES,
                "eval_samples": EVAL_SAMPLES,
                "device": DEVICE,
                "training_framework": "HuggingFace Transformers",
                "orchestrator": "ZenML",
            }
        )

        # ----------------------------------------------------
        # Log evaluation metrics
        # ----------------------------------------------------

        mlflow.log_metrics(
            {
                "accuracy": ACCURACY,
                "precision": PRECISION,
                "recall": RECALL,
                "f1": F1,
            }
        )

        # ----------------------------------------------------
        # Log model metadata
        #
        # We intentionally do NOT upload model.safetensors.
        # The trained model already exists locally.
        # ----------------------------------------------------

        mlflow.set_tags(
            {
                "model_type": "DistilBERT",
                "task": "binary_sentiment_classification",
                "dataset": "IMDB",
                "pipeline": "ZenML",
                "status": "baseline",
                "model_path": str(MODEL_DIR),
            }
        )

        # ----------------------------------------------------
        # Log lightweight model files as artifacts
        # ----------------------------------------------------
        #
        # These files are tiny compared with model weights.
        #

        lightweight_files = [
            MODEL_DIR / "config.json",
            MODEL_DIR / "tokenizer_config.json",
            MODEL_DIR / "special_tokens_map.json",
        ]

        for file_path in lightweight_files:

            if file_path.exists():

                mlflow.log_artifact(
                    str(file_path),
                    artifact_path="model_metadata",
                )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("MLFLOW BASELINE LOGGED")
        print("=" * 60)

        print(
            f"Run ID: {run.info.run_id}"
        )

        print(
            f"Experiment: IMDB-DistilBERT-Sentiment"
        )

        print(
            f"Accuracy:  {ACCURACY:.4f}"
        )

        print(
            f"Precision: {PRECISION:.4f}"
        )

        print(
            f"Recall:    {RECALL:.4f}"
        )

        print(
            f"F1:        {F1:.4f}"
        )

        print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()