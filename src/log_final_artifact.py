from pathlib import Path

import mlflow


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "final_model"

BEST_PARAMS_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_optuna_params.txt"
)

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


# ============================================================
# MLflow configuration
# ============================================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

mlflow.set_experiment(
    "IMDB-DistilBERT-Sentiment"
)


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("LOGGING FINAL MODEL TO MLFLOW")
    print("=" * 60)

    # --------------------------------------------------------
    # Verify model exists
    # --------------------------------------------------------

    if not MODEL_DIR.exists():

        raise FileNotFoundError(
            f"Final model not found:\n{MODEL_DIR}"
        )

    print(
        f"\nModel directory:\n{MODEL_DIR}"
    )

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="final-optimized-model-artifact"
    ) as run:

        print(
            f"\nMLflow Run ID: "
            f"{run.info.run_id}"
        )

        # ----------------------------------------------------
        # Log final model metrics
        # ----------------------------------------------------

        mlflow.log_metrics({

            "accuracy": 0.9104,

            "precision": 0.9097444089456869,

            "recall": 0.9112,

            "f1": 0.9104716227018386,

        })

        # ----------------------------------------------------
        # Log Optuna parameters
        # ----------------------------------------------------

        mlflow.log_params({

            "model_name": "distilbert-base-uncased",

            "learning_rate":
                1.827226177606625e-05,

            "batch_size": 4,

            "num_epochs": 1,

            "weight_decay":
                0.015599452033620266,

            "warmup_ratio":
                0.011616722433639893,

            "optimization": "Optuna",

            "optuna_best_trial": 0,

        })

        # ----------------------------------------------------
        # Log model directory as MLflow artifact
        # ----------------------------------------------------

        mlflow.log_artifacts(
            str(MODEL_DIR),
            artifact_path="final_model",
        )

        # ----------------------------------------------------
        # Log Optuna parameter file
        # ----------------------------------------------------

        if BEST_PARAMS_FILE.exists():

            mlflow.log_artifact(
                str(BEST_PARAMS_FILE),
                artifact_path="optuna",
            )

        # ----------------------------------------------------
        # Completion
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("FINAL MODEL LOGGED TO MLFLOW")
        print("=" * 60)

        print(
            f"Run ID: {run.info.run_id}"
        )

        print(
            "Artifact: final_model/"
        )

        print(
            "\nMetrics:"
        )

        print(
            "Accuracy : 0.9104"
        )

        print(
            "Precision: 0.9097"
        )

        print(
            "Recall   : 0.9112"
        )

        print(
            "F1       : 0.9105"
        )

        print("=" * 60)


if __name__ == "__main__":
    main()