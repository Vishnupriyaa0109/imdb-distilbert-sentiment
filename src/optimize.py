from pathlib import Path

import mlflow
import numpy as np
import optuna

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
# Project / MLflow Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MLFLOW_URI = f"sqlite:///{MLFLOW_DB}"

EXPERIMENT_NAME = "IMDB-DistilBERT-Optuna"

mlflow.set_tracking_uri(MLFLOW_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Optimization Configuration
# ============================================================

N_TRIALS = 20

TRAIN_SAMPLES = 500

EVAL_SAMPLES = 200


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):
    """
    Calculate accuracy, precision, recall and F1.
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
# Model
# ============================================================

def build_model():
    """
    Load a fresh DistilBERT model for each Optuna trial.
    """

    return AutoModelForSequenceClassification.from_pretrained(
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


# ============================================================
# Main Optimization
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    print(f"Number of trials: {N_TRIALS}")
    print(f"Training examples per trial: {TRAIN_SAMPLES}")
    print(f"Evaluation examples per trial: {EVAL_SAMPLES}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_imdb_dataset()

    tokenized_dataset, tokenizer = tokenize_dataset(
        dataset
    )

    # --------------------------------------------------------
    # Select shuffled development datasets
    # --------------------------------------------------------

    train_dataset = (
        tokenized_dataset["train"]
        .shuffle(seed=SEED)
        .select(range(TRAIN_SAMPLES))
    )

    eval_dataset = (
        tokenized_dataset["test"]
        .shuffle(seed=SEED)
        .select(range(EVAL_SAMPLES))
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # Optuna objective
    # --------------------------------------------------------

    def objective(trial):

        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-5,
            5e-5,
            log=True,
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            [4, 8, 16],
        )

        num_epochs = trial.suggest_int(
            "num_epochs",
            1,
            3,
        )

        weight_decay = trial.suggest_float(
            "weight_decay",
            0.0,
            0.1,
        )

        warmup_ratio = trial.suggest_float(
            "warmup_ratio",
            0.0,
            0.2,
        )

        print("\n" + "-" * 60)
        print(f"OPTUNA TRIAL {trial.number}")
        print("-" * 60)

        print(
            f"learning_rate: {learning_rate:.8f}"
        )

        print(
            f"batch_size: {batch_size}"
        )

        print(
            f"num_epochs: {num_epochs}"
        )

        print(
            f"weight_decay: {weight_decay:.4f}"
        )

        print(
            f"warmup_ratio: {warmup_ratio:.4f}"
        )

        # ----------------------------------------------------
        # Fresh model for this trial
        # ----------------------------------------------------

        model = build_model()

        # ----------------------------------------------------
        # MLflow nested run
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=f"optuna-trial-{trial.number}",
            nested=True,
        ):

            mlflow.log_params(
                {
                    "trial_number": trial.number,
                    "model_name": MODEL_NAME,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "weight_decay": weight_decay,
                    "warmup_ratio": warmup_ratio,
                    "train_samples": TRAIN_SAMPLES,
                    "eval_samples": EVAL_SAMPLES,
                    "device": "cpu",
                }
            )

            # ------------------------------------------------
            # Training arguments
            # ------------------------------------------------

            output_dir = (
                MODEL_DIR
                / "optuna_trials"
                / f"trial_{trial.number}"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            training_args = TrainingArguments(
                output_dir=str(output_dir),

                learning_rate=learning_rate,

                per_device_train_batch_size=batch_size,

                per_device_eval_batch_size=batch_size,

                num_train_epochs=num_epochs,

                warmup_ratio=warmup_ratio,

                weight_decay=weight_decay,

                eval_strategy="epoch",

                save_strategy="no",

                logging_strategy="steps",

                logging_steps=25,

                report_to="none",

                seed=SEED,

                use_cpu=True,
            )

            # ------------------------------------------------
            # Trainer
            # ------------------------------------------------

            trainer = Trainer(
                model=model,

                args=training_args,

                train_dataset=train_dataset,

                eval_dataset=eval_dataset,

                processing_class=tokenizer,

                data_collator=data_collator,

                compute_metrics=compute_metrics,
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            trainer.train()

            # ------------------------------------------------
            # Evaluate
            # ------------------------------------------------

            evaluation_metrics = trainer.evaluate()

            accuracy = float(
                evaluation_metrics.get(
                    "eval_accuracy",
                    0.0,
                )
            )

            precision = float(
                evaluation_metrics.get(
                    "eval_precision",
                    0.0,
                )
            )

            recall = float(
                evaluation_metrics.get(
                    "eval_recall",
                    0.0,
                )
            )

            f1 = float(
                evaluation_metrics.get(
                    "eval_f1",
                    0.0,
                )
            )

            loss = float(
                evaluation_metrics.get(
                    "eval_loss",
                    0.0,
                )
            )

            # ------------------------------------------------
            # Log metrics to MLflow
            # ------------------------------------------------

            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "loss": loss,
                }
            )

            print(
                f"Trial {trial.number} "
                f"F1: {f1:.4f}"
            )

            print(
                f"Trial {trial.number} "
                f"Accuracy: {accuracy:.4f}"
            )

        # ----------------------------------------------------
        # Optuna objective
        # ----------------------------------------------------

        return f1

    # ========================================================
    # Optuna Study
    # ========================================================

    sampler = optuna.samplers.TPESampler(
        seed=SEED
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0,
    )

    study = optuna.create_study(
        study_name="distilbert_sentiment_optimization",

        direction="maximize",

        sampler=sampler,

        pruner=pruner,
    )

    # --------------------------------------------------------
    # Run optimization
    # --------------------------------------------------------

    study.optimize(
        objective,
        n_trials=N_TRIALS,
    )

    # ========================================================
    # Best Trial
    # ========================================================

    best_trial = study.best_trial

    print("\n" + "=" * 60)
    print("OPTUNA OPTIMIZATION COMPLETED")
    print("=" * 60)

    print(
        f"Best trial: "
        f"{best_trial.number}"
    )

    print(
        f"Best F1: "
        f"{best_trial.value:.4f}"
    )

    print("\nBest hyperparameters:")

    for parameter, value in (
        best_trial.params.items()
    ):

        print(
            f"{parameter}: {value}"
        )

    print("=" * 60)

    # ========================================================
    # Save best parameters
    # ========================================================

    best_params_path = (
        PROJECT_ROOT
        / "models"
        / "best_optuna_params.txt"
    )

    with open(
        best_params_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"best_trial={best_trial.number}\n"
        )

        file.write(
            f"best_f1={best_trial.value}\n"
        )

        for parameter, value in (
            best_trial.params.items()
        ):

            file.write(
                f"{parameter}={value}\n"
            )

    print(
        f"\nBest parameters saved to:"
        f"\n{best_params_path}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()