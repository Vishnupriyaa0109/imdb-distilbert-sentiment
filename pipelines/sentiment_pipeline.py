from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.validate_data import validate_data
from steps.transform_data import transform_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.deploy_model import deploy_model


@pipeline
def sentiment_pipeline(
    train_samples: int = 50,
    eval_samples: int = 20,
    deploy: bool = False,
):
    """
    End-to-end ZenML sentiment analysis pipeline.

    Stages:
        1. Ingest IMDB dataset
        2. Validate dataset
        3. Tokenize dataset
        4. Fine-tune DistilBERT
        5. Evaluate model
        6. Optional deployment
    """

    # ========================================================
    # 1. INGEST
    # ========================================================

    dataset = ingest_data()

    # ========================================================
    # 2. VALIDATE
    # ========================================================

    validated_dataset = validate_data(
        dataset
    )

    # ========================================================
    # 3. TRANSFORM
    # ========================================================

    tokenized_dataset = transform_data(
        validated_dataset
    )

    # ========================================================
    # 4. TRAIN
    # ========================================================

    model_path = train_model(
        tokenized_dataset=tokenized_dataset,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # ========================================================
    # 5. EVALUATE
    # ========================================================

    metrics = evaluate_model(
        model_path=model_path,
        tokenized_dataset=tokenized_dataset,
        eval_samples=eval_samples,
        )

    # ========================================================
    # 6. OPTIONAL DEPLOYMENT
    # ========================================================

    if deploy:
        deploy_model(
            model_path=model_path,
        )

    return metrics


# ============================================================
# LOCAL EXECUTION
# ============================================================

if __name__ == "__main__":

    sentiment_pipeline(
        train_samples=50,
        eval_samples=20,
        deploy=False,
    ) 