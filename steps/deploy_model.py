from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from zenml import step

from src.config import MODEL_DIR


@step
def deploy_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
) -> str:
    """
    Prepare the trained sentiment model for deployment.

    The model and tokenizer are saved as a production-ready
    Hugging Face model directory that will later be consumed
    by the FastAPI inference service.

    Returns:
        Path to the deployment-ready model directory.
    """

    print("\n" + "=" * 60)
    print("PREPARING MODEL FOR DEPLOYMENT")
    print("=" * 60)

    deployment_dir = (
        MODEL_DIR / "deployment"
    )

    deployment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print("\nSaving model...")

    model.save_pretrained(
        str(deployment_dir),
        safe_serialization=True,
    )

    # --------------------------------------------------------
    # Save tokenizer
    # --------------------------------------------------------

    print("Saving tokenizer...")

    tokenizer.save_pretrained(
        str(deployment_dir)
    )

    # --------------------------------------------------------
    # Validate deployment artifact
    # --------------------------------------------------------

    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (deployment_dir / filename).exists()
    ]

    if missing_files:
        raise RuntimeError(
            "Model deployment artifact is incomplete. "
            f"Missing files: {missing_files}"
        )

    print("\nDeployment artifact created successfully.")

    print(
        f"Deployment directory: {deployment_dir}"
    )

    print("\nFiles:")

    for file_path in sorted(
        deployment_dir.iterdir()
    ):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (
                1024 * 1024
            )

            print(
                f"  {file_path.name}: "
                f"{size_mb:.2f} MB"
            )

    print("=" * 60)

    return str(deployment_dir)