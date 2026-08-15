from pathlib import Path
import torch


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


# Create required directories automatically
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Hugging Face Configuration
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

# Official Hugging Face IMDB dataset repository
DATASET_NAME = "stanfordnlp/imdb"


# ============================================================
# Dataset Configuration
# ============================================================

MAX_LENGTH = 256

NUM_LABELS = 2

LABEL_NAMES = {
    0: "negative",
    1: "positive",
}


# ============================================================
# Training Configuration
# ============================================================

LEARNING_RATE = 2e-5

BATCH_SIZE = 8

NUM_EPOCHS = 2

WARMUP_RATIO = 0.1

WEIGHT_DECAY = 0.01

SEED = 42


# ============================================================
# Device Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")