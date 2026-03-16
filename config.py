from __future__ import annotations

from pathlib import Path
import torch

# =========================
# Update these three paths on your machine
# =========================
LOL_PATH = r"C:/Users/flore/OneDrive/Documents/CS615 Final Project/all files/Datasets/LOL"
LOLI_PATH = r"C:/Users/flore/OneDrive/Documents/CS615 Final Project/all files/Datasets/LoLI-Street"
EXDARK_PATH = r"C:/Users/flore/OneDrive/Documents/CS615 Final Project/all files/Datasets/ExDark"

# =========================
# General settings
# =========================
IMAGE_SIZE = 256
CLASSIFIER_IMAGE_SIZE = 224
BATCH_SIZE = 8
CLASSIFIER_BATCH_SIZE = 32
EPOCHS_ENHANCEMENT = 20
EPOCHS_CLASSIFIER = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
SEED = 42
AMP = True
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Output locations
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / 'runs'
CHECKPOINT_DIR = RUNS_DIR / 'checkpoints'
RESULTS_DIR = RUNS_DIR / 'results'
SAMPLES_DIR = RUNS_DIR / 'samples'

# Device selection
PREFERRED_DEVICE = 'cuda'
DEVICE = torch.device(PREFERRED_DEVICE if torch.cuda.is_available() and PREFERRED_DEVICE == 'cuda' else 'cpu')

# Optional override for very small GPUs
ENHANCEMENT_BATCH_SIZE_GRAY = 8
ENHANCEMENT_BATCH_SIZE_RGB = 8


def using_cuda() -> bool:
    return DEVICE.type == 'cuda'


def pretty_device_name() -> str:
    if DEVICE.type == 'cuda':
        return torch.cuda.get_device_name(0)
    return str(DEVICE)
