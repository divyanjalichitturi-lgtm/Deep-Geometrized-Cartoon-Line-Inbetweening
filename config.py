"""Shared, project-relative defaults for DeepLine AI."""
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset"
IMAGE_SIZE = 256
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 20
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_WEIGHTS = {"l1": 1.0, "mse": 0.15, "edge": 0.35, "ssim": 0.05}

