# src/config.py
import os
from pathlib import Path

# Project root is one level up from src/
PROJECT_ROOT = Path(__file__).parent.parent

# Core directories
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Optional: You can add more configuration here later
# e.g., RANDOM_STATE = 42, LOG_LEVEL = "INFO", etc.
RANDOM_STATE = 42
TEST_SIZE = 0.2