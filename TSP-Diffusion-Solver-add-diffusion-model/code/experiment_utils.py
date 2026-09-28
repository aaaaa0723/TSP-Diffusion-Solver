"""Shared filesystem locations used by the data-generation scripts."""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CODE_DIR.parent / "results"
DATASET_PATH = RESULTS_DIR / "tsp_dataset_lite.npz"
