"""Shared filesystem locations used by the data-generation scripts."""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CODE_DIR.parent / "results"
DATASET_PATH = Path(os.environ.get("TSP_DATASET_PATH", str(RESULTS_DIR / "tsp_dataset_7eleven_300nodes_30s.npz"))).resolve()
DATA_DIR = CODE_DIR.parent / "data"
TAICHUNG_BOUNDARY_PATH = DATA_DIR / "taichung_city_boundary.geojson"
