"""共用路徑與實驗紀錄工具：讓 train/test/plot 的輸出都集中在同一個 run 資料夾。"""
import json
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = (CODE_DIR.parent / "results").resolve()
RUNS_DIR = RESULTS_DIR / "runs"
LATEST_RUN_POINTER = RUNS_DIR / "latest.txt"

DATASET_PATH = RESULTS_DIR / "tsp_dataset_lite.npz"
OPTUNA_DB_PATH = RESULTS_DIR / "optuna_mean_p01_p95.db"


def start_new_run(params: dict) -> Path:
    """建立新的 timestamp run 資料夾，寫入 params.json，並標記為最新 run。"""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    LATEST_RUN_POINTER.write_text(run_id, encoding="utf-8")
    return run_dir


def get_latest_run() -> Path:
    """取得最近一次 train.py 建立的 run 資料夾，供 test/plot 腳本沿用同一份輸出目錄。"""
    if not LATEST_RUN_POINTER.exists():
        raise FileNotFoundError(
            f"找不到任何實驗紀錄，請先執行 train.py 產生 run 資料夾（{LATEST_RUN_POINTER} 不存在）"
        )
    run_id = LATEST_RUN_POINTER.read_text(encoding="utf-8").strip()
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"最新的 run 資料夾不存在：{run_dir}")
    return run_dir
