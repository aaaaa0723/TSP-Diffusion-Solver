"""Train the completed route-cost dataset on CUDA and require all five plots."""
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results/tsp_dataset_youbike_300nodes_toll_signals.npz"
OUTPUT = ROOT / "results/training_toll_signals"
PLOTS = ("loss_curve.png", "mean_gap_histogram.png", "best_case_comparison.png",
         "average_case_comparison.png", "worst_case_comparison.png")


def status(phase, **details):
    document = {"phase": phase,
                "updated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "dataset": str(DATASET), "runner_pid": os.getpid(), **details}
    temporary = OUTPUT / "status.tmp.json"
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT / "status.json")


def run_stage(name, command, env, run_dir):
    with (OUTPUT / (name + ".log")).open("a", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith(("epoch ", "test ", "Loading dataset:", "Training ")):
                    status(name, run_dir=str(run_dir), process_pid=process.pid,
                           latest_progress=line.strip(), log=str(OUTPUT / (name + ".log")))
            if process.wait() != 0:
                raise RuntimeError(name + " failed; inspect " + str(OUTPUT / (name + ".log")))


def verify_run(run_dir, expected_cases, epochs):
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "gap_metrics.json").read_text(encoding="utf-8"))
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    if report["status"] != "passed" or not metrics["all_routes_valid"]:
        raise ValueError("Training or route validation failed")
    if metrics["objective_unit"] != "equivalent_km" or metrics["num_test_cases"] != expected_cases:
        raise ValueError("Incorrect evaluation objective or test count")
    if params["device"] != "cuda" or params["num_nodes"] != 300:
        raise ValueError("Training must use CUDA and 300 nodes")
    if len(report["train_loss"]) != epochs or len(report["validation_loss"]) != epochs:
        raise ValueError("Incomplete training history")
    if not np.isfinite(report["train_loss"] + report["validation_loss"]).all():
        raise ValueError("Non-finite training loss")
    for name in PLOTS:
        with Image.open(run_dir / name) as image:
            image.verify()
    if not (run_dir / "difusco_tsp.pth").is_file():
        raise ValueError("Model checkpoint is missing")
    return metrics


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S") + "_cost_300nodes_cuda"
    run_dir = ROOT / "results/runs" / run_name
    smoke_dir = run_dir.with_name(run_name + "_smoke")
    try:
        status("checking", run_dir=str(run_dir))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to train on CPU")
        test = torch.ones((32, 32), device="cuda")
        torch.mm(test, test)
        torch.cuda.synchronize()
        print("GPU: " + torch.cuda.get_device_name(0), flush=True)
        with np.load(DATASET, allow_pickle=False) as data:
            if (str(data["cost_mode"]) != "osm_distance_duration_toll_signals_v3"
                    or int(data["num_samples"]) != 1000 or int(data["num_nodes"]) != 300):
                raise ValueError("Dataset settings do not match the complete route-cost dataset")
        env = {**os.environ, "TSP_DATASET_PATH": str(DATASET), "TSP_OSRM_URL": "http://localhost:5000",
               "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        common = [sys.executable, "-B", "-u", "code/train.py", "--no-latest"]
        status("smoke", run_dir=str(run_dir))
        run_stage("smoke", common + ["--smoke-test", "--samples", "3", "--eval-samples", "1",
                  "--run-dir", str(smoke_dir)], env, smoke_dir)
        verify_run(smoke_dir, 1, 1)
        status("training", run_dir=str(run_dir))
        run_stage("training", common + ["--epochs", "20", "--validation-limit", "0",
                  "--eval-samples", "0", "--run-dir", str(run_dir)], env, run_dir)
        status("verifying_artifacts", run_dir=str(run_dir))
        metrics = verify_run(run_dir, 150, 20)
        (ROOT / "results/runs/latest.txt").write_text(run_name, encoding="utf-8")
        status("complete", run_dir=str(run_dir), gpu=torch.cuda.get_device_name(0),
               plots=list(PLOTS), metrics=metrics)
        print("Training, evaluation and all five plots complete: " + str(run_dir), flush=True)
    except BaseException as exc:
        status("failed", run_dir=str(run_dir), error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
