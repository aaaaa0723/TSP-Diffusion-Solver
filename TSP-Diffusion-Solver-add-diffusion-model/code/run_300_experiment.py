"""Generate 1,000 300-store instances with 30s references, then train/evaluate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "experiment_300nodes_30s"
DATASET = ROOT / "results" / "tsp_dataset_7eleven_300nodes_30s.npz"


def status(phase, **details):
    document = {"phase": phase, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": str(DATASET), **details}
    temp = OUTPUT / "status.tmp.json"
    temp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temp.replace(OUTPUT / "status.json")


def run_stage(command, name, env):
    with (OUTPUT / f"{name}.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(f"{name} failed; inspect {OUTPUT / (name + '.log')}")


def validate_dataset():
    with np.load(DATASET, allow_pickle=False) as data:
        assert int(data["num_samples"]) == 1000 and int(data["num_nodes"]) == 300
        assert int(data["solve_seconds"]) == 30
        assert data["coords"].shape == (1000, 300, 2)
        assert data["distances"].shape == (1000, 300, 300)
        assert np.isfinite(data["distances"]).all()
        assert all(len(set(row)) == 300 for row in data["store_ids"])
        for adjacency in data["adjacencies"]:
            assert np.all(adjacency.sum(axis=0) == 1) and np.all(adjacency.sum(axis=1) == 1)
            node, visited = 0, set()
            for _ in range(300):
                assert node not in visited
                visited.add(node)
                node = int(np.argmax(adjacency[node]))
            assert node == 0 and len(visited) == 300


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime("%Y%m%d_%H%M%S") + "_300nodes_30s"
    run_dir = ROOT / "results" / "runs" / run_name
    env = {**os.environ, "TSP_NUM_SAMPLES": "1000", "TSP_NUM_NODES": "300",
           "TSP_SOLVE_SECONDS": "30", "TSP_NUM_WORKERS": "8",
           "TSP_DATASET_PATH": str(DATASET), "PYTHONIOENCODING": "utf-8"}
    try:
        status("generating", run_dir=str(run_dir))
        if not DATASET.exists():
            run_stage([sys.executable, "-u", "code/generate_tsp.py"], "generate", env)
        status("validating_dataset", run_dir=str(run_dir))
        validate_dataset()
        status("training", run_dir=str(run_dir))
        run_stage([sys.executable, "-u", "code/train.py", "--epochs", "20",
                   "--eval-samples", "0", "--run-dir", str(run_dir), "--no-latest"], "train", env)
        metrics = json.loads((run_dir / "gap_metrics.json").read_text(encoding="utf-8"))
        assert metrics["num_test_cases"] == 150 and metrics["all_routes_valid"]
        (ROOT / "results" / "runs" / "latest.txt").write_text(run_name, encoding="utf-8")
        status("complete", run_dir=str(run_dir), metrics=metrics)
        print(f"Experiment complete: {run_dir}", flush=True)
    except BaseException as exc:
        status("failed", run_dir=str(run_dir), error=str(exc))
        raise


if __name__ == "__main__":
    main()
