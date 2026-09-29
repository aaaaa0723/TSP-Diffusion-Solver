"""Tune training parameters with Optuna using held-out mean route gap."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent
TRAIN = CODE / "train.py"
OUTPUT = ROOT / "results" / "optuna"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100,
                        help="dataset instances used for tuning (not nodes per instance)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--validation-limit", type=int, default=32,
                        help="validation instances per epoch; 0 uses the full validation set")
    parser.add_argument("--eval-samples", type=int, default=30,
                        help="validation instances used for route gaps; 0 uses the full validation set")
    args = parser.parse_args()
    if args.trials < 1 or args.samples < 3 or args.epochs < 1:
        parser.error("trials and epochs must be positive; samples must be at least three")
    if args.validation_limit < 0 or args.eval_samples < 0:
        parser.error("validation-limit and eval-samples must be nonnegative")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        config = {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 8e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "hidden": trial.suggest_categorical("hidden", [8, 16]),
            "layers": trial.suggest_int("layers", 1, 2),
            "diffusion_steps": trial.suggest_categorical("diffusion_steps", [16, 32]),
            "inference_steps": trial.suggest_categorical("inference_steps", [4, 8]),
        }
        with tempfile.TemporaryDirectory(prefix=f"tsp_diffusion_trial_{trial.number}_") as temp:
            run_dir = Path(temp) / "run"
            command = [sys.executable, str(TRAIN), "--samples", str(args.samples),
                       "--epochs", str(args.epochs), "--validation-limit", str(args.validation_limit),
                       "--eval-samples", str(args.eval_samples), "--evaluation-split", "validation",
                       "--run-dir", str(run_dir), "--no-latest", "--no-eval-plots"]
            for key, value in config.items():
                command.extend(("--" + key.replace("_", "-"), str(value)))
            result = subprocess.run(command, cwd=CODE, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"Trial {trial.number} failed.\n{result.stdout}\n{result.stderr}")
            metrics = json.loads((run_dir / "validation" / "gap_metrics.json").read_text(encoding="utf-8"))
            objective_value = metrics["objective_mean_plus_0_1_p95"]
            trial.set_user_attr("mean_gap_percent", metrics["mean_gap_percent"])
            trial.set_user_attr("p95_gap_percent", metrics["p95_gap_percent"])
            print(f"trial {trial.number}: objective={objective_value:.3f}, "
                  f"mean_gap={metrics['mean_gap_percent']:.2f}%, params={config}")
            return objective_value

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials)
    best = {**study.best_params,
            "objective_mean_plus_0_1_p95": study.best_value,
            "mean_gap_percent": study.best_trial.user_attrs.get("mean_gap_percent"),
            "p95_gap_percent": study.best_trial.user_attrs.get("p95_gap_percent"),
            "tuning_samples": args.samples, "tuning_epochs": args.epochs,
            "validation_limit": args.validation_limit, "eval_samples": args.eval_samples,
            "trials": args.trials}
    target = OUTPUT / "best_params.json"
    target.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Best tuning result saved to {target}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
