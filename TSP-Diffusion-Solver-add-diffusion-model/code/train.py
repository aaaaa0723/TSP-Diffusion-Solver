"""Train the categorical edge-diffusion TSP solver; supports --smoke-test."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from difusco_model import CategoricalEdgeDiffusion, DIFUSCOTSP

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "results" / "tsp_dataset_lite.npz"
RUNS = ROOT / "results" / "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="one training batch plus reverse diffusion and route decoding")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=0, help="limit training data; 0 uses all records")
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--validation-limit", type=int, default=32,
                        help="maximum validation examples used per epoch; 0 evaluates all")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--eval-samples", type=int, default=30,
                        help="held-out examples used for gap metrics and plots; 0 evaluates all")
    parser.add_argument("--evaluation-split", choices=("validation", "test"), default="test",
                        help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-latest", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use-best-params", action="store_true",
                        help="load the best Optuna configuration from results/optuna/best_params.json")
    args = parser.parse_args()

    if args.use_best_params:
        best_path = ROOT / "results" / "optuna" / "best_params.json"
        if not best_path.exists():
            raise FileNotFoundError(f"Run tune.py first; missing {best_path}")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        for name in ("learning_rate", "weight_decay", "hidden", "layers",
                     "diffusion_steps", "inference_steps"):
            if name in best:
                setattr(args, name, best[name])

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}. Run python generate_tsp.py first.")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = np.load(DATASET_PATH)
    coords = torch.from_numpy(arrays["coords"])
    labels = torch.from_numpy(arrays["adjacencies"])
    count = min(len(coords), args.samples) if args.samples else len(coords)
    coords, labels = coords[:count], labels[:count]
    if count < 3:
        raise ValueError("At least three instances are required for train/validation/test splits.")
    if args.smoke_test:
        train_ids, validation_ids, test_ids = [0], [1], [2]
        epochs, reverse_steps = 1, min(4, args.inference_steps)
    else:
        perm = torch.randperm(count, generator=torch.Generator().manual_seed(42))
        train_end = min(max(1, int(count * .7)), count - 2)
        validation_end = min(count - 1, max(train_end + 1, int(count * .85)))
        train_ids = perm[:train_end].tolist()
        validation_ids = perm[train_end:validation_end].tolist()
        test_ids = perm[validation_end:].tolist()
        epochs, reverse_steps = args.epochs, args.inference_steps

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir if args.run_dir else RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if not args.no_latest:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "latest.txt").write_text(run_id, encoding="utf-8")
    params = vars(args).copy()
    params["run_dir"] = str(run_dir) if args.run_dir else None
    params |= {"run_id": run_id, "device": str(device), "dataset_size": count,
                           "train_size": len(train_ids), "validation_size": len(validation_ids),
                           "validation_indices": validation_ids,
                           "test_size": len(test_ids), "test_indices": test_ids,
                           "num_nodes": int(coords.shape[1]),
                           "diffusion_type": "categorical", "edge_representation": "symmetric binary adjacency"}
    (run_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    model = DIFUSCOTSP(hidden=args.hidden, layers=args.layers).to(device)
    diffusion = CategoricalEdgeDiffusion(steps=args.diffusion_steps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    validation_history = []
    best_validation_loss = float("inf")
    best_state = None
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        order = train_ids if args.smoke_test else np.random.permutation(train_ids).tolist()
        for index in order:
            points = coords[index:index + 1].to(device)
            target = torch.maximum(labels[index], labels[index].T).to(device=device, dtype=torch.long).unsqueeze(0)
            noisy, t = diffusion.corrupt(target)
            logits = model(points, noisy, t)
            valid = ~torch.eye(target.shape[-1], device=device, dtype=torch.bool).unsqueeze(0)
            loss = F.cross_entropy(logits.permute(0, 2, 3, 1)[valid], target[valid])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(epoch_losses)))
        model.eval()
        val_ids = validation_ids
        if args.validation_limit > 0:
            val_ids = val_ids[:args.validation_limit]
        val_losses = []
        with torch.no_grad():
            for index in val_ids:
                points = coords[index:index + 1].to(device)
                target = torch.maximum(labels[index], labels[index].T).to(device=device, dtype=torch.long).unsqueeze(0)
                noisy, t = diffusion.corrupt(target)
                logits = model(points, noisy, t)
                valid = ~torch.eye(target.shape[-1], device=device, dtype=torch.bool).unsqueeze(0)
                val_losses.append(float(F.cross_entropy(logits.permute(0, 2, 3, 1)[valid], target[valid]).cpu()))
        val_loss = float(np.mean(val_losses))
        validation_history.append(val_loss)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{epochs} train_loss={history[-1]:.5f} val_loss={val_loss:.5f}")

    save_loss_curve(history, validation_history, run_dir)
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint selected by validation loss.")
    torch.save(best_state, run_dir / "difusco_tsp.pth")
    report = {"status": "trained", "train_loss": history, "validation_loss": validation_history,
              "best_validation_loss": best_validation_loss,
              "elapsed_seconds": round(time.perf_counter() - started, 2)}
    # Import lazily so Optuna's objective and the command-line training path share one evaluator.
    from evaluate import evaluate_checkpoint
    metrics = evaluate_checkpoint(run_dir, sample_limit=args.eval_samples, split=args.evaluation_split)
    report["status"] = "passed"
    report["decoded_test_cases"] = metrics["num_test_cases"]
    report["mean_gap_percent"] = metrics["mean_gap_percent"]
    report["p95_gap_percent"] = metrics["p95_gap_percent"]
    report["mean_extra_km_vs_ortools"] = metrics["mean_extra_km_vs_ortools"]
    report["route_closed"] = metrics["all_routes_valid"]
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved checkpoint and report to {run_dir}")


def save_loss_curve(train_losses, validation_losses, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train loss")
    ax.plot(epochs, validation_losses, marker="s", label="Validation loss")
    ax.set(title="DIFUSCO TSP training loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
