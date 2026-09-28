"""Evaluate a trained diffusion model and plot its route gaps against OR-Tools."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from difusco_model import CategoricalEdgeDiffusion, DIFUSCOTSP
from evaluation import calculate_path_distance, divide_and_conquer_decoder

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "results" / "tsp_dataset_lite.npz"
RUNS = ROOT / "results" / "runs"


def split_indices(count):
    if count < 3:
        raise ValueError("Evaluation requires at least three dataset instances")
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(42)).tolist()
    train_end = min(max(1, int(count * .7)), count - 2)
    validation_end = min(count - 1, max(train_end + 1, int(count * .85)))
    return permutation[:train_end], permutation[train_end:validation_end], permutation[validation_end:]


def adjacency_to_tour(adjacency):
    n = len(adjacency)
    path = [0]
    visited = {0}
    current = 0
    for _ in range(n - 1):
        next_nodes = np.flatnonzero(adjacency[current] > 0)
        next_nodes = [int(node) for node in next_nodes if int(node) not in visited]
        if not next_nodes:
            raise ValueError("OR-Tools reference adjacency is not a single Hamiltonian cycle")
        current = next_nodes[0]
        path.append(current)
        visited.add(current)
    path.append(0)
    return path


def evaluate_checkpoint(run_dir, sample_limit=30, split="test"):
    run_dir = Path(run_dir)
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    dataset = np.load(DATASET_PATH)
    coords = dataset["coords"][:params["dataset_size"]]
    labels = dataset["adjacencies"][:params["dataset_size"]]
    train_ids, validation_ids, test_ids = split_indices(len(coords))
    if split == "validation":
        test_ids = params.get("validation_indices", validation_ids)
    else:
        test_ids = params.get("test_indices", test_ids)
    if sample_limit > 0:
        test_ids = test_ids[:sample_limit]
    if not test_ids:
        raise ValueError("No held-out test examples available")

    device = torch.device(params.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = DIFUSCOTSP(hidden=params["hidden"], layers=params["layers"]).to(device)
    checkpoint = torch.load(run_dir / "difusco_tsp.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    diffusion = CategoricalEdgeDiffusion(steps=params["diffusion_steps"], device=device)
    reverse_steps = params["inference_steps"]
    records = []

    for index in test_ids:
        points = torch.from_numpy(coords[index:index + 1]).to(device)
        with torch.no_grad():
            heatmap = diffusion.sample(model, points, inference_steps=reverse_steps)[0].cpu().numpy()
        route = divide_and_conquer_decoder(heatmap, coords=coords[index])
        if len(set(route[:-1])) != len(coords[index]) or route[0] != route[-1]:
            raise RuntimeError(f"Decoder returned an invalid tour for instance {index}")
        reference = adjacency_to_tour(labels[index])
        predicted_length = float(calculate_path_distance(route, coords[index]))
        reference_length = float(calculate_path_distance(reference, coords[index]))
        extra_km = predicted_length - reference_length
        records.append({"index": int(index), "coords": coords[index], "route": route,
                        "reference": reference, "route_length": predicted_length,
                        "reference_length": reference_length,
                        "extra_km": extra_km,
                        "gap_percent": (predicted_length - reference_length) / reference_length * 100})
        print(f"test {index}: DIFUSCO={predicted_length:.3f} km, OR-Tools={reference_length:.3f} km, "
              f"difference={extra_km:+.3f} km, gap={records[-1]['gap_percent']:.2f}%")

    gaps = np.array([item["gap_percent"] for item in records])
    mean_gap = float(gaps.mean())
    p95_gap = float(np.percentile(gaps, 95))
    mean_extra_km = float(np.mean([item["extra_km"] for item in records]))
    metrics = {"evaluation_split": split, "num_test_cases": len(records), "mean_gap_percent": mean_gap,
               "p95_gap_percent": p95_gap,
               "all_routes_valid": True,
               "mean_route_length": float(np.mean([x["route_length"] for x in records])),
               "mean_ortools_length": float(np.mean([x["reference_length"] for x in records])),
               "mean_extra_km_vs_ortools": mean_extra_km,
               "objective_mean_plus_0_1_p95": mean_gap + 0.1 * p95_gap}
    (run_dir / "gap_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path = run_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update({"mean_gap_percent": mean_gap, "p95_gap_percent": p95_gap,
                       "mean_extra_km_vs_ortools": mean_extra_km,
                       "decoded_test_cases": len(records), "route_closed": True})
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gaps, bins=min(20, max(5, len(gaps))), color="#4C78A8", edgecolor="white")
    ax.axvline(mean_gap, color="#E45756", linestyle="--", linewidth=2,
               label=f"Mean gap: {mean_gap:.2f}% | mean extra: {mean_extra_km:+.2f} km")
    ax.set(title="DIFUSCO vs OR-Tools: test route gap", xlabel="Route gap (%)", ylabel="Test instances")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(run_dir / "mean_gap_histogram.png", dpi=160)
    plt.close(fig)

    cases = [("best", int(np.argmin(gaps))), ("average", int(np.argmin(np.abs(gaps - mean_gap)))),
             ("worst", int(np.argmax(gaps)))]
    for label, record_idx in cases:
        item = records[record_idx]
        fig, ax = plt.subplots(figsize=(8, 7))
        points = item["coords"]
        ax.scatter(points[:, 0], points[:, 1], s=12, color="#333333", zorder=3)
        for key, color, style in (("reference", "#999999", "--"), ("route", "#E45756", "-")):
            path = item[key]
            line = points[np.asarray(path)]
            name = "OR-Tools" if key == "reference" else "DIFUSCO"
            length = item["reference_length"] if key == "reference" else item["route_length"]
            ax.plot(line[:, 0], line[:, 1], color=color, linestyle=style, linewidth=1.2,
                    alpha=.85, label=f"{name}: {length:.2f} km")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{label.title()} case (instance {item['index']}): "
                     f"gap {item['gap_percent']:.2f}%, extra {item['extra_km']:+.2f} km")
        ax.legend()
        ax.grid(alpha=.15)
        fig.tight_layout()
        fig.savefig(run_dir / f"{label}_case_comparison.png", dpi=160)
        plt.close(fig)

    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest", help="run ID or 'latest'")
    parser.add_argument("--samples", type=int, default=30, help="held-out cases; 0 evaluates all")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    run_id = (RUNS / "latest.txt").read_text(encoding="utf-8").strip() if args.run == "latest" else args.run
    evaluate_checkpoint(RUNS / run_id, sample_limit=args.samples, split=args.split)


if __name__ == "__main__":
    main()
