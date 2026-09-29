"""Compare decoders with identical seeded heatmaps and untouched reference tours."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from difusco_model import DIFUSCOTSP, CategoricalEdgeDiffusion
from evaluation import (divide_and_conquer_kmeans_beam_decoder,
                        road_multistart_decoder, calculate_path_distance)
from evaluate import adjacency_to_tour, plot_road_cases

ROOT = Path(__file__).resolve().parents[1]


def export_benchmark(run_dir, split):
    """Render the already measured tours without drawing new diffusion samples."""
    import matplotlib.pyplot as plt
    summary = json.loads((run_dir / f"decoder_benchmark_{split}.json").read_text())
    params = json.loads((run_dir / "params.json").read_text())
    with np.load(ROOT / params["dataset_path"], allow_pickle=False) as dataset:
        coords, distances, labels = (dataset[key] for key in ("coords", "distances", "adjacencies"))
        reference_seconds = int(dataset["solve_seconds"]) if "solve_seconds" in dataset.files else None
    records = []
    for record in summary["records"]:
        index = record["index"]
        length = calculate_path_distance(record["route"], distances[index])
        reference = record["reference_length"]
        records.append({"index": index, "coords": coords[index].tolist(), "route": record["route"],
                        "reference": adjacency_to_tour(labels[index]), "route_length": length,
                        "reference_length": reference, "extra_km": length - reference,
                        "gap_percent": record["hybrid_gap_percent"]})
    output = run_dir / "validation" if split == "validation" else run_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics = {"evaluation_split": split, "num_test_cases": len(records),
               "reference_solve_seconds": reference_seconds,
               "decoder": "road_multistart", "decoder_starts": summary["starts"],
               "local_search_moves": summary["max_moves"], "inference_seed": summary["inference_seed"],
               "mean_gap_percent": summary["hybrid_mean_gap_percent"],
               "p95_gap_percent": summary["hybrid_p95_gap_percent"], "all_routes_valid": True,
               "mean_route_length": float(np.mean([r["route_length"] for r in records])),
               "mean_ortools_length": float(np.mean([r["reference_length"] for r in records])),
               "mean_extra_km_vs_ortools": float(np.mean([r["extra_km"] for r in records]))}
    metrics["objective_mean_plus_0_1_p95"] = metrics["mean_gap_percent"] + .1 * metrics["p95_gap_percent"]
    (output / "gap_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps({"status": "evaluated", **metrics,
        "weights_retrained": False, "source_run": params.get("source_run", params["run_id"])}, indent=2), encoding="utf-8")
    (output / "plot_records.json").write_text(json.dumps(records), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([r["gap_percent"] for r in records], bins=20, color="#4C78A8", edgecolor="white")
    ax.axvline(metrics["mean_gap_percent"], color="#E45756", linestyle="--",
               label=f"Mean gap: {metrics['mean_gap_percent']:.2f}%")
    reference_label = f"{reference_seconds}s OR-Tools" if reference_seconds is not None else "OR-Tools"
    ax.set(title=f"DIFUSCO + road search: {split} gap", xlabel=f"Gap vs {reference_label} (%)", ylabel="Instances")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "mean_gap_histogram.png", dpi=160)
    plt.close(fig)
    plot_road_cases(output, records)
    print(f"Exported measured {split} tours and metrics to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--samples", type=int, default=30, help="0 evaluates the entire split")
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--road-only-ablation", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="plot saved benchmark tours without rerunning inference")
    args = parser.parse_args()
    run_dir = ROOT / "results" / "runs" / args.run
    if args.export_only:
        export_benchmark(run_dir, args.split)
        return
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    with np.load(ROOT / params["dataset_path"], allow_pickle=False) as dataset:
        coords, distances, labels = (dataset[key] for key in ("coords", "distances", "adjacencies"))
    model = DIFUSCOTSP(params["hidden"], params["layers"])
    model.load_state_dict(torch.load(run_dir / "difusco_tsp.pth", map_location="cpu", weights_only=True))
    model.eval()
    diffusion = CategoricalEdgeDiffusion(params["diffusion_steps"])
    indices = params[f"{args.split}_indices"]
    if args.samples:
        indices = indices[:args.samples]
    records = []
    for index in indices:
        torch.manual_seed(42 + index)
        began = time.perf_counter()
        heatmap = diffusion.sample(model, torch.from_numpy(coords[index:index + 1]),
                                   torch.from_numpy(distances[index:index + 1]),
                                   params["inference_steps"])[0].numpy()
        legacy = divide_and_conquer_kmeans_beam_decoder(heatmap, distance_matrix=distances[index])
        legacy_seconds = time.perf_counter() - began
        began = time.perf_counter()
        route = road_multistart_decoder(heatmap, distances[index], legacy,
                                       starts=args.starts, max_moves=args.max_moves)
        improvement_seconds = time.perf_counter() - began
        assert len(route) == len(coords[index]) + 1
        assert route[0] == route[-1] and len(set(route[:-1])) == len(coords[index])
        if args.road_only_ablation:
            road_only = road_multistart_decoder(
                np.ones_like(heatmap), distances[index], list(range(len(heatmap))) + [0],
                starts=args.starts, max_moves=args.max_moves)
        # Reference labels enter only here, after both predicted tours are final.
        reference_cost = calculate_path_distance(adjacency_to_tour(labels[index]), distances[index])
        old_cost = calculate_path_distance(legacy, distances[index])
        new_cost = calculate_path_distance(route, distances[index])
        record = {"index": index, "legacy_gap_percent": (old_cost / reference_cost - 1) * 100,
                  "hybrid_gap_percent": (new_cost / reference_cost - 1) * 100,
                  "legacy_seconds": legacy_seconds, "additional_search_seconds": improvement_seconds,
                  "route": route, "legacy_route": legacy, "reference_length": reference_cost}
        records.append(record)
        if args.road_only_ablation:
            record["road_only_gap_percent"] = (
                calculate_path_distance(road_only, distances[index]) / reference_cost - 1) * 100
        print(f"{args.split} {index}: {record['legacy_gap_percent']:.2f}% -> "
              f"{record['hybrid_gap_percent']:.2f}% ({improvement_seconds:.2f}s search)", flush=True)
    summary = {"split": args.split, "num_cases": len(records), "starts": args.starts,
               "max_moves": args.max_moves, "inference_seed": "42 + dataset index",
               "legacy_mean_gap_percent": float(np.mean([r["legacy_gap_percent"] for r in records])),
               "hybrid_mean_gap_percent": float(np.mean([r["hybrid_gap_percent"] for r in records])),
               "hybrid_p95_gap_percent": float(np.percentile([r["hybrid_gap_percent"] for r in records], 95)),
               "mean_additional_search_seconds": float(np.mean([r["additional_search_seconds"] for r in records])),
               "all_routes_valid": True, "records": records}
    if args.road_only_ablation:
        summary["road_only_mean_gap_percent"] = float(np.mean([r["road_only_gap_percent"] for r in records]))
    suffix = "_ablation" if args.road_only_ablation else ""
    output = run_dir / f"decoder_benchmark_{args.split}{suffix}.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
