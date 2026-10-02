"""One all-Taichung SPX case: 30-second OR-Tools vs an existing checkpoint."""
import argparse
import csv
import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch

import generate_tsp as generation
from difusco_model import DIFUSCOTSP, CategoricalEdgeDiffusion
from evaluate import adjacency_to_tour
from evaluation import calculate_path_distance, divide_and_conquer_decoder
from shopee_taichung import collect

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="20260929_191743")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "shopee_taichung_case")
    parser.add_argument("--dataset", type=Path, help="reuse an existing single-case dataset and reference tour")
    parser.add_argument("--legacy-euclidean", action="store_true",
                        help="old coordinate-distance features and original KMeans/beam/2-opt decoder")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset or output / "dataset.npz"
    if args.dataset and not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    run_dir = ROOT / "results" / "runs" / args.run
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    if not dataset_path.exists():
        catalog = collect()
        stores = catalog["stores"]
        print(f"Aligning all {len(stores)} SPX locations to roads", flush=True)
        with requests.Session() as session:
            for i, store in enumerate(stores):
                lon, lat = store["original_coords"]
                response = session.get(f"{generation.OSRM_URL}/nearest/v1/driving/{lon},{lat}", timeout=30)
                response.raise_for_status()
                data = response.json()
                if data.get("code") != "Ok":
                    raise ValueError(f"Cannot snap {store['id']}: {data}")
                waypoint = data["waypoints"][0]
                store["coords"] = waypoint["location"]
                store["snap_distance_m"] = float(waypoint["distance"])
                if (i + 1) % 50 == 0:
                    print(f"Aligned {i + 1}/{len(stores)}", flush=True)
        # Preserve every distinct store ID, even when road positions coincide.
        (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
        generation.NUM_NODES = len(stores)
        generation.NUM_SAMPLES = 1
        generation.SOLVE_SECONDS = 30
        generation.NODE_SOURCE = "taichung_shopee_spx"
        generation.STORE_CATALOG = catalog
        generation.CATALOG_HASH = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
        print("Requesting road matrix and solving OR-Tools (30 seconds)", flush=True)
        seed, coords, adjacency, distances, indices = generation.solve_single_tsp(42)
        generation.save_samples(dataset_path, {seed: (coords, adjacency, distances, indices)})
    with np.load(dataset_path, allow_pickle=False) as data:
        coords, adjacency, distances = data["coords"][0], data["adjacencies"][0], data["distances"][0]
        ids, names = data["store_ids"][0], data["store_names"][0]
    assert np.isfinite(distances).all()
    assert len(set(ids)) == len(coords)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    model = DIFUSCOTSP(hidden=params["hidden"], layers=params["layers"]).to(device)
    weights = run_dir / "difusco_tsp.pth"
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    torch.manual_seed(42)
    diffusion = CategoricalEdgeDiffusion(steps=params["diffusion_steps"], device=device)
    print(f"Running checkpoint {args.run} on {len(coords)} nodes ({device})", flush=True)
    started = time.perf_counter()
    with torch.no_grad():
        points = torch.from_numpy(coords[None]).to(device)
        if args.legacy_euclidean:
            # Verified against the retained CPython 3.11 model bytecode:
            # center coordinates, then torch.cdist; forward normalizes by mean.
            centered = points - points.mean(dim=1, keepdim=True)
            feature_distances = torch.cdist(centered, centered)
        else:
            feature_distances = torch.from_numpy(distances[None]).to(device)
        heatmap = diffusion.sample(model, points, feature_distances,
                                   inference_steps=params["inference_steps"])[0].cpu().numpy()
    inference_seconds = time.perf_counter() - started
    print(f"Diffusion completed in {inference_seconds:.2f}s; decoding", flush=True)
    started = time.perf_counter()
    if args.legacy_euclidean:
        local_dependencies = ROOT / ".local-tools" / "legacy-deps"
        if importlib.util.find_spec("sklearn") is None and local_dependencies.exists():
            sys.path.insert(0, str(local_dependencies))
        loader = SourceFileLoader("legacy_evaluation", str(ROOT / "code" / "evaluation.py.orig"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        legacy = importlib.util.module_from_spec(spec)
        loader.exec_module(legacy)
        route = legacy.divide_and_conquer_decoder(heatmap, coords=coords)
    else:
        route = divide_and_conquer_decoder(heatmap, distance_matrix=distances,
                                           decoder=params.get("decoder", "road_multistart"),
                                           starts=params.get("decoder_starts", 16),
                                           max_moves=params.get("local_search_moves", 200))
    decode_seconds = time.perf_counter() - started
    reference = adjacency_to_tour(adjacency)
    for tour in (route, reference):
        assert len(tour) == len(coords) + 1 and tour[0] == tour[-1]
        assert set(tour[:-1]) == set(range(len(coords)))
    model_km = float(calculate_path_distance(route, distances))
    reference_km = float(calculate_path_distance(reference, distances))
    report = {"checkpoint": str(weights), "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "num_instances": 1, "num_nodes": len(coords), "ortools_limit_seconds": 30,
              "ortools_km": reference_km, "model_decoder_km": model_km,
              "extra_km": model_km - reference_km, "gap_percent": (model_km / reference_km - 1) * 100,
              "inference_seconds": inference_seconds, "decode_seconds": decode_seconds,
              "device": str(device), "seed": 42, "all_routes_valid": True,
              "dataset": str(dataset_path.resolve()),
              "feature_mode": "legacy_centered_coordinate_cdist" if args.legacy_euclidean else "osrm_driving",
              "decoder": "legacy_kmeans_beam_euclidean_2opt" if args.legacy_euclidean else params.get("decoder", "road_multistart"),
              "scoring_distance": "osrm_driving",
              "model_params": params, "scope": "All Taichung SPX locations listed in saved catalog; no subsampling"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    record = {"index": 0, "coords": coords.tolist(), "route": list(map(int, route)),
              "reference": list(map(int, reference)), "route_length": model_km,
              "reference_length": reference_km, "extra_km": report["extra_km"], "gap_percent": report["gap_percent"]}
    (output / "plot_records.json").write_text(json.dumps([record]), encoding="utf-8")
    for label, tour in (("model", route), ("ortools", reference)):
        with (output / f"{label}_tour.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["visit", "node", "store_id", "name", "longitude", "latitude"])
            for visit, node in enumerate(tour):
                writer.writerow([visit, node, ids[node], names[node], *coords[node]])
    print(json.dumps({k: v for k, v in report.items() if k != "model_params"}, indent=2), flush=True)
    from plot_shopee_case import plot_case
    plot_case(output)


if __name__ == "__main__":
    main()
