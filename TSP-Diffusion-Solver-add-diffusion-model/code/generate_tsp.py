import os
import shutil
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import requests
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from experiment_utils import DATASET_PATH


NUM_SAMPLES = int(os.environ.get("TSP_NUM_SAMPLES", "1000"))
NUM_NODES = int(os.environ.get("TSP_NUM_NODES", "494"))
SOLVE_SECONDS = max(1, int(os.environ.get("TSP_SOLVE_SECONDS", os.environ.get("TSP_SOLVER_TIME_LIMIT_SECONDS", "5"))))
NUM_WORKERS = max(1, min(cpu_count(), int(os.environ.get("TSP_NUM_WORKERS", "8"))))
CHECKPOINT_EVERY = max(1, int(os.environ.get("TSP_CHECKPOINT_EVERY", "10")))
DISTANCE_MODE = os.environ.get("TSP_DISTANCE_MODE", "auto").lower()

LON_MIN, LON_MAX = 120.5500, 120.8500
LAT_MIN, LAT_MAX = 24.0500, 24.3500
OSRM_SESSION = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def init_worker():
    """Reuse one HTTP connection per worker for repeated OSRM requests."""
    global OSRM_SESSION
    OSRM_SESSION = requests.Session()


def get_osrm_distance_matrix(coords):
    """Request an OSRM road-distance matrix for the supplied coordinates."""
    url = "http://localhost:5000/table/v1/driving"
    payload = {
        "coordinates": np.asarray(coords, dtype=float).tolist(),
        "annotations": ["distance"],
    }

    try:
        session = OSRM_SESSION or requests
        response = session.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != "Ok":
            raise ValueError(f"OSRM returned {data.get('code')}: {data}")

        matrix = np.asarray(data["distances"], dtype=np.float32)
        # Keep distances in kilometers to reduce storage while retaining road costs.
        matrix = np.nan_to_num(
            matrix, nan=9999999, posinf=9999999, neginf=9999999
        )
        return (matrix / 1000.0).astype(np.float32)
    except Exception as exc:
        detail = exc
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            detail = f"HTTP {exc.response.status_code}: {exc.response.text[:1000]}"
        raise RuntimeError(
            f"OSRM request failed for {len(coords)} coordinates: {detail}. "
            "Check localhost:5000 and ensure --max-table-size >= NUM_NODES."
        ) from exc


def get_haversine_distance_matrix(coords):
    """Return an aerial-distance matrix in kilometres for lon/lat coordinates."""
    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])
    delta_lon = lon[:, None] - lon[None, :]
    delta_lat = lat[:, None] - lat[None, :]
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(delta_lon / 2) ** 2
    return (6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).astype(np.float32)


def solve_single_tsp(seed):
    rng = np.random.default_rng(seed)
    lons = rng.uniform(LON_MIN, LON_MAX, NUM_NODES)
    lats = rng.uniform(LAT_MIN, LAT_MAX, NUM_NODES)
    coords = np.column_stack((lons, lats)).astype(np.float32)

    if DISTANCE_MODE == "road":
        distance_matrix = get_osrm_distance_matrix(coords)
    else:
        distance_matrix = get_haversine_distance_matrix(coords)
    manager = pywrapcp.RoutingIndexManager(NUM_NODES, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(float(distance_matrix[from_node, to_node]) * 1000.0))

    callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = SOLVE_SECONDS

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError(f"OR-Tools failed to find a route for seed {seed}")

    adjacency = np.zeros((NUM_NODES, NUM_NODES), dtype=np.int8)
    index = routing.Start(0)
    while not routing.IsEnd(index):
        from_node = manager.IndexToNode(index)
        index = solution.Value(routing.NextVar(index))
        to_node = manager.IndexToNode(index)
        adjacency[from_node, to_node] = 1

    return seed, coords, adjacency, distance_matrix


def save_samples(path, results):
    """Atomically save completed samples so an interrupted run can resume."""
    seeds = np.asarray(sorted(results), dtype=np.int32)
    coords = np.stack([results[int(seed)][0] for seed in seeds]).astype(np.float32)
    adjacencies = np.stack([results[int(seed)][1] for seed in seeds]).astype(np.int8)
    distances = np.stack([results[int(seed)][2] for seed in seeds]).astype(np.float32)
    temp_path = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(
        temp_path,
        seeds=seeds,
        coords=coords,
        adjacencies=adjacencies,
        distances=distances,
        num_samples=np.int32(NUM_SAMPLES),
        num_nodes=np.int32(NUM_NODES),
        solve_seconds=np.int32(SOLVE_SECONDS),
    )
    os.replace(temp_path, path)


def load_progress(path):
    if not path.exists():
        return {}

    results = {}
    for checkpoint_path in sorted(path.glob("sample_*.npz")):
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            expected = (NUM_SAMPLES, NUM_NODES, SOLVE_SECONDS)
            actual = (
                int(saved["num_samples"]),
                int(saved["num_nodes"]),
                int(saved["solve_seconds"]),
            )
            if actual != expected:
                raise ValueError(
                    f"Checkpoint settings {actual} do not match {expected}; "
                    f"move or remove {path} before starting a new dataset."
                )
            if "distances" not in saved.files:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} has no OSRM distance matrix; "
                    "remove it and restart generation."
                )
            seed = int(saved["seed"])
            results[seed] = (
                saved["coords"], saved["adjacencies"], saved["distances"]
            )
    return results


def save_progress_sample(path, seed, coords, adjacency, distances):
    path.mkdir(parents=True, exist_ok=True)
    sample_path = path / f"sample_{seed:06d}.npz"
    temp_path = path / f"sample_{seed:06d}.tmp.npz"
    np.savez_compressed(
        temp_path,
        seed=np.int32(seed),
        coords=coords,
        adjacencies=adjacency,
        distances=distances,
        num_samples=np.int32(NUM_SAMPLES),
        num_nodes=np.int32(NUM_NODES),
        solve_seconds=np.int32(SOLVE_SECONDS),
    )
    os.replace(temp_path, sample_path)


def main():
    if NUM_SAMPLES < 1:
        raise ValueError("TSP_NUM_SAMPLES must be at least 1.")
    if NUM_NODES < 3:
        raise ValueError("TSP_NUM_NODES must be at least 3.")
    if DISTANCE_MODE not in {"auto", "road", "haversine"}:
        raise ValueError("TSP_DISTANCE_MODE must be auto, road, or haversine.")

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    progress_path = DATASET_PATH.with_name(
        f"{DATASET_PATH.stem}.progress_{NUM_SAMPLES}x{NUM_NODES}_{SOLVE_SECONDS}s"
    )
    results = load_progress(progress_path)
    pending_seeds = [seed for seed in range(NUM_SAMPLES) if seed not in results]
    workers = min(NUM_WORKERS, max(1, len(pending_seeds)))
    completed_before = len(results)
    run_start = time.time()

    print(
        f"Generating {NUM_SAMPLES} TSP samples ({NUM_NODES} nodes each); "
        f"workers={workers}, solver limit={SOLVE_SECONDS}s."
    )
    if completed_before:
        print(f"Loaded checkpoint: {completed_before}/{NUM_SAMPLES}; resuming.")

    try:
        if pending_seeds:
            with Pool(processes=workers, initializer=init_worker) as pool:
                for seed, coords, adjacency, distances in pool.imap_unordered(
                    solve_single_tsp, pending_seeds, chunksize=1
                ):
                    results[seed] = (coords, adjacency, distances)
                    save_progress_sample(
                        progress_path, seed, coords, adjacency, distances
                    )
                    completed = len(results)
                    run_completed = completed - completed_before

                    if completed % CHECKPOINT_EVERY == 0 or completed == NUM_SAMPLES:
                        elapsed = time.time() - run_start
                        rate = run_completed / elapsed if elapsed else 0.0
                        remaining = NUM_SAMPLES - completed
                        eta_seconds = remaining / rate if rate else 0.0
                        eta_minutes, eta_remainder = divmod(int(eta_seconds), 60)
                        print(
                            f"Progress {completed}/{NUM_SAMPLES} "
                            f"({completed / NUM_SAMPLES:.1%}) | "
                            f"{rate:.2f} samples/s | "
                            f"ETA about {eta_minutes}m {eta_remainder}s | "
                            "checkpoint saved",
                            flush=True,
                        )
    except KeyboardInterrupt:
        if results:
            print(
                f"Interrupted; saved checkpoint with {len(results)}/{NUM_SAMPLES} "
                f"samples at {progress_path}."
            )
        raise
    except Exception:
        if results:
            print(f"Saved checkpoint with {len(results)}/{NUM_SAMPLES} samples.")
        raise

    save_samples(DATASET_PATH, results)
    if progress_path.exists():
        shutil.rmtree(progress_path)

    print(
        f"Dataset complete: {DATASET_PATH}; "
        f"coords={NUM_SAMPLES}x{NUM_NODES}x2, "
        f"adjacencies={NUM_SAMPLES}x{NUM_NODES}x{NUM_NODES}; "
        f"elapsed {time.time() - run_start:.1f}s."
    )


if __name__ == "__main__":
    main()
