import os
import json
import hashlib
import shutil
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import requests
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from experiment_utils import DATASET_PATH, TAICHUNG_BOUNDARY_PATH
from seven_eleven import prepare_stores


NUM_SAMPLES = int(os.environ.get("TSP_NUM_SAMPLES", "1000"))
NUM_NODES = int(os.environ.get("TSP_NUM_NODES", "500"))
SOLVE_SECONDS = max(1, int(os.environ.get("TSP_SOLVE_SECONDS", os.environ.get("TSP_SOLVER_TIME_LIMIT_SECONDS", "30"))))
NUM_WORKERS = max(1, min(cpu_count(), int(os.environ.get("TSP_NUM_WORKERS", "8"))))
CHECKPOINT_EVERY = max(1, int(os.environ.get("TSP_CHECKPOINT_EVERY", "10")))
OSRM_URL = os.environ.get("TSP_OSRM_URL", "http://localhost:5000").rstrip("/")
BOUNDARY_URL = os.environ.get(
    "TSP_TAICHUNG_BOUNDARY_URL",
    "https://nominatim.openstreetmap.org/search?format=geojson&polygon_geojson=1&limit=5&q=Taichung%2C%20Taiwan",
)
OSRM_SESSION = None
STORE_CATALOG = None
CATALOG_HASH = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def init_worker(catalog, catalog_hash):
    """Share a fixed store pool across Windows/Linux workers."""
    global OSRM_SESSION, STORE_CATALOG, CATALOG_HASH
    OSRM_SESSION = requests.Session()
    STORE_CATALOG = catalog
    CATALOG_HASH = catalog_hash


class UnreachableStoresError(ValueError):
    pass


def get_osrm_distance_matrix(coords):
    """Request an OSRM road-distance matrix for the supplied coordinates."""
    url = f"{OSRM_URL}/table/v1/driving"
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
        if matrix.shape != (len(coords), len(coords)):
            raise ValueError("OSRM returned an invalid matrix shape")
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise UnreachableStoresError("Some selected stores have no driving route between them")
        return (matrix / 1000.0).astype(np.float32)
    except UnreachableStoresError:
        raise
    except Exception as exc:
        detail = exc
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            detail = f"HTTP {exc.response.status_code}: {exc.response.text[:1000]}"
        raise RuntimeError(
            f"OSRM request failed for {len(coords)} coordinates: {detail}. "
            f"Check {OSRM_URL} and ensure --max-table-size >= NUM_NODES."
        ) from exc


def _extract_taichung_geometry(feature_collection):
    """Return the city relation geometry from a Nominatim response."""
    for feature in feature_collection.get("features", []):
        properties = feature.get("properties", {})
        if (properties.get("osm_type") == "relation" and
                properties.get("osm_id") == 2921154 and
                properties.get("name") in {"臺中市", "台中市"}):
            return feature["geometry"]
    raise ValueError("The boundary response did not contain OSM relation 2921154 (臺中市).")


def load_taichung_boundary():
    """Load cached WGS84 city boundary, downloading it once when absent."""
    if TAICHUNG_BOUNDARY_PATH.exists():
        document = json.loads(TAICHUNG_BOUNDARY_PATH.read_text(encoding="utf-8"))
    else:
        response = requests.get(
            BOUNDARY_URL,
            headers={"User-Agent": "TSP-Diffusion-Solver/1.0 (educational project)"},
            timeout=60,
        )
        response.raise_for_status()
        geometry = _extract_taichung_geometry(response.json())
        document = {"type": "Feature", "properties": {
            "name": "臺中市", "source": "OpenStreetMap relation 2921154 via Nominatim",
            "license": "ODbL 1.0"}, "geometry": geometry}
        TAICHUNG_BOUNDARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        TAICHUNG_BOUNDARY_PATH.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        print(f"Downloaded and cached Taichung boundary: {TAICHUNG_BOUNDARY_PATH}")
    geometry = document.get("geometry", document)
    if geometry.get("type") == "Polygon":
        return [np.asarray(ring, dtype=np.float64) for ring in geometry["coordinates"]]
    if geometry.get("type") == "MultiPolygon":
        return [np.asarray(ring, dtype=np.float64) for polygon in geometry["coordinates"] for ring in polygon]
    raise ValueError("Taichung boundary must be a Polygon or MultiPolygon.")


def solve_single_tsp(seed):
    rng = np.random.default_rng(seed)
    for attempt in range(20):
        store_indices = rng.choice(len(STORE_CATALOG["stores"]), NUM_NODES, replace=False)
        coords = np.asarray([STORE_CATALOG["stores"][i]["coords"] for i in store_indices], dtype=np.float32)
        try:
            distance_matrix = get_osrm_distance_matrix(coords)
            break
        except UnreachableStoresError:
            if attempt == 19:
                raise RuntimeError(f"Could not find {NUM_NODES} mutually reachable stores for seed {seed}")
    # Only finite, fully reachable samples are passed to the solver.
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

    return seed, coords, adjacency, distance_matrix, store_indices



def replace_checkpoint(temp_path, path):
    """Retry transient Windows file locks without deleting the previous dataset."""
    for attempt in range(6):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError as exc:
            if attempt == 5:
                raise PermissionError(
                    f"Cannot replace {path}; it may be open in a training/evaluation "
                    f"process or not writable. Close processes using this file and "
                    f"check its permissions, then rerun generation to resume. "
                    f"Generated data is preserved at {temp_path}; "
                    "progress checkpoints are also retained."
                ) from exc
            time.sleep(0.5)


def save_samples(path, results):
    """Atomically save completed samples so an interrupted run can resume."""
    seeds = np.asarray(sorted(results), dtype=np.int32)
    coords = np.stack([results[int(seed)][0] for seed in seeds]).astype(np.float32)
    adjacencies = np.stack([results[int(seed)][1] for seed in seeds]).astype(np.int8)
    distances = np.stack([results[int(seed)][2] for seed in seeds]).astype(np.float32)
    sample_indices = np.stack([results[int(seed)][3] for seed in seeds])
    catalog_stores = STORE_CATALOG["stores"]
    temp_path = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(
        temp_path,
        seeds=seeds,
        store_indices=sample_indices,
        store_ids=np.asarray([store["id"] for store in catalog_stores])[sample_indices],
        store_names=np.asarray([store["name"] for store in catalog_stores])[sample_indices],
        store_coords=np.asarray([store["original_coords"] for store in catalog_stores], dtype=np.float64)[sample_indices],
        store_catalog_json=np.asarray(json.dumps(STORE_CATALOG, ensure_ascii=False)),
        catalog_hash=np.asarray(CATALOG_HASH),
        node_source=np.asarray("taichung_7eleven_osm"),
        coords=coords,
        adjacencies=adjacencies,
        distances=distances,
        num_samples=np.int32(NUM_SAMPLES),
        num_nodes=np.int32(NUM_NODES),
        solve_seconds=np.int32(SOLVE_SECONDS),
        distance_mode=np.asarray("osrm_driving"),
        boundary_name=np.asarray("臺中市"),
    )
    replace_checkpoint(temp_path, path)


def load_progress(path):
    if not path.exists():
        return {}

    results = {}
    for checkpoint_path in sorted(path.glob("sample_*.npz")):
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            if "catalog_hash" not in saved.files or str(saved["catalog_hash"]) != CATALOG_HASH:
                raise ValueError(f"Store catalog mismatch in {checkpoint_path}; do not mix datasets.")
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
                saved["coords"], saved["adjacencies"], saved["distances"], saved["store_indices"]
            )
    return results


def save_progress_sample(path, seed, coords, adjacency, distances, store_indices):
    path.mkdir(parents=True, exist_ok=True)
    sample_path = path / f"sample_{seed:06d}.npz"
    temp_path = path / f"sample_{seed:06d}.tmp.npz"
    np.savez_compressed(
        temp_path,
        seed=np.int32(seed),
        store_indices=store_indices,
        catalog_hash=np.asarray(CATALOG_HASH),
        node_source=np.asarray("taichung_7eleven_osm"),
        coords=coords,
        adjacencies=adjacency,
        distances=distances,
        num_samples=np.int32(NUM_SAMPLES),
        num_nodes=np.int32(NUM_NODES),
        solve_seconds=np.int32(SOLVE_SECONDS),
    )
    replace_checkpoint(temp_path, sample_path)


def main():
    if NUM_SAMPLES < 1:
        raise ValueError("TSP_NUM_SAMPLES must be at least 1.")
    if NUM_NODES < 3:
        raise ValueError("TSP_NUM_NODES must be at least 3.")
    global STORE_CATALOG, CATALOG_HASH
    load_taichung_boundary()
    STORE_CATALOG = prepare_stores(OSRM_URL, NUM_NODES)
    CATALOG_HASH = hashlib.sha256(json.dumps(
        {key: value for key, value in STORE_CATALOG.items() if key != "prepared_at"},
        sort_keys=True).encode("utf-8")).hexdigest()

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    progress_path = DATASET_PATH.with_name(
        f"{DATASET_PATH.stem}.progress_{NUM_SAMPLES}x{NUM_NODES}_{SOLVE_SECONDS}s_{CATALOG_HASH[:12]}"
    )
    progress_path.mkdir(parents=True, exist_ok=True)
    (progress_path / "store_catalog.json").write_text(
        json.dumps(STORE_CATALOG, ensure_ascii=False, indent=2), encoding="utf-8")
    results = load_progress(progress_path)
    pending_seeds = [seed for seed in range(NUM_SAMPLES) if seed not in results]
    workers = min(NUM_WORKERS, max(1, len(pending_seeds)))
    completed_before = len(results)
    run_start = time.time()

    print(
        f"Generating {NUM_SAMPLES} TSP samples ({NUM_NODES} 7-ELEVEN stores each); "
        f"workers={workers}, solver limit={SOLVE_SECONDS}s, distance=osrm_driving, boundary=臺中市."
    )
    if completed_before:
        print(f"Loaded checkpoint: {completed_before}/{NUM_SAMPLES}; resuming.")

    try:
        if pending_seeds:
            with Pool(processes=workers, initializer=init_worker, initargs=(STORE_CATALOG, CATALOG_HASH)) as pool:
                for seed, coords, adjacency, distances, store_indices in pool.imap_unordered(
                    solve_single_tsp, pending_seeds, chunksize=1
                ):
                    results[seed] = (coords, adjacency, distances, store_indices)
                    save_progress_sample(
                        progress_path, seed, coords, adjacency, distances, store_indices
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
