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
from youbike import NODE_SOURCE, prepare_stores
from osm_route_cost import get_feature_matrices, UnreachableRouteError
from road_requests import configure_session


NUM_SAMPLES = int(os.environ.get("TSP_NUM_SAMPLES", "1000"))
NUM_NODES = int(os.environ.get("TSP_NUM_NODES", "300"))
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
COST_MODE = "osm_distance_duration_toll_signals_v3"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def init_worker(catalog, catalog_hash):
    """Share a fixed store pool across Windows/Linux workers."""
    global OSRM_SESSION, STORE_CATALOG, CATALOG_HASH
    OSRM_SESSION = configure_session(requests.Session())
    STORE_CATALOG = catalog
    CATALOG_HASH = catalog_hash


class UnreachableStoresError(ValueError):
    pass


def get_osrm_matrices(coords):
    """Request an OSRM road-distance matrix for the supplied coordinates."""
    url = f"{OSRM_URL}/table/v1/driving"
    payload = {
        "coordinates": np.asarray(coords, dtype=float).tolist(),
        "annotations": ["distance", "duration"],
    }

    try:
        if os.environ.get("TSP_PAIR_STORE_PATH"):
            from pair_store import get_pair_store
            distances, durations, toll_km, signal_counts = get_pair_store().matrices(coords)
            costs, metadata = build_route_cost(distances, durations, toll_km, signal_counts)
            return distances, durations, costs, metadata
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
        durations = np.asarray(data["durations"], dtype=np.float32)
        distances = (matrix / 1000.0).astype(np.float32)
        toll_km, signal_counts = get_feature_matrices(coords, distances, durations, session, OSRM_URL)
        costs, metadata = build_route_cost(distances, durations, toll_km, signal_counts)
        return distances, durations, costs, metadata
    except UnreachableRouteError as exc:
        raise UnreachableStoresError(str(exc)) from exc
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


def get_osrm_distance_matrix(coords):
    return get_osrm_matrices(coords)[0]


def build_route_cost(distances, durations, toll_km=None, signal_counts=None):
    """Balance distance and OSM-profile travel time in equivalent kilometers."""
    distances = np.asarray(distances, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1] or durations.shape != distances.shape:
        raise ValueError("Distance and duration matrices must have matching square shapes")
    if any(not np.isfinite(a).all() or (a < 0).any() for a in (distances, durations)):
        raise UnreachableStoresError("Invalid or unreachable distance/duration")
    mask = ~np.eye(len(distances), dtype=bool) & (distances > 0) & (durations > 0)
    if not mask.any():
        raise ValueError("No positive road pairs available to calibrate costs")
    km_per_second = float(np.median(distances[mask]) / np.median(durations[mask]))
    toll_km = np.zeros_like(distances) if toll_km is None else np.asarray(toll_km, dtype=float)
    signal_counts = np.zeros_like(distances) if signal_counts is None else np.asarray(signal_counts, dtype=float)
    for feature in (toll_km, signal_counts):
        if feature.shape != distances.shape or not np.isfinite(feature).all() or (feature < 0).any():
            raise ValueError("Invalid toll/signal feature matrix")
    if (toll_km > distances + .002).any() or (signal_counts != np.floor(signal_counts)).any():
        raise ValueError("Toll length exceeds route length or signal count is not an integer")
    costs = .35 * distances + .65 * durations * km_per_second + .25 * toll_km + .02 * signal_counts
    np.fill_diagonal(costs, 0)
    return costs.astype(np.float32), {"mode": COST_MODE, "distance_weight": .35,
        "duration_weight": .65, "duration_km_per_second": km_per_second,
        "unit": "equivalent_km", "traffic": "static_osrm_profile",
        "toll_penalty_equivalent_km_per_km": .25,
        "signal_penalty_equivalent_km_per_signal": .02,
        "toll_km": toll_km.astype(np.float32), "signal_counts": signal_counts.astype(np.int32),
        "feature_source": "OSM toll ways and highway=traffic_signals on OSRM route nodes",
        "pair_route_source": "annotated_osrm_route",
        "osm_feature_index": os.environ.get("TSP_OSM_FEATURE_INDEX", "overpass")}


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
            distance_matrix, durations, costs, cost_metadata = get_osrm_matrices(coords)
            break
        except UnreachableStoresError:
            print(f"Seed {seed}: unreachable station pair; resampling ({attempt + 1}/20)", flush=True)
            if attempt == 19:
                raise RuntimeError(f"Could not find {NUM_NODES} mutually reachable stores for seed {seed}")
    # Only finite, fully reachable samples are passed to the solver.
    manager = pywrapcp.RoutingIndexManager(NUM_NODES, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(float(costs[from_node, to_node]) * 1000.0))

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

    return seed, coords, adjacency, distance_matrix, store_indices, durations, costs, cost_metadata



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
    durations = np.stack([results[int(seed)][4] for seed in seeds])
    costs = np.stack([results[int(seed)][5] for seed in seeds])
    cost_metadata_json = [json.dumps({k: v for k, v in results[int(seed)][6].items() if k not in ("toll_km", "signal_counts")}, sort_keys=True) for seed in seeds]
    toll_km = [results[int(seed)][6].get("toll_km", np.zeros_like(results[int(seed)][2])) for seed in seeds]
    signal_counts = [results[int(seed)][6].get("signal_counts", np.zeros_like(results[int(seed)][2])) for seed in seeds]
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
        node_source=np.asarray(NODE_SOURCE),
        coords=coords,
        adjacencies=adjacencies,
        distances=distances,
        durations=durations,
        costs=costs,
        cost_mode=np.asarray(COST_MODE),
        cost_metadata_json=np.asarray(cost_metadata_json),
        toll_km=np.asarray(toll_km, dtype=np.float32),
        signal_counts=np.asarray(signal_counts, dtype=np.int32),
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
            if "cost_mode" not in saved.files or str(saved["cost_mode"]) != COST_MODE:
                raise ValueError("Checkpoint cost mode mismatch; regenerate in a separate progress directory")
            if "distances" not in saved.files:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} has no OSRM distance matrix; "
                    "remove it and restart generation."
                )
            seed = int(saved["seed"])
            results[seed] = (
                saved["coords"], saved["adjacencies"], saved["distances"], saved["store_indices"],
                saved["durations"], saved["costs"], {**json.loads(str(saved["cost_metadata_json"])), "toll_km": saved["toll_km"], "signal_counts": saved["signal_counts"]}
            )
    return results


def save_progress_sample(path, seed, coords, adjacency, distances, store_indices, durations, costs, cost_metadata):
    toll_km = cost_metadata.get("toll_km", np.zeros_like(distances))
    signal_counts = cost_metadata.get("signal_counts", np.zeros_like(distances))
    cost_metadata_json = json.dumps({k: v for k, v in cost_metadata.items() if k not in ("toll_km", "signal_counts")}, sort_keys=True)
    path.mkdir(parents=True, exist_ok=True)
    sample_path = path / f"sample_{seed:06d}.npz"
    temp_path = path / f"sample_{seed:06d}.tmp.npz"
    np.savez_compressed(
        temp_path,
        seed=np.int32(seed),
        store_indices=store_indices,
        catalog_hash=np.asarray(CATALOG_HASH),
        node_source=np.asarray(NODE_SOURCE),
        coords=coords,
        adjacencies=adjacency,
        distances=distances,
        durations=durations,
        costs=costs,
        cost_mode=np.asarray(COST_MODE),
        cost_metadata_json=np.asarray(cost_metadata_json),
        toll_km=np.asarray(toll_km, dtype=np.float32),
        signal_counts=np.asarray(signal_counts, dtype=np.int32),
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
        f"{DATASET_PATH.stem}.progress_{NUM_SAMPLES}x{NUM_NODES}_{SOLVE_SECONDS}s_{CATALOG_HASH[:12]}_{COST_MODE}"
    )
    progress_path.mkdir(parents=True, exist_ok=True)
    (progress_path / "store_catalog.json").write_text(
        json.dumps(STORE_CATALOG, ensure_ascii=False, indent=2), encoding="utf-8")
    if os.environ.get("TSP_PAIR_STORE_DIRECTORY"):
        from pair_store import create_pair_store
        create_pair_store(STORE_CATALOG, CATALOG_HASH, OSRM_URL, progress_path)
    results = load_progress(progress_path)
    pending_seeds = [seed for seed in range(NUM_SAMPLES) if seed not in results]
    workers = min(NUM_WORKERS, max(1, len(pending_seeds)))
    completed_before = len(results)
    run_start = time.time()

    print(
        f"Generating {NUM_SAMPLES} TSP samples ({NUM_NODES} YouBike stations each); "
        f"workers={workers}, solver limit={SOLVE_SECONDS}s, distance=osrm_driving, boundary=臺中市."
    )
    if completed_before:
        print(f"Loaded checkpoint: {completed_before}/{NUM_SAMPLES}; resuming.")

    try:
        if pending_seeds:
            with Pool(processes=workers, initializer=init_worker, initargs=(STORE_CATALOG, CATALOG_HASH)) as pool:
                for seed, coords, adjacency, distances, store_indices, durations, costs, cost_metadata in pool.imap_unordered(
                    solve_single_tsp, pending_seeds, chunksize=1
                ):
                    results[seed] = (coords, adjacency, distances, store_indices, durations, costs, cost_metadata)
                    save_progress_sample(
                        progress_path, seed, coords, adjacency, distances, store_indices, durations, costs, cost_metadata
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
    # Retain checkpoints until the independent runner validation is complete.

    print(
        f"Dataset complete: {DATASET_PATH}; "
        f"coords={NUM_SAMPLES}x{NUM_NODES}x2, "
        f"adjacencies={NUM_SAMPLES}x{NUM_NODES}x{NUM_NODES}; "
        f"elapsed {time.time() - run_start:.1f}s."
    )


if __name__ == "__main__":
    main()
