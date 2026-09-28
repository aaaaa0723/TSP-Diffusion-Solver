"""Generate random Euclidean TSP instances and OR-Tools reference tours."""
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT, "results", "tsp_dataset_lite.npz")
NUM_SAMPLES = int(os.getenv("TSP_NUM_SAMPLES", "1000"))
NUM_NODES = int(os.getenv("TSP_NUM_NODES", "494"))
TIME_LIMIT = int(os.getenv("TSP_SOLVER_TIME_LIMIT_SECONDS", "30"))
TAICHUNG_AREA_KM2 = 2215.0
SIDE_KM = np.sqrt(TAICHUNG_AREA_KM2)


def solve_single(seed):
    rng = np.random.default_rng(seed)
    coords = (rng.random((NUM_NODES, 2)) * SIDE_KM).astype(np.float32)
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.rint(np.sqrt(np.square(delta).sum(axis=-1)) * 1000).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(NUM_NODES, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance(from_index, to_index):
        return int(distances[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    callback = routing.RegisterTransitCallback(distance)
    routing.SetArcCostEvaluatorOfAllVehicles(callback)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = TIME_LIMIT
    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError(f"OR-Tools did not find a route for seed {seed}")

    adjacency = np.zeros((NUM_NODES, NUM_NODES), dtype=np.int8)
    index = routing.Start(0)
    while not routing.IsEnd(index):
        source = manager.IndexToNode(index)
        index = solution.Value(routing.NextVar(index))
        target = manager.IndexToNode(index)
        adjacency[source, target] = 1
    return coords, adjacency


def main():
    if NUM_SAMPLES < 1 or NUM_NODES < 3 or TIME_LIMIT < 1:
        raise ValueError("Sample count, node count, and solver time limit must be positive (at least 3 nodes).")
    workers = min(cpu_count(), NUM_SAMPLES, 16)
    print(f"Generating {NUM_SAMPLES} TSP instances with {NUM_NODES} cities; {workers} workers.")
    with Pool(workers) as pool:
        rows = list(pool.imap(solve_single, range(NUM_SAMPLES)))
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    np.savez_compressed(
        DATASET_PATH,
        coords=np.stack([row[0] for row in rows]),
        adjacencies=np.stack([row[1] for row in rows]),
    )
    print(f"Saved reference instances to {DATASET_PATH}")
    print("Reference tours use an OR-Tools time limit and are not guaranteed globally optimal.")


if __name__ == "__main__":
    main()
