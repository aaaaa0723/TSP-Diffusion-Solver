"""Fetch full OSRM road shapes in bounded batches for large TSP tours."""
import hashlib
import json
import os

import numpy as np
import requests


def road_geometry(points, tour, cache_dir):
    url = os.environ.get("TSP_OSRM_URL", "http://localhost:5000").rstrip("/")
    ordered = np.asarray(points, dtype=float)[np.asarray(tour, dtype=int)]
    cache_key = hashlib.sha256(
        (url + json.dumps(ordered.tolist())).encode("utf-8")
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        return np.asarray(json.loads(cache_path.read_text(encoding="utf-8")), dtype=float)

    lines = []
    try:
        with requests.Session() as session:
            # Adjacent batches share one waypoint and include the closing edge.
            # Stay below the usual OSRM 100-waypoint route limit.
            for start in range(0, len(ordered) - 1, 79):
                batch = ordered[start:start + 80]
                coordinates = ";".join(f"{lon:.8f},{lat:.8f}" for lon, lat in batch)
                response = session.get(
                    f"{url}/route/v1/driving/{coordinates}",
                    params={"geometries": "geojson", "overview": "full",
                            "steps": "false", "alternatives": "false",
                            "continue_straight": "false"},
                    timeout=(10, 120),
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("code") != "Ok":
                    raise ValueError(f"OSRM returned {payload.get('code')}: {payload.get('message', '')}")
                geometry = payload["routes"][0]["geometry"]
                line = np.asarray(geometry["coordinates"], dtype=float)
                if (geometry["type"] != "LineString" or line.ndim != 2
                        or line.shape[1] != 2 or len(line) < 2
                        or not np.isfinite(line).all()):
                    raise ValueError("Invalid road geometry")
                lines.append(line.tolist())
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"OSRM road geometry unavailable at {url}: {exc}. "
            "Start the same OSRM service used for generation, then rerun "
            "evaluate.py --run <run_id> --plots-only. Metrics still use the saved distance matrix."
        ) from exc

    # Separate batches to avoid drawing artificial straight connectors if snapping differs.
    joined = []
    for line in lines:
        if joined:
            joined.append([None, None])
        joined.extend(line)
    cache_path.write_text(json.dumps(joined), encoding="utf-8")
    return np.asarray(joined, dtype=float)
