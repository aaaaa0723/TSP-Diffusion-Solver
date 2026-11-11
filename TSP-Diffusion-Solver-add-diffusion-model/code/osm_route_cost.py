"""Match OSRM route node annotations to OSM toll ways and signal nodes."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import warnings

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / "data" / "osm_route_cost_cache"


def cached_json(key, fetch, persist=True):
    # Set a new namespace when rebuilding OSRM or refreshing the OSM snapshot.
    key = os.environ.get("TSP_OSM_COST_CACHE_VERSION", "v1") + key
    path = CACHE / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    value = fetch()
    if not persist:
        return value
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(value), encoding="utf-8")
        try:
            os.replace(temporary, path)
        except PermissionError:
            # Another worker may have published the same key while this worker
            # fetched it. Windows cannot replace a file held open by a reader.
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise
    except OSError:
        # The route response is valid; an optional cache failure must not discard
        # hours of work. Dataset checkpoints remain mandatory and fail on errors.
        warnings.warn("Route cache unavailable; using fetched road features", RuntimeWarning)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return value


class UnreachableRouteError(ValueError):
    """An OSRM pair has no usable driving route; resample the TSP instance."""


def request_json(response):
    # OSRM reports NoRoute with HTTP 400. Classify it before HTTP checking.
    try:
        data = response.json()
    except ValueError:
        response.raise_for_status()
        raise
    if data.get("code") == "NoRoute":
        raise UnreachableRouteError("OSRM found no driving route for this station pair")
    response.raise_for_status()
    if data.get("remark") or ("code" in data and data["code"] != "Ok"):
        raise ValueError(f"Road feature request failed: {data}")
    return data


def feature_index(elements):
    nodes = {}
    toll_edges = set()
    for element in elements:
        if element["type"] == "node":
            nodes[int(element["id"])] = element.get("tags", {})
        elif element["type"] == "way":
            tags = element.get("tags", {})
            if tags.get("toll:motorcar", tags.get("toll:motor_vehicle", tags.get("toll"))) == "yes":
                ids = element["nodes"]
                for a, b in zip(ids, ids[1:]):
                    toll_edges.add((a, b))
                    toll_edges.add((b, a))
    return nodes, toll_edges


def route_features(annotation, nodes, toll_edges):
    ids = annotation["nodes"]
    lengths = annotation["distance"]
    if len(ids) != len(lengths) + 1:
        raise ValueError("OSRM node/distance annotations do not align")
    valid_ids = [node for node in ids if node > 0]
    missing = not nodes.contains_all(valid_ids) if hasattr(nodes, "contains_all") else any(node not in nodes for node in valid_ids)
    if missing:
        raise ValueError("OSM nodes missing: use an OSM snapshot matching the OSRM graph")
    if not np.isfinite(lengths).all() or any(d < 0 for d in lengths):
        raise ValueError("Invalid route segment lengths")
    toll_km = sum(d for a, b, d in zip(ids, ids[1:], lengths) if (a, b) in toll_edges) / 1000
    # Count actual traversals, even if a node is revisited; omit final node so
    # a signal at a TSP stop is charged on departure only.
    signals = sum(nodes.get(node, {}).get("highway") == "traffic_signals" for node in ids[:-1])
    return toll_km, signals


_LOCAL_INDEX = None


def local_feature_index():
    global _LOCAL_INDEX
    if _LOCAL_INDEX is not None:
        return _LOCAL_INDEX
    path = Path(os.environ["TSP_OSM_FEATURE_INDEX"])
    document = json.loads((path / "features.json").read_text(encoding="utf-8"))
    known_ids = np.load(path / "node_ids.npy", mmap_mode="r")
    signals = set(document["signals"])
    class LocalNodes:
        def contains_all(self, ids):
            query = np.asarray(ids, dtype=np.int64)
            indices = np.searchsorted(known_ids, query)
            return bool((indices < len(known_ids)).all() and np.array_equal(known_ids[indices], query))
        def __contains__(self, node):
            index = np.searchsorted(known_ids, node)
            return index < len(known_ids) and known_ids[index] == node
        def get(self, node, default=None):
            return {"highway": "traffic_signals"} if node in signals else {}
    edges = set(map(tuple, document["toll_edges"]))
    edges.update((b, a) for a, b in list(edges))
    fingerprint = f"{document['source_size']}:{document['source_mtime_ns']}"
    _LOCAL_INDEX = LocalNodes(), edges, fingerprint
    return _LOCAL_INDEX


def local_feature_matrices(coords, distances, durations, session, osrm_url):
    nodes, edges, fingerprint = local_feature_index()
    toll = np.zeros_like(distances)
    signals = np.zeros_like(distances)
    for i, origin in enumerate(coords):
        for j, destination in enumerate(coords):
            if i == j or distances[i, j] == 0:
                continue
            coordinates = ";".join(f"{float(lon):.8f},{float(lat):.8f}" for lon, lat in (origin, destination))
            url = f"{osrm_url}/route/v1/driving/{coordinates}"
            def fetch():
                data = request_json(session.get(url, params={"annotations": "nodes,distance",
                    "overview": "false", "alternatives": "false"}, timeout=120))
                route = data["routes"][0]
                toll_km, count = route_features(route["legs"][0]["annotation"], nodes, edges)
                return {"distance": route["distance"], "duration": route["duration"],
                        "toll_km": toll_km, "signal_counts": count}
            feature = cached_json("local-v2:" + fingerprint + url, fetch)
            # Table and route can choose different equal-time paths. Use the
            # annotated route for every factor instead of mixing their distances.
            distances[i, j] = feature["distance"] / 1000
            durations[i, j] = feature["duration"]
            toll[i, j], signals[i, j] = feature["toll_km"], feature["signal_counts"]
        if i % 5 == 0 or i + 1 == len(coords):
            print(f"Road cost rows: {i + 1}/{len(coords)} (worker {os.getpid()})", flush=True)
    return toll, signals


def get_feature_matrices(coords, distances, durations, session, osrm_url):
    if os.environ.get("TSP_OSM_FEATURE_INDEX"):
        return local_feature_matrices(coords, distances, durations, session, osrm_url)
    routes = {}
    all_ids = set()
    for i, origin in enumerate(coords):
        for j, destination in enumerate(coords):
            if i == j or distances[i, j] == 0:
                continue
            coordinates = ";".join(f"{float(lon):.8f},{float(lat):.8f}" for lon, lat in (origin, destination))
            url = f"{osrm_url}/route/v1/driving/{coordinates}"
            data = cached_json(url, lambda: request_json(session.get(url, params={
                "annotations": "nodes,distance", "overview": "false", "alternatives": "false"}, timeout=120)))
            route = data["routes"][0]
            distances[i, j] = route["distance"] / 1000
            durations[i, j] = route["duration"]
            annotation = route["legs"][0]["annotation"]
            routes[i, j] = annotation
            all_ids.update(node for node in annotation["nodes"] if node > 0)
    endpoint = os.environ.get("TSP_OVERPASS_URL", "https://overpass.private.coffee/api/interpreter")
    elements = []
    ordered_ids = sorted(all_ids)
    for start in range(0, len(ordered_ids), 1000):
        ids = ",".join(map(str, ordered_ids[start:start + 1000]))
        query = f'[out:json][timeout:120];node(id:{ids})->.n;(.n;way(bn.n)["toll"];way(bn.n)["toll:motorcar"];way(bn.n)["toll:motor_vehicle"];);out body;'
        data = cached_json(endpoint + query, lambda: request_json(session.post(endpoint, data={"data": query}, timeout=150)))
        elements.extend(data["elements"])
    nodes, toll_edges = feature_index(elements)
    toll = np.zeros_like(distances)
    signals = np.zeros_like(distances)
    for (i, j), annotation in routes.items():
        toll[i, j], signals[i, j] = route_features(annotation, nodes, toll_edges)
    return toll, signals
