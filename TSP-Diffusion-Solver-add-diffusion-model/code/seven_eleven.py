"""Taichung 7-ELEVEN candidates from OpenStreetMap, aligned to OSRM roads."""
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import requests

from experiment_utils import DATA_DIR

CATALOG_PATH = DATA_DIR / "taichung_7eleven_osm.json"
QUERY = '''[out:json][timeout:90];area(3602921154)->.a;
(nwr(area.a)[shop=convenience][brand~"7.?eleven|7-11",i];
 nwr(area.a)[shop=convenience][name~"7.?eleven|7-11",i];);out center tags;'''


def load_candidates():
    if not CATALOG_PATH.exists():
        endpoint = os.environ.get("TSP_OVERPASS_URL", "https://overpass.private.coffee/api/interpreter")
        with urlopen(endpoint + "?" + urlencode({"data": QUERY}), timeout=120) as response:
            document = json.load(response)
        if document.get("remark") or not document.get("elements"):
            raise ValueError(f"Incomplete or empty Overpass response: {document.get('remark')}")
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CATALOG_PATH.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        temporary.replace(CATALOG_PATH)
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if document.get("remark"):
        raise ValueError("Cached Overpass response is incomplete; refresh the store catalog.")
    stores, seen = [], set()
    for item in sorted(document["elements"], key=lambda x: (x["type"], x["id"])):
        location = item.get("center", item)
        if "lon" not in location or "lat" not in location:
            continue
        tags = item.get("tags", {})
        coords = [float(location["lon"]), float(location["lat"])]
        # Nodes and building polygons can describe the same branch.
        identity = tags.get("ref") or (tags.get("name", ""), tuple(round(v, 5) for v in coords))
        if identity in seen:
            continue
        seen.add(identity)
        stores.append({"id": f"{item['type']}/{item['id']}",
                       "name": tags.get("name", "7-ELEVEN"),
                       "original_coords": coords})
    return stores


def prepare_stores(osrm_url, minimum):
    """Exclude distant road snaps and duplicate road positions before sampling."""
    maximum = float(os.environ.get("TSP_STORE_MAX_SNAP_METERS", "100"))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("TSP_STORE_MAX_SNAP_METERS must be positive and finite")
    candidates = load_candidates()
    stores, positions = [], set()
    with requests.Session() as session:
        for index, store in enumerate(candidates):
            lon, lat = store["original_coords"]
            response = session.get(f"{osrm_url}/nearest/v1/driving/{lon:.8f},{lat:.8f}",
                                   params={"number": 1}, timeout=(10, 30))
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") == "NoSegment":
                continue
            if payload.get("code") != "Ok":
                raise RuntimeError(f"OSRM nearest failed: {payload}")
            waypoint = payload["waypoints"][0]
            snap_distance = float(waypoint["distance"])
            coords = np.asarray(waypoint["location"], dtype=np.float32)
            if (coords.shape != (2,) or not np.isfinite(coords).all()
                    or not np.isfinite(snap_distance) or snap_distance > maximum):
                continue
            position = tuple(coords.tolist())
            if position in positions:
                continue
            positions.add(position)
            stores.append({**store, "coords": coords.tolist(), "snap_distance_m": snap_distance})
            if (index + 1) % 100 == 0:
                print(f"Road alignment: {index + 1}/{len(candidates)} stores checked", flush=True)
    if len(stores) < minimum:
        raise ValueError(f"Only {len(stores)} unique road-aligned stores; need {minimum}. "
                         "Refresh the OSM catalog or inspect the OSRM road data.")
    print(f"7-ELEVEN pool: {len(stores)}/{len(candidates)} stores within {maximum:g} m of a road.")
    source = json.loads(CATALOG_PATH.read_text(encoding="utf-8")).get("source", "OpenStreetMap contributors / Overpass")
    return {"source": source, "license": "ODbL 1.0",
            "region": "Taichung, OSM relation 2921154", "osrm_url": osrm_url,
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "max_snap_meters": maximum, "stores": stores}
