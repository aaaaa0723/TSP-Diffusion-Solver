"""Fixed Taichung YouBike station snapshot, aligned to driving roads."""
import json
import os
from urllib.request import urlopen

import numpy as np

from experiment_utils import DATA_DIR
from seven_eleven import prepare_stores as align_locations

CATALOG_PATH = DATA_DIR / "taichung_youbike.json"
SOURCE_URL = "https://ybjson02.youbike.com.tw:60008/yb2/taichung/gwjs.json"
NODE_SOURCE = "taichung_youbike"


def parse_candidates(document):
    if str(document.get("retCode")) != "1" or not isinstance(document.get("retVal"), list):
        raise ValueError("Invalid Taichung YouBike response")
    stations, seen = [], set()
    for item in document["retVal"]:
        if item.get("scityen") != "Taichung City" or str(item.get("act")) != "1":
            continue
        station_id = str(item.get("sno", "")).strip()
        try:
            coords = [float(item["lng"]), float(item["lat"])]
        except (KeyError, TypeError, ValueError):
            continue
        if (not station_id or station_id in seen or not np.isfinite(coords).all()
                or not (-180 <= coords[0] <= 180 and -90 <= coords[1] <= 90)
                or coords == [0, 0]):
            continue
        seen.add(station_id)
        stations.append({"id": f"youbike/{station_id}", "name": item.get("sna", station_id),
                         "original_coords": coords})
    if not stations:
        raise ValueError("No active Taichung YouBike stations in response")
    return sorted(stations, key=lambda station: station["id"])


def load_candidates():
    if CATALOG_PATH.exists():
        return parse_candidates(json.loads(CATALOG_PATH.read_text(encoding="utf-8")))
    url = os.environ.get("TSP_YOUBIKE_URL", SOURCE_URL)
    with urlopen(url, timeout=60) as response:
        document = json.load(response)
    stations = parse_candidates(document)
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CATALOG_PATH.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    temporary.replace(CATALOG_PATH)
    return stations


def prepare_stores(osrm_url, minimum):
    # Keep the existing NPZ store_* keys for training/checkpoint compatibility.
    return align_locations(osrm_url, minimum, candidates=load_candidates(),
                           source=os.environ.get("TSP_YOUBIKE_URL", SOURCE_URL),
                           label="YouBike", license_name="Open Government Data License 1.0")
