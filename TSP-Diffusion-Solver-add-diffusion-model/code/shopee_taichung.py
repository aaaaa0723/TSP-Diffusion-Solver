"""Collect publicly listed SPX points with spatial subdivision of capped queries."""
import json
import math
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from experiment_utils import DATA_DIR, TAICHUNG_BOUNDARY_PATH

ENDPOINT = "https://spx.tw/api/service-point/point/around/list"
CATALOG = DATA_DIR / "taichung_shopee.json"


def collect():
    if CATALOG.exists():
        return json.loads(CATALOG.read_text(encoding="utf-8"))
    boundary = json.loads(TAICHUNG_BOUNDARY_PATH.read_text(encoding="utf-8"))["geometry"]
    polygons = boundary["coordinates"] if boundary["type"] == "MultiPolygon" else [boundary["coordinates"]]
    points = np.asarray([xy for polygon in polygons for ring in polygon for xy in ring])
    low, high = points.min(axis=0), points.max(axis=0)
    queue = [(float(low[0]), float(low[1]), float(high[0]), float(high[1]), 0)]
    found, queries = {}, []
    cache = DATA_DIR / "shopee_queries"
    cache.mkdir(exist_ok=True)
    while queue:
        west, south, east, north, depth = queue.pop()
        lon, lat = (west + east) / 2, (south + north) / 2
        radius = math.ceil(math.hypot(east - west, north - south) * 111320 / 2) + 100
        params = {"latitude": lat, "longitude": lon, "radius": radius, "selected_radius": radius}
        target = cache / f"{west:.6f}_{south:.6f}_{east:.6f}_{north:.6f}.json"
        if target.exists():
            payload = json.loads(target.read_text(encoding="utf-8"))
        else:
            with urlopen(ENDPOINT + "?" + urlencode(params), timeout=60) as response:
                payload = json.load(response)
            if payload.get("retcode") != 0:
                raise ValueError(f"SPX query failed: {payload}")
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        rows = payload["data"]["list"]
        queries.append({**params, "returned": len(rows), "depth": depth})
        for item in rows:
            address = json.loads(item.get("address_data") or "{}")
            city = address.get("city", "").replace("台", "臺")
            if city != "臺中市" or not item.get("is_spx_point"):
                continue
            found[str(item["id"])] = item
        if len(rows) >= 100:
            if depth >= 12:
                raise RuntimeError("SPX query cap unresolved; cannot claim full coverage")
            queue.extend([(w, s, e, n, depth + 1) for w, e in [(west, lon), (lon, east)]
                          for s, n in [(south, lat), (lat, north)]])
        print(f"SPX queries={len(queries)}, pending={len(queue)}, Taichung points={len(found)}", flush=True)
    stores = [{"id": f"spx/{key}", "name": item["alias"], "address": item["address"],
               "original_coords": [float(item["longitude"]), float(item["latitude"])],
               "point_status": item.get("point_status"), "is_display": item.get("is_display")}
              for key, item in sorted(found.items(), key=lambda pair: int(pair[0]))]
    if not stores:
        raise ValueError("No Taichung SPX points found")
    document = {"source": ENDPOINT, "collected_at": datetime.now(timezone.utc).isoformat(),
                "scope": "All Taichung SPX points returned by uncapped spatial queries covering city bounds",
                "queries": queries, "stores": stores}
    CATALOG.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


if __name__ == "__main__":
    print(f"Collected {len(collect()['stores'])} Taichung SPX points")
