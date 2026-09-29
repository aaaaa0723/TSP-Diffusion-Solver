"""Optional offline catalog refresh: pip install osmium, then run this script."""
import json
import re

import osmium
from matplotlib.path import Path as PolygonPath

from experiment_utils import DATA_DIR, TAICHUNG_BOUNDARY_PATH
from seven_eleven import CATALOG_PATH


def main():
    document = json.loads(TAICHUNG_BOUNDARY_PATH.read_text(encoding="utf-8"))
    geometry = document.get("geometry", document)
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    polygons = [[PolygonPath(ring) for ring in polygon] for polygon in polygons]

    class Stores(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.elements = []

        def node(self, node):
            if node.tags.get("shop") != "convenience":
                return
            if not re.search(r"7.?eleven|7-11", node.tags.get("brand", "") + " " + node.tags.get("name", ""), re.I):
                return
            coords = (node.location.lon, node.location.lat)
            if not any(polygon[0].contains_point(coords) and not any(
                    hole.contains_point(coords) for hole in polygon[1:]) for polygon in polygons):
                return
            self.elements.append({"type": "node", "id": node.id, "lon": coords[0],
                                  "lat": coords[1], "tags": dict(node.tags)})

    handler = Stores()
    handler.apply_file(str(DATA_DIR / "osrm" / "taiwan-latest.osm.pbf"))
    if not handler.elements:
        raise ValueError("No 7-ELEVEN node POIs found; existing catalog was not changed")
    catalog = {"source": "OpenStreetMap contributors / local taiwan-latest.osm.pbf, node POIs within Taichung boundary",
               "license": "ODbL 1.0", "elements": handler.elements}
    temporary = CATALOG_PATH.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(CATALOG_PATH)
    print(f"Extracted {len(handler.elements)} store records to {CATALOG_PATH}")


if __name__ == "__main__":
    main()
