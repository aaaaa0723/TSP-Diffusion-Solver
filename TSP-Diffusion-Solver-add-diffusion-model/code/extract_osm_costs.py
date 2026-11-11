"""Extract cost tags and node coverage from the same OSM PBF used by OSRM."""
import argparse
from array import array
import json
from pathlib import Path
import numpy as np
import osmium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pbf', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    class CostTags(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ids = array('q')
            self.signals = []
            self.edges = set()
        def node(self, node):
            self.ids.append(node.id)
            if len(self.ids) % 5000000 == 0:
                print(f"OSM nodes indexed: {len(self.ids):,}", flush=True)
            if node.tags.get('highway') == 'traffic_signals':
                self.signals.append(node.id)
        def way(self, way):
            value = way.tags.get('toll:motorcar', way.tags.get('toll:motor_vehicle', way.tags.get('toll')))
            if value == 'yes':
                ids = [node.ref for node in way.nodes]
                self.edges.update(zip(ids, ids[1:]))
    tags = CostTags()
    print(f'Extracting OSM cost tags: {args.pbf}', flush=True)
    # libosmium on Windows does not reliably open Unicode paths.
    tags.apply_buffer(args.pbf.read_bytes(), 'pbf')
    args.output.mkdir(parents=True, exist_ok=True)
    ids = np.frombuffer(tags.ids, dtype=np.int64).copy()
    ids.sort()
    np.save(args.output / 'node_ids.npy', ids)
    document = {'source': str(args.pbf), 'source_size': args.pbf.stat().st_size,
                'source_mtime_ns': args.pbf.stat().st_mtime_ns,
                'signals': tags.signals, 'toll_edges': sorted(tags.edges), 'node_count': len(ids)}
    (args.output / 'features.json').write_text(json.dumps(document), encoding='utf-8')
    print(f'Index ready: {len(ids)} nodes, {len(tags.signals)} signals, {len(tags.edges)} toll edges', flush=True)

if __name__ == '__main__':
    main()
