"""Shared, directed route features reused across random TSP station subsets."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np
import requests

from osm_route_cost import cached_json, local_feature_index, request_json, route_features, UnreachableRouteError
from road_requests import configure_session

PAIR_STORE = None


class PairStore:
    def __init__(self, path):
        self.path = Path(path)
        self.document = json.loads((self.path / 'catalog.json').read_text(encoding='utf-8'))
        self.coords = np.asarray(self.document['coords'], dtype=np.float32)
        self.lookup = {tuple(point): i for i, point in enumerate(self.coords.tolist())}
        self.values = np.load(self.path / 'values.npy', mmap_mode='r+')
        self.ready = np.load(self.path / 'ready.npy', mmap_mode='r+')
        self.local = threading.local()
        self.features = local_feature_index()

    def fetch_pair(self, pair):
        i, j = pair
        state = int(self.ready[i, j])
        if state == 1:
            return
        if state == 2:
            raise UnreachableRouteError(f'Unreachable catalog pair {i}->{j}')
        if not hasattr(self.local, 'session'):
            self.local.session = configure_session(requests.Session())
        nodes, edges, fingerprint = self.features
        coordinates = ';'.join(f'{float(lon):.8f},{float(lat):.8f}' for lon, lat in (self.coords[i], self.coords[j]))
        url = f"{self.document['osrm_url']}/route/v1/driving/{coordinates}"
        def fetch():
            data = request_json(self.local.session.get(url, params={'annotations': 'nodes,distance',
                'overview': 'false', 'alternatives': 'false'}, timeout=120))
            route = data['routes'][0]
            toll, signals = route_features(route['legs'][0]['annotation'], nodes, edges)
            return {'distance': route['distance'], 'duration': route['duration'],
                    'toll_km': toll, 'signal_counts': signals}
        try:
            feature = cached_json('local-v2:' + fingerprint + url, fetch, persist=False)
        except UnreachableRouteError:
            self.ready[i, j] = 2
            self.ready.flush()
            raise
        value = [feature['distance'] / 1000, feature['duration'], feature['toll_km'], feature['signal_counts']]
        if not np.isfinite(value).all() or min(value) < 0:
            raise ValueError(f'Invalid catalog route features for {i}->{j}')
        # Publish readiness only after every feature has been assigned.
        self.values[i, j] = value
        self.ready[i, j] = 1

    def matrices(self, coords):
        indices = np.asarray([self.lookup[tuple(point)] for point in np.asarray(coords, dtype=np.float32).tolist()])
        grid = np.ix_(indices, indices)
        state = self.ready[grid].copy()
        if (state == 2).any():
            raise UnreachableRouteError('Station subset contains a known unreachable pair')
        rows, cols = np.where(state == 0)
        pairs = list(zip(indices[rows].tolist(), indices[cols].tolist()))
        if pairs:
            print(f'New road pairs: {len(pairs)} (worker {os.getpid()})', flush=True)
            with ThreadPoolExecutor(max_workers=int(os.environ.get('TSP_ROUTE_THREADS', '8'))) as pool:
                for start in range(0, len(pairs), 256):
                    list(pool.map(self.fetch_pair, pairs[start:start + 256]))
                    if start % 2048 == 0:
                        print(f'Road pairs filled: {min(start + 256, len(pairs))}/{len(pairs)} (worker {os.getpid()})', flush=True)
            self.values.flush()
            self.ready.flush()
        values = self.values[grid].copy()
        return tuple(values[:, :, field].astype(np.float32) for field in range(4))


def create_pair_store(catalog, catalog_hash, osrm_url, progress_path):
    base = Path(os.environ['TSP_PAIR_STORE_DIRECTORY'])
    _, _, fingerprint = local_feature_index()
    identity = hashlib.sha256((catalog_hash + fingerprint + osrm_url +
        os.environ.get('TSP_OSM_COST_CACHE_VERSION', 'v1')).encode()).hexdigest()[:20]
    path = base / identity
    path.mkdir(parents=True, exist_ok=True)
    catalog_path = path / 'catalog.json'
    size = len(catalog['stores'])
    if not catalog_path.exists():
        values = np.lib.format.open_memmap(path / 'values.npy', mode='w+', dtype=np.float32, shape=(size, size, 4))
        ready = np.lib.format.open_memmap(path / 'ready.npy', mode='w+', dtype=np.uint8, shape=(size, size))
        values[:] = 0
        ready[:] = 0
        np.fill_diagonal(ready, 1)
        values.flush()
        ready.flush()
        document = {'catalog_hash': catalog_hash, 'osm_fingerprint': fingerprint, 'osrm_url': osrm_url,
                    'coords': [store['coords'] for store in catalog['stores']]}
        catalog_path.write_text(json.dumps(document), encoding='utf-8')
    store = PairStore(path)
    imported = 0
    for checkpoint in sorted(progress_path.glob('sample_*.npz')):
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved['catalog_hash']) != catalog_hash or str(saved['cost_mode']) != 'osm_distance_duration_toll_signals_v3':
                raise ValueError('Cannot import incompatible checkpoint route features')
            indices = saved['store_indices']
            grid = np.ix_(indices, indices)
            values = np.stack([saved['distances'], saved['durations'], saved['toll_km'], saved['signal_counts']], axis=-1)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'Invalid route features in {checkpoint}')
            store.values[grid] = values
            store.ready[grid] = 1
            imported += 1
    store.values.flush()
    store.ready.flush()
    print(f'Shared road matrix ready: {np.count_nonzero(store.ready == 1):,}/{size * size:,} pairs; imported {imported} checkpoints', flush=True)
    os.environ['TSP_PAIR_STORE_PATH'] = str(path)
    return store


def get_pair_store():
    global PAIR_STORE
    if PAIR_STORE is None:
        PAIR_STORE = PairStore(os.environ['TSP_PAIR_STORE_PATH'])
    return PAIR_STORE
