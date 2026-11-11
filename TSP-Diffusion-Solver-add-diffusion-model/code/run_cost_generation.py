"""Generate the toll/signal dataset with logs, status and validation; no training."""
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import re
import time
import uuid
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results' / 'generation_toll_signals'
DATASET = ROOT / 'results' / 'tsp_dataset_youbike_300nodes_toll_signals.npz'
SMOKE = OUTPUT / 'smoke_30nodes.npz'
COST_MODE = 'osm_distance_duration_toll_signals_v3'
COMPLETED_SAMPLES = 0


def status(phase, **details):
    document = {'phase': phase, 'updated_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'dataset': str(DATASET), 'runner_pid': os.getpid(), 'num_samples': 1000,
                'num_nodes': 300, 'cost_mode': COST_MODE, **details}
    document['completed_samples'] = max(COMPLETED_SAMPLES, details.get('completed_samples', 0))
    payload = json.dumps(document, indent=2, ensure_ascii=False)
    temporary = OUTPUT / f'status.{uuid.uuid4().hex}.tmp.json'
    try:
        temporary.write_text(payload, encoding='utf-8')
        for attempt in range(5):
            try:
                temporary.replace(OUTPUT / 'status.json')
                return True
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.1 * (attempt + 1))
    except OSError as exc:
        # Monitoring must never terminate the dataset generation process.
        print(f'Status update deferred: {exc}', flush=True)
        try:
            with (OUTPUT / 'status_events.jsonl').open('a', encoding='utf-8') as events:
                events.write(json.dumps(document, ensure_ascii=False) + '\n')
        except OSError:
            pass
        return False
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def stage(name, env):
    global COMPLETED_SAMPLES
    with (OUTPUT / f'{name}.log').open('a', encoding='utf-8') as log:
        log.write(f"\nRun started: {datetime.now(timezone(timedelta(hours=8))).isoformat()}\n")
        log.flush()
        with subprocess.Popen([sys.executable, '-B', '-u', 'code/generate_tsp.py'], cwd=ROOT,
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            status(name, process_pid=process.pid, log=str(OUTPUT / f'{name}.log'))
            completed = COMPLETED_SAMPLES
            last_error = ""
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.strip():
                    last_error = line.strip()
                match = re.search(r"Progress (\d+)/(\d+)", line)
                if match:
                    completed = max(completed, int(match.group(1)))
                    if name == "generating":
                        COMPLETED_SAMPLES = completed
                if 'Road cost rows:' in line or 'Road pairs filled:' in line or 'Shared road matrix ready:' in line or 'Progress ' in line or 'Generating ' in line:
                    status(name, process_pid=process.pid, latest_progress=line.strip(), completed_samples=completed,
                           log=str(OUTPUT / f'{name}.log'))
            if process.wait() != 0:
                raise RuntimeError(f'{name} failed: {last_error}; inspect {OUTPUT / (name + ".log")}')


def validate(path, count, nodes):
    with np.load(path, allow_pickle=False) as data:
        assert str(data['cost_mode']) == COST_MODE
        arrays = {key: data[key] for key in ('distances', 'durations', 'costs', 'toll_km', 'signal_counts', 'adjacencies', 'cost_metadata_json')}
        for key in ('distances', 'durations', 'costs', 'toll_km', 'signal_counts'):
            array = arrays[key]
            assert array.shape == (count, nodes, nodes), key
            assert np.isfinite(array).all() and (array >= 0).all(), key
        for index in range(count):
            costs = arrays['costs'][index]
            metadata = json.loads(str(arrays['cost_metadata_json'][index]))
            expected = (.35 * arrays['distances'][index] + .65 * arrays['durations'][index] *
                        metadata['duration_km_per_second'] + .25 * arrays['toll_km'][index] +
                        .02 * arrays['signal_counts'][index])
            np.fill_diagonal(expected, 0)
            np.testing.assert_allclose(costs, expected, rtol=2e-6, atol=2e-6)
            adjacency = arrays['adjacencies'][index]
            assert (adjacency.sum(axis=0) == 1).all() and (adjacency.sum(axis=1) == 1).all()
            node, visited = 0, set()
            for _ in range(nodes):
                assert node not in visited
                visited.add(node)
                node = int(np.argmax(adjacency[node]))
            assert node == 0
        return {'samples': count, 'nodes': nodes,
                'toll_pairs': int(np.count_nonzero(arrays['toll_km'])),
                'signal_pairs': int(np.count_nonzero(arrays['signal_counts'])),
                'bytes': path.stat().st_size}


def main():
    global COMPLETED_SAMPLES
    OUTPUT.mkdir(parents=True, exist_ok=True)
    progress_dirs = list(DATASET.parent.glob(f"{DATASET.stem}.progress_1000x300_30s_*_{COST_MODE}"))
    COMPLETED_SAMPLES = max((sum(1 for entry in directory.glob("sample_*.npz"))
                             for directory in progress_dirs), default=0)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'TSP_NUM_SAMPLES': '1000',
           'TSP_NUM_NODES': '300', 'TSP_SOLVE_SECONDS': '30', 'TSP_NUM_WORKERS': os.environ.get('TSP_NUM_WORKERS', '16'),
           'TSP_CHECKPOINT_EVERY': '1', 'TSP_DATASET_PATH': str(DATASET),
           'TSP_OSRM_URL': 'http://localhost:5000',
           'TSP_OSM_FEATURE_INDEX': str(ROOT / 'data' / 'osm_cost_index_260925'),
           'TSP_OSM_COST_CACHE_VERSION': 'taiwan-260925-v2',
           'TSP_PAIR_STORE_DIRECTORY': str(ROOT / 'data' / 'osm_pair_store'),
           'TSP_ROUTE_THREADS': os.environ.get('TSP_ROUTE_THREADS', '2')}
    try:
        for attempt in range(36):
            try:
                response = requests.get(env['TSP_OSRM_URL'] + '/nearest/v1/driving/120.67,24.15', timeout=3)
                response.raise_for_status()
                if response.json().get('code') == 'Ok':
                    break
            except requests.RequestException:
                pass
            status('waiting_osrm', service=env['TSP_OSRM_URL'])
            time.sleep(5)
        else:
            raise RuntimeError('OSRM service did not become ready within startup timeout')
        while not (Path(env["TSP_OSM_FEATURE_INDEX"]) / "features.json").exists():
            status("indexing_osm", feature_index=env["TSP_OSM_FEATURE_INDEX"])
            time.sleep(5)
        smoke_env = {**env, 'TSP_NUM_SAMPLES': '1', 'TSP_NUM_NODES': '30',
                     'TSP_SOLVE_SECONDS': '1', 'TSP_NUM_WORKERS': '1',
                     'TSP_DATASET_PATH': str(SMOKE)}
        if not SMOKE.exists():
            stage('smoke', smoke_env)
        status('smoke_validated', validation=validate(SMOKE, 1, 30))
        if not DATASET.exists():
            stage('generating', env)
        status('validating')
        result = validate(DATASET, 1000, 300)
        COMPLETED_SAMPLES = 1000
        status('complete', validation=result)
    except BaseException as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
