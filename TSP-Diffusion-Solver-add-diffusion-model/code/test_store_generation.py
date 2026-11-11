"""Run with python -m unittest discover -s code -p test_store_generation.py."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import generate_tsp as generation
import seven_eleven


class StoreGenerationTests(unittest.TestCase):
    def test_unreachable_distances_are_rejected(self):
        response = Mock()
        response.json.return_value = {"code": "Ok", "distances": [[0, None], [10, 0]]}
        with patch.object(generation, "OSRM_SESSION") as session:
            session.post.return_value = response
            with self.assertRaises(generation.UnreachableStoresError):
                generation.get_osrm_distance_matrix([[120, 24], [121, 24]])

    def test_sample_and_checkpoint_preserve_store_identity(self):
        stores = [{"id": f"node/{i}", "name": f"store {i}",
                   "coords": [120 + i * .001, 24], "original_coords": [120 + i * .001, 24]}
                  for i in range(110)]
        catalog = {"stores": stores}
        matrix = np.ones((100, 100), dtype=np.float32)
        np.fill_diagonal(matrix, 0)
        with patch.multiple(generation, NUM_NODES=100, NUM_SAMPLES=1, SOLVE_SECONDS=1,
                            STORE_CATALOG=catalog, CATALOG_HASH="test_catalog"), \
                patch.object(generation, "get_osrm_matrices", return_value=(matrix, matrix * 120, matrix, {"mode": generation.COST_MODE})):
            seed, coords, adj, distances, indices, durations, costs, metadata = generation.solve_single_tsp(42)
            self.assertEqual(len(set(indices.tolist())), 100)
            np.testing.assert_equal(coords, np.asarray([stores[i]["coords"] for i in indices], dtype=np.float32))
            np.testing.assert_equal(adj.sum(axis=0), np.ones(100))
            np.testing.assert_equal(adj.sum(axis=1), np.ones(100))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                generation.save_progress_sample(path, seed, coords, adj, distances, indices, durations, costs, metadata)
                results = generation.load_progress(path)
                np.testing.assert_equal(results[seed][3], indices)
                generation.save_samples(path / "dataset.npz", results)
                with np.load(path / "dataset.npz") as data:
                    self.assertEqual(str(data["node_source"]), "taichung_youbike")
                    self.assertEqual(json.loads(str(data["store_catalog_json"])), catalog)
                    np.testing.assert_equal(data["store_indices"][0], indices)
                with patch.object(generation, "CATALOG_HASH", "different"):
                    with self.assertRaisesRegex(ValueError, "catalog mismatch"):
                        generation.load_progress(path)

    def test_cost_prefers_faster_route_and_preserves_direction(self):
        distances = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        durations = np.array([[0, 60, 120], [180, 0, 120], [120, 120, 0]])
        costs, metadata = generation.build_route_cost(distances, durations)
        self.assertLess(costs[0, 1], costs[0, 2])
        self.assertNotEqual(costs[0, 1], costs[1, 0])
        self.assertAlmostEqual(metadata["duration_km_per_second"], 1 / 120)
        np.testing.assert_equal(np.diag(costs), 0)
        durations[0, 1] = -1
        with self.assertRaises(generation.UnreachableStoresError):
            generation.build_route_cost(distances, durations)

    def test_toll_and_signal_penalties(self):
        distances = np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]])
        durations = distances * 60
        toll = np.zeros((3, 3))
        signals = np.zeros((3, 3))
        toll[0, 1] = 1
        signals[0, 2] = 1
        base, _ = generation.build_route_cost(distances, durations)
        costs, metadata = generation.build_route_cost(distances, durations, toll, signals)
        self.assertAlmostEqual(float(costs[0, 1] - base[0, 1]), .25, places=5)
        self.assertAlmostEqual(float(costs[0, 2] - base[0, 2]), .02, places=5)
        self.assertGreater(costs[0, 1], costs[0, 2])
        self.assertEqual(metadata["signal_counts"][0][2], 1)
        with self.assertRaises(ValueError):
            generation.build_route_cost(distances, durations, toll * 10, signals)

    def test_osm_tags_match_traversed_edges_only(self):
        from osm_route_cost import feature_index, route_features
        elements = [{"type": "node", "id": i, "tags": {"highway": "traffic_signals"} if i == 2 else {}}
                    for i in range(1, 5)]
        elements += [{"type": "way", "id": 10, "nodes": [1, 2, 4], "tags": {"toll": "yes"}},
                     {"type": "way", "id": 11, "nodes": [2, 3], "tags": {"toll": "yes", "toll:motorcar": "no"}}]
        nodes, edges = feature_index(elements)
        toll, signals = route_features({"nodes": [1, 2, 3], "distance": [100, 200]}, nodes, edges)
        self.assertAlmostEqual(toll, .1)
        self.assertEqual(signals, 1)
        self.assertEqual(route_features({"nodes": [3, 2, 1], "distance": [200, 100]}, nodes, edges), (.1, 1))
        with self.assertRaisesRegex(ValueError, "missing"):
            route_features({"nodes": [1, 999], "distance": [100]}, nodes, edges)

    def test_osrm_cost_pipeline_preserves_penalties(self):
        response = Mock()
        response.json.return_value = {"code": "Ok", "distances": [[0, 1000], [1000, 0]],
                                      "durations": [[0, 60], [60, 0]]}
        with patch.object(generation, "OSRM_SESSION") as session, patch.object(
                generation, "get_feature_matrices", return_value=(np.array([[0, 1], [0, 0]]),
                                                                   np.array([[0, 0], [1, 0]]))):
            session.post.return_value = response
            distances, durations, costs, metadata = generation.get_osrm_matrices([[120, 24], [121, 24]])
        self.assertAlmostEqual(float(costs[0, 1]), 1.25)
        self.assertAlmostEqual(float(costs[1, 0]), 1.02, places=5)

    def test_local_pbf_index_drives_cost_matrices(self):
        import osm_route_cost as features
        response = Mock()
        response.json.return_value = {"code": "Ok", "routes": [{"distance": 1000, "duration": 60,
            "legs": [{"annotation": {"nodes": [1, 2, 3], "distance": [500, 500]}}]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "node_ids.npy", np.array([1, 2, 3], dtype=np.int64))
            (path / "features.json").write_text(json.dumps({"signals": [2], "toll_edges": [[1, 2]],
                "source_size": 100, "source_mtime_ns": 1}))
            with patch.dict(features.os.environ, {"TSP_OSM_FEATURE_INDEX": directory}), \
                    patch.object(features, "_LOCAL_INDEX", None), patch.object(features, "CACHE", path / "cache"):
                session = Mock()
                session.get.return_value = response
                distances = np.array([[0, 1], [1, 0]], dtype=float)
                toll, signals = features.get_feature_matrices([[120, 24], [121, 24]], distances,
                                                             distances * 60, session, "http://test")
                np.testing.assert_equal(toll, [[0, .5], [.5, 0]])
                np.testing.assert_equal(signals, [[0, 1], [1, 0]])
                # Different equal-time Table paths must not mix distance with
                # the annotated route's toll/signal features.
                mismatched = np.array([[0, .95], [.95, 0]], dtype=float)
                times = np.array([[0, 61], [61, 0]], dtype=float)
                features.get_feature_matrices([[120, 24], [121, 24]], mismatched,
                                             times, session, 'http://test')
                np.testing.assert_equal(mismatched, distances)
                np.testing.assert_equal(times, distances * 60)
                session.get.reset_mock()
                features.get_feature_matrices([[120, 24], [121, 24]], distances,
                                             distances * 60, session, "http://test")
                session.get.assert_not_called()
            import gc
            gc.collect()

    def test_cache_write_lock_does_not_discard_valid_route(self):
        import osm_route_cost as features
        value = {"distance": 1000, "duration": 60, "toll_km": .5, "signal_counts": 1}
        with tempfile.TemporaryDirectory() as directory, patch.object(features, "CACHE", Path(directory)), \
                patch.object(features.os, "replace", side_effect=PermissionError("Windows file lock")):
            fetch = Mock(return_value=value)
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(features.cached_json("locked-pair", fetch), value)
            fetch.assert_called_once()
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_status_file_lock_does_not_terminate_generation(self):
        import run_cost_generation as runner
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "OUTPUT", Path(directory)), \
                patch.object(runner, "COMPLETED_SAMPLES", 12), \
                patch.object(Path, "replace", side_effect=PermissionError("IDE file lock")), \
                patch.object(runner.time, "sleep"):
            self.assertFalse(runner.status("generating"))
            event = json.loads((Path(directory) / "status_events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["phase"], "generating")
            self.assertEqual(event["completed_samples"], 12)
            self.assertEqual(list(Path(directory).glob("*.tmp.json")), [])

    def test_no_route_http_400_is_classified_for_resampling(self):
        import osm_route_cost as features
        response = Mock()
        response.json.return_value = {"code": "NoRoute", "message": "No route found between points"}
        response.raise_for_status.side_effect = RuntimeError("HTTP 400 must not hide NoRoute")
        with self.assertRaises(features.UnreachableRouteError):
            features.request_json(response)
        response.raise_for_status.assert_not_called()
        table = Mock()
        table.json.return_value = {"code": "Ok", "distances": [[0, 1000], [1000, 0]],
                                  "durations": [[0, 60], [60, 0]]}
        with patch.object(generation, "OSRM_SESSION") as session, patch.object(generation,
                "get_feature_matrices", side_effect=features.UnreachableRouteError("NoRoute")):
            session.post.return_value = table
            with self.assertRaises(generation.UnreachableStoresError):
                generation.get_osrm_matrices([[120, 24], [121, 24]])

    def test_shared_pair_store_import_preserves_directed_features(self):
        import pair_store as shared
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            progress = path / "progress"
            progress.mkdir()
            coords = [[120, 24], [121, 24], [122, 24]]
            distances = np.array([[0, 3], [4, 0]], dtype=np.float32)
            np.savez(progress / "sample_000000.npz", catalog_hash="test", cost_mode=generation.COST_MODE,
                     store_indices=[2, 0], distances=distances, durations=distances * 60,
                     toll_km=distances / 2, signal_counts=np.array([[0, 2], [1, 0]]))
            with patch.dict(shared.os.environ, {"TSP_PAIR_STORE_DIRECTORY": str(path / "shared")}), \
                    patch.object(shared, "local_feature_index", return_value=(None, None, "test")):
                store = shared.create_pair_store({"stores": [{"coords": c} for c in coords]},
                                                 "test", "http://test", progress)
                try:
                    with patch.object(store, "fetch_pair") as fetch:
                        result = store.matrices([coords[2], coords[0]])
                        fetch.assert_not_called()
                    np.testing.assert_equal(result[0], distances)
                    np.testing.assert_equal(result[1], distances * 60)
                    np.testing.assert_equal(result[2], distances / 2)
                    np.testing.assert_equal(result[3], [[0, 2], [1, 0]])
                    store.ready[2, 0] = 2
                    with self.assertRaises(shared.UnreachableRouteError):
                        store.matrices([coords[2], coords[0]])
                finally:
                    store.values._mmap.close()
                    store.ready._mmap.close()

    def test_bad_snaps_and_duplicate_positions_are_excluded(self):
        candidates = [{"id": str(i), "original_coords": [120, 24]} for i in range(3)]
        replies = [Mock(), Mock(), Mock()]
        for response, distance in zip(replies, [2, 2, 200]):
            response.json.return_value = {"code": "Ok", "waypoints": [
                {"distance": distance, "location": [120, 24]}]}
        with patch.object(seven_eleven, "load_candidates", return_value=candidates), \
                patch.object(seven_eleven.requests, "Session") as factory, \
                patch.object(seven_eleven, "CATALOG_PATH") as catalog_path:
            catalog_path.read_text.return_value = '{"source": "test catalog"}'
            factory.return_value.__enter__.return_value.get.side_effect = replies
            with patch.dict(seven_eleven.os.environ, {"TSP_STORE_MAX_SNAP_METERS": "100"}):
                catalog = seven_eleven.prepare_stores("http://example.test", 1)
        self.assertEqual(len(catalog["stores"]), 1)


if __name__ == "__main__":
    unittest.main()
