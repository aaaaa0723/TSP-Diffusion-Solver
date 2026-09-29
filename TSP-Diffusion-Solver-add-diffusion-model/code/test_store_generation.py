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
                patch.object(generation, "get_osrm_distance_matrix", return_value=matrix):
            seed, coords, adj, distances, indices = generation.solve_single_tsp(42)
            self.assertEqual(len(set(indices.tolist())), 100)
            np.testing.assert_equal(coords, np.asarray([stores[i]["coords"] for i in indices], dtype=np.float32))
            np.testing.assert_equal(adj.sum(axis=0), np.ones(100))
            np.testing.assert_equal(adj.sum(axis=1), np.ones(100))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                generation.save_progress_sample(path, seed, coords, adj, distances, indices)
                results = generation.load_progress(path)
                np.testing.assert_equal(results[seed][3], indices)
                generation.save_samples(path / "dataset.npz", results)
                with np.load(path / "dataset.npz") as data:
                    self.assertEqual(str(data["node_source"]), "taichung_7eleven_osm")
                    self.assertEqual(json.loads(str(data["store_catalog_json"])), catalog)
                    np.testing.assert_equal(data["store_indices"][0], indices)
                with patch.object(generation, "CATALOG_HASH", "different"):
                    with self.assertRaisesRegex(ValueError, "catalog mismatch"):
                        generation.load_progress(path)

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
