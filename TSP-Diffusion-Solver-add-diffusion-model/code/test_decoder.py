import unittest
import numpy as np

from evaluation import directed_local_search, road_multistart_decoder, calculate_path_distance


class DirectedDecoderTests(unittest.TestCase):
    def test_one_move_matches_brute_force_directed_neighborhood(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            n = 8
            distance = rng.uniform(.1, 100, (n, n))
            route = [0] + rng.permutation(np.arange(1, n)).tolist()
            candidates = [route]
            for i in range(1, n):
                for j in range(i + 1, n):
                    candidates.append(route[:i] + route[i:j + 1][::-1] + route[j + 1:])
            for length in (1, 2, 3):
                for i in range(1, n - length + 1):
                    block = route[i:i + length]
                    rest = route[:i] + route[i + length:]
                    for after in range(len(rest)):
                        candidates.append(rest[:after + 1] + block + rest[after + 1:])
            expected = min(calculate_path_distance(p + [0], distance) for p in candidates)
            actual = directed_local_search(route + [0], distance, max_moves=1)
            self.assertAlmostEqual(calculate_path_distance(actual, distance), expected, places=8)

    def test_search_preserves_cycle_and_never_worsens_initial(self):
        rng = np.random.default_rng(11)
        for n in (3, 10, 30):
            distance = rng.uniform(.1, 100, (n, n))
            initial = list(range(n)) + [0]
            result = road_multistart_decoder(rng.uniform(0, 1, (n, n)), distance, initial)
            self.assertEqual(sorted(result[:-1]), list(range(n)))
            self.assertEqual(result[0], result[-1])
            self.assertLessEqual(calculate_path_distance(result, distance),
                                 calculate_path_distance(initial, distance) + 1e-8)


if __name__ == "__main__":
    unittest.main()
