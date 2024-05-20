import math
import itertools
import unittest

from Oinfo import chunked_combinations

class TestChunkedCombinations(unittest.TestCase):
    def test_chunked_combinations(self):
        n = 10
        k = 3
        chunk_size = 5

        all_combinations = list(itertools.combinations(range(n), k))
        total_combinations = math.comb(n, k)

        # Verify the total number of combinations
        self.assertEqual(total_combinations, len(all_combinations), f"Expected {total_combinations} combinations, but got {len(all_combinations)}.")

        # Verify chunked combinations
        chunked_combinations_list = []
        for chunk in chunked_combinations(n, k, chunk_size):
            chunked_combinations_list.extend(chunk)

        self.assertEqual(len(chunked_combinations_list), total_combinations, f"Expected {total_combinations} total combinations, but got {len(chunked_combinations_list)}.")

        for comb in chunked_combinations_list:
            self.assertIn(tuple(comb), all_combinations, f"Combination {comb} not found in expected combinations.")

if __name__ == '__main__':
    unittest.main()