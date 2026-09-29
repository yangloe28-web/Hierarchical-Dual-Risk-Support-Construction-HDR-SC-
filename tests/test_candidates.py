import unittest

from hdr_sc.candidates import generate_candidates


class CandidateGenerationTests(unittest.TestCase):
    def test_enumerates_spaces_below_budget(self) -> None:
        self.assertEqual(len(generate_candidates(32, 1, 500, seed=0)), 32)
        self.assertEqual(len(generate_candidates(32, 2, 500, seed=0)), 496)

    def test_samples_large_space_deterministically(self) -> None:
        first = generate_candidates(32, 4, 500, seed=7)
        second = generate_candidates(32, 4, 500, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 500)
        self.assertEqual(len(set(first)), 500)
        self.assertTrue(all(len(candidate) == 4 for candidate in first))


if __name__ == "__main__":
    unittest.main()
