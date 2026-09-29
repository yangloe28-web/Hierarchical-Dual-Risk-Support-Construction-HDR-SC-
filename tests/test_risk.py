import unittest

from hdr_sc.risk import normal_risk


class NormalRiskTests(unittest.TestCase):
    def test_mean_tail_risk(self) -> None:
        risk = normal_risk([1, 2, 3, 4, 5], tail_fraction=0.20, beta=1.0)
        self.assertAlmostEqual(risk.mean_risk, 3.0)
        self.assertAlmostEqual(risk.tail_risk, 5.0)
        self.assertAlmostEqual(risk.total_risk, 8.0)
        self.assertEqual(risk.tail_count, 1)

    def test_tail_uses_ceiling(self) -> None:
        risk = normal_risk([1, 2, 3, 4, 5, 6], tail_fraction=0.20, beta=0.5)
        self.assertEqual(risk.tail_count, 2)
        self.assertAlmostEqual(risk.tail_risk, 5.5)
        self.assertAlmostEqual(risk.total_risk, 3.5 + 0.5 * 5.5)


if __name__ == "__main__":
    unittest.main()
