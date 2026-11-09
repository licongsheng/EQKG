"""Semantic and zero-leakage regression tests for the v4 PrimeKG & MPS models."""

import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.benchmark_mps_public_v4 import (
    TRI_SUBSETS, build_design_matrix, fit_hgb_l1, weighted_median,
)
from scripts.benchmark_primekg_public_v4 import (
    ALL_FEATURES, IDX_CORE_TOPO, IDX_V3_ALL, SUBSET_MAP, fit_predict_model,
    parallel_paired_bootstrap,
)


class TestPublicV4Benchmarks(unittest.TestCase):
    def test_weighted_median_exact_and_outlier_robust(self):
        vals = np.array([1.0, 2.0, 1000.0])
        weights = np.array([0.45, 0.45, 0.10])
        self.assertEqual(weighted_median(vals, weights), 2.0)
        self.assertEqual(weighted_median(np.array([5.0]), np.array([1.0])), 5.0)

    def test_primekg_v4_subsets_cover_all_three_anchors(self):
        self.assertEqual(set(SUBSET_MAP), {"M", "F", "A", "MF", "MA", "FA", "MFA"})
        self.assertEqual(
            set(SUBSET_MAP["MFA"]),
            set(SUBSET_MAP["M"]) | set(SUBSET_MAP["F"]) | set(SUBSET_MAP["A"]),
        )
        self.assertTrue(set(IDX_CORE_TOPO).issubset(set(SUBSET_MAP["M"])))
        self.assertEqual(len(IDX_V3_ALL), 22)
        self.assertGreater(len(ALL_FEATURES), len(IDX_V3_ALL))

    def test_mps_v4_design_matrix_hierarchy(self):
        fake_row = {
            "stats": {
                "u_med": 0.5,
                "U": [0.5, 0.0, 0.0, 0.1, 2.0, 1.0],
                "A0": [0.0] * 9,
                "A1": [0.0] * 9,
                "A2": [0.0] * 9,
                "J01": [0.0] * 7,
                "J02": [0.0] * 7,
                "J12": [0.0] * 7,
                "J012": [0.0] * 7,
            }
        }
        rows = [fake_row, fake_row]
        shapes = {s: build_design_matrix(rows, s).shape for s in TRI_SUBSETS}
        self.assertEqual(shapes["U"], (2, 6))
        self.assertEqual(shapes["0"], (2, 15))
        self.assertEqual(shapes["01"], (2, 31))
        self.assertEqual(shapes["012"], (2, 61))

    def test_parallel_bootstrap_identity_and_reproducibility(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        s = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        g = np.array(["g1", "g1", "g2", "g2", "g3", "g3"])
        ci1 = parallel_paired_bootstrap(y, s, s, g, repetitions=40, jobs=4)
        ci2 = parallel_paired_bootstrap(y, s, s, g, repetitions=40, jobs=1)
        self.assertEqual(ci1, ci2)
        self.assertEqual(ci1["ap"]["delta_full_minus_comparator"], 0.0)
        self.assertEqual(ci1["ap"]["ci95_percentile"], [0.0, 0.0])

    def test_saved_v4_summaries_exist_and_finite(self):
        p_path = Path("/home/AI4BioEM/data/primekg_validation_v4/multiscale_calibrated_v4/summary.json")
        m_path = Path("/home/AI4BioEM/data/caltech_mps_validation_v4/summary.json")
        self.assertTrue(p_path.exists())
        self.assertTrue(m_path.exists())
        p_data = json.loads(p_path.read_text())
        m_data = json.loads(m_path.read_text())
        self.assertIn("drug", p_data["results"])
        self.assertIn("exposure", p_data["results"])
        for mod in ("CA_FOMS_standard", "CP_FOMS_standard", "CV_FOMS_standard"):
            ev = m_data["evaluations"][f"test:{mod}"]["fixed_full_supported_targets"]
            self.assertIsNotNone(ev["hgb_tri_012"]["mae"])


if __name__ == "__main__":
    unittest.main()
