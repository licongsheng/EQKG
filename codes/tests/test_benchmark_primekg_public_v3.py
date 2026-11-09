"""Correctness/leakage invariants, deliberately no desired-performance assertions."""
import csv
import inspect
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.benchmark_primekg_public_v3 import (
    Covariates, FEATURES, FeatureMatrix, binary_metrics, drug_family,
    exposure_family, frozen_memberships, load_graph, macro_metrics,
    make_dataset, paired_bootstrap, select_baseline, subset_scores,
)


class PrimeKGPublicV3Tests(unittest.TestCase):
    def covariates(self):
        cov = Covariates()
        cov.drug["q"] = {"p1", "p2"}
        cov.exposure["e"] = {"p1", "p2"}
        cov.disease["d"] = {"p1"}
        cov.protein_disease["p1"] = {"d"}
        cov.annotations["mf"]["p1"] = {"f"}
        cov.annotations["present"]["p1"] = {"a"}
        cov.annotations["bp"]["p1"] = {"b"}
        cov.exposure_annotations["bp"]["e"] = {"b"}
        return cov

    def test_ties_and_permutation_match_sklearn(self):
        y = np.array([1, 0, 1, 0, 0, 1])
        s = np.array([2, 2, 1, 1, 1, 0])
        actual = binary_metrics(y, s)
        self.assertEqual(actual["auroc"], roc_auc_score(y, s))
        self.assertEqual(actual["ap"], average_precision_score(y, s))
        self.assertEqual(actual, binary_metrics(y[::-1], s[::-1]))
        self.assertEqual(binary_metrics(y, np.ones(6)), {"auroc": 0.5, "ap": 0.5})

    def test_single_class_explicitly_undefined(self):
        self.assertEqual(binary_metrics([0, 0], [1, 2]), {"auroc": None, "ap": None})
        self.assertEqual(binary_metrics([1, 1], [1, 2]), {"auroc": None, "ap": 1.0})
        m = macro_metrics(np.array([1, 0, 0]), np.ones(3), np.array(["a", "a", "b"]))
        self.assertEqual(m["auroc_eligible"], 1)
        self.assertEqual(m["ap_eligible"], 1)

    def test_frozen_split_label_free_stable_and_disjoint(self):
        mapping = {"a": set(), "b": set(), "c": set(), "d": set()}
        keys = {"a": "family1", "b": "family1", "c": "family2", "d": "family3"}
        rows = frozen_memberships(mapping, keys)
        self.assertEqual(rows, frozen_memberships(dict(reversed(list(mapping.items()))), keys))
        sets = [{r["group_id"] for r in rows if r["split"] == split} for split in ("dev", "test")]
        self.assertFalse(sets[0] & sets[1])
        self.assertEqual(rows[0]["split"], rows[1]["split"])
        self.assertEqual(drug_family("a", {"atc_3": "Drug A is pharmacologically related to Class X."}),
                 drug_family("b", {"atc_3": "Drug B is pharmacologically related to Class X."}))
        self.assertNotEqual(drug_family("a", {"atc_3": "Drug A is pharmacologically related to Class X."}),
                    drug_family("b", {"atc_3": "Drug B is pharmacologically related to Class Y."}))
        self.assertEqual(drug_family("a", {}), "id_a")
        self.assertEqual(exposure_family("Polychlorinated biphenyl"), "biphenyl")

    def test_original_v2_formulas_and_all_subsets(self):
        extractor = FeatureMatrix(self.covariates())
        drug = np.array(extractor.pair("drug", "q", "d"))
        exp = np.array(extractor.pair("exposure", "e", "d"))
        np.testing.assert_allclose(drug[-3:], [math.log1p(3000) / math.sqrt(2), 0.3 / math.sqrt(2), 0.25])
        np.testing.assert_allclose(exp[-3:], [1 / math.log(3), 1.5 * math.log(2), 0.15])
        scores = subset_scores(np.vstack([drug, exp]))
        self.assertEqual(len(scores), 8)
        np.testing.assert_allclose(scores["graph_MFA"], scores["graph_M"] + scores["graph_FA"])
        self.assertEqual(len(drug), len(FEATURES))
        self.assertFalse(hasattr(extractor.cov, "labels"))

    def test_label_exclusivity_and_unknown_handling(self):
        cov = self.covariates()
        members = frozen_memberships(cov.drug, {"q": "family"})
        labels = {k: set() for k in ("indication", "contraindication", "off-label use", "exposure_disease")}
        labels["indication"].add(("q", "d"))
        labels["contraindication"].add(("q", "d"))
        rows, _, _ = make_dataset(cov, labels, "drug", members, FeatureMatrix(cov))
        self.assertEqual(rows, [])  # Conflicting labels excluded, never invented as negatives.
        members = frozen_memberships(cov.exposure, {"e": "family"})
        rows, _, _ = make_dataset(cov, labels, "exposure", members, FeatureMatrix(cov))
        self.assertEqual(rows[0]["label_meaning"], "unlabeled")

    def test_task_edges_both_directions_do_not_enter_features(self):
        fields = ["relation", "display_relation", "x_index", "x_type", "x_name", "x_source",
                  "y_index", "y_type", "y_name", "y_source"]
        def edge(rel, x, xt, y, yt, display=""):
            return [rel, display, x, xt, "", "src", y, yt, "", "src"]
        base = [edge("drug_protein", "q", "drug", "p", "gene/protein", "target"),
                edge("disease_protein", "p", "gene/protein", "d", "disease")]
        extras = [edge(rel, "q" if rel != "exposure_disease" else "e", "drug" if rel != "exposure_disease" else "exposure", "d", "disease")
                  for rel in ("indication", "contraindication", "off-label use", "exposure_disease")]
        extras += [r[:2] + r[6:10] + r[2:6] for r in extras]
        with tempfile.TemporaryDirectory() as tmp:
            vectors = []
            for i, rows in enumerate((base, base + extras)):
                path = Path(tmp) / f"{i}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    writer.writerows(rows)
                cov, labels, _, sources = load_graph(path)
                vectors.append(FeatureMatrix(cov).pair("drug", "q", "d"))
                self.assertEqual(len(sources), 2)
                if i:
                    self.assertEqual(labels["indication"], {("q", "d")})
            np.testing.assert_equal(*vectors)

    def test_selection_api_cannot_read_test_labels(self):
        self.assertEqual(list(inspect.signature(select_baseline).parameters),
                         ["X_dev", "y_dev", "groups_dev", "jobs"])

    def test_group_oof_selection_uses_only_development_scores(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(36, len(FEATURES)))
        y = np.tile([0, 1], 18)
        groups = np.repeat(["a", "b", "c", "d", "e", "f"], 6)
        selected, scores, audit = select_baseline(X, y, groups, 1)
        for fold in audit["folds"]:
            self.assertFalse(set(fold["train_groups"]) & set(fold["validation_groups"]))
        expected = max(audit["eligible_order"], key=lambda name: average_precision_score(y, scores[name]))
        self.assertEqual(selected, expected)
        self.assertTrue(all(np.isfinite(s).all() for s in scores.values()))

    def test_bootstrap_weighted_metrics_equal_repeated_entire_groups(self):
        y = np.array([1, 0, 1, 0, 0])
        s = np.array([1, 1, 2, 0, 2])
        weights = np.array([2, 2, 0, 0, 1])
        idx = np.repeat(np.arange(len(y)), weights)
        self.assertEqual(binary_metrics(y, s, weights), binary_metrics(y[idx], s[idx]))

    def test_paired_group_bootstrap_identity_and_reproducibility(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        s = np.array([1., 1., 3., 2., 0., 2.])
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        ci = paired_bootstrap(y, s, s, groups, 30)
        self.assertEqual(ci, paired_bootstrap(y, s, s, groups, 30))
        for metric in ci.values():
            self.assertEqual(metric["delta_full_minus_comparator"], 0)
            self.assertEqual(metric["ci95_percentile"], [0, 0])


if __name__ == "__main__":
    unittest.main()