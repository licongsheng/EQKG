#!/usr/bin/env python3
"""Refined Tri-Anchor Calibrated & Multi-Scale PrimeKG Benchmark (v4).

Strictly adheres to all v3 leakage-audit, family-disjoint split, and zero-test-peeking
invariants, while resolving the three modeling bottlenecks diagnosed in v3:
1. Disease-identity shortcut control: explicitly separates topology-only baselines
   from full same-information learners, and evaluates whether multi-scale Tri-Anchor
   mechanistic features improve over core topology under the exact same learner family.
2. Calibrated Tri-Anchor Regime Model (CTAR): replaces uncalibrated 1:1:1 raw sum
   with a development-fitted standardized logistic calibration over the 3 anchor blocks
   plus an interaction-aware Gradient Boosted / ExtraTrees Tri-Anchor ensemble.
3. Multi-scale biological & pharmacokinetic feature completion: incorporates
   protein-protein interaction (PPI) neighborhood convergence, pathway & biological
   process triangulation, drug side-effect/disease-phenotype alignment, and
   pharmacokinetic/ADME (enzyme, transporter, carrier, half-life, compartment/BBB)
   attributes already present in PrimeKG without reading any indication/label fields.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = (Path(__file__).resolve().parents[4] if Path(__file__).resolve().parents[1].name == 'codes' else Path(__file__).resolve().parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.benchmark_primekg_public_v3 import (
    ANNOTATIONS, CORE, EXP_ANNOTATIONS, FEATURES as V3_FEATURES,
    HEURISTICS, LABELS, PRIMITIVES, SEED, FeatureMatrix as V3FeatureMatrix,
    binary_metrics, drug_family, exposure_family, frozen_memberships,
    load_graph as load_v3_graph, macro_metrics, metrics, oriented, sha256, union,
)
from scripts.validate_primekg_rigorous_v2 import classify_disease, parse_drug_profile_v2

ROOT = (Path(__file__).resolve().parents[4] if Path(__file__).resolve().parents[1].name == 'codes' else Path(__file__).resolve().parents[1])
DEFAULT_DATA = Path("/data/AI4BioEM/PrimmeKG_dataverse")
OUT_ROOT = ROOT / "data/primekg_validation_v4"

# Anchor-grouped multi-scale columns added in v4 (on top of the 22 v3 columns):
# Anchor M (Molecular & Physicochemical / PPI):
ANCHOR_M_EXTRA = [
    "m_ppi_overlap", "m_ppi_jaccard", "m_min_bridge_deg_inv",
    "m_mw", "m_tpsa", "m_clogp", "m_has_physchem",
    "m_bbb_cns_block", "m_local_skin_mismatch", "m_local_mucosal_mismatch", "m_domain_match",
]
# Anchor F (Functional, Pathway, Biological Process & Phenotype/Side-Effect):
ANCHOR_F_EXTRA = [
    "f_pathway_overlap", "f_pathway_ratio", "f_bridge_pathways",
    "f_bioproc_overlap", "f_bioproc_ratio", "f_bridge_bioprocs",
    "f_sideeffect_pheno_overlap", "f_sideeffect_pheno_ratio", "f_sideeffect_count",
    "f_sideeffect_neg_pheno_overlap", "f_bridge_pheno_overlap",
]
# Anchor A (Anatomical, Subcellular & Pharmacokinetic/ADME Persistence):
ANCHOR_A_EXTRA = [
    "a_anat_present_ratio", "a_anat_Selectivity", "a_cellcomp_overlap", "a_cellcomp_ratio",
    "a_enzyme_count", "a_transporter_count", "a_carrier_count",
    "a_adme_disease_overlap", "a_adme_ppi_disease_overlap",
    "a_log_halflife_hours", "a_has_halflife", "a_invitro_systemic_mismatch",
]

ALL_FEATURES = V3_FEATURES + ANCHOR_M_EXTRA + ANCHOR_F_EXTRA + ANCHOR_A_EXTRA

# Column indices for the 3 anchors in the full multi-scale space:
IDX_CORE_TOPO = list(range(len(CORE)))  # [0..9]
IDX_V3_ALL = list(range(len(V3_FEATURES)))  # [0..21]
IDX_V3_BLOCKS = [19, 20, 21]

IDX_M_FULL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19] + [
    ALL_FEATURES.index(k) for k in ANCHOR_M_EXTRA
]
IDX_F_FULL = [10, 11, 15, 16, 17, 20] + [
    ALL_FEATURES.index(k) for k in ANCHOR_F_EXTRA
]
IDX_A_FULL = [12, 13, 14, 18, 21] + [
    ALL_FEATURES.index(k) for k in ANCHOR_A_EXTRA
]

SUBSET_MAP = {
    "M": sorted(set(IDX_M_FULL)),
    "F": sorted(set(IDX_F_FULL)),
    "A": sorted(set(IDX_A_FULL)),
    "MF": sorted(set(IDX_M_FULL) | set(IDX_F_FULL)),
    "MA": sorted(set(IDX_M_FULL) | set(IDX_A_FULL)),
    "FA": sorted(set(IDX_F_FULL) | set(IDX_A_FULL)),
    "MFA": sorted(set(IDX_M_FULL) | set(IDX_F_FULL) | set(IDX_A_FULL)),
}


def load_multiscale_graph(data_dir):
    """Load v3 graph plus admissible PPI, pathway, phenotype, and ADME relations."""
    cov, labels, exp_names, sources = load_v3_graph(data_dir / "kg.csv")
    extra = {
        "drug_enzyme": defaultdict(set),
        "drug_transporter": defaultdict(set),
        "drug_carrier": defaultdict(set),
        "drug_sideeffect": defaultdict(set),
        "dis_pheno_pos": defaultdict(set),
        "dis_pheno_neg": defaultdict(set),
        "prot_pheno": defaultdict(set),
        "prot_pathway": defaultdict(set),
        "prot_ppi": defaultdict(set),
        "node_names": {},
    }
    with (data_dir / "kg.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rel = row["relation"]
            extra["node_names"][row["x_index"]] = row["x_name"]
            extra["node_names"][row["y_index"]] = row["y_name"]
            if rel in LABELS:
                continue
            if rel == "drug_protein":
                pair = oriented(row, "drug", "gene/protein")
                if pair:
                    drel = row["display_relation"]
                    if drel == "enzyme":
                        extra["drug_enzyme"][pair[0]].add(pair[1])
                    elif drel == "transporter":
                        extra["drug_transporter"][pair[0]].add(pair[1])
                    elif drel == "carrier":
                        extra["drug_carrier"][pair[0]].add(pair[1])
            elif rel == "drug_effect":
                pair = oriented(row, "drug", "effect/phenotype")
                if pair:
                    extra["drug_sideeffect"][pair[0]].add(pair[1])
            elif rel == "disease_phenotype_positive":
                pair = oriented(row, "disease", "effect/phenotype")
                if pair:
                    extra["dis_pheno_pos"][pair[0]].add(pair[1])
            elif rel == "disease_phenotype_negative":
                pair = oriented(row, "disease", "effect/phenotype")
                if pair:
                    extra["dis_pheno_neg"][pair[0]].add(pair[1])
            elif rel == "phenotype_protein":
                pair = oriented(row, "gene/protein", "effect/phenotype")
                if pair:
                    extra["prot_pheno"][pair[0]].add(pair[1])
            elif rel == "pathway_protein":
                pair = oriented(row, "gene/protein", "pathway")
                if pair:
                    extra["prot_pathway"][pair[0]].add(pair[1])
            elif rel == "protein_protein":
                u, v = row["x_index"], row["y_index"]
                extra["prot_ppi"][u].add(v)
                extra["prot_ppi"][v].add(u)

    drug_raw = {}
    with (data_dir / "drug_features.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            drug_raw[row["node_index"]] = row

    dis_raw = {}
    with (data_dir / "disease_features.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dis_raw.setdefault(row["node_index"], row)

    return cov, labels, exp_names, sources, extra, drug_raw, dis_raw


class MultiScaleExtractor:
    """Computes both exact v3 22-col features and v4 multi-scale Tri-Anchor features."""

    def __init__(self, cov, extra, drug_raw, dis_raw):
        self.v3 = V3FeatureMatrix(cov)
        self.cov = cov
        self.extra = extra
        self.drug_profiles = {
            d: parse_drug_profile_v2(d, drug_raw.get(d, {})) for d in cov.drug
        }
        self.dis_profiles = {}
        for d, ps in cov.disease.items():
            self.dis_profiles[d] = {
                "pw": union(extra["prot_pathway"], ps),
                "bp": self.v3.profiles[d]["bp"],
                "mf": self.v3.profiles[d]["mf"],
                "cc": self.v3.profiles[d]["cc"],
                "present": self.v3.profiles[d]["present"],
                "absent": self.v3.profiles[d]["absent"],
                "pheno": extra["dis_pheno_pos"].get(d, set()) | union(extra["prot_pheno"], ps),
                "neg_pheno": extra["dis_pheno_neg"].get(d, set()),
                "cls": classify_disease(d, extra["node_names"].get(d, ""), dis_raw.get(d, {})),
            }
        self.query_cache = {}
        for task in ("drug", "exposure"):
            mapping = getattr(cov, task)
            for q, ps in mapping.items():
                adme = (
                    extra["drug_enzyme"].get(q, set())
                    | extra["drug_transporter"].get(q, set())
                    | extra["drug_carrier"].get(q, set())
                ) if task == "drug" else set()
                self.query_cache[(task, q)] = {
                    "ppi": union(extra["prot_ppi"], ps),
                    "pw": union(extra["prot_pathway"], ps),
                    "bp": union(cov.annotations["bp"], ps),
                    "mf": union(cov.annotations["mf"], ps),
                    "cc": union(cov.annotations["cc"], ps),
                    "pheno": union(extra["prot_pheno"], ps),
                    "adme": adme,
                    "adme_ppi": union(extra["prot_ppi"], adme) if adme else set(),
                    "se": extra["drug_sideeffect"].get(q, set()) if task == "drug" else set(),
                }

    def pair(self, task, query, disease):
        base22 = self.v3.pair(task, query, disease)
        cov = self.cov
        extra = self.extra
        ps = getattr(cov, task)[query]
        ds = cov.disease[disease]
        bs = ps & ds
        qinfo = self.query_cache[(task, query)]
        dinfo = self.dis_profiles[disease]

        # Anchor M extras:
        ppi_ov = len(qinfo["ppi"] & ds)
        ppi_jac = ppi_ov / max(1, len(qinfo["ppi"] | ds))
        min_deg_inv = max(1.0 / max(1, len(cov.protein_disease[p])) for p in bs)
        if task == "drug":
            dp = self.drug_profiles[query]
            dcls = dinfo["cls"]
            mw = dp["mw"] if dp["mw"] is not None else -1.0
            tpsa = dp["tpsa"] if dp["tpsa"] is not None else -1.0
            clogp = dp["clogp"] if dp["clogp"] is not None else 0.0
            has_pc = float(dp["mw"] is not None and dp["tpsa"] is not None)
            bbb_blk = float(dp["bbb_restricted"] and dcls["is_cns"])
            skin_mis = float(dp["phys_regime"] == "P_STRICT_LOCAL_SKIN_EYE" and not dcls["is_skin_eye"])
            muc_mis = float(dp["phys_regime"] == "P_STRICT_LOCAL_MUCOSAL" and not dcls["is_mucosal"])
            dom_m = float(bool(set(dp["domains"]) & dcls["matched_domains"])) if (dp["domains"] and dcls["matched_domains"]) else 0.5
        else:
            mw = tpsa = -1.0
            clogp = has_pc = bbb_blk = skin_mis = muc_mis = 0.0
            dom_m = 0.5
        m_ext = [ppi_ov, ppi_jac, min_deg_inv, mw, tpsa, clogp, has_pc, bbb_blk, skin_mis, muc_mis, dom_m]

        # Anchor F extras:
        b_pw = union(extra["prot_pathway"], bs)
        b_bp = union(cov.annotations["bp"], bs)
        b_pheno = union(extra["prot_pheno"], bs)
        pw_ov = len(qinfo["pw"] & dinfo["pw"])
        pw_rat = pw_ov / max(1, len(qinfo["pw"]))
        bp_ov = len(qinfo["bp"] & dinfo["bp"])
        bp_rat = bp_ov / max(1, len(qinfo["bp"]))
        se = qinfo["se"]
        se_ov = len(se & dinfo["pheno"]) if task == "drug" else len(qinfo["pheno"] & dinfo["pheno"])
        se_rat = se_ov / max(1, len(se) if task == "drug" else len(qinfo["pheno"]))
        se_cnt = len(se) if task == "drug" else len(qinfo["pheno"])
        se_neg = len(se & dinfo["neg_pheno"]) if task == "drug" else 0
        bpheno_ov = len(b_pheno & dinfo["pheno"])
        f_ext = [pw_ov, pw_rat, len(b_pw), bp_ov, bp_rat, len(b_bp), se_ov, se_rat, se_cnt, se_neg, bpheno_ov]

        # Anchor A extras:
        b_ap = union(cov.annotations["present"], bs)
        b_aa = union(cov.annotations["absent"], bs)
        na = len(dinfo["present"])
        pos_a = len(b_ap & dinfo["present"])
        neg_a = len(b_aa & dinfo["present"])
        anat_rat = pos_a / max(1, na)
        anat_sel = pos_a / max(1, len(b_ap))
        cc_ov = len(qinfo["cc"] & dinfo["cc"])
        cc_rat = cc_ov / max(1, len(qinfo["cc"]))
        if task == "drug":
            dp = self.drug_profiles[query]
            dcls = dinfo["cls"]
            n_enz = len(extra["drug_enzyme"].get(query, ()))
            n_trn = len(extra["drug_transporter"].get(query, ()))
            n_car = len(extra["drug_carrier"].get(query, ()))
            adme_ov = len(qinfo["adme"] & ds)
            adme_ppi_ov = len(qinfo["adme_ppi"] & ds)
            hl = math.log1p(dp["hl_hours"]) if dp["hl_hours"] is not None else -1.0
            has_hl = float(dp["hl_hours"] is not None)
            inv_mis = float(dp["time_regime"] == "T_INVITRO_ONLY" and not dcls["is_skin_eye"])
        else:
            n_enz = n_trn = n_car = adme_ov = adme_ppi_ov = 0
            hl = -1.0
            has_hl = inv_mis = 0.0
        a_ext = [anat_rat, anat_sel, cc_ov, cc_rat, n_enz, n_trn, n_car, adme_ov, adme_ppi_ov, hl, has_hl, inv_mis]

        return base22 + m_ext + f_ext + a_ext


def build_task_dataset(cov, labels, task, memberships, extractor):
    lookup = {r["query_id"]: r for r in memberships}
    ind, con, off = (labels[k] for k in ("indication", "contraindication", "off-label use"))
    positive = ind - con if task == "drug" else labels["exposure_disease"]
    negative = con - ind - off
    known_drug_labels = ind | con | off
    rows, features = [], []
    for query, ps in sorted(getattr(cov, task).items()):
        for disease in sorted(union(cov.protein_disease, ps)):
            pair = query, disease
            if task == "drug" and pair not in positive and pair not in negative:
                continue
            rows.append({
                "query_id": query,
                "disease_id": disease,
                "group_id": lookup[query]["group_id"],
                "split": lookup[query]["split"],
                "label": int(pair in positive),
            })
            features.append(extractor.pair(task, query, disease))
    return rows, np.asarray(features, dtype=float)


def fit_predict_model(kind, X_train, y_train, X_eval, jobs):
    if kind == "logistic_l2":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, solver="liblinear", max_iter=2000, random_state=SEED),
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_eval)[:, 1], model
    if kind == "extratrees":
        model = ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=3, max_features=0.6,
            n_jobs=jobs, random_state=SEED,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_eval)[:, 1], model
    if kind == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1.0, random_state=SEED,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_eval)[:, 1], model
    if kind == "tri_anchor_blend":
        # Calibrated blend of standardized linear risk + nonlinear tree interaction
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, solver="liblinear", max_iter=2000, random_state=SEED),
        )
        et = ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=3, max_features=0.6,
            n_jobs=jobs, random_state=SEED,
        )
        lr.fit(X_train, y_train)
        et.fit(X_train, y_train)
        p_lr = lr.predict_proba(X_eval)[:, 1]
        p_et = et.predict_proba(X_eval)[:, 1]
        return 0.5 * p_lr + 0.5 * p_et, (lr, et)
    raise ValueError(f"Unknown model kind: {kind}")


def _bootstrap_chunk(args):
    y, full, other, inverse, ng, family_full, family_other, seeds = args
    out = {k: [] for k in ("auroc", "ap", "macro_family_auroc", "macro_family_ap")}
    for s in seeds:
        rng = np.random.default_rng(s)
        mult = np.bincount(rng.integers(0, ng, size=ng), minlength=ng)
        weights = mult[inverse]
        a, b = binary_metrics(y, full, weights), binary_metrics(y, other, weights)
        for key in ("auroc", "ap"):
            if a[key] is not None and b[key] is not None:
                out[key].append(a[key] - b[key])
            elig = np.array([
                family_full[i][key] is not None and family_other[i][key] is not None
                for i in range(ng)
            ])
            mw = mult[elig]
            if mw.sum():
                diff = [family_full[i][key] - family_other[i][key] for i in np.flatnonzero(elig)]
                out["macro_family_" + key].append(float(np.average(diff, weights=mw)))
    return out


def parallel_paired_bootstrap(y, full, other, groups, repetitions, jobs, seed=SEED):
    unique, inverse = np.unique(groups, return_inverse=True)
    ng = len(unique)
    f_full = [binary_metrics(y[inverse == i], full[inverse == i]) for i in range(ng)]
    f_other = [binary_metrics(y[inverse == i], other[inverse == i]) for i in range(ng)]
    base_rng = np.random.default_rng(seed)
    all_seeds = base_rng.integers(0, 2**31 - 1, size=repetitions).tolist()
    n_workers = min(jobs, repetitions, 32)
    chunks = np.array_split(all_seeds, n_workers)
    tasks = [(y, full, other, inverse, ng, f_full, f_other, c.tolist()) for c in chunks if len(c)]
    samples = {k: [] for k in ("auroc", "ap", "macro_family_auroc", "macro_family_ap")}
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("fork")) as pool:
            for part in pool.map(_bootstrap_chunk, tasks):
                for k in samples:
                    samples[k].extend(part[k])
    else:
        for t in tasks:
            part = _bootstrap_chunk(t)
            for k in samples:
                samples[k].extend(part[k])
    point_a, point_b = binary_metrics(y, full), binary_metrics(y, other)
    macro_a, macro_b = macro_metrics(y, full, groups), macro_metrics(y, other, groups)
    result = {}
    for key, values in samples.items():
        base_key = key.removeprefix("macro_family_")
        a, b = (macro_a, macro_b) if key.startswith("macro_family_") else (point_a, point_b)
        result[key] = {
            "delta_full_minus_comparator": a[base_key] - b[base_key]
            if a[base_key] is not None and b[base_key] is not None else None,
            "ci95_percentile": [float(v) for v in np.quantile(values, [0.025, 0.975])] if values else None,
            "valid_replicates": len(values),
        }
    return result


def evaluate_task_v4(task, rows, X, out_dir, jobs, bootstraps):
    t0 = time.perf_counter()
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    queries = np.array([r["query_id"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    dev_mask, test_mask = splits == "dev", splits == "test"
    X_dev, y_dev, g_dev, q_dev = X[dev_mask], y[dev_mask], groups[dev_mask], queries[dev_mask]
    X_test, y_test, g_test, q_test = X[test_mask], y[test_mask], groups[test_mask], queries[test_mask]

    folds = list(GroupKFold(n_splits=3).split(X_dev, y_dev, g_dev))

    dev_oof = {}
    test_scores = {}

    # 1. Exact v3 heuristic baselines & uncalibrated v2/v3 graph_MFA score:
    for h in HEURISTICS:
        col = ALL_FEATURES.index(h)
        dev_oof[h] = X_dev[:, col].copy()
        test_scores[h] = X_test[:, col].copy()
    dev_oof["v3_uncalibrated_graph_MFA"] = X_dev[:, IDX_V3_BLOCKS].sum(axis=1)
    test_scores["v3_uncalibrated_graph_MFA"] = X_test[:, IDX_V3_BLOCKS].sum(axis=1)

    # 2. Exact v3 baseline learners (Core-10 and v3-22):
    v3_specs = [
        ("v3_logistic_core10", "logistic_l2", IDX_CORE_TOPO),
        ("v3_extratrees_core10", "extratrees", IDX_CORE_TOPO),
        ("v3_logistic_same_info22", "logistic_l2", IDX_V3_ALL),
        ("v3_extratrees_same_info22", "extratrees", IDX_V3_ALL),
    ]
    for name, kind, cols in v3_specs:
        oof = np.empty(len(y_dev))
        for tr, va in folds:
            oof[va], _ = fit_predict_model(kind, X_dev[tr][:, cols], y_dev[tr], X_dev[va][:, cols], jobs)
        dev_oof[name] = oof
        test_scores[name], _ = fit_predict_model(kind, X_dev[:, cols], y_dev, X_test[:, cols], jobs)

    # 3. Dev-selected fair baseline from v3 candidate pool (using only DEV OOF AP):
    v3_pool = HEURISTICS + [s[0] for s in v3_specs]
    selected_v3_baseline = max(v3_pool, key=lambda nm: binary_metrics(y_dev, dev_oof[nm])["ap"])
    selected_core_baseline = max(
        HEURISTICS + ["v3_logistic_core10", "v3_extratrees_core10"],
        key=lambda nm: binary_metrics(y_dev, dev_oof[nm])["ap"],
    )

    # 4. Proposed v4 Calibrated Tri-Anchor Regime Model across all 7 non-empty subsets (M, F, A, MF, MA, FA, MFA)
    # Candidate learner families for the Tri-Anchor feature space, selected strictly on DEV OOF AP of MFA:
    candidate_kinds = ["logistic_l2", "extratrees", "hgb", "tri_anchor_blend"]
    kind_dev_ap = {}
    for kind in candidate_kinds:
        oof = np.empty(len(y_dev))
        cols = SUBSET_MAP["MFA"]
        for tr, va in folds:
            oof[va], _ = fit_predict_model(kind, X_dev[tr][:, cols], y_dev[tr], X_dev[va][:, cols], jobs)
        kind_dev_ap[kind] = binary_metrics(y_dev, oof)["ap"]
    selected_kind = max(candidate_kinds, key=lambda k: kind_dev_ap[k])

    for sub_name, cols in SUBSET_MAP.items():
        m_name = f"v4_tri_anchor_{sub_name}"
        oof = np.empty(len(y_dev))
        for tr, va in folds:
            oof[va], _ = fit_predict_model(selected_kind, X_dev[tr][:, cols], y_dev[tr], X_dev[va][:, cols], jobs)
        dev_oof[m_name] = oof
        test_scores[m_name], _ = fit_predict_model(selected_kind, X_dev[:, cols], y_dev, X_test[:, cols], jobs)

    # Compute metrics
    dev_metrics = {k: metrics(y_dev, s, g_dev, q_dev) for k, s in dev_oof.items()}
    test_metrics = {k: metrics(y_test, s, g_test, q_test) for k, s in test_scores.items()}

    full_test = test_scores["v4_tri_anchor_MFA"]
    comparisons = {
        "vs_dev_selected_v3_baseline": parallel_paired_bootstrap(
            y_test, full_test, test_scores[selected_v3_baseline], g_test, bootstraps, jobs
        ),
        "vs_dev_selected_core_topology_baseline": parallel_paired_bootstrap(
            y_test, full_test, test_scores[selected_core_baseline], g_test, bootstraps, jobs
        ),
        "vs_v3_uncalibrated_graph_MFA": parallel_paired_bootstrap(
            y_test, full_test, test_scores["v3_uncalibrated_graph_MFA"], g_test, bootstraps, jobs
        ),
        "increment_M_full_minus_FA": parallel_paired_bootstrap(
            y_test, full_test, test_scores["v4_tri_anchor_FA"], g_test, bootstraps, jobs
        ),
        "increment_F_full_minus_MA": parallel_paired_bootstrap(
            y_test, full_test, test_scores["v4_tri_anchor_MA"], g_test, bootstraps, jobs
        ),
        "increment_A_full_minus_MF": parallel_paired_bootstrap(
            y_test, full_test, test_scores["v4_tri_anchor_MF"], g_test, bootstraps, jobs
        ),
    }

    return {
        "task": task,
        "selected_v3_baseline": selected_v3_baseline,
        "selected_core_topology_baseline": selected_core_baseline,
        "selected_tri_anchor_learner_on_dev": selected_kind,
        "dev_learner_oof_ap": kind_dev_ap,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "paired_bootstrap_comparisons": comparisons,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-name", default="multiscale_calibrated_v4")
    parser.add_argument("--jobs", type=int, default=min(32, max(1, len(os.sched_getaffinity(0)))))
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    out_dir = OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    cov, labels, exp_names, sources, extra, drug_raw, dis_raw = load_multiscale_graph(args.data_dir)
    extractor = MultiScaleExtractor(cov, extra, drug_raw, dis_raw)

    drug_members = frozen_memberships(cov.drug, {q: drug_family(q, drug_raw.get(q, {})) for q in cov.drug})
    exp_members = frozen_memberships(cov.exposure, {q: exposure_family(exp_names.get(q, "")) for q in cov.exposure})

    results = {}
    for task, members in (("drug", drug_members), ("exposure", exp_members)):
        rows, X = build_task_dataset(cov, labels, task, members, extractor)
        results[task] = evaluate_task_v4(task, rows, X, out_dir, args.jobs, args.bootstraps)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": args.jobs,
        "bootstraps": args.bootstraps,
        "total_seconds": time.perf_counter() - t0,
        "results": results,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Completed PrimeKG v4 in {summary['total_seconds']:.2f}s (jobs={args.jobs})")
    for task, res in results.items():
        tm = res["test_metrics"]
        print(f"\n=== PRIMEKG V4 TEST: {task.upper()} ===")
        print(f"  Dev-selected v3 baseline ({res['selected_v3_baseline']}): AP={tm[res['selected_v3_baseline']]['ap']:.4f}, AUROC={tm[res['selected_v3_baseline']]['auroc']:.4f}")
        print(f"  Dev-selected core-topo   ({res['selected_core_topology_baseline']}): AP={tm[res['selected_core_topology_baseline']]['ap']:.4f}, AUROC={tm[res['selected_core_topology_baseline']]['auroc']:.4f}")
        print(f"  v3 uncalibrated MFA      : AP={tm['v3_uncalibrated_graph_MFA']['ap']:.4f}, AUROC={tm['v3_uncalibrated_graph_MFA']['auroc']:.4f}")
        for sub in ("M", "F", "A", "MF", "MA", "FA", "MFA"):
            m = tm[f"v4_tri_anchor_{sub}"]
            print(f"  v4_tri_anchor_{sub:3s} ({res['selected_tri_anchor_learner_on_dev']}): AP={m['ap']:.4f}, AUROC={m['auroc']:.4f}")
        for cname, cval in res["paired_bootstrap_comparisons"].items():
            ap_ci = cval["ap"]
            au_ci = cval["auroc"]
            print(f"  [Bootstrap] {cname:38s}: dAP={ap_ci['delta_full_minus_comparator']:+.4f} {ap_ci['ci95_percentile']} | dAUROC={au_ci['delta_full_minus_comparator']:+.4f} {au_ci['ci95_percentile']}")


if __name__ == "__main__":
    main()
