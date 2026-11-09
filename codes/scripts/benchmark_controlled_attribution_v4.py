#!/usr/bin/env python3
"""Controlled Same-Data, Same-Learner, Same-Budget Attribution Experiment Across
BioEM, PrimeKG, and Caltech MPS (v4-controlled).

Directly tests whether the Conditional Knowledge Graph (CondKG) representation
itself provides statistically significant, positive incremental value when
data, learner family, and training budget are held strictly identical across:
  - R1_Unconditioned: Unconditioned graph proximity / cross-plate statistics
  - R2_Flat_SameInfo: Complete flat node/condition attributes from the exact
    same data tables, without relational Tri-Anchor alignment operators
  - R3_CondKG_Shuffled: Flat attributes + relational Tri-Anchor alignment
    features computed under permuted condition-entity pairings (breaking
    relational condition alignment while keeping feature dimensionality and
    marginal distributions identical to R4)
  - R4_CondKG_TriAnchor: Full relational Tri-Anchor Conditional KG representation
    (plus all 7 non-empty anchor subsets for complete ablation)
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.bib_1103_evaluations import DEV_STUDIES, HOLDOUT_STUDIES
from scripts.bib_1103_hierarchy_and_data import build_five_level_hierarchy
from scripts.bib_1103_relations_and_rules import (
    build_repaired_relations_and_change_log,
    compute_structured_modifiers,
    map_b2_to_mechanism_id,
)
from scripts.benchmark_primekg_public_v3 import (
    SEED,
    binary_metrics,
    drug_family,
    exposure_family,
    frozen_memberships,
    macro_metrics,
    metrics as primekg_metrics,
)
from scripts.benchmark_primekg_public_v4 import (
    ALL_FEATURES as PRIMEKG_ALL_FEATURES,
    DEFAULT_DATA as PRIMEKG_DATA,
    IDX_CORE_TOPO,
    SUBSET_MAP as PRIMEKG_SUBSET_MAP,
    MultiScaleExtractor,
    build_task_dataset,
    fit_predict_model as fit_primekg_model,
    load_multiscale_graph,
    parallel_paired_bootstrap,
)
from scripts.benchmark_mps_public_v3 import (
    ENDPOINTS as MPS_ENDPOINTS,
    METHODS as MPS_V3_METHODS,
    STRONG as MPS_V3_STRONG,
    SUBSETS as MPS_V3_SUBSETS,
    cluster_bootstrap as mps_cluster_bootstrap,
    encoded_features,
    metrics as mps_metrics,
    split_family,
)
from scripts.benchmark_mps_public_v4 import (
    TRI_SUBSETS as MPS_TRI_SUBSETS,
    V3_OBS_CACHE as MPS_OBS_CACHE,
    weighted_median,
)

OUT_Controlled = ROOT / "data/controlled_attribution_v4"


# =============================================================================
# PART 1: BIOEM CONTROLLED ATTRIBUTION EXPERIMENT
# =============================================================================

def run_bioem_controlled(jobs: int = 32, bootstraps: int = 1000) -> dict:
    t0 = time.perf_counter()
    hdata = build_five_level_hierarchy()
    eobs = hdata["parent_observations"]
    repaired_rels, pruned_pairs, change_log = build_repaired_relations_and_change_log(eobs)
    ret_set = {(r["source_obs"], r["target_obs"]) for r in repaired_rels} | {
        (r["target_obs"], r["source_obs"]) for r in repaired_rels
    }
    cand_set = {(c["source_obs"], c["target_obs"]) for c in change_log} | {
        (c["target_obs"], c["source_obs"]) for c in change_log
    }

    domains = sorted({o["domain"] for o in eobs})
    scales = ["micro", "meso", "macro"]
    species_list = sorted({o["species"] for o in eobs})
    preps = ["in_vitro", "ex_vivo", "in_vivo"]
    modalities = sorted({o["A_phys_coupling_modality"] for o in eobs})
    waveforms = sorted({o["waveform_structured"] for o in eobs})
    dos_statuses = sorted({o["dosimetry_status_structured"] for o in eobs})
    norm_apps = sorted({o["normalization_applicability"] for o in eobs})
    ref_procs = sorted({str(o["reference_process_id"]) for o in eobs})
    targets = sorted({o["controlled_target_id"] for o in eobs})
    coarse_buckets = sorted({o["bio_concept"] for o in eobs})

    def prep_tier(o):
        p = o["preparation"].lower()
        if "in vitro" in p:
            return "in_vitro"
        if "ex vivo" in p:
            return "ex_vivo"
        return "in_vivo"

    def one_hot(val, vocab):
        return [float(val == v) for v in vocab]

    def node_flat_vec(o):
        f_hz = math.log10(1.0 + float(o["frequency_hz"])) if o["frequency_hz"] is not None else -1.0
        i_val = math.log10(1.0 + abs(float(o["intensity_value"]))) if o["intensity_value"] is not None else -1.0
        s_min = math.log1p(float(o["session_duration_min"])) if o["session_duration_min"] is not None else -1.0
        e_days = math.log1p(float(o["exposure_days"])) if o["exposure_days"] is not None else -1.0
        lag_d = math.log1p(float(o["post_exposure_lag_days"])) if o["post_exposure_lag_days"] is not None else 0.0
        tau_e = float(o["normalized_exposure_ratio"]) if o["normalized_exposure_ratio"] is not None else -1.0
        tau_l = float(o["normalized_lag_ratio"]) if o["normalized_lag_ratio"] is not None else -1.0
        is_f1 = float("F1" in str(o["generation"]))
        return (
            [f_hz, i_val, s_min, e_days, lag_d, tau_e, tau_l, is_f1]
            + one_hot(o["domain"], domains)
            + one_hot(o["scale"], scales)
            + one_hot(o["species"], species_list)
            + one_hot(prep_tier(o), preps)
            + one_hot(o["A_phys_coupling_modality"], modalities)
            + one_hot(o["waveform_structured"], waveforms)
            + one_hot(o["dosimetry_status_structured"], dos_statuses)
            + one_hot(o["normalization_applicability"], norm_apps)
            + one_hot(str(o["reference_process_id"]), ref_procs)
            + one_hot(o["bio_concept"], coarse_buckets)
            + one_hot(o["controlled_target_id"], targets)
        )

    def pair_uncond_vec(a, b):
        return [
            float(a["domain"] == b["domain"]),
            float(a["scale"] == b["scale"]),
            float(a["species"] == b["species"]),
            float(a["bio_concept"] == b["bio_concept"]),
            float(a["A_phys_coupling_modality"] == b["A_phys_coupling_modality"]),
        ]

    def pair_tri_anchor_blocks(a, b):
        mods = compute_structured_modifiers(a, b)
        p_code = mods["physical_status_structured"].split("_")[0]
        p_vec = one_hot(p_code, ["P1", "P1b", "P2", "P2b", "P3", "P4", "P5"]) + [
            float(mods["frequency_relation"] == "same_exact_frequency"),
            float(mods["waveform_relation"] == "same_waveform_and_modulation"),
            float(mods["dose_relation"] in ("same_nominal_exposure_arm", "same_nominal_external_value_and_unit")),
            float("unreported" in mods["dose_relation"]),
        ]
        same_ctrl_target = float(a["controlled_target_id"] == b["controlled_target_id"])
        sub_conflict = float(
            (same_ctrl_target
             and a["controlled_target_id"] == "REPRO_TESTIS_AND_PLACENTA_HISTOLOGY"
             and ("placenta" in a["organ_or_compartment"].lower()) != ("placenta" in b["organ_or_compartment"].lower()))
            or (same_ctrl_target
                and a["controlled_target_id"] == "REPRO_GERMCELL_SPHASE_PLACENTAL_EPIGENETICS"
                and a["study_id"] != b["study_id"])
        )
        mech_id = map_b2_to_mechanism_id(a, b)
        has_mech = float(mech_id is not None)
        same_coarse_diff_ctrl = float(a["bio_concept"] == b["bio_concept"] and not same_ctrl_target)
        b_vec = [
            same_ctrl_target * (1.0 - sub_conflict),
            sub_conflict,
            has_mech,
            same_coarse_diff_ctrl,
            float(a["paper_id"] == b["paper_id"]),
            float(mods["batch_relation"] == "independent_replicate_batch_same_nominal_protocol"),
            float(mods["unit_overlap"] == "same_subjects_or_split_aliquots"),
            float(mods["unit_overlap"] == "paternal_F0_to_offspring_F1_within_study"),
        ]
        t_code = mods["temporal_relation"].split("_")[0]
        ra, rb = a["normalized_exposure_ratio"], b["normalized_exposure_ratio"]
        la, lb = a["normalized_lag_ratio"], b["normalized_lag_ratio"]
        dtau = (abs(float(ra) - float(rb)) if (ra is not None and rb is not None) else 1.0) + (
            abs(float(la) - float(lb)) if (la is not None and lb is not None) else 0.5
        )
        delta_homog = float(p_code in ("P1", "P2") and t_code == "T1" and mods["same_preparation"])
        t_vec = one_hot(t_code, ["T1", "T2", "T3", "T4a", "T4b", "T5"]) + [
            dtau,
            float(mods["same_preparation"]),
            float(mods["same_compartment"]),
            delta_homog,
        ]
        return p_vec, b_vec, t_vec

    pairs_all = list(itertools.combinations(eobs, 2))
    universes = {
        "candidate_320_pool": [
            p for p in pairs_all if (p[0]["observation_id"], p[1]["observation_id"]) in cand_set
        ],
        "all_2775_pairs": pairs_all,
    }

    out = {"universes": {}}
    for u_name, pair_list in universes.items():
        dev_mask = np.array([a["study_id"] in DEV_STUDIES and b["study_id"] in DEV_STUDIES for a, b in pair_list])
        test_mask = ~dev_mask
        y = np.array([int((a["observation_id"], b["observation_id"]) in ret_set) for a, b in pair_list])
        g_test = np.array([
            f"{a['study_id']}|{b['study_id']}" for idx, (a, b) in enumerate(pair_list) if test_mask[idx]
        ])

        X_r1 = np.array([pair_uncond_vec(a, b) for a, b in pair_list], dtype=float)
        X_r2 = np.array([pair_uncond_vec(a, b) + node_flat_vec(a) + node_flat_vec(b) for a, b in pair_list], dtype=float)
        blocks_list = [pair_tri_anchor_blocks(a, b) for a, b in pair_list]
        P_blk = np.array([b[0] for b in blocks_list], dtype=float)
        B_blk = np.array([b[1] for b in blocks_list], dtype=float)
        T_blk = np.array([b[2] for b in blocks_list], dtype=float)
        Phi_true = np.hstack([P_blk, B_blk, T_blk])

        rng = np.random.default_rng(SEED)
        Phi_shuf = Phi_true.copy()
        Phi_shuf[dev_mask] = Phi_true[dev_mask][rng.permutation(np.sum(dev_mask))]
        Phi_shuf[test_mask] = Phi_true[test_mask][rng.permutation(np.sum(test_mask))]

        reps = {
            "R1_Unconditioned": X_r1,
            "R2_Flat_SameInfo": X_r2,
            "R3_CondKG_Shuffled": np.hstack([X_r2, Phi_shuf]),
            "R4_CondKG_P": np.hstack([X_r2, P_blk]),
            "R4_CondKG_B": np.hstack([X_r2, B_blk]),
            "R4_CondKG_T": np.hstack([X_r2, T_blk]),
            "R4_CondKG_PB": np.hstack([X_r2, P_blk, B_blk]),
            "R4_CondKG_PT": np.hstack([X_r2, P_blk, T_blk]),
            "R4_CondKG_BT": np.hstack([X_r2, B_blk, T_blk]),
            "R4_CondKG_PBT": np.hstack([X_r2, Phi_true]),
        }

        learners = {
            "hgb": lambda: HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=5, l2_regularization=1.0, random_state=SEED,
            ),
            "extratrees": lambda: ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=2, max_features=0.6,
                n_jobs=jobs, random_state=SEED,
            ),
            "logistic_l2": lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=SEED),
            ),
        }

        u_res = {
            "n_dev": int(np.sum(dev_mask)),
            "n_test": int(np.sum(test_mask)),
            "pos_dev": int(np.sum(y[dev_mask])),
            "pos_test": int(np.sum(y[test_mask])),
            "learners": {},
        }
        for l_name, make_clf in learners.items():
            l_metrics = {}
            test_probs = {}
            for r_name, Xm in reps.items():
                clf = make_clf()
                clf.fit(Xm[dev_mask], y[dev_mask])
                prob = clf.predict_proba(Xm[test_mask])[:, 1]
                pred = (prob >= 0.5).astype(int)
                test_probs[r_name] = prob
                l_metrics[r_name] = {
                    "dim": int(Xm.shape[1]),
                    "test_ap": float(average_precision_score(y[test_mask], prob)),
                    "test_auroc": float(roc_auc_score(y[test_mask], prob)),
                    "test_f1": float(f1_score(y[test_mask], pred)),
                }
            # Paired cluster bootstrap on test study pairs
            boot_comp = {}
            for comp_key in ("R1_Unconditioned", "R2_Flat_SameInfo", "R3_CondKG_Shuffled"):
                boot_comp[f"R4_CondKG_PBT_vs_{comp_key}"] = parallel_paired_bootstrap(
                    y[test_mask], test_probs["R4_CondKG_PBT"], test_probs[comp_key],
                    g_test, bootstraps, jobs
                )
            u_res["learners"][l_name] = {
                "representations": l_metrics,
                "paired_bootstrap": boot_comp,
            }
        out["universes"][u_name] = u_res

    out["elapsed_seconds"] = time.perf_counter() - t0
    return out


# =============================================================================
# PART 2: PRIMEKG CONTROLLED ATTRIBUTION EXPERIMENT
# =============================================================================

def run_primekg_controlled(jobs: int = 32, bootstraps: int = 1000) -> dict:
    t0 = time.perf_counter()
    cov, labels, exp_names, sources, extra, drug_raw, dis_raw = load_multiscale_graph(PRIMEKG_DATA)
    extractor = MultiScaleExtractor(cov, extra, drug_raw, dis_raw)
    drug_members = frozen_memberships(cov.drug, {q: drug_family(q, drug_raw.get(q, {})) for q in cov.drug})
    exp_members = frozen_memberships(cov.exposure, {q: exposure_family(exp_names.get(q, "")) for q in cov.exposure})

    flat_names = [
        "query_mf", "disease_anatomy",
        "m_mw", "m_tpsa", "m_clogp", "m_has_physchem",
        "f_sideeffect_count",
        "a_enzyme_count", "a_transporter_count", "a_carrier_count",
        "a_log_halflife_hours", "a_has_halflife",
    ]
    flat_cols = sorted(set(IDX_CORE_TOPO + [PRIMEKG_ALL_FEATURES.index(c) for c in flat_names]))
    rel_cols = [i for i in range(len(PRIMEKG_ALL_FEATURES)) if i not in flat_cols]

    out = {"tasks": {}}
    for task, members in (("drug", drug_members), ("exposure", exp_members)):
        rows, X = build_task_dataset(cov, labels, task, members, extractor)
        y = np.array([r["label"] for r in rows])
        splits = np.array([r["split"] for r in rows])
        groups = np.array([r["group_id"] for r in rows])
        queries = np.array([r["query_id"] for r in rows])
        diseases = np.array([r["disease_id"] for r in rows])
        dev, test = splits == "dev", splits == "test"

        rng = np.random.default_rng(SEED + (11 if task == "drug" else 22))
        X_topo = X[:, IDX_CORE_TOPO]
        X_flat = X[:, flat_cols]
        X_shuf = X.copy()
        X_shuf[np.ix_(dev, rel_cols)] = X[np.ix_(dev, rel_cols)][rng.permutation(np.sum(dev))]
        X_shuf[np.ix_(test, rel_cols)] = X[np.ix_(test, rel_cols)][rng.permutation(np.sum(test))]

        reps = {
            "R1_Unconditioned_Topo10": X_topo,
            "R2_Flat_SameInfo": X_flat,
            "R3_CondKG_Shuffled": X_shuf,
            **{f"R4_CondKG_{sub}": X[:, cols] for sub, cols in PRIMEKG_SUBSET_MAP.items()},
        }

        task_res = {
            "n_dev": int(np.sum(dev)),
            "n_test": int(np.sum(test)),
            "pos_dev": int(np.sum(y[dev])),
            "pos_test": int(np.sum(y[test])),
            "learners": {},
        }
        for kind in ("hgb", "extratrees", "logistic_l2", "tri_anchor_blend"):
            l_metrics = {}
            test_preds = {}
            for r_name, Xm in reps.items():
                pred, _ = fit_primekg_model(kind, Xm[dev], y[dev], Xm[test], jobs)
                test_preds[r_name] = pred
                m = primekg_metrics(y[test], pred, groups[test], queries[test])
                m["macro_disease"] = macro_metrics(y[test], pred, diseases[test])
                m["dim"] = int(Xm.shape[1])
                l_metrics[r_name] = m

            boot_comp = {}
            for comp_key in ("R1_Unconditioned_Topo10", "R2_Flat_SameInfo", "R3_CondKG_Shuffled",
                             "R4_CondKG_FA", "R4_CondKG_MA", "R4_CondKG_MF"):
                boot_comp[f"R4_CondKG_MFA_vs_{comp_key}"] = parallel_paired_bootstrap(
                    y[test], test_preds["R4_CondKG_MFA"], test_preds[comp_key],
                    groups[test], bootstraps, jobs
                )
            task_res["learners"][kind] = {
                "representations": l_metrics,
                "paired_bootstrap": boot_comp,
            }
        out["tasks"][task] = task_res

    out["elapsed_seconds"] = time.perf_counter() - t0
    return out


# =============================================================================
# PART 3: CALTECH MPS CONTROLLED ATTRIBUTION EXPERIMENT
# =============================================================================

_MPS_GROUPS_CTRL = {}


def _extract_mps_controlled_group(key):
    obs = _MPS_GROUPS_CTRL[key]
    n_obs = len(obs)
    if n_obs < 2:
        return []
    data, valid = encoded_features(obs)
    ys = np.array([o["y"] for o in obs], dtype=float)
    plates = [set(o["plates"]) for o in obs]
    plate_keys = [tuple(o["plates"]) for o in obs]
    sids = [o["sample_id"] for o in obs]
    aids = [o["analysis_id"] for o in obs]
    pids = [o["process_data_id"] for o in obs]
    n_f = len(obs[0]["features"])
    col_anc = np.array([anc for anc, _, _, _, _ in obs[0]["features"]])
    col_tol = np.array([tol is not None for _, _, _, tol, _ in obs[0]["features"]])
    anc_counts = np.array([(col_anc == a).sum() for a in range(3)], dtype=float)
    data_clean = np.where(valid, data, 0.0)

    records = []
    for i, target in enumerate(obs):
        refs = [
            j for j in range(n_obs)
            if j != i
            and sids[j] != sids[i]
            and aids[j] != aids[i]
            and pids[j] != pids[i]
            and plates[i].isdisjoint(plates[j])
        ]
        if not refs:
            continue
        refs_arr = np.array(refs)
        ref_y = ys[refs_arr]
        ref_p = [plate_keys[j] for j in refs]

        p_ys = defaultdict(list)
        for idx_r, pk in enumerate(ref_p):
            p_ys[pk].append(ref_y[idx_r])
        p_med_all = float(np.median([np.median(v) for v in p_ys.values()]))
        u_med = float(np.median(ref_y))
        u_mean = float(np.mean(ref_y))
        u_iqr = float(np.percentile(ref_y, 75) - np.percentile(ref_y, 25))
        feat_U = [
            u_med,
            p_med_all - u_med,
            u_mean - u_med,
            u_iqr,
            math.log1p(len(ref_y)),
            math.log1p(len(p_ys)),
        ]

        # Flat non-relational summary of target vs reference pool average conditions (6 cols):
        ref_mean_data = np.mean(data_clean[refs_arr], axis=0)
        ref_mean_valid = np.mean(valid[refs_arr].astype(float), axis=0)
        flat_diff = []
        for a in range(3):
            cols_a = col_anc == a
            both_a = valid[i, cols_a] & (ref_mean_valid[cols_a] > 0)
            diff_a = np.minimum(np.abs(data_clean[i, cols_a] - ref_mean_data[cols_a]), 5.0)
            flat_diff.extend([
                float(np.mean(np.where(both_a, diff_a, 1.0))),
                float(np.mean(valid[i, cols_a].astype(float)) - np.mean(ref_mean_valid[cols_a])),
            ])

        # Compute pairwise relational condition compatibility & distance ONCE:
        anchor_ok = np.ones((len(refs), 3), dtype=bool)
        anc_dist = np.zeros((len(refs), 3), dtype=float)
        for col in range(n_f):
            anc = col_anc[col]
            both = valid[i, col] & valid[refs_arr, col]
            delta = np.abs(data[i, col] - data[refs_arr, col])
            ok = both & ((delta <= 1 + 1e-12) if col_tol[col] else (delta == 0))
            anchor_ok[:, anc] &= ok
            comp_d = np.minimum(delta, 5.0) if col_tol[col] else (delta != 0).astype(float)
            anc_dist[:, anc] += np.where(both, comp_d, 1.0)
        for a in range(3):
            anc_dist[:, a] /= anc_counts[a]

        v3_dist = (anc_dist[:, 0] * anc_counts[0] + anc_dist[:, 1] * anc_counts[1] + anc_dist[:, 2] * anc_counts[2]) / n_f
        rank = np.argsort(v3_dist, kind="stable")
        v3_preds = {m: None for m in MPS_V3_METHODS}
        v3_preds["mean"] = u_mean
        v3_preds["median"] = u_med
        v3_preds["soft_knn_8"] = float(np.mean(ref_y[rank[:8]]))
        for bw in (0.25, 1.0):
            w = np.exp(-(v3_dist - v3_dist.min()) / bw)
            v3_preds[f"soft_kernel_{bw:g}"] = float(np.dot(w, ref_y) / w.sum())
        for mask, name in enumerate(MPS_V3_SUBSETS):
            ancs = [a for a in range(3) if mask & (1 << a)]
            sel = np.all(anchor_ok[:, ancs], axis=1) if ancs else np.ones(len(refs), dtype=bool)
            if np.any(sel):
                v3_preds[name] = float(np.mean(ref_y[sel]))

        rng = np.random.default_rng((hash(target["id"]) & 0xffffffff) ^ SEED)
        ref_y_shuf = rng.permutation(ref_y)

        def build_stats_for_y(y_vec):
            f_A = []
            for a in range(3):
                sel = anchor_ok[:, a]
                da = anc_dist[:, a]
                w1 = np.exp(-(da - da.min()) / 0.15)
                w2 = np.exp(-(da - da.min()) / 0.50)
                wm1 = weighted_median(y_vec, w1)
                wm2 = weighted_median(y_vec, w2)
                wmn = float(np.dot(w1, y_vec) / w1.sum())
                if np.any(sel):
                    gm = float(np.median(y_vec[sel]))
                    gmn = float(np.mean(y_vec[sel]))
                    gn = math.log1p(int(sel.sum()))
                    gnp = math.log1p(len({ref_p[k] for k in np.flatnonzero(sel)}))
                else:
                    gm, gmn, gn, gnp = wm1, wmn, 0.0, 0.0
                f_A.append([
                    gm - u_med, gmn - u_med, wm1 - u_med, wm2 - u_med, wmn - u_med,
                    gn, gnp, float(da.min()), float(da.mean()),
                ])

            def j_stats(ancs):
                sel = np.all(anchor_ok[:, ancs], axis=1)
                dj = anc_dist[:, ancs].mean(axis=1)
                wj = np.exp(-(dj - dj.min()) / 0.20)
                wm = weighted_median(y_vec, wj)
                wmn = float(np.dot(wj, y_vec) / wj.sum())
                if np.any(sel):
                    gm = float(np.median(y_vec[sel]))
                    gmn = float(np.mean(y_vec[sel]))
                    gn = math.log1p(int(sel.sum()))
                    gnp = math.log1p(len({ref_p[k] for k in np.flatnonzero(sel)}))
                else:
                    gm, gmn, gn, gnp = wm, wmn, 0.0, 0.0
                return [gm - u_med, gmn - u_med, wm - u_med, wmn - u_med, gn, gnp, float(dj.min())]

            return {
                "A0": f_A[0], "A1": f_A[1], "A2": f_A[2],
                "J01": j_stats([0, 1]), "J02": j_stats([0, 2]),
                "J12": j_stats([1, 2]), "J012": j_stats([0, 1, 2]),
            }

        records.append({
            "target_id": target["id"],
            "y": target["y"],
            "u_med": u_med,
            "family": target["family"],
            "plates": target["plates"],
            "split": split_family(target["family"]),
            "modality": target["modality"],
            "v3_preds": v3_preds,
            "U": feat_U,
            "flat_diff": flat_diff,
            "true_stats": build_stats_for_y(ref_y),
        })
    return records


def _build_mps_cond_matrix(rows, subset: str, key_name: str = "true_stats"):
    mats = [np.array([r["U"][1:] for r in rows], dtype=float)]
    if "0" in subset:
        mats.append(np.array([r[key_name]["A0"] for r in rows], dtype=float))
    if "1" in subset:
        mats.append(np.array([r[key_name]["A1"] for r in rows], dtype=float))
    if "2" in subset:
        mats.append(np.array([r[key_name]["A2"] for r in rows], dtype=float))
    if "0" in subset and "1" in subset:
        mats.append(np.array([r[key_name]["J01"] for r in rows], dtype=float))
    if "0" in subset and "2" in subset:
        mats.append(np.array([r[key_name]["J02"] for r in rows], dtype=float))
    if "1" in subset and "2" in subset:
        mats.append(np.array([r[key_name]["J12"] for r in rows], dtype=float))
    if subset == "012":
        mats.append(np.array([r[key_name]["J012"] for r in rows], dtype=float))
    return np.hstack(mats)


def _mps_ref_envelope(rows, mad_val: float):
    X_c = _build_mps_cond_matrix(rows, "012", "true_stats")
    # u_iqr is at index 2 of U[1:]
    u_iqr = X_c[:, 2]
    # Median shift columns in X_c (excluding arithmetic mean columns):
    # U[1:] has 5 cols (0..4); A0(5..13): 5,7,8; A1(14..22): 14,16,17; A2(23..31): 23,25,26;
    # J01(32..38): 32,34; J02(39..45): 39,41; J12(46..52): 46,48; J012(53..59): 53,55
    med_cols = [5, 7, 8, 14, 16, 17, 23, 25, 26, 32, 34, 39, 41, 46, 48, 53, 55]
    max_cond_med = np.max(np.abs(X_c[:, med_cols]), axis=1)
    return np.maximum(3.0 * max_cond_med + 2.0 * u_iqr, 8.0 * mad_val)


def _fit_hgb_l1(X_tr, r_tr):
    reg = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=150, learning_rate=0.05,
        max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=19471,
    )
    reg.fit(X_tr, r_tr)
    return reg


def run_mps_controlled(workers: int = 32, bootstraps: int = 1000) -> dict:
    t0 = time.monotonic()
    groups = defaultdict(list)
    with gzip.open(MPS_OBS_CACHE, "rt", encoding="utf-8") as stream:
        for line in stream:
            o = json.loads(line)
            groups[(tuple(tuple(x) for x in o["composition"]), o["modality"])].append(o)

    global _MPS_GROUPS_CTRL
    _MPS_GROUPS_CTRL = dict(groups)
    tasks = sorted(groups.keys(), key=lambda k: (-len(groups[k]), k))

    by_split_mod = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for chunk in pool.map(_extract_mps_controlled_group, tasks, chunksize=40):
            for rec in chunk:
                by_split_mod[(rec["split"], rec["modality"])].append(rec)

    out = {"modalities": {}}
    for modality in MPS_ENDPOINTS:
        dev_elig = by_split_mod[("dev", modality)]
        test_elig = by_split_mod[("test", modality)]
        dev_full = [r for r in dev_elig if r["v3_preds"]["gate_012"] is not None]
        test_full = [r for r in test_elig if r["v3_preds"]["gate_012"] is not None]

        mod_res = {}
        for cohort_name, dev_rows, test_rows in (
            ("fixed_full_supported", dev_full, test_full),
            ("all_eligible", dev_elig, test_elig),
        ):
            y_dev = np.array([r["y"] for r in dev_rows])
            u_dev = np.array([r["u_med"] for r in dev_rows])
            f_dev = np.array([r["family"] for r in dev_rows])
            y_test = np.array([r["y"] for r in test_rows])
            u_test = np.array([r["u_med"] for r in test_rows])
            p_test = np.array(["|".join(r["plates"]) for r in test_rows])
            f_test = np.array([r["family"] for r in test_rows])

            folds = list(GroupKFold(n_splits=3).split(dev_rows, y_dev, f_dev))

            # Stage 1: R1_Unconditioned Multi-Plate Prior
            X_u_dev = np.array([r["U"] for r in dev_rows], dtype=float)
            X_u_test = np.array([r["U"] for r in test_rows], dtype=float)
            oof_u = np.empty(len(dev_rows))
            for tr, va in folds:
                m_u = _fit_hgb_l1(X_u_dev[tr], y_dev[tr] - u_dev[tr])
                oof_u[va] = u_dev[va] + m_u.predict(X_u_dev[va])
            m_u_full = _fit_hgb_l1(X_u_dev, y_dev - u_dev)
            pred_u_test = u_test + m_u_full.predict(X_u_test)

            mad_dev = float(np.median(np.abs(y_dev - oof_u)))
            env_dev = _mps_ref_envelope(dev_rows, mad_dev)
            env_test = _mps_ref_envelope(test_rows, mad_dev)
            r1_env = np.clip(y_dev - oof_u, -env_dev, env_dev)

            preds_test = {
                "uncond_median": u_test,
                "v3_gate_012": np.array([
                    r["v3_preds"]["gate_012"] if r["v3_preds"]["gate_012"] is not None else np.nan
                    for r in test_rows
                ]),
                "R1_Unconditioned_HGB": pred_u_test,
            }

            # Stage 2A: R2_Flat_SameInfo (target vs reference pool average condition diff, no pairwise alignment)
            X_flat_dev = np.hstack([X_u_dev[:, 1:], np.array([r["flat_diff"] for r in dev_rows], dtype=float)])
            X_flat_test = np.hstack([X_u_test[:, 1:], np.array([r["flat_diff"] for r in test_rows], dtype=float)])
            m_flat = _fit_hgb_l1(X_flat_dev, r1_env)
            preds_test["R2_Flat_SameInfo_HGB"] = pred_u_test + np.clip(m_flat.predict(X_flat_test), -env_test, env_test)

            # Stage 2B: R3_CondKG_Shuffled (cross-target permutation of relational Tri-Anchor blocks,
            # keeping U[1:] and flat_diff intact while breaking relational condition-target alignment,
            # exactly matching BioEM and PrimeKG R3_CondKG_Shuffled!)
            rng_shuf = np.random.default_rng(SEED + 303)
            Xc_full_dev = _build_mps_cond_matrix(dev_rows, "012", "true_stats")[:, 5:]
            Xc_full_test = _build_mps_cond_matrix(test_rows, "012", "true_stats")[:, 5:]
            X_shuf_dev = np.hstack([
                X_flat_dev,
                Xc_full_dev[rng_shuf.permutation(len(dev_rows))],
            ])
            X_shuf_test = np.hstack([
                X_flat_test,
                Xc_full_test[rng_shuf.permutation(len(test_rows))],
            ])
            m_shuf = _fit_hgb_l1(X_shuf_dev, r1_env)
            preds_test["R3_CondKG_Shuffled_HGB"] = pred_u_test + np.clip(m_shuf.predict(X_shuf_test), -env_test, env_test)

            # Stage 2C: R4_CondKG_TriAnchor across all 7 non-empty subsets (0, 1, 2, 01, 02, 12, 012)
            for s in ("0", "1", "2", "01", "02", "12", "012"):
                Xc_dev = np.hstack([X_flat_dev, _build_mps_cond_matrix(dev_rows, s, "true_stats")[:, 5:]])
                Xc_test = np.hstack([X_flat_test, _build_mps_cond_matrix(test_rows, s, "true_stats")[:, 5:]])
                ms = _fit_hgb_l1(Xc_dev, r1_env)
                preds_test[f"R4_CondKG_{s}_HGB"] = pred_u_test + np.clip(ms.predict(Xc_test), -env_test, env_test)

            # Evaluate all representations & compute paired cluster bootstrap CIs:
            eval_metrics = {
                k: mps_metrics(y_test, v, p_test, f_test)
                for k, v in preds_test.items()
                if np.all(np.isfinite(v))
            }
            comparisons = {}
            for comp_name in (
                "uncond_median", "R1_Unconditioned_HGB", "R2_Flat_SameInfo_HGB",
                "R3_CondKG_Shuffled_HGB", "R4_CondKG_01_HGB", "R4_CondKG_02_HGB", "R4_CondKG_12_HGB",
            ):
                delta = (
                    np.abs(y_test - preds_test["R4_CondKG_012_HGB"])
                    - np.abs(y_test - preds_test[comp_name])
                )
                comparisons[f"R4_CondKG_012_vs_{comp_name}"] = {
                    "plate_cluster": mps_cluster_bootstrap(delta, p_test, repeats=bootstraps),
                    "family_cluster": mps_cluster_bootstrap(delta, f_test, repeats=bootstraps),
                }
            mod_res[cohort_name] = {
                "n_dev": len(dev_rows),
                "n_test": len(test_rows),
                "metrics": eval_metrics,
                "paired_comparisons": comparisons,
            }
        out["modalities"][modality] = mod_res

    out["elapsed_seconds"] = time.monotonic() - t0
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(32, max(1, len(os.sched_getaffinity(0)))))
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    OUT_Controlled.mkdir(parents=True, exist_ok=True)

    print("1/3 Running Controlled Same-Learner Attribution on BioEM...", flush=True)
    bioem_res = run_bioem_controlled(jobs=args.jobs, bootstraps=args.bootstraps)
    print(f"    BioEM done in {bioem_res['elapsed_seconds']:.2f}s", flush=True)

    print("2/3 Running Controlled Same-Learner Attribution on PrimeKG...", flush=True)
    primekg_res = run_primekg_controlled(jobs=args.jobs, bootstraps=args.bootstraps)
    print(f"    PrimeKG done in {primekg_res['elapsed_seconds']:.2f}s", flush=True)

    print("3/3 Running Controlled Same-Learner Attribution on Caltech MPS...", flush=True)
    mps_res = run_mps_controlled(workers=args.jobs, bootstraps=args.bootstraps)
    print(f"    Caltech MPS done in {mps_res['elapsed_seconds']:.2f}s", flush=True)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": args.jobs,
        "bootstraps": args.bootstraps,
        "total_seconds": time.perf_counter() - t0,
        "bioem": bioem_res,
        "primekg": primekg_res,
        "mps": mps_res,
    }
    out_path = OUT_Controlled / "controlled_attribution_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved complete controlled attribution summary to {out_path} in {summary['total_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
