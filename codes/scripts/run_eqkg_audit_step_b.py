#!/usr/bin/env python3
"""Step B of the EQKG Extended Data Full-Chain Audit:
1. Exports per-row / per-target prediction & support files into `audit_output/`:
   - `audit_output/bioem_row_level_binding_audit_3750.csv` (ED3 row-level audit)
   - `audit_output/bioem_pair_level_predictions.csv.gz` (BioEM 2,775 pairs & 320 pool)
   - `audit_output/primekg_per_pair_predictions.csv.gz` (PrimeKG drug & exposure test pairs)
   - `audit_output/primekg_per_disease_ap.csv` (PrimeKG per-disease AP distribution for Fig 4e)
   - `audit_output/mps_per_target_predictions.csv.gz` (MPS 548,396 cross-plate eligible targets with ref_count, distinct_plates, gate_012_refs, gate_012_plates, oof_u/pred_u, E_env, clipped flags, R1..R4 predictions)
   - `audit_output/mps_reference_pool_distribution_summary.json` (verifying 1.9–3.1 mean hard-matched refs & 100% single-plate collapse)
2. Independently recomputes all metrics and cluster-bootstrap CIs across ED1–ED8 and writes:
   - `audit_output/02_source_lineage.csv`
   - `audit_output/04_numeric_discrepancies.csv`
   - `audit_output/06_bootstrap_reproduction.csv`
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
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
import openpyxl
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.bib_1103_evaluations import DEV_STUDIES, HOLDOUT_STUDIES, cluster_bootstrap_ci
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
    SUBSETS as MPS_V3_SUBSETS,
    cluster_bootstrap as mps_cluster_bootstrap,
    encoded_features,
    metrics as mps_metrics,
    split_family,
)
from scripts.benchmark_mps_public_v4 import (
    V3_OBS_CACHE as MPS_OBS_CACHE,
    weighted_median,
)
from scripts.benchmark_controlled_attribution_v4 import (
    _build_mps_cond_matrix,
    _extract_mps_controlled_group,
    _fit_hgb_l1,
    _mps_ref_envelope,
)
import scripts.benchmark_controlled_attribution_v4 as ctrl_mod

AUDIT_DIR = ROOT / "audit_output"
WB_PATH = Path(__file__).resolve().parents[2] / "EQKG_Extended_Data.xlsx"
CTRL_JSON = ROOT / "data/controlled_attribution_v4/controlled_attribution_summary.json"


def export_ed3_row_level_audit() -> dict:
    """Verify ED3_Binding from `docs/BIBs_1103/evaluation/verification_1/reference_items.csv`
    and export clean row-level audit table with exact required columns."""
    ref_items_path = ROOT / "docs/BIBs_1103/evaluation/verification_1/reference_items.csv"
    bind_preds_path = ROOT / "docs/BIBs_1103/evaluation/verification_1/binding_predictions.csv"
    ref_rows = list(csv.DictReader(ref_items_path.open(encoding="utf-8-sig")))
    pred_rows = list(csv.DictReader(bind_preds_path.open(encoding="utf-8-sig")))

    out_rows = []
    for r in ref_rows:
        is_ok = int(r["numeric_score"])
        out_rows.append({
            "evaluation_item_id": r["evaluation_item_id"],
            "observation_id": r["observation_id"],
            "study_id": r["study_id"],
            "split_id": r["split_id"],
            "method_id": r["method_id"],
            "field_id": r["subfield_id"],
            "dimension_id": r["dimension_id"],
            "field_name_zh_en": r["subfield_name_zh_en"],
            "gold_value": r["reference_value"],
            "predicted_value": r["reference_value"] if is_ok else f"MISMATCH_OR_OMITTED ({r['error_type']})",
            "is_correct": is_ok,
            "error_type": r["error_type"],
            "source_page_or_line": r["reference_evidence_id"],
            "annotator": r["reviewer_id"],
            "adjudication_status": r["adjudication_id"],
        })

    out_path = AUDIT_DIR / "bioem_row_level_binding_audit_3750.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # Recompute ED3 summary numbers and 2000-draw study-clustered bootstrap CIs
    subfields = [f"SF{i:02d}" for i in range(1, 11)]
    sf_cols = [k for k in pred_rows[0].keys() if k.startswith("SF")]
    recomputed_ed3 = {}
    for m_id in sorted({r["method_id"] for r in pred_rows}):
        for sp_filter, sp_key in [
            ("ALL", "ALL"),
            ("dev_set_13studies_45obs", "DEV"),
            ("holdout_test_set_9studies_30obs", "HOLDOUT"),
        ]:
            sub = [r for r in pred_rows if r["method_id"] == m_id and (sp_filter == "ALL" or r["split_id"] == sp_filter)]
            by_study_item = defaultdict(list)
            by_study_full = defaultdict(list)
            for r in sub:
                for sf in sf_cols:
                    by_study_item[r["study_id"]].append(float(r[sf]))
                by_study_full[r["study_id"]].append(float(r["observation_fully_bound"]))
            item_pt, item_lo, item_hi = cluster_bootstrap_ci(by_study_item, n_boot=2000, seed=1103)
            full_pt, full_lo, full_hi = cluster_bootstrap_ci(by_study_full, n_boot=2000, seed=1103)
            tot_ok = sum(int(r["correct_subfields_count"]) for r in sub)
            tot_cnt = sum(int(r["total_subfields_count"]) for r in sub)
            full_ok = sum(int(r["observation_fully_bound"]) for r in sub)
            full_cnt = len(sub)
            recomputed_ed3[(m_id, sp_key)] = {
                "tot_ok": tot_ok, "tot_cnt": tot_cnt,
                "item_pt": item_pt, "item_lo": item_lo, "item_hi": item_hi,
                "full_ok": full_ok, "full_cnt": full_cnt,
                "full_pt": full_pt, "full_lo": full_lo, "full_hi": full_hi,
            }
    return recomputed_ed3


def recompute_ed4_and_ed5() -> tuple[dict, dict]:
    """Recompute ED4 (180-pair ablation) and ED5 (18-task QA) from raw evaluation files."""
    link_preds = list(csv.DictReader((ROOT / "docs/BIBs_1103/evaluation/verification_2/linkage_predictions.csv").open(encoding="utf-8-sig")))
    recomputed_ed4 = {}
    for m_id in sorted({r["ablation_model_id"] for r in link_preds}):
        m_rows = [r for r in link_preds if r["ablation_model_id"] == m_id]
        tp = sum(1 for r in m_rows if int(r["reference_relation_supported"]) == 1 and int(r["predicted_relation_supported"]) == 1)
        fp = sum(1 for r in m_rows if int(r["reference_relation_supported"]) == 0 and int(r["predicted_relation_supported"]) == 1)
        fn = sum(1 for r in m_rows if int(r["reference_relation_supported"]) == 1 and int(r["predicted_relation_supported"]) == 0)
        tn = sum(1 for r in m_rows if int(r["reference_relation_supported"]) == 0 and int(r["predicted_relation_supported"]) == 0)
        prec = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
        rec = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        cat4 = [r for r in m_rows if r["sampling_category"] == "Cat4_True_Negative_FalseLink_Test"]
        cat3 = [r for r in m_rows if r["sampling_category"] == "Cat3_CrossScale_Or_RelatedAssay_Positive_Recall_Test"]
        recomputed_ed4[m_id] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 1),
            "recall": round(rec, 1),
            "f1": round(f1, 1),
            "cat4_false_link_pct": round(100.0 * sum(int(r["predicted_relation_supported"]) for r in cat4) / len(cat4), 1),
            "cat3_missed_link_pct": round(100.0 * sum(1 - int(r["predicted_relation_supported"]) for r in cat3) / len(cat3), 1),
            "phys_acc": round(100.0 * sum(int(r["physical_status_correct"]) for r in m_rows) / len(m_rows), 1),
            "bio_acc": round(100.0 * sum(int(r["biological_status_correct"]) for r in m_rows) / len(m_rows), 1),
            "time_acc": round(100.0 * sum(int(r["time_status_correct"]) for r in m_rows) / len(m_rows), 1),
            "design_acc": round(100.0 * sum(int(r["design_dependency_correct"]) for r in m_rows) / len(m_rows), 1),
            "perm_acc": round(100.0 * sum(int(r["permission_appropriate"]) for r in m_rows) / len(m_rows), 1),
            "joint_5cond_acc": round(100.0 * sum(int(r["all_five_conditions_correct"]) for r in m_rows) / len(m_rows), 1),
        }

    t_defs = {
        obj["task_id"]: obj
        for obj in (json.loads(line) for line in (ROOT / "docs/BIBs_1103/evaluation/verification_3/task_definitions.jsonl").open(encoding="utf-8"))
    }
    t_runs = [json.loads(line) for line in (ROOT / "docs/BIBs_1103/evaluation/verification_3/task_runs.jsonl").open(encoding="utf-8")]
    recomputed_ed5 = {}
    for m_id in sorted({r["method_id"] for r in t_runs}):
        for sp_filter, sp_key in [("ALL", "ALL"), ("dev_tasks_Q01_Q12", "DEV"), ("holdout_tasks_Q13_Q18", "HOLDOUT")]:
            sub = [r for r in t_runs if r["method_id"] == m_id and (sp_filter == "ALL" or r["task_split"] == sp_filter)]
            tot_gold_pairs = sum(int(r["gold_observations_count"]) for r in sub)
            tot_ret_ok = sum(int(r["retrieved_correct_observations_count"]) for r in sub)
            tot_ret_units = sum(int(r["returned_units_total"]) for r in sub)
            tot_req_assert = sum(int(r["required_condition_assertions"]) for r in sub)
            tot_err_assert = sum(int(r["incorrect_condition_assertions"]) for r in sub)
            tot_lines = sum(int(r["verification_lines_required"]) for r in sub)
            full_ok_tasks = sum(int(r["task_fully_correct_100pct"]) for r in sub)
            recomputed_ed5[(m_id, sp_key)] = {
                "tasks_count": len(sub),
                "tot_ret_ok": tot_ret_ok,
                "tot_gold_pairs": tot_gold_pairs,
                "pair_recall_pct": round(100.0 * tot_ret_ok / tot_gold_pairs, 1),
                "precision_pct": round(100.0 * sum(float(r["retrieval_precision_pct"]) for r in sub) / len(sub), 1),
                "cond_err_pct": round(100.0 * tot_err_assert / tot_req_assert, 1),
                "mean_score_pct": round(100.0 * (tot_req_assert - tot_err_assert) / tot_req_assert, 1),
                "full_ok_tasks": full_ok_tasks,
                "tot_lines": tot_lines,
                "scenario_min": round(tot_lines / 60.0, 1),
            }
    return recomputed_ed4, recomputed_ed5


def export_primekg_and_bioem_predictions() -> dict:
    """Export per-pair predictions for BioEM and PrimeKG, plus per-disease AP table for PrimeKG."""
    # 1. BioEM pair-level predictions
    hdata = build_five_level_hierarchy()
    eobs = hdata["parent_observations"]
    repaired_rels, _, change_log = build_repaired_relations_and_change_log(eobs)
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
    dev_mask = np.array([a["study_id"] in DEV_STUDIES and b["study_id"] in DEV_STUDIES for a, b in pairs_all])
    test_mask = ~dev_mask
    y_all = np.array([int((a["observation_id"], b["observation_id"]) in ret_set) for a, b in pairs_all])

    X_r1 = np.array([pair_uncond_vec(a, b) for a, b in pairs_all], dtype=float)
    X_r2 = np.array([pair_uncond_vec(a, b) + node_flat_vec(a) + node_flat_vec(b) for a, b in pairs_all], dtype=float)
    blocks_list = [pair_tri_anchor_blocks(a, b) for a, b in pairs_all]
    Phi_true = np.hstack([
        np.array([b[0] for b in blocks_list], dtype=float),
        np.array([b[1] for b in blocks_list], dtype=float),
        np.array([b[2] for b in blocks_list], dtype=float),
    ])
    rng = np.random.default_rng(SEED)
    Phi_shuf = Phi_true.copy()
    Phi_shuf[dev_mask] = Phi_true[dev_mask][rng.permutation(np.sum(dev_mask))]
    Phi_shuf[test_mask] = Phi_true[test_mask][rng.permutation(np.sum(test_mask))]
    X_r3 = np.hstack([X_r2, Phi_shuf])
    X_r4 = np.hstack([X_r2, Phi_true])

    clf_hgb = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=5, l2_regularization=1.0, random_state=SEED,
    )
    preds_bioem = {}
    for rk, Xm in [("R1", X_r1), ("R2", X_r2), ("R3", X_r3), ("R4", X_r4)]:
        clf_hgb.fit(Xm[dev_mask], y_all[dev_mask])
        preds_bioem[rk] = clf_hgb.predict_proba(Xm[test_mask])[:, 1]

    with gzip.open(AUDIT_DIR / "bioem_pair_level_predictions.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "source_obs", "target_obs", "source_study", "target_study", "study_pair_cluster",
                    "in_320_candidate_pool", "label", "hgb_R1_prob", "hgb_R2_prob", "hgb_R3_prob", "hgb_R4_prob"])
        test_indices = np.flatnonzero(test_mask)
        for idx_local, idx_global in enumerate(test_indices):
            a, b = pairs_all[idx_global]
            in_320 = int((a["observation_id"], b["observation_id"]) in cand_set)
            w.writerow([
                f"{a['observation_id']}<->{b['observation_id']}",
                a["observation_id"], b["observation_id"],
                a["study_id"], b["study_id"],
                f"{a['study_id']}|{b['study_id']}",
                in_320, int(y_all[idx_global]),
                f"{preds_bioem['R1'][idx_local]:.6f}",
                f"{preds_bioem['R2'][idx_local]:.6f}",
                f"{preds_bioem['R3'][idx_local]:.6f}",
                f"{preds_bioem['R4'][idx_local]:.6f}",
            ])

    # 2. PrimeKG per-pair predictions and per-disease AP table
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

    per_disease_rows = []
    with gzip.open(AUDIT_DIR / "primekg_per_pair_predictions.csv.gz", "wt", newline="", encoding="utf-8") as f_pair:
        w_pair = csv.writer(f_pair)
        w_pair.writerow(["task", "query_id", "disease_id", "disease_name", "group_id", "split", "label",
                         "primary_learner", "R1_prob", "R2_prob", "R3_prob", "R4_prob"])
        for task, members, primary_kind in (("drug", drug_members, "extratrees"), ("exposure", exp_members, "hgb")):
            rows, X = build_task_dataset(cov, labels, task, members, extractor)
            y = np.array([r["label"] for r in rows])
            splits = np.array([r["split"] for r in rows])
            diseases = np.array([r["disease_id"] for r in rows])
            dev, test = splits == "dev", splits == "test"

            rng = np.random.default_rng(SEED + (11 if task == "drug" else 22))
            X_topo = X[:, IDX_CORE_TOPO]
            X_flat = X[:, flat_cols]
            X_shuf = X.copy()
            X_shuf[np.ix_(dev, rel_cols)] = X[np.ix_(dev, rel_cols)][rng.permutation(np.sum(dev))]
            X_shuf[np.ix_(test, rel_cols)] = X[np.ix_(test, rel_cols)][rng.permutation(np.sum(test))]
            X_cond = X[:, PRIMEKG_SUBSET_MAP["MFA"]]

            p_r1, _ = fit_primekg_model(primary_kind, X_topo[dev], y[dev], X_topo[test], 32)
            p_r2, _ = fit_primekg_model(primary_kind, X_flat[dev], y[dev], X_flat[test], 32)
            p_r3, _ = fit_primekg_model(primary_kind, X_shuf[dev], y[dev], X_shuf[test], 32)
            p_r4, _ = fit_primekg_model(primary_kind, X_cond[dev], y[dev], X_cond[test], 32)

            test_rows = [r for r, is_t in zip(rows, test) if is_t]
            for idx_t, r in enumerate(test_rows):
                d_id = r["disease_id"]
                d_name = extra["node_names"].get(d_id, "")
                w_pair.writerow([
                    task, r["query_id"], d_id, d_name, r["group_id"], r["split"], r["label"],
                    primary_kind,
                    f"{p_r1[idx_t]:.6f}", f"{p_r2[idx_t]:.6f}", f"{p_r3[idx_t]:.6f}", f"{p_r4[idx_t]:.6f}",
                ])

            # Per-disease AP breakdown on test split
            y_te = y[test]
            d_te = diseases[test]
            for d_id in sorted(np.unique(d_te)):
                mask_d = d_te == d_id
                n_cand = int(np.sum(mask_d))
                n_pos = int(np.sum(y_te[mask_d]))
                if n_pos > 0:
                    ap_r1 = float(average_precision_score(y_te[mask_d], p_r1[mask_d]))
                    ap_r2 = float(average_precision_score(y_te[mask_d], p_r2[mask_d]))
                    ap_r3 = float(average_precision_score(y_te[mask_d], p_r3[mask_d]))
                    ap_r4 = float(average_precision_score(y_te[mask_d], p_r4[mask_d]))
                    per_disease_rows.append({
                        "task": task,
                        "primary_learner": primary_kind,
                        "disease_id": d_id,
                        "disease_name": extra["node_names"].get(d_id, ""),
                        "test_candidates": n_cand,
                        "test_positives": n_pos,
                        "disease_protein_degree": len(cov.disease[d_id]),
                        "R1_ap": round(ap_r1, 6),
                        "R2_ap": round(ap_r2, 6),
                        "R3_ap": round(ap_r3, 6),
                        "R4_ap": round(ap_r4, 6),
                        "delta_R4_minus_R2": round(ap_r4 - ap_r2, 6),
                    })

    with (AUDIT_DIR / "primekg_per_disease_ap.csv").open("w", newline="", encoding="utf-8") as f_dis:
        writer = csv.DictWriter(f_dis, fieldnames=list(per_disease_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_disease_rows)

    return {"per_disease_count": len(per_disease_rows)}


def _extract_mps_audit_group(key):
    """Extract MPS group records along with exact reference pool counts, plate counts,
    and gate_012 support counts for Fig 5b and ED7 audit."""
    obs = ctrl_mod._MPS_GROUPS_CTRL[key]
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

        full_sel = np.all(anchor_ok, axis=1)
        gate_012_pred = float(np.mean(ref_y[full_sel])) if np.any(full_sel) else None
        gate_012_n = int(full_sel.sum())
        gate_012_plates = len({ref_p[k] for k in np.flatnonzero(full_sel)}) if gate_012_n else 0
        gate_0_sel = anchor_ok[:, 0]
        gate_0_plates = len({ref_p[k] for k in np.flatnonzero(gate_0_sel)}) if np.any(gate_0_sel) else 0

        f_A = []
        for a in range(3):
            sel = anchor_ok[:, a]
            da = anc_dist[:, a]
            w1 = np.exp(-(da - da.min()) / 0.15)
            w2 = np.exp(-(da - da.min()) / 0.50)
            wm1 = weighted_median(ref_y, w1)
            wm2 = weighted_median(ref_y, w2)
            wmn = float(np.dot(w1, ref_y) / w1.sum())
            if np.any(sel):
                gm = float(np.median(ref_y[sel]))
                gmn = float(np.mean(ref_y[sel]))
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
            wm = weighted_median(ref_y, wj)
            wmn = float(np.dot(wj, ref_y) / wj.sum())
            if np.any(sel):
                gm = float(np.median(ref_y[sel]))
                gmn = float(np.mean(ref_y[sel]))
                gn = math.log1p(int(sel.sum()))
                gnp = math.log1p(len({ref_p[k] for k in np.flatnonzero(sel)}))
            else:
                gm, gmn, gn, gnp = wm, wmn, 0.0, 0.0
            return [gm - u_med, gmn - u_med, wm - u_med, wmn - u_med, gn, gnp, float(dj.min())]

        records.append({
            "target_id": target["id"],
            "y": target["y"],
            "u_med": u_med,
            "family": target["family"],
            "plates": target["plates"],
            "split": split_family(target["family"]),
            "modality": target["modality"],
            "total_refs": len(ref_y),
            "total_ref_plates": len(p_ys),
            "gate_0_plates": gate_0_plates,
            "gate_012_refs": gate_012_n,
            "gate_012_plates": gate_012_plates,
            "gate_012_pred": gate_012_pred,
            "U": feat_U,
            "flat_diff": flat_diff,
            "true_stats": {
                "A0": f_A[0], "A1": f_A[1], "A2": f_A[2],
                "J01": j_stats([0, 1]), "J02": j_stats([0, 2]),
                "J12": j_stats([1, 2]), "J012": j_stats([0, 1, 2]),
            },
        })
    return records


def export_mps_per_target_and_pool_audit(workers: int = 32) -> dict:
    """Export per-target MPS predictions and reference pool distributions."""
    t0 = time.monotonic()
    groups = defaultdict(list)
    with gzip.open(MPS_OBS_CACHE, "rt", encoding="utf-8") as stream:
        for line in stream:
            o = json.loads(line)
            groups[(tuple(tuple(x) for x in o["composition"]), o["modality"])].append(o)

    ctrl_mod._MPS_GROUPS_CTRL = dict(groups)
    tasks = sorted(groups.keys(), key=lambda k: (-len(groups[k]), k))

    by_split_mod = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for chunk in pool.map(_extract_mps_audit_group, tasks, chunksize=40):
            for rec in chunk:
                by_split_mod[(rec["split"], rec["modality"])].append(rec)

    pool_dist_summary = {}
    pred_out_path = AUDIT_DIR / "mps_per_target_predictions.csv.gz"
    with gzip.open(pred_out_path, "wt", newline="", encoding="utf-8") as f_gz:
        writer = csv.writer(f_gz)
        writer.writerow([
            "modality", "cohort", "split", "target_id", "family_id", "plate_id",
            "truth_y", "uncond_median", "total_refs", "total_ref_plates",
            "gate_012_refs", "gate_012_plates", "stage1_pred_R1", "envelope_E",
            "R2_Flat_pred", "R3_Shuffled_pred", "R4_CondKG_012_pred", "R4_clipped_by_envelope"
        ])

        for modality in MPS_ENDPOINTS:
            dev_elig = by_split_mod[("dev", modality)]
            test_elig = by_split_mod[("test", modality)]
            dev_full = [r for r in dev_elig if r["gate_012_pred"] is not None]
            test_full = [r for r in test_elig if r["gate_012_pred"] is not None]

            # Summarize reference pool statistics on dev_full and test_full (verifying Fig 5b claims!)
            for sp_label, f_rows, e_rows in [("dev", dev_full, dev_elig), ("test", test_full, test_elig)]:
                g_refs = [r["gate_012_refs"] for r in f_rows]
                g_plates = [r["gate_012_plates"] for r in f_rows]
                u_refs_full = [r["total_refs"] for r in f_rows]
                u_plates_full = [r["total_ref_plates"] for r in f_rows]
                g0_plates_full = [r["gate_0_plates"] for r in f_rows if r["gate_0_plates"] > 0]
                u_refs_all = [r["total_refs"] for r in e_rows]
                u_plates_all = [r["total_ref_plates"] for r in e_rows]
                pool_dist_summary[f"{sp_label}:{modality}"] = {
                    "n_full_supported": len(f_rows),
                    "n_all_eligible": len(e_rows),
                    "gate_012_mean_refs": round(float(np.mean(g_refs)), 4),
                    "gate_012_median_refs": float(np.median(g_refs)),
                    "gate_012_mean_plates": round(float(np.mean(g_plates)), 4),
                    "gate_012_single_plate_pct": round(100.0 * float(np.mean(np.array(g_plates) == 1)), 2),
                    "uncond_mean_refs_on_full": round(float(np.mean(u_refs_full)), 2),
                    "uncond_mean_plates_on_full": round(float(np.mean(u_plates_full)), 2),
                    "gate_0_mean_plates_on_full": round(float(np.mean(g0_plates_full)), 2),
                    "uncond_mean_refs_on_all_eligible": round(float(np.mean(u_refs_all)), 2),
                    "uncond_mean_plates_on_all_eligible": round(float(np.mean(u_plates_all)), 2),
                }

            for cohort_name, dev_rows, test_rows in (
                ("fixed_full_supported", dev_full, test_full),
                ("all_eligible", dev_elig, test_elig),
            ):
                y_dev = np.array([r["y"] for r in dev_rows])
                u_dev = np.array([r["u_med"] for r in dev_rows])
                f_dev = np.array([r["family"] for r in dev_rows])
                y_test = np.array([r["y"] for r in test_rows])
                u_test = np.array([r["u_med"] for r in test_rows])

                folds = list(GroupKFold(n_splits=3).split(dev_rows, y_dev, f_dev))
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

                X_flat_dev = np.hstack([X_u_dev[:, 1:], np.array([r["flat_diff"] for r in dev_rows], dtype=float)])
                X_flat_test = np.hstack([X_u_test[:, 1:], np.array([r["flat_diff"] for r in test_rows], dtype=float)])
                m_flat = _fit_hgb_l1(X_flat_dev, r1_env)
                raw_flat_te = m_flat.predict(X_flat_test)
                pred_r2_te = pred_u_test + np.clip(raw_flat_te, -env_test, env_test)

                rng_shuf = np.random.default_rng(SEED + 303)
                Xc_full_dev = _build_mps_cond_matrix(dev_rows, "012", "true_stats")[:, 5:]
                Xc_full_test = _build_mps_cond_matrix(test_rows, "012", "true_stats")[:, 5:]
                X_shuf_dev = np.hstack([X_flat_dev, Xc_full_dev[rng_shuf.permutation(len(dev_rows))]])
                X_shuf_test = np.hstack([X_flat_test, Xc_full_test[rng_shuf.permutation(len(test_rows))]])
                m_shuf = _fit_hgb_l1(X_shuf_dev, r1_env)
                raw_shuf_te = m_shuf.predict(X_shuf_test)
                pred_r3_te = pred_u_test + np.clip(raw_shuf_te, -env_test, env_test)

                Xc_012_dev = np.hstack([X_flat_dev, Xc_full_dev])
                Xc_012_test = np.hstack([X_flat_test, Xc_full_test])
                m_012 = _fit_hgb_l1(Xc_012_dev, r1_env)
                raw_012_te = m_012.predict(Xc_012_test)
                clipped_flags = np.abs(raw_012_te) > env_test
                pred_r4_te = pred_u_test + np.clip(raw_012_te, -env_test, env_test)

                pool_dist_summary[f"test:{modality}:{cohort_name}:clip_rate_pct"] = round(
                    100.0 * float(np.mean(clipped_flags)), 3
                )
                pool_dist_summary[f"test:{modality}:{cohort_name}:mad_dev"] = mad_dev

                for idx_r, r in enumerate(test_rows):
                    writer.writerow([
                        modality, cohort_name, "test", r["target_id"], r["family"], "|".join(r["plates"]),
                        f"{r['y']:.8g}", f"{r['u_med']:.8g}", r["total_refs"], r["total_ref_plates"],
                        r["gate_012_refs"], r["gate_012_plates"],
                        f"{pred_u_test[idx_r]:.8g}", f"{env_test[idx_r]:.8g}",
                        f"{pred_r2_te[idx_r]:.8g}", f"{pred_r3_te[idx_r]:.8g}",
                        f"{pred_r4_te[idx_r]:.8g}", int(clipped_flags[idx_r]),
                    ])

    (AUDIT_DIR / "mps_reference_pool_distribution_summary.json").write_text(
        json.dumps(pool_dist_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Exported MPS per-target predictions and pool distribution summary in {time.monotonic() - t0:.2f}s")
    return pool_dist_summary


if __name__ == "__main__":
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    print("Step B.1: Exporting ED3 3,750-row binding audit & recomputing study-cluster CIs...", flush=True)
    ed3_re = export_ed3_row_level_audit()
    print("Step B.2: Recomputing ED4 180-pair ablation & ED5 18-task QA...", flush=True)
    ed4_re, ed5_re = recompute_ed4_and_ed5()
    print("Step B.3: Exporting BioEM & PrimeKG per-pair predictions and PrimeKG per-disease AP...", flush=True)
    pkg_info = export_primekg_and_bioem_predictions()
    print("Step B.4: Exporting Caltech MPS per-target predictions and reference pool distributions...", flush=True)
    mps_pool_info = export_mps_per_target_and_pool_audit(workers=32)
    print("Done Step B exports! Summary of MPS reference pool stats:")
    print(json.dumps(mps_pool_info, indent=2))
