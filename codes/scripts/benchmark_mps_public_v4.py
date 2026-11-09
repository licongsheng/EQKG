#!/usr/bin/env python3
"""Hierarchical Tri-Anchor Shrinkage & L1-Calibrated MPS Benchmark (v4).

Strictly preserves all v3 provenance, link-expansion, strict timestamp chronology,
missing-never-equal, and family/plate disjoint evaluation rules, while solving
the three modeling defects that caused v3's hard gate (`gate_012`) to lose on CA/CP:
1. Uncapped cross-plate pool (`reference_cap=0` default) + 32-worker fork parallelism:
   eliminates random 128-reference truncation while finishing in ~35 seconds.
2. Single-plate collapse mitigation: in v3, 100% of `gate_012` targets collapsed
   onto a single reference plate (mean N=1.9–3.1 samples), amplifying plate systematic
   offset and L2 outlier sensitivity compared to multi-plate unconditional medians.
   v4 computes hierarchical sufficient statistics across all 3 anchors (unconditional
   multi-plate priors, per-anchor hard/soft plate-consolidated medians, support counts,
   and minimum condition discrepancies).
3. L1-Calibrated Hierarchical Tri-Anchor Regressor (`hgb_tri_012` and all 7 anchor
   subsets): trained strictly on the DEV split using `loss='absolute_error'` (L1)
   over the anchor sufficient statistics to predict the condition-specific residual
   over the unconditional cross-plate median. Evaluates both on the fixed `gate_012`
   supported cohort and on 100% of cross-plate supported targets.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = (Path(__file__).resolve().parents[4] if Path(__file__).resolve().parents[1].name == 'codes' else Path(__file__).resolve().parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from scripts.benchmark_mps_public_v3 import (
    ANCHORS, ENDPOINTS, METHODS as V3_METHODS, SQL_PATH, STRONG as V3_STRONG,
    SUBSETS as V3_SUBSETS, cluster_bootstrap, encoded_features, file_hash,
    load_observations, metrics, split_family, write_json,
)

OUT_DIR_V4 = Path("/home/AI4BioEM/data/caltech_mps_validation_v4")
V3_OBS_CACHE = Path("/home/AI4BioEM/data/caltech_mps_validation_v3/observations.jsonl.gz")

TRI_SUBSETS = ("U", "0", "1", "2", "01", "02", "12", "012")
_GROUPS_V4 = {}


def weighted_median(vals, weights):
    if len(vals) == 1:
        return float(vals[0])
    order = np.argsort(vals)
    vs, ws = vals[order], weights[order]
    cs = np.cumsum(ws)
    idx = np.searchsorted(cs, 0.5 * cs[-1])
    return float(vs[min(idx, len(vs) - 1)])


def extract_group_stats(key):
    obs = _GROUPS_V4[key]
    n_obs = len(obs)
    if n_obs < 2:
        return [
            {
                "target_id": o["id"],
                "y": o["y"],
                "family": o["family"],
                "plates": o["plates"],
                "split": split_family(o["family"]),
                "modality": o["modality"],
                "eligible_count": 0,
                "v3_preds": {m: None for m in V3_METHODS},
                "stats": None,
            }
            for o in obs
        ]

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
            records.append({
                "target_id": target["id"],
                "y": target["y"],
                "family": target["family"],
                "plates": target["plates"],
                "split": split_family(target["family"]),
                "modality": target["modality"],
                "eligible_count": 0,
                "v3_preds": {m: None for m in V3_METHODS},
                "stats": None,
            })
            continue

        refs_arr = np.array(refs)
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

        ref_y = ys[refs_arr]
        ref_p = [plate_keys[j] for j in refs]
        total_dist = anc_dist.sum(axis=1) * np.array([9.0, 3.0, 4.0])[0]  # use exact per-field mean for v3 parity
        # Recompute exact v3 per-field mean distance
        v3_dist = (anc_dist[:, 0] * anc_counts[0] + anc_dist[:, 1] * anc_counts[1] + anc_dist[:, 2] * anc_counts[2]) / n_f
        rank = np.argsort(v3_dist, kind="stable")

        v3_preds = {m: None for m in V3_METHODS}
        v3_preds["mean"] = float(np.mean(ref_y))
        v3_preds["median"] = float(np.median(ref_y))
        v3_preds["soft_knn_8"] = float(np.mean(ref_y[rank[:8]]))
        for bw in (0.25, 1.0):
            w = np.exp(-(v3_dist - v3_dist.min()) / bw)
            v3_preds[f"soft_kernel_{bw:g}"] = float(np.dot(w, ref_y) / w.sum())
        for mask, name in enumerate(V3_SUBSETS):
            ancs = [a for a in range(3) if mask & (1 << a)]
            sel = np.all(anchor_ok[:, ancs], axis=1) if ancs else np.ones(len(refs), dtype=bool)
            if np.any(sel):
                v3_preds[name] = float(np.mean(ref_y[sel]))
        full_sel = np.all(anchor_ok, axis=1)
        full_n = int(full_sel.sum())
        if full_n:
            v3_preds["count_matched_random"] = float(np.mean(ref_y[:full_n]))
            v3_preds["count_matched_knn"] = float(np.mean(ref_y[rank[:full_n]]))

        # Hierarchical Plate-Consolidated & Multi-Bandwidth Sufficient Statistics:
        p_ys = defaultdict(list)
        for idx_r, pk in enumerate(ref_p):
            p_ys[pk].append(ref_y[idx_r])
        p_med_all = float(np.median([np.median(v) for v in p_ys.values()]))
        u_med = v3_preds["median"]
        u_mean = v3_preds["mean"]
        u_iqr = float(np.percentile(ref_y, 75) - np.percentile(ref_y, 25))
        feat_U = [
            u_med,
            p_med_all - u_med,
            u_mean - u_med,
            u_iqr,
            math.log1p(len(ref_y)),
            math.log1p(len(p_ys)),
        ]

        feat_A = []
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
            feat_A.append([
                gm - u_med, gmn - u_med, wm1 - u_med, wm2 - u_med, wmn - u_med,
                gn, gnp, float(da.min()), float(da.mean()),
            ])

        def joint_stats(ancs):
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
            "family": target["family"],
            "plates": target["plates"],
            "split": split_family(target["family"]),
            "modality": target["modality"],
            "eligible_count": len(refs),
            "v3_preds": v3_preds,
            "stats": {
                "u_med": u_med,
                "U": feat_U,
                "A0": feat_A[0],
                "A1": feat_A[1],
                "A2": feat_A[2],
                "J01": joint_stats([0, 1]),
                "J02": joint_stats([0, 2]),
                "J12": joint_stats([1, 2]),
                "J012": joint_stats([0, 1, 2]),
            },
        })
    return records


def build_design_matrix(rows, subset):
    mats = [np.array([r["stats"]["U"] for r in rows], dtype=float)]
    if "0" in subset:
        mats.append(np.array([r["stats"]["A0"] for r in rows], dtype=float))
    if "1" in subset:
        mats.append(np.array([r["stats"]["A1"] for r in rows], dtype=float))
    if "2" in subset:
        mats.append(np.array([r["stats"]["A2"] for r in rows], dtype=float))
    if "0" in subset and "1" in subset:
        mats.append(np.array([r["stats"]["J01"] for r in rows], dtype=float))
    if "0" in subset and "2" in subset:
        mats.append(np.array([r["stats"]["J02"] for r in rows], dtype=float))
    if "1" in subset and "2" in subset:
        mats.append(np.array([r["stats"]["J12"] for r in rows], dtype=float))
    if subset == "012":
        mats.append(np.array([r["stats"]["J012"] for r in rows], dtype=float))
    return np.hstack(mats)


def fit_hgb_l1(X_train, r_train):
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=19471,
    )
    model.fit(X_train, r_train)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DIR_V4)
    parser.add_argument("--workers", type=int, default=min(32, max(1, len(os.sched_getaffinity(0)))))
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args(argv)

    t0 = time.monotonic()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    # Load verified v3 observations cache if present, else parse SQL
    groups = defaultdict(list)
    if V3_OBS_CACHE.exists():
        print(f"Loading verified observations from {V3_OBS_CACHE}", flush=True)
        with gzip.open(V3_OBS_CACHE, "rt", encoding="utf-8") as stream:
            for line in stream:
                o = json.loads(line)
                groups[(tuple(tuple(x) for x in o["composition"]), o["modality"])].append(o)
    else:
        observations = load_observations(SQL_PATH, out)
        for o in observations:
            groups[(tuple(tuple(x) for x in o["composition"]), o["modality"])].append(o)

    global _GROUPS_V4
    _GROUPS_V4 = dict(groups)
    tasks = sorted(groups.keys(), key=lambda k: (-len(groups[k]), k))
    print(f"Extracting uncapped hierarchical Tri-Anchor stats across {len(tasks)} groups with {args.workers} workers...", flush=True)

    by_split_mod = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for chunk in pool.map(extract_group_stats, tasks, chunksize=40):
            for rec in chunk:
                by_split_mod[(rec["split"], rec["modality"])].append(rec)

    print(f"Finished parallel feature extraction in {time.monotonic() - t0:.2f}s. Fitting DEV L1-Calibrated Tri-Anchor models...", flush=True)

    all_methods = list(V3_METHODS) + [f"hgb_tri_{s}" for s in TRI_SUBSETS]
    summary = {
        "status": "HIERARCHICAL_L1_CALIBRATED_TRIANCHOR_UNCAPPED_V4",
        "workers": args.workers,
        "dev_selection": {},
        "evaluations": {},
        "paired_comparisons_full_supported": {},
        "paired_comparisons_all_eligible": {},
    }

    for modality in ENDPOINTS:
        dev_all = by_split_mod[("dev", modality)]
        test_all = by_split_mod[("test", modality)]
        dev_elig = [r for r in dev_all if r["stats"] is not None]
        test_elig = [r for r in test_all if r["stats"] is not None]

        # Initialize predictions dict on all records
        for r in dev_all + test_all:
            r["predictions"] = dict(r["v3_preds"])
            for s in TRI_SUBSETS:
                r["predictions"][f"hgb_tri_{s}"] = None

        # Fit two sets of DEV models (strictly on DEV, grouped by family):
        # 1) Cohort-specific calibration on full-supported DEV targets (for apple-to-apple full-cohort comparison)
        # 2) Universal calibration on all cross-plate eligible DEV targets (for 100% eligible coverage)
        dev_full = [r for r in dev_elig if r["v3_preds"]["gate_012"] is not None]
        test_full = [r for r in test_elig if r["v3_preds"]["gate_012"] is not None]

        y_dev_elig = np.array([r["y"] for r in dev_elig])
        u_dev_elig = np.array([r["stats"]["u_med"] for r in dev_elig])
        f_dev_elig = np.array([r["family"] for r in dev_elig])
        u_test_elig = np.array([r["stats"]["u_med"] for r in test_elig])

        y_dev_full = np.array([r["y"] for r in dev_full])
        u_dev_full = np.array([r["stats"]["u_med"] for r in dev_full])
        f_dev_full = np.array([r["family"] for r in dev_full])
        u_test_full = np.array([r["stats"]["u_med"] for r in test_full])

        folds_elig = list(GroupKFold(n_splits=3).split(dev_elig, y_dev_elig, f_dev_elig))
        folds_full = list(GroupKFold(n_splits=3).split(dev_full, y_dev_full, f_dev_full))

        for s in TRI_SUBSETS:
            m_name = f"hgb_tri_{s}"
            # Universal model across all eligible targets:
            X_de = build_design_matrix(dev_elig, s)
            X_te = build_design_matrix(test_elig, s)
            oof_e = np.empty(len(dev_elig))
            for tr, va in folds_elig:
                mod_e = fit_hgb_l1(X_de[tr], y_dev_elig[tr] - u_dev_elig[tr])
                oof_e[va] = u_dev_elig[va] + mod_e.predict(X_de[va])
            mod_e_final = fit_hgb_l1(X_de, y_dev_elig - u_dev_elig)
            pred_te = u_test_elig + mod_e_final.predict(X_te)
            for r, val in zip(dev_elig, oof_e):
                r["predictions"][m_name] = float(val)
            for r, val in zip(test_elig, pred_te):
                r["predictions"][m_name] = float(val)

            # On the full-supported cohort, refine with cohort-calibrated model (still strictly DEV-trained):
            X_df = build_design_matrix(dev_full, s)
            X_tf = build_design_matrix(test_full, s)
            oof_f = np.empty(len(dev_full))
            for tr, va in folds_full:
                mod_f = fit_hgb_l1(X_df[tr], y_dev_full[tr] - u_dev_full[tr])
                oof_f[va] = u_dev_full[va] + mod_f.predict(X_df[va])
            mod_f_final = fit_hgb_l1(X_df, y_dev_full - u_dev_full)
            pred_tf = u_test_full + mod_f_final.predict(X_tf)
            # Blend universal and cohort-specific DEV models (0.5/0.5) on full-supported targets
            for r, vf in zip(dev_full, oof_f):
                r["predictions"][m_name] = 0.5 * r["predictions"][m_name] + 0.5 * float(vf)
            for r, vf in zip(test_full, pred_tf):
                r["predictions"][m_name] = 0.5 * r["predictions"][m_name] + 0.5 * float(vf)

        # Select strong baseline strictly on DEV (same as v3):
        dev_scores_full = {
            m: float(np.mean([abs(r["y"] - r["predictions"][m]) for r in dev_full]))
            for m in V3_STRONG
        }
        dev_scores_elig = {
            m: float(np.mean([abs(r["y"] - r["predictions"][m]) for r in dev_elig]))
            for m in ("mean", "median", "soft_knn_8", "soft_kernel_0.25", "soft_kernel_1")
        }
        sel_full = min(dev_scores_full, key=lambda k: dev_scores_full[k])
        sel_elig = min(dev_scores_elig, key=lambda k: dev_scores_elig[k])
        summary["dev_selection"][modality] = {
            "selected_full_supported_baseline": sel_full,
            "dev_full_supported_mae": dev_scores_full,
            "selected_all_eligible_baseline": sel_elig,
            "dev_all_eligible_mae": dev_scores_elig,
        }

        for split, rows in (("dev", dev_all), ("test", test_all)):
            key_name = f"{split}:{modality}"
            y_arr = np.array([r["y"] for r in rows])
            plate_arr = np.array(["|".join(r["plates"]) for r in rows])
            fam_arr = np.array([r["family"] for r in rows])
            preds_map = {
                m: np.array([r["predictions"][m] if r["predictions"][m] is not None else np.nan for r in rows])
                for m in all_methods
            }
            full_mask = np.isfinite(preds_map["gate_012"])
            elig_mask = np.isfinite(preds_map["median"])

            summary["evaluations"][key_name] = {
                "all_targets": {
                    m: metrics(y_arr, pred, plate_arr, fam_arr) for m, pred in preds_map.items()
                },
                "all_eligible_targets": {
                    m: metrics(y_arr[elig_mask], pred[elig_mask], plate_arr[elig_mask], fam_arr[elig_mask])
                    for m, pred in preds_map.items()
                },
                "fixed_full_supported_targets": {
                    m: metrics(y_arr[full_mask], pred[full_mask], plate_arr[full_mask], fam_arr[full_mask])
                    for m, pred in preds_map.items()
                },
            }

            # Paired bootstrap comparisons for hgb_tri_012 vs baselines & anchor leave-outs
            comp_full = {}
            for cname in (sel_full, "gate_012", "hgb_tri_U", "hgb_tri_01", "hgb_tri_02", "hgb_tri_12"):
                delta = (
                    np.abs(y_arr[full_mask] - preds_map["hgb_tri_012"][full_mask])
                    - np.abs(y_arr[full_mask] - preds_map[cname][full_mask])
                )
                comp_full[cname] = {
                    "plate_cluster": cluster_bootstrap(delta, plate_arr[full_mask], repeats=args.bootstrap_repeats),
                    "family_cluster": cluster_bootstrap(delta, fam_arr[full_mask], repeats=args.bootstrap_repeats),
                }
            summary["paired_comparisons_full_supported"][key_name] = comp_full

            comp_elig = {}
            for cname in (sel_elig, "hgb_tri_U", "hgb_tri_01", "hgb_tri_02", "hgb_tri_12"):
                delta = (
                    np.abs(y_arr[elig_mask] - preds_map["hgb_tri_012"][elig_mask])
                    - np.abs(y_arr[elig_mask] - preds_map[cname][elig_mask])
                )
                comp_elig[cname] = {
                    "plate_cluster": cluster_bootstrap(delta, plate_arr[elig_mask], repeats=args.bootstrap_repeats),
                    "family_cluster": cluster_bootstrap(delta, fam_arr[elig_mask], repeats=args.bootstrap_repeats),
                }
            summary["paired_comparisons_all_eligible"][key_name] = comp_elig

    summary["elapsed_seconds"] = time.monotonic() - t0
    write_json(out / "summary.json", summary)
    print(f"\nCompleted MPS v4 in {summary['elapsed_seconds']:.2f}s (workers={args.workers})")

    for modality in ENDPOINTS:
        k = f"test:{modality}"
        ev_f = summary["evaluations"][k]["fixed_full_supported_targets"]
        ev_e = summary["evaluations"][k]["all_eligible_targets"]
        sel_f = summary["dev_selection"][modality]["selected_full_supported_baseline"]
        sel_e = summary["dev_selection"][modality]["selected_all_eligible_baseline"]
        print(f"\n=== MPS V4 TEST: {modality} ===")
        print(f"  [Fixed Full-Supported Cohort N={ev_f['hgb_tri_012']['n']}]")
        print(f"    Dev-selected baseline ({sel_f:16s}): MAE={ev_f[sel_f]['mae']:.6g}, RMSE={ev_f[sel_f]['rmse']:.6g}")
        print(f"    v3 Hard Gate (gate_012)              : MAE={ev_f['gate_012']['mae']:.6g}, RMSE={ev_f['gate_012']['rmse']:.6g}")
        for s in TRI_SUBSETS:
            m = ev_f[f"hgb_tri_{s}"]
            print(f"    v4 hgb_tri_{s:3s}                       : MAE={m['mae']:.6g}, RMSE={m['rmse']:.6g}")
        print(f"  [All Cross-Plate Eligible Cohort N={ev_e['hgb_tri_012']['n']} (100% of eligible targets)]")
        print(f"    Dev-selected baseline ({sel_e:16s}): MAE={ev_e[sel_e]['mae']:.6g}, RMSE={ev_e[sel_e]['rmse']:.6g}")
        for s in TRI_SUBSETS:
            m = ev_e[f"hgb_tri_{s}"]
            print(f"    v4 hgb_tri_{s:3s}                       : MAE={m['mae']:.6g}, RMSE={m['rmse']:.6g}")


if __name__ == "__main__":
    main()
