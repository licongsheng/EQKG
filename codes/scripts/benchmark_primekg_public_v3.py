#!/usr/bin/env python3
"""Frozen exploratory PrimeKG graph ranking; no claim of independent validation.

Standalone implementation: never imports/runs legacy scripts or writes their files.
Task edges are labels only. Full admissible covariate graph is transductive.
"""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import platform
import re
import resource
import shlex
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = (Path(__file__).resolve().parents[4] if Path(__file__).resolve().parents[1].name == 'codes' else Path(__file__).resolve().parents[1])
OUTPUT = ROOT / "data/primekg_validation_v3"
SEED = 20261106
LABELS = {"indication", "contraindication", "off-label use", "exposure_disease"}
ANNOTATIONS = {"bioprocess_protein": ("bp", "biological_process"),
               "molfunc_protein": ("mf", "molecular_function"),
               "cellcomp_protein": ("cc", "cellular_component"),
               "anatomy_protein_present": ("present", "anatomy"),
               "anatomy_protein_absent": ("absent", "anatomy")}
EXP_ANNOTATIONS = {"exposure_bioprocess": ("bp", "biological_process"),
                   "exposure_molfunc": ("mf", "molecular_function"),
                   "exposure_cellcomp": ("cc", "cellular_component")}
CORE = ["bridge", "resource_allocation", "adamic_adar", "cosine", "jaccard",
        "query_proteins", "disease_proteins", "disease_only_resource_allocation",
        "v2_inverse_hub", "v2_idf_normalized"]
PRIMITIVES = ["query_mf", "bridge_mf", "disease_anatomy", "bridge_present_overlap",
              "bridge_absent_overlap", "tri_bp", "pair_bp", "mf_overlap", "cc_overlap"]
FEATURES = CORE + PRIMITIVES + ["block_M", "block_F", "block_A"]
HEURISTICS = CORE[:5] + ["disease_only_resource_allocation", "v2_inverse_hub"]
LEARNERS = ["logistic_core", "logistic_blocks", "logistic_same_information",
            "extratrees_same_information"]
BASELINES = HEURISTICS + LEARNERS
CAVEATS = [
    "Previously exposed public data: exploratory retrospective grouped test, NOT independent blind validation.",
    "Drug task: indication-minus-contra versus contra-minus-indication-minus-offlabel, only two-hop candidates.",
    "Exposure task: observed-positive versus UNLABELED; zero labels are not confirmed negatives or clinical harms.",
    "Full covariate graph is transductive; no disease-family/source/time holdout; shared-source CTD circularity unresolved.",
    "Drug groups use normalized complete ATC3 class profiles (category fallback), removing drug-name prefixes; NOT molecular scaffolds. Different multi-class profiles may still share individual classes.",
    "Exposure families reproduce v2 name-stem/six-character fallback; NOT validated chemical families.",
    "ATC/category are used ONLY for split grouping, never predictors; no route/PK/physical gating tested; ATC-only baseline not applicable.",
    "M/F/A are graph specificity/function/context blocks, NOT physical/time anchors; annotation overlap can be structurally redundant.",
    "Bootstrap intervals are conditional on fixed fitted models/split, unadjusted for multiplicity, and cannot repair prior data exposure.",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def exposure_family(name):
    name = name.lower()
    for stem in ["biphenyl", "diphenyl ether", "dibenzodioxin", "dibenzofuran",
                 "arsenic", "mercury", "cadmium", "lead", "air pollutant", "smoke",
                 "particulate", "benzophenone", "phthalate", "bisphenol", "perfluoro",
                 "trichloro", "tetrachloro", "naphthol", "pyrene", "phenanthrene",
                 "pesticide", "organophosph"]:
        if stem in name:
            return stem
    return name[:6]


def drug_family(query, row):
    """Remove drug-name prose prefix; never split truncated drug-specific text.

    Exact complete class profiles are proxy families, not chemical scaffolds.
    Unrecognized prose is not interpreted as a known pharmacological family.
    """
    for field_name, marker in (("atc_3", "is pharmacologically related to "),
                               ("category", "is part of ")):
        value = (row.get(field_name) or "").lower().strip()
        if marker in value:
            value = value.split(marker, 1)[1].strip().rstrip(".")
            parts = sorted(set(re.sub(r"\s+", " ", s.strip()) for s in value.split(" and ") if s.strip()))
            if parts:
                return field_name + ":" + " | ".join(parts)
    return f"id_{query}"


def frozen_memberships(query_map, keys):
    """v2 largest-family-first alternating DEV/TEST; independent of labels."""
    groups = defaultdict(list)
    for query in sorted(query_map):
        groups[keys[query]].append(query)
    rows = []
    for i, (key, queries) in enumerate(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
        group = hashlib.sha256(key.encode()).hexdigest()
        rows.extend({"query_id": q, "family_key": key, "group_id": group,
                     "split": "dev" if i % 2 == 0 else "test"} for q in queries)
    return sorted(rows, key=lambda row: row["query_id"])


@dataclass
class Covariates:
    drug: dict = field(default_factory=lambda: defaultdict(set))
    exposure: dict = field(default_factory=lambda: defaultdict(set))
    disease: dict = field(default_factory=lambda: defaultdict(set))
    protein_disease: dict = field(default_factory=lambda: defaultdict(set))
    annotations: dict = field(default_factory=lambda: {
        k: defaultdict(set) for k in ("bp", "mf", "cc", "present", "absent")})
    exposure_annotations: dict = field(default_factory=lambda: {
        k: defaultdict(set) for k in ("bp", "mf", "cc")})


def oriented(row, left_type, right_type):
    if row["x_type"] == left_type and row["y_type"] == right_type:
        return row["x_index"], row["y_index"]
    if row["y_type"] == left_type and row["x_type"] == right_type:
        return row["y_index"], row["x_index"]
    return None


def load_graph(path):
    cov = Covariates()
    labels = {rel: set() for rel in LABELS}
    exposure_names, sources = {}, Counter()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rel = row["relation"]
            for side in ("x", "y"):
                if row[f"{side}_type"] == "exposure":
                    exposure_names[row[f"{side}_index"]] = row[f"{side}_name"]
            if rel in LABELS:
                pair = oriented(row, "exposure" if rel == "exposure_disease" else "drug", "disease")
                if pair:
                    labels[rel].add(pair)
                continue  # Both directions of ALL task edges forbidden as predictors.
            pair = None
            if rel == "drug_protein" and row["display_relation"] == "target":
                pair = oriented(row, "drug", "gene/protein")
                if pair:
                    cov.drug[pair[0]].add(pair[1])
            elif rel == "exposure_protein":
                pair = oriented(row, "exposure", "gene/protein")
                if pair:
                    cov.exposure[pair[0]].add(pair[1])
            elif rel == "disease_protein":
                pair = oriented(row, "disease", "gene/protein")
                if pair:
                    cov.disease[pair[0]].add(pair[1])
                    cov.protein_disease[pair[1]].add(pair[0])
            elif rel in ANNOTATIONS:
                key, typ = ANNOTATIONS[rel]
                pair = oriented(row, "gene/protein", typ)
                if pair:
                    cov.annotations[key][pair[0]].add(pair[1])
            elif rel in EXP_ANNOTATIONS:
                key, typ = EXP_ANNOTATIONS[rel]
                pair = oriented(row, "exposure", typ)
                if pair:
                    cov.exposure_annotations[key][pair[0]].add(pair[1])
            if pair:
                sources[(rel, row["x_source"], row["y_source"])] += 1
    return cov, labels, exposure_names, [
        {"relation": r, "x_source": x, "y_source": y, "rows": n}
        for (r, x, y), n in sorted(sources.items())]


def union(mapping, nodes):
    result = set()
    for node in nodes:
        result.update(mapping.get(node, ()))
    return result


class FeatureMatrix:
    """Predictor has no labels, names, family keys or drug metadata access."""

    def __init__(self, cov):
        self.cov = cov
        self.profiles = {d: {k: union(m, ps) for k, m in cov.annotations.items()}
                         for d, ps in cov.disease.items()}
        self.query_mf = {q: union(cov.annotations["mf"], ps) for q, ps in cov.drug.items()}
        self.query_degree = {}
        for task in ("drug", "exposure"):
            self.query_degree[task] = Counter(p for ps in getattr(cov, task).values() for p in ps)

    def pair(self, task, query, disease):
        cov = self.cov
        ps, ds = getattr(cov, task)[query], cov.disease[disease]
        bridge = ps & ds
        n, nq, nd = len(bridge), len(ps), len(ds)
        if not n:
            raise ValueError("Only two-hop candidates are admissible")
        deg = [len(cov.protein_disease[p]) for p in sorted(bridge)]
        total_deg = [len(cov.protein_disease[p]) + self.query_degree[task][p] for p in sorted(bridge)]
        invhub = sum(1 / math.log(2 + d) for d in deg)
        idf = sum(math.log1p(3000 / d) for d in deg) / math.sqrt(nq * nd)
        core = [n, sum(1 / d for d in total_deg), sum(1 / math.log(d) for d in total_deg),
                n / math.sqrt(nq * nd), n / (nq + nd - n), nq, nd,
                sum(1 / d for d in deg), invhub, idf]
        bp = union(cov.annotations["bp"], bridge) if task == "exposure" else set()
        mf = union(cov.annotations["mf"], bridge) if task == "drug" else set()
        present = union(cov.annotations["present"], bridge)
        absent = union(cov.annotations["absent"], bridge)
        profile = self.profiles[disease]
        na = len(profile["present"])
        pos, neg = len(present & profile["present"]), len(absent & profile["present"])
        qmf = len(self.query_mf.get(query, ())) if task == "drug" else 0
        tri = pair = mf_co = cc_co = 0
        if task == "drug":
            blocks = [idf, 0.30 * len(mf) / max(1, qmf) / math.sqrt(nq),
                      0.25 * (pos - 2 * neg) / max(1, na)]
        else:
            exp = cov.exposure_annotations
            ebp = exp["bp"].get(query, set())
            tri, pair = len(ebp & bp & profile["bp"]), len(ebp & profile["bp"])
            mf_co = len(exp["mf"].get(query, set()) & profile["mf"])
            cc_co = len(exp["cc"].get(query, set()) & profile["cc"])
            blocks = [invhub, math.log1p(tri) + 0.5 * math.log1p(pair),
                      0.5 * math.log1p(mf_co + cc_co) + 0.15 * (pos - neg) / max(1, na)]
        return core + [qmf, len(mf), na, pos, neg, tri, pair, mf_co, cc_co] + blocks


def make_dataset(cov, labels, task, memberships, extractor):
    lookup = {r["query_id"]: r for r in memberships}
    ind, con, off = (labels[k] for k in ("indication", "contraindication", "off-label use"))
    positive = ind - con if task == "drug" else labels["exposure_disease"]
    negative = con - ind - off
    known_drug_labels = ind | con | off
    counts = Counter()
    rows, features = [], []
    for query, ps in sorted(getattr(cov, task).items()):
        for disease in sorted(union(cov.protein_disease, ps)):
            pair = query, disease
            counts["two_hop_candidates"] += 1
            counts["candidate_observed_positive"] += int(pair in positive)
            counts["candidate_unlabeled"] += int(pair not in known_drug_labels if task == "drug" else pair not in positive)
            if task == "drug" and pair not in positive and pair not in negative:
                counts["excluded_unknown_offlabel_or_conflicting"] += 1
                continue
            rows.append({"query_id": query, "disease_id": disease,
                         "group_id": lookup[query]["group_id"], "split": lookup[query]["split"],
                         "label": int(pair in positive),
                         "label_meaning": ("indication" if pair in positive else "exclusive_contraindication")
                         if task == "drug" else ("observed_positive" if pair in positive else "unlabeled")})
            features.append(extractor.pair(task, query, disease))
    counts["all_positive_labels"] = len(positive)
    counts["positive_labels_outside_candidate_universe"] = len(positive) - counts["candidate_observed_positive"]
    if task == "drug":
        counts["indication_contra_overlap"] = len(ind & con)
        counts["contra_offlabel_overlap"] = len(con & off)
    return rows, np.asarray(features, dtype=float), dict(counts)


def subset_scores(X):
    blocks = X[:, -3:]
    scores = {}
    for size in range(4):
        for subset in itertools.combinations(range(3), size):
            name = "graph_" + ("".join("MFA"[i] for i in subset) or "empty")
            scores[name] = blocks[:, subset].sum(axis=1)
    return scores


def binary_metrics(y, score, weights=None):
    y, score = np.asarray(y), np.asarray(score)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    pos, neg = float(w[y == 1].sum()), float(w[y == 0].sum())
    return {"auroc": float(roc_auc_score(y, score, sample_weight=w)) if pos and neg else None,
            "ap": float(average_precision_score(y, score, sample_weight=w)) if pos else None}


def macro_metrics(y, score, groups):
    values = [binary_metrics(y[groups == g], score[groups == g]) for g in np.unique(groups)]
    return {**{k: float(np.mean(v)) if (v := [m[k] for m in values if m[k] is not None]) else None
               for k in ("auroc", "ap")},
            "groups": len(values), "auroc_eligible": sum(m["auroc"] is not None for m in values),
            "ap_eligible": sum(m["ap"] is not None for m in values)}


def metrics(y, score, groups, queries):
    return {**binary_metrics(y, score), "macro_family": macro_metrics(y, score, groups),
            "macro_query": macro_metrics(y, score, queries)}


def model_spec(name, jobs):
    if name == "logistic_core":
        columns = list(range(len(CORE)))
    elif name == "logistic_blocks":
        columns = list(range(len(FEATURES) - 3, len(FEATURES)))
    else:
        columns = list(range(len(FEATURES)))
    if name.startswith("logistic"):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, solver="liblinear", max_iter=2000, random_state=SEED))
    else:
        model = ExtraTreesClassifier(n_estimators=256, min_samples_leaf=5,
                                    max_features=1.0, n_jobs=jobs, random_state=SEED)
    return model, columns


def select_baseline(X_dev, y_dev, groups_dev, jobs):
    """No test data/labels accepted. Fixed hyperparameters; shared group-OOF AP selection."""
    if len(np.unique(groups_dev)) < 3:
        raise ValueError("Need three development families")
    folds = list(GroupKFold(n_splits=3).split(X_dev, y_dev, groups_dev))
    scores = {name: X_dev[:, FEATURES.index(name)].copy() for name in HEURISTICS}
    audit_folds = []
    for train, valid in folds:
        if set(groups_dev[train]) & set(groups_dev[valid]):
            raise AssertionError("Family overlap in development folds")
        if len(np.unique(y_dev[train])) != 2:
            raise ValueError("One-class development training fold; refusing adaptive split repair")
        audit_folds.append({"train_groups": sorted(set(groups_dev[train])),
                            "validation_groups": sorted(set(groups_dev[valid]))})
    for name in LEARNERS:
        score = np.empty(len(y_dev))
        for train, valid in folds:
            model, cols = model_spec(name, jobs)
            model.fit(X_dev[train][:, cols], y_dev[train])
            score[valid] = model.predict_proba(X_dev[valid][:, cols])[:, 1]
        scores[name] = score
    aps = {name: binary_metrics(y_dev, score)["ap"] for name, score in scores.items()}
    winner = max(BASELINES, key=lambda name: aps[name])
    return winner, scores, {"criterion": "pooled development 3-fold group-OOF AP; ties use predefined baseline order",
                           "eligible_order": BASELINES, "oof_ap": aps, "folds": audit_folds,
                           "selected": winner}


def paired_bootstrap(y, full, other, groups, repetitions, seed=SEED):
    """Resample entire families with replacement, paired scores, no model refit."""
    unique, inverse = np.unique(groups, return_inverse=True)
    ng = len(unique)
    family = [[binary_metrics(y[inverse == i], s[inverse == i]) for i in range(ng)]
              for s in (full, other)]
    samples = {k: [] for k in ("auroc", "ap", "macro_family_auroc", "macro_family_ap")}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        multiplicity = np.bincount(rng.integers(0, ng, size=ng), minlength=ng)
        weights = multiplicity[inverse]
        a, b = binary_metrics(y, full, weights), binary_metrics(y, other, weights)
        for key in ("auroc", "ap"):
            if a[key] is not None and b[key] is not None:
                samples[key].append(a[key] - b[key])
            eligible = np.array([family[0][i][key] is not None and family[1][i][key] is not None for i in range(ng)])
            mw = multiplicity[eligible]
            if mw.sum():
                diff = [family[0][i][key] - family[1][i][key] for i in np.flatnonzero(eligible)]
                samples["macro_family_" + key].append(float(np.average(diff, weights=mw)))
    point_a, point_b = binary_metrics(y, full), binary_metrics(y, other)
    macro_a, macro_b = macro_metrics(y, full, groups), macro_metrics(y, other, groups)
    result = {}
    for key, values in samples.items():
        base_key = key.removeprefix("macro_family_")
        a, b = (macro_a, macro_b) if key.startswith("macro_family_") else (point_a, point_b)
        result[key] = {"delta_full_minus_comparator": a[base_key] - b[base_key]
                       if a[base_key] is not None and b[base_key] is not None else None,
                       "ci95_percentile": [float(v) for v in np.quantile(values, [0.025, 0.975])] if values else None,
                       "valid_replicates": len(values), "requested_replicates": repetitions}
    return result


def export_rows(path, rows, X, scores):
    with gzip.open(path, "xt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(rows[0]) + ["feature_" + k for k in FEATURES] + list(scores))
        for i, row in enumerate(rows):
            writer.writerow(list(row.values()) + X[i].tolist() + [float(s[i]) for s in scores.values()])


def evaluate_task(task, rows, X, out, jobs, bootstraps):
    start = time.perf_counter()
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    queries = np.array([r["query_id"] for r in rows])
    dev = np.array([r["split"] == "dev" for r in rows])
    test = ~dev
    assert not set(groups[dev]) & set(groups[test])
    for mask in (dev, test):
        if len(np.unique(y[mask])) != 2:
            raise ValueError("One-class split; refusing label-dependent split changes")
    winner, oof, selection = select_baseline(X[dev], y[dev], groups[dev], jobs)
    write_json(out / f"{task}_selection.json", selection)  # freeze before test evaluation
    scores = subset_scores(X)
    scores.update({name: X[:, FEATURES.index(name)] for name in HEURISTICS})
    for name in LEARNERS:
        model, cols = model_spec(name, jobs)
        model.fit(X[dev][:, cols], y[dev])
        score = np.empty(len(y))
        score[dev] = oof[name]
        score[test] = model.predict_proba(X[test][:, cols])[:, 1]
        scores[name] = score
        joblib.dump({"model": model, "columns": [FEATURES[c] for c in cols]}, out / f"{task}_{name}.joblib")
    scores["dev_selected_baseline"] = scores[winner].copy()
    export_rows(out / f"{task}_rows.csv.gz", rows, X, scores)
    write_json(out / f"{task}_dev_oof.json", {
        name: metrics(y[dev], s, groups[dev], queries[dev]) for name, s in oof.items()})
    results = {}
    for split, mask in (("dev", dev), ("test", test)):
        results[split] = {"counts": {"rows": int(mask.sum()), "positive": int(y[mask].sum()),
                                    "negative_or_unlabeled": int(mask.sum() - y[mask].sum()),
                                    "families": len(set(groups[mask])), "queries": len(set(queries[mask])),
                                    "prevalence": float(y[mask].mean())},
                          "metrics": {name: metrics(y[mask], s[mask], groups[mask], queries[mask])
                                      for name, s in scores.items()}}
    contrasts = ["dev_selected_baseline", "graph_FA", "graph_MA", "graph_MF"]
    # Each worker handles an entire contrast; BLAS capped at one, <= jobs workers.
    cis = joblib.Parallel(n_jobs=jobs, prefer="threads")(
        joblib.delayed(paired_bootstrap)(y[test], scores["graph_MFA"][test], scores[name][test],
                                       groups[test], bootstraps) for name in contrasts)
    results["paired_family_bootstrap"] = dict(zip(contrasts, cis))
    results["selected_baseline"] = winner
    results["seconds"] = time.perf_counter() - start
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/AI4BioEM/PrimmeKG_dataverse"))
    parser.add_argument("--run-name", default="frozen_full")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 8 or args.bootstraps < 1:
        parser.error("jobs must be 1..8 and bootstraps positive")
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be a single safe directory name")
    start = time.perf_counter()
    out = OUTPUT / args.run_name
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a frozen run.
    config = {"status": "EXPLORATORY_PREVIOUSLY_EXPOSED_PUBLIC_DATA", "caveats": CAVEATS,
              "created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
              "jobs": args.jobs, "bootstraps": args.bootstraps,
              "features": FEATURES, "baseline_order": BASELINES,
              "learner_hyperparameters": {n: model_spec(n, args.jobs)[0].get_params(deep=False).__repr__() for n in LEARNERS},
              "feature_definition": "Exact v2 M/F/A blocks. RA/AA use disease+query protein degree in task bipartite graph; disease-only RA and v2 inverse-hub also reported. IDF retains v2 constant 3000.",
              "split": "Frozen largest-family-first alternating dev/test; normalized complete ATC3 class profiles without drug-name prefixes, category fallback then singleton; exposure v2 name stems. Corrects v2 all-singleton drug-family bug; no label-dependent split search.",
              "selection": "Fixed hyperparameters, pooled dev 3-fold group-OOF AP; no test selection or tuning",
              "command": shlex.join([sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]]),
              "versions": {"python": sys.version, "executable": sys.executable,
                           "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                           "joblib": joblib.__version__, "platform": platform.platform()},
              "input_sha256": {name: sha256(args.data_dir / name) for name in ("kg.csv", "drug_features.csv")},
              "code_sha256": sha256(__file__),
              "context_code_sha256": {n: sha256(ROOT / "scripts" / n) for n in
                                       ("validate_primekg_rigorous_v2.py", "validate_primekg_cross_domain.py")}}
    write_json(out / "config.json", config)
    with (out / "benchmark_primekg_public_v3.py").open("x") as handle:
        handle.write(Path(__file__).read_text())
    print("Frozen config saved; loading covariates and labels separately", flush=True)
    cov, labels, names, sources = load_graph(args.data_dir / "kg.csv")
    with (args.data_dir / "drug_features.csv").open(newline="", encoding="utf-8") as handle:
        family_keys = {row["node_index"]: drug_family(row["node_index"], row) for row in csv.DictReader(handle)}
    membership = {
        "drug": frozen_memberships(cov.drug, {q: family_keys.get(q, f"id_{q}") for q in cov.drug}),
        "exposure": frozen_memberships(cov.exposure, {q: exposure_family(names.get(q, "")) for q in cov.exposure})}
    write_json(out / "membership.json", membership)
    write_json(out / "grouping_audit.json", {
        task: {"queries": len(rows), "families": len(set(r["group_id"] for r in rows)),
               "singleton_families": sum(n == 1 for n in Counter(r["group_id"] for r in rows).values()),
               "largest_family_sizes": sorted(Counter(r["group_id"] for r in rows).values(), reverse=True)[:10]}
        for task, rows in membership.items()})
    write_json(out / "feature_sources.json", sources)
    print("Frozen membership saved; building full two-hop datasets", flush=True)
    extractor = FeatureMatrix(cov)
    result = {"status": config["status"], "caveats": CAVEATS, "tasks": {}}
    with threadpool_limits(limits=1):
        for task in ("drug", "exposure"):
            rows, X, counts = make_dataset(cov, labels, task, membership[task], extractor)
            print(f"{task}: {len(rows)} rows; fitting dev-only comparators", flush=True)
            task_result = evaluate_task(task, rows, X, out, args.jobs, args.bootstraps)
            task_result["candidate_accounting"] = counts
            result["tasks"][task] = task_result
            write_json(out / f"{task}_results.json", task_result)
            print(f"{task}: selected={task_result['selected_baseline']}; test="
                  f"{task_result['test']['metrics']['graph_MFA']}; seconds={task_result['seconds']:.1f}", flush=True)
    result["total_seconds"] = time.perf_counter() - start
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(out / "summary.json", result)
    write_json(out / "artifact_sha256.json", {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(f"Complete: {out}; seconds={result['total_seconds']:.1f}; peak RSS={result['peak_rss_mib']:.1f} MiB", flush=True)
    return result


if __name__ == "__main__":
    main()