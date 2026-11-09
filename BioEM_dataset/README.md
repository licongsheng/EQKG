# BioEM Dataset & Source-Verifiable Evidence Benchmark (`EQKG`)

**Manuscript Title**: *Evidence-Qualified Knowledge Graphs for Reliable Inference across Experimental Contexts*  
**Domain I**: Physics-Driven Biomedicine (Bioelectromagnetics: Central Nervous System Plasticity & Male Reproductive / Developmental Biology)  
**Dataset Path**: [BioEM_dataset](.)  
**File Checksums**: [DATASET_MANIFEST_SHA256.csv](DATASET_MANIFEST_SHA256.csv) (92 files, 34.54 MiB, 100% SHA-256 verified)

---

## 1. Ready-to-Use Data Availability Statement

> **Data Availability**  
> All datasets supporting the findings of this study are publicly available or included in the submission archive:
>
> 1. **BioEM Evidence-Qualified Knowledge Graph Benchmark (`BioEM_dataset/`)**: Contains the 4,450-study domain corpus overview (`01_corpus_4450_overview/`), the verbatim page-and-line-anchored Markdown source texts of all 22 primary experimental studies (`02_raw_22_studies_markdowns/`), the legacy single-record extraction baselines (`03_legacy_single_record_jsons_M1/`), the five-level experimental provenance hierarchy $\mathcal{H} = (\mathcal{S}, \mathcal{X}, \mathcal{A}, \mathcal{C}, \mathcal{M})$ covering 22 studies, 26 experiments, 34 exposure arms, 75 parent comparisons, 87 atomic measurements, 28 controlled measurement targets, and 8 external mechanistic modules (`04_five_level_hierarchy_H/`), the 203 retained parent conditional relations, 294 atomic conditional relations, 117 pruned false-homology candidates, and 320-pair transition audit (`05_qualified_relations_and_audit_E_Q/`), the formal YAML qualification and non-transitivity governance rules (`06_rules_and_governance_Gamma/`), the frozen 13-study development / 9-study holdout split, all 2,775 unordered observation-pair splits, the 3,750-row subfield binding audit, the 180-pair ablation benchmark, and the 18-task multiscale QA benchmark (`07_evaluations_and_splits/`), together with the complete machine-readable graph JSON and interactive HTML review workbench (`08_complete_json_workbooks_and_workbench/`). Every file is indexed with SHA-256 checksums in `DATASET_MANIFEST_SHA256.csv`.  
> 2. **PrimeKG**: Obtained from the Harvard Dataverse precision medicine knowledge graph release (`doi:10.7910/DVN/IXA7BM`, 129,375 nodes and 8,100,498 directed edges / 4,050,249 undirected relations).  
> 3. **Caltech Materials Provenance Store (MPS)**: Obtained from the CaltechDATA eighth public release (`doi:10.22002/D1.20079`, 668,898 provenance-linked electrochemical observations across 99,305 composition–modality groups and 186 element families).

---

## 2. Directory Architecture & Exact Denominators

| Subdirectory | Files | Key Contents & Exact Verified Denominators | Manuscript Sections, Figures & Extended Data Tables |
| :--- | ---: | :--- | :--- |
| [01_corpus_4450_overview/](01_corpus_4450_overview/) | 3 | `bioem_4450_corpus_features.csv` (4,450 primary experimental studies metadata & dosimetry/thermometry reporting landscape), `fig1_hierarchy_counts.csv`, `fig2_scale_and_dosimetry_distribution.csv` | **Methods 5; Results 2; Extended Data Table 1 (`D1_BioEM`)** |
| [02_raw_22_studies_markdowns/](02_raw_22_studies_markdowns/) | 22 | Verbatim converted Markdown files (`*.md`) for all 22 curated primary studies (`E1–E8, N5, N6, F1–F7, F8a, F8b, R2, R4, R6`) with physical `<!-- PAGE N -->` markers verified by `provenance.csv` | **Results 2 & 5; Methods 2 (Eq. 4, $\rho_u$)** |
| [03_legacy_single_record_jsons_M1/](03_legacy_single_record_jsons_M1/) | 22 | Historical single-record LLM extraction JSONs (`*.json`) evaluated as Baseline `M1` in the 750-check scope-binding benchmark | **Results 2; Fig. 3b; Extended Data Table 3 (`ED3_Binding`)** |
| [04_five_level_hierarchy_H/](04_five_level_hierarchy_H/) | 12 | `studies.csv` (**22**), `experiments.csv` (**26**), `arms.csv` (**34**), `comparisons.csv` (**75** parent comparisons: 34 Neuro `[13 micro / 12 meso / 9 macro]` + 41 Repro `[30 micro / 5 meso / 6 macro]`), `measurements.csv` (**87** atomic measurements), `observation_id_mapping.csv` (**87**), `target_dictionary.csv` (**28** controlled targets), `mechanism_evidence.csv` (**8** modules `MECH_01–08`), `provenance.csv` (**87** verbatim quotes), plus bilingual CN/EN master tables | **Results 2; Methods 2 & 3.1 (Eqs. 3–4, 8–9); Fig. 3a,c,d; Extended Data Table 1** |
| [05_qualified_relations_and_audit_E_Q/](05_qualified_relations_and_audit_E_Q/) | 9 | `relation_assertions.csv` (**203** retained parent edges: **105** within-study + **98** cross-study, including **65** cross-species and **55** cross-scale edges), `atomic_relation_assertions.csv` (**294** atomic edges), `rejected_relation_candidates.csv` (**117** pruned false-homology pairs), `candidate_change_log.csv` (**320** candidate transition audit: `185+111=296` old, `18+6=24` new), `summary_metrics.json`, plus bilingual CN/EN relation tables | **Results 5; Methods 3.1; Fig. 6a,c,d,e; Extended Data Tables 1 & 4** |
| [06_rules_and_governance_Gamma/](06_rules_and_governance_Gamma/) | 5 | Formal machine-readable YAML specifications: `physical_comparison_rules.yaml` (`P1–P5`), `biological_relation_rules.yaml`, `temporal_comparison_rules.yaml` (`T1–T5`), `candidate_rules.yaml`, `inference_permissions.yaml` ($\lambda(p), \omega(p,\xi)$) | **Methods 3.1 & 4.1 (Eqs. 8–9, 14–15); Fig. 1c,d** |
| [07_evaluations_and_splits/](07_evaluations_and_splits/) | 15 | `study_split_membership_22.csv` (**13 Dev studies [45 obs] vs 9 Holdout studies [30 obs]**), `unordered_pairs_2775_splits.csv` (**2,775 pairs = 990 Dev + 1,785 Holdout Test**), `bioem_row_level_binding_audit_3750.csv` (**3,750 subfield checks**), `bioem_pair_level_predictions.csv.gz` (**1,785 holdout pair R1–R4 predictions**), plus full Verification 1 (`750 checks`), Verification 2 (`180 stratified pairs`), and Verification 3 (`18 tasks, 120 task-obs pairs, 75 unique obs`) raw & summary files | **Results 1, 2, 5; Fig. 2b,e, Fig. 3b,c, Fig. 6b,c,f; Extended Data Tables 2, 3, 4, 5, 8** |
| [08_complete_json_workbooks_and_workbench/](08_complete_json_workbooks_and_workbench/) | 4 | `BioEM_EQKG_Complete_Graph_22Studies.json` (full bilingual graph JSON), `manual_review_workbench.html` (interactive browser audit tool), and 2 multi-sheet Excel workbooks | **Complete Standalone Graph & Interactive Review Package** |

---

## 3. Quick Integrity Verification

To verify all 92 dataset files against [DATASET_MANIFEST_SHA256.csv](DATASET_MANIFEST_SHA256.csv) and run the 12 BioEM structural/provenance acceptance tests:

```bash
python3 -B - <<'PY'
import csv, hashlib
from pathlib import Path

ds_dir = Path("BioEM_dataset") if Path("BioEM_dataset").exists() else Path(".")
rows = list(csv.DictReader((ds_dir / "DATASET_MANIFEST_SHA256.csv").open(encoding="utf-8")))
for r in rows:
    p = ds_dir / r["dataset_relative_path"]
    assert p.exists(), f"Missing {p}"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"], f"Hash mismatch on {p}"
print(f"Verified all {len(rows)}/92 BioEM_dataset files match SHA-256 checksums (100% pass).")
PY
```
