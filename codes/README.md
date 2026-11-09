# Code Availability & Reproducibility Package for *Evidence-Qualified Knowledge Graphs*

**Manuscript Title**: *Evidence-Qualified Knowledge Graphs for Reliable Inference across Experimental Contexts*  
**Framework**: Evidence-Qualified Knowledge Graphs ($\text{EQKG}$, $G_{\text{EQ}} = (\mathcal{H}, \mathcal{E}, \mathcal{Q}, \Gamma)$)  
**Package Path**: [codes](.)  
**Manifest & Checksums**: [CODE_MANIFEST_SHA256.csv](CODE_MANIFEST_SHA256.csv)  
**Environment Lock**: [environment_lock.json](environment_lock.json) | [requirements.txt](requirements.txt)

---

## 1. Ready-to-Use Code Availability Statement

> **Code Availability**  
> All source code for the Evidence-Qualified Knowledge Graph ($\text{EQKG}$) framework—including the five-level experimental scope parser and non-transitive path governance engine for BioEM, the multi-scale mechanistic qualification and family-disjoint evaluation pipeline for PrimeKG, the streaming PostgreSQL provenance parser and two-stage reference-envelope calibration pipeline for the Caltech Materials Provenance Store (MPS), the four-representation controlled attribution suite ($\mathcal{R}_1$–$\mathcal{R}_4$), the cluster-paired bootstrap statistical inference engine, and all figure/table generation scripts—is archived in `codes/` with SHA-256 checksums (`CODE_MANIFEST_SHA256.csv`) and deterministic random seeds (`20261106`, `19471`, `1103`).

---

## 2. Directory Structure & Script-to-Manuscript Mapping

All 16 core Python modules and 4 automated verification test suites are organized as follows:

| File in `codes/` | Role | Manuscript Sections, Figures & Extended Data Tables Supported |
| :--- | :--- | :--- |
| [reproduce_all_eqkg.sh](reproduce_all_eqkg.sh) | One-click reproduction & verification shell script | Verifies SHA-256 checksums, runs all unit tests, executes `--verify-only` or `--full-recompute`, and exports all tables/figures |
| [CODE_MANIFEST_SHA256.csv](CODE_MANIFEST_SHA256.csv) | Machine-readable file manifest with byte sizes & SHA-256 hashes | Full provenance audit of all 20 code and test files |
| [requirements.txt](requirements.txt) / [environment_lock.json](environment_lock.json) | Pinned Python 3.12.3 dependencies & thread-isolation flags | `numpy==2.3.2`, `scipy==1.16.1`, `scikit-learn==1.7.1`, `joblib==1.5.1`, `matplotlib==3.10.5`, `openpyxl==3.1.5` |
| [scripts/benchmark_controlled_attribution_v4.py](scripts/benchmark_controlled_attribution_v4.py) | **Master Controlled Attribution Pipeline ($\mathcal{R}_1$–$\mathcal{R}_4$)** | **Results 1, 3, 4; Methods 6; Fig. 2a–e, Fig. 3c, Fig. 4b–e, Fig. 5c–e; Extended Data Tables 2, 6, 7, 8** |
| [scripts/export_eqkg_package.py](scripts/export_eqkg_package.py) | **Master Table & Figure Exporter** | Generates all CSVs in [../tables/](../tables/), [../figures_data/](../figures_data/), and PDFs/PNGs in [../figures/](../figures/) |
| [scripts/bib_1103_hierarchy_and_data.py](scripts/bib_1103_hierarchy_and_data.py) | **Domain I (BioEM) Stage 1 Hierarchy Builder** | **Results 2; Methods 2 & 3.1; Fig. 3a,c; Extended Data Table 1 (`D1_BioEM`)** (22 studies $\to$ 26 experiments $\to$ 34 arms $\to$ 75 comparisons $\to$ 87 atomic measurements) |
| [scripts/bib_1103_relations_and_rules.py](scripts/bib_1103_relations_and_rules.py) | **Domain I (BioEM) Stage 2 & 3 Tri-Anchor & Non-Transitive Guard** | **Results 5; Methods 3.1 & 4.1; Fig. 6a,d,e** (203 retained parent edges, 294 atomic edges, 117 pruned pairs, 320 transition log) |
| [scripts/bib_1103_evaluations.py](scripts/bib_1103_evaluations.py) | **Domain I (BioEM) Verification 1–3 Suite** | **Results 2 & 5; Fig. 3b, Fig. 6b,c,f; Extended Data Tables 3 (`ED3_Binding`), 4 (`ED4_Anchor_Abl`), 5 (`ED5_QA`)** |
| [scripts/build_multiscale_assets.py](scripts/build_multiscale_assets.py) | BioEM verbatim source-quote locator & hash verifier | Verifies exact physical page and line spans in raw Markdown papers |
| [scripts/build_tri_anchored_weak_linkage_kg.py](scripts/build_tri_anchored_weak_linkage_kg.py) | BioEM primary study extractor | Base corpus definitions and source anchor probes across 22 BioEM studies |
| [scripts/tri_anchored_bilingual_and_verification.py](scripts/tri_anchored_bilingual_and_verification.py) | BioEM bilingual metadata & candidate pruning auditor | Supports the 320-candidate transition audit between 10 coarse buckets and 28 controlled targets |
| [scripts/benchmark_primekg_public_v3.py](scripts/benchmark_primekg_public_v3.py) | **Domain II (PrimeKG) Base Graph Loader & Split Engine** | **Methods 3.2 & 5; Extended Data Table 1 (`D2_PrimeKG`)** (bidirectional task-edge exclusion, family-disjoint grouping) |
| [scripts/benchmark_primekg_public_v4.py](scripts/benchmark_primekg_public_v4.py) | **Domain II (PrimeKG) Multi-Scale Tri-Anchor (`M, F, A`) Engine** | **Results 3; Methods 3.2; Fig. 4a–e; Extended Data Table 6 (`ED6_PrimeKG`)** |
| [scripts/validate_primekg_rigorous_v2.py](scripts/validate_primekg_rigorous_v2.py) | PrimeKG physicochemical & pharmacokinetic parser | Parses MW, TPSA, cLogP, BBB restriction, and half-life ($t_{1/2}$) without accessing indication text |
| [scripts/validate_primekg_cross_domain.py](scripts/validate_primekg_cross_domain.py) | PrimeKG anatomical/compartment classifier | Classifies CNS, skin/eye, and mucosal disease compartments |
| [scripts/benchmark_mps_public_v3.py](scripts/benchmark_mps_public_v3.py) | **Domain III (Caltech MPS) SQL Parser & Chronology Engine** | **Methods 2, 3.3 & 5; Extended Data Table 1 (`D3_Caltech_MPS`)** (668,898 provenance observations, strict past-timestamp synthesis history) |
| [scripts/benchmark_mps_public_v4.py](scripts/benchmark_mps_public_v4.py) | **Domain III (Caltech MPS) Hierarchical Tri-Anchor Engine** | **Results 4; Methods 3.3 & 4.2; Fig. 5a–e; Extended Data Table 7 (`ED7_MPS`)** |
| [scripts/run_eqkg_audit_step_a.py](scripts/run_eqkg_audit_step_a.py) | Full-Chain Audit Step A (File Manifest & Workbook Schema) | Generates `audit_output/00_file_manifest.csv` and `01_sheet_schema_report.md` |
| [scripts/run_eqkg_audit_step_b.py](scripts/run_eqkg_audit_step_b.py) | Full-Chain Audit Step B (Row/Pair/Target-Level Exporter) | Exports 3,750 BioEM binding checks, per-pair BioEM/PrimeKG predictions, per-disease AP, and 548,396 MPS target predictions |
| [tests/test_benchmark_primekg_public_v3.py](tests/test_benchmark_primekg_public_v3.py) | 10 semantic tests for PrimeKG | Verifies bidirectional label-edge exclusion, ATC prefix fix, and family isolation |
| [tests/test_benchmark_mps_public_v3.py](tests/test_benchmark_mps_public_v3.py) | 10 semantic tests for Caltech MPS | Verifies PostgreSQL COPY unescaping, 1-to-many join expansion, and past-only chronology |
| [tests/test_benchmark_public_v4.py](tests/test_benchmark_public_v4.py) | 5 semantic tests for v4 models & parallel bootstrap | Verifies outlier-robust weighted median, 7-subset coverage, and parallel bootstrap identity |
| [tests/test_bib_1103_acceptance.py](tests/test_bib_1103_acceptance.py) | Acceptance test suite for BioEM | Verifies 5-level foreign keys, quote provenance, 203/117 split, and non-transitivity lock |

---

## 3. Reproducibility & Execution Instructions

### 3.1 Fast Verification Mode (~5 seconds)

Verifies all 20 file SHA-256 checksums, runs the complete unit test suite across BioEM, PrimeKG, and Caltech MPS, and regenerates all summary tables and figures from the verified benchmark outputs:

```bash
bash codes/reproduce_all_eqkg.sh --verify-only
```

### 3.2 Full End-to-End Multi-Core Recomputation (~12–16 minutes on 32 CPU cores)

Re-executes all three domain benchmarks ($\text{BioEM}$, $\text{PrimeKG}$, and $\text{Caltech MPS}$), including the 1,000-replicate paired cluster bootstraps across $\mathcal{R}_1, \mathcal{R}_2, \mathcal{R}_3, \mathcal{R}_4$, and exports all tables and figures:

```bash
bash codes/reproduce_all_eqkg.sh --full-recompute 32
```

### 3.3 Exporting Row-Level / Target-Level Audit Artifacts

To generate all granular row-level, pair-level, per-disease, and per-target prediction files into `audit_output/`:

```bash
python3 -B codes/scripts/run_eqkg_audit_step_a.py
python3 -B codes/scripts/run_eqkg_audit_step_b.py
```
