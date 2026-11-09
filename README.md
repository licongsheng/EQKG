# Evidence-Qualified Knowledge Graphs (EQKG) — Code & BioEM Dataset Release

This repository provides the official **Code Availability** (`codes/`) and **Data Availability** (`BioEM_dataset/`) materials for the study:

> **Evidence-Qualified Knowledge Graphs for Reliable Inference across Experimental Contexts**

## Repository Structure

- **[`codes/`](codes/README.md)**: Complete reproducibility code package (16 core Python scripts, 4 unit/acceptance test suites with 37 automated checks, locked environment manifests, and one-click reproduction script `reproduce_all_eqkg.sh`).
- **[`BioEM_dataset/`](BioEM_dataset/README.md)**: Complete BioEM benchmark dataset (92 files across 8 subdirectories with SHA-256 checksums in `DATASET_MANIFEST_SHA256.csv`, including the 4,450-study corpus overview, 22 primary study full-text Markdowns, five-level hierarchy $\mathcal{H}$, 203 parent / 294 atomic conditional relations, 117 pruned false-homology candidates, YAML governance rules $\Gamma$, frozen 13-dev / 9-holdout splits, 2,775 unordered pairs, 3,750-row subfield binding audit, and interactive HTML review workbench).

## Quick Verification

```bash
bash codes/reproduce_all_eqkg.sh --verify-only
```
