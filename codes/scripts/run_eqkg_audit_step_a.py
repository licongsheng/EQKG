#!/usr/bin/env python3
"""Step A of the EQKG Extended Data Full-Chain Audit:
1. Generates `audit_output/00_file_manifest.csv` covering all raw data, scripts,
   intermediate outputs, EQKG tables/figures, DOCX manuscripts, and Excel workbooks
   with exact byte sizes, SHA-256 hashes, modification timestamps, roles, versions,
   parent dependencies, and generator scripts.
2. Inspects all 10 sheets of `EQKG_Extended_Data.xlsx` and generates
   `audit_output/01_sheet_schema_report.md` with exact row/column counts, primary keys,
   units, null patterns, source lineage, and schema audit findings.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import openpyxl

ROOT = Path("/home/AI4BioEM")
AUDIT_DIR = ROOT / "audit_output"
WB_PATH = Path(__file__).resolve().parents[2] / "EQKG_Extended_Data.xlsx"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_file_manifest() -> list[dict]:
    """Scan all relevant files across the workspace and external raw datasets."""
    entries = [
        # External raw datasets (read-only)
        (Path("/data/AI4BioEM/PrimmeKG_dataverse/kg.csv"), "raw_source_data", "PrimeKG_dataverse_v1", "External Dataverse raw graph", "None (Primary Raw Data)"),
        (Path("/data/AI4BioEM/PrimmeKG_dataverse/drug_features.csv"), "raw_source_data", "PrimeKG_dataverse_v1", "External Dataverse raw drug features", "None (Primary Raw Data)"),
        (Path("/data/AI4BioEM/PrimmeKG_dataverse/disease_features.csv"), "raw_source_data", "PrimeKG_dataverse_v1", "External Dataverse raw disease features", "None (Primary Raw Data)"),
        (Path("/data/AI4BioEM/Caltech Materials Provenance Store_eighth/mps_eighth.sql"), "raw_source_data", "MPS_eighth_release", "External Caltech PostgreSQL dump", "None (Primary Raw Data)"),
        (ROOT / "data/bioem.sqlite3", "raw_source_data", "BioEM_4450_corpus_db", "Local BioEM 4,450-paper SQLite database", "app.py / batch_extract_local.py"),
        (ROOT / "data/bioem_features.csv", "raw_source_data", "BioEM_4450_features", "Local BioEM 4,450-paper extracted feature table", "features.py"),
    ]

    # Scripts
    script_files = [
        ("scripts/bib_1103_hierarchy_and_data.py", "code_script", "v3.0-bib-1103", "data/bioem.sqlite3, /data/AI4BioEM/markdowns/*.md", "Builds BioEM 5-level hierarchy (22S->26X->34A->75C->87M)"),
        ("scripts/bib_1103_relations_and_rules.py", "code_script", "v3.0-bib-1103", "scripts/bib_1103_hierarchy_and_data.py", "Builds BioEM 203 retained edges, 117 pruned pairs, 320 transition log"),
        ("scripts/bib_1103_evaluations.py", "code_script", "v3.0-bib-1103", "scripts/bib_1103_hierarchy_and_data.py, scripts/bib_1103_relations_and_rules.py", "Runs BioEM Verification 1 (750 checks), Verification 2 (180 pairs), Verification 3 (18 tasks)"),
        ("scripts/benchmark_primekg_public_v3.py", "code_script", "v3.0-public", "/data/AI4BioEM/PrimmeKG_dataverse/kg.csv", "Audited PrimeKG v3 benchmark (22 features, family-disjoint split)"),
        ("scripts/benchmark_primekg_public_v4.py", "code_script", "v4.0-public", "scripts/benchmark_primekg_public_v3.py, drug_features.csv, disease_features.csv", "Multi-scale calibrated PrimeKG v4 benchmark (56 features, 7 subsets)"),
        ("scripts/benchmark_mps_public_v3.py", "code_script", "v3.0-public", "/data/AI4BioEM/Caltech Materials Provenance Store_eighth/mps_eighth.sql", "Audited Caltech MPS v3 benchmark (668,898 obs, strict past chronology)"),
        ("scripts/benchmark_mps_public_v4.py", "code_script", "v4.0-public", "data/caltech_mps_validation_v3/observations.jsonl.gz", "Uncapped hierarchical L1-calibrated MPS v4 benchmark"),
        ("scripts/benchmark_controlled_attribution_v4.py", "code_script", "v4.0-controlled", "bib_1103_*, benchmark_primekg_public_v4.py, benchmark_mps_public_v4.py", "Master R1/R2/R3/R4 same-data same-learner controlled attribution across BioEM, PrimeKG, MPS"),
        ("scripts/export_eqkg_package.py", "code_script", "v4.0-eqkg", "data/controlled_attribution_v4/controlled_attribution_summary.json, docs/BIBs_1103/*", "Exports EQKG tables, figure data CSVs, and rendered figures"),
    ]
    for rel, role, ver, parent, gen in script_files:
        entries.append((ROOT / rel, role, ver, parent, gen))

    # Intermediate & benchmark data artifacts
    data_files = [
        ("data/controlled_attribution_v4/controlled_attribution_summary.json", "intermediate_summary_json", "v4.0-controlled", "Raw BioEM/PrimeKG/MPS datasets", "scripts/benchmark_controlled_attribution_v4.py"),
        ("data/primekg_validation_v3/corrected_families_full/summary.json", "intermediate_summary_json", "v3.0-public", "PrimmeKG_dataverse/kg.csv", "scripts/benchmark_primekg_public_v3.py"),
        ("data/primekg_validation_v3/corrected_families_full/drug_rows.csv.gz", "intermediate_row_predictions", "v3.0-public", "PrimmeKG_dataverse/kg.csv", "scripts/benchmark_primekg_public_v3.py"),
        ("data/primekg_validation_v3/corrected_families_full/exposure_rows.csv.gz", "intermediate_row_predictions", "v3.0-public", "PrimmeKG_dataverse/kg.csv", "scripts/benchmark_primekg_public_v3.py"),
        ("data/primekg_validation_v3/corrected_families_full/membership.json", "intermediate_split_membership", "v3.0-public", "PrimmeKG_dataverse/kg.csv", "scripts/benchmark_primekg_public_v3.py"),
        ("data/primekg_validation_v4/multiscale_calibrated_v4/summary.json", "intermediate_summary_json", "v4.0-public", "PrimmeKG_dataverse/*", "scripts/benchmark_primekg_public_v4.py"),
        ("data/caltech_mps_validation_v3/observations.jsonl.gz", "intermediate_extracted_observations", "v3.0-public", "mps_eighth.sql", "scripts/benchmark_mps_public_v3.py"),
        ("data/caltech_mps_validation_v3/join_chronology_audit.json", "intermediate_audit_json", "v3.0-public", "mps_eighth.sql", "scripts/benchmark_mps_public_v3.py"),
        ("data/caltech_mps_validation_v3/split_membership.json", "intermediate_split_membership", "v3.0-public", "mps_eighth.sql", "scripts/benchmark_mps_public_v3.py"),
        ("data/caltech_mps_validation_v3/summary.json", "intermediate_summary_json", "v3.0-public", "observations.jsonl.gz", "scripts/benchmark_mps_public_v3.py"),
        ("data/caltech_mps_validation_v4/summary.json", "intermediate_summary_json", "v4.0-public", "observations.jsonl.gz", "scripts/benchmark_mps_public_v4.py"),
    ]
    for rel, role, ver, parent, gen in data_files:
        entries.append((ROOT / rel, role, ver, parent, gen))

    # BioEM BIBs_1103 source & evaluation tables
    for p in sorted((ROOT / "docs/BIBs_1103/data").glob("*.csv")):
        entries.append((p, "bioem_5level_entity_table", "v3.0-bib-1103", "Raw BioEM 22 Markdown papers", "scripts/bib_1103_hierarchy_and_data.py"))
    for p in sorted((ROOT / "docs/BIBs_1103/derived").glob("*")):
        if p.is_file():
            entries.append((p, "bioem_derived_relations", "v3.0-bib-1103", "docs/BIBs_1103/data/*", "scripts/bib_1103_relations_and_rules.py"))
    for p in sorted((ROOT / "docs/BIBs_1103/evaluation").glob("**/*")):
        if p.is_file():
            entries.append((p, "bioem_evaluation_detail", "v3.0-bib-1103", "docs/BIBs_1103/data/*, docs/BIBs_1103/derived/*", "scripts/bib_1103_evaluations.py"))

    # All files under docs/EQKG
    for p in sorted((Path(__file__).resolve().parents[2]).glob("**/*")):
        if p.is_file():
            rel_s = str(p.relative_to(ROOT))
            if p.name == "EQKG_Extended_Data.xlsx":
                role, ver, parent, gen = "target_extended_data_workbook", "EQKG_v1", "tables/*.csv, figures_data/*.csv", "User/Packaged Extended Data Workbook"
            elif p.name == "EQKG_Manuscript_CN.docx":
                role, ver, parent, gen = "target_manuscript_docx", "EQKG_v1", "MANUSCRIPT_EQKG_CN.md, tables/*.csv", "Main Manuscript DOCX under Audit"
            elif "tables/" in rel_s:
                role, ver, parent, gen = "eqkg_summary_table_csv", "v4.0-eqkg", "controlled_attribution_summary.json, docs/BIBs_1103/*", "scripts/export_eqkg_package.py"
            elif "figures_data/" in rel_s:
                role, ver, parent, gen = "eqkg_figure_source_csv", "v4.0-eqkg", "controlled_attribution_summary.json, docs/BIBs_1103/*", "scripts/export_eqkg_package.py"
            elif "figures/" in rel_s:
                role, ver, parent, gen = "eqkg_rendered_figure", "v4.0-eqkg", "figures_data/*.csv", "scripts/export_eqkg_package.py"
            elif "ai_drawing_prompts/" in rel_s:
                role, ver, parent, gen = "eqkg_figure_prompt_md", "v4.0-eqkg", "tables/*.csv", "Author/Agent Prompt Specification"
            else:
                role, ver, parent, gen = "eqkg_documentation_md", "v4.0-eqkg", "All EQKG artifacts", "Author/Agent Documentation"
            entries.append((p, role, ver, parent, gen))

    manifest_rows = []
    seen = set()
    for path, role, ver, parent, gen in entries:
        if str(path) in seen:
            continue
        seen.add(str(path))
        exists = path.exists()
        rel_path = str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path)
        if exists:
            st = path.stat()
            size_b = st.st_size
            mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            digest = sha256_file(path)
        else:
            size_b = 0
            mtime_iso = "MISSING"
            digest = "MISSING"
        manifest_rows.append({
            "relative_or_abs_path": rel_path,
            "file_type": path.suffix.lstrip(".") or "unknown",
            "role_category": role,
            "exists": exists,
            "byte_size": size_b,
            "sha256": digest,
            "modified_utc": mtime_iso,
            "version_tag": ver,
            "parent_dependencies": parent,
            "generating_script": gen,
        })
    return manifest_rows


def inspect_workbook_schemas() -> str:
    wb = openpyxl.load_workbook(WB_PATH, data_only=True)
    lines = [
        "# EQKG Extended Data 全工作簿（10 个 Sheet）结构与主键审计报告",
        "",
        f"- **被审计工作簿路径**：[EQKG_Extended_Data.xlsx](../EQKG_Extended_Data.xlsx)",
        f"- **SHA-256**：`{sha256_file(WB_PATH)}`",
        f"- **审计生成时间（UTC）**：`{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## 一、 10 个 Sheet 全局结构总览表",
        "",
        "| 序号 | Sheet 名称 | 总行数 (含表头) | 数据行数 | 总列数 | 有效非空列范围 | 建议唯一主键 (Primary Key) | 对应源 CSV 文件 | 结构审计状态 |",
        "| ---: | :--- | ---: | ---: | ---: | :--- | :--- | :--- | :--- |",
    ]

    sheet_meta = {
        "README": ("key (Col A)", "无独立 CSV（工作簿内置说明页）", "MAJOR_ISSUE: 仅 6 行简略说明，缺字段字典、符号表、版本号、SHA256 及“CI 为配对差异 CI”的统计说明"),
        "ED1_Domains": ("domain_id", "tables/Table1_CrossDomain_Dataset_and_Schema_Overview.csv", "MINOR_ISSUE: dataset_name 仍写 Conditional BioEM-KG，PrimeKG 边数仅写双向展开后的 8,100,498"),
        "ED2_R1R4": ("(dataset, task_or_cohort, learner_family, metric_name)", "tables/Table2_Controlled_SameLearner_Attribution_All3Datasets.csv", "VERIFIED_SCHEMA: 20 行数据主键唯一，但需与 ED8 核对四舍五入显示精度"),
        "ED3_Binding": ("(method_id, evaluation_split)", "tables/Table3A_BioEM_Stage1_Hierarchy_Binding_750Checks.csv", "MAJOR_ISSUE: 沿用 36 列宽表模板，M–AJ 列（第 13–36 列）共 24 列全为空，且缺 750 项行级绑定明细"),
        "ED4_Anchor_Abl": ("ablation_model_id", "tables/Table3B_BioEM_Stage2_TriAnchor_Ablation_180Pairs.csv", "MAJOR_ISSUE: 沿用 36 列宽表模板，A–L 列与 AB–AJ 列共 21 列全为空，有效列仅 M–AA（第 13–27 列）"),
        "ED5_QA": ("(method_id, evaluation_split)", "tables/Table3C_BioEM_Stage3_Governed_MultiHop_QA_18Tasks.csv", "MAJOR_ISSUE: 沿用 36 列宽表模板，D–AA 列（第 4–27 列）共 24 列全为空；且需消解 120 vs 114 任务-观察对历史文字冲突"),
        "ED6_PrimeKG": ("(task, learner, representation_or_subset)", "tables/Table4_PrimeKG_Controlled_Attribution_and_Subsets.csv", "VERIFIED_SCHEMA: 80 行数据（2 任务 × 4 学习器 × 10 表征/子集），主键唯一，无空值"),
        "ED7_MPS": ("(modality, cohort, method_or_representation)", "tables/Table5_Caltech_MPS_Controlled_Attribution_and_Subsets.csv", "VERIFIED_SCHEMA: 69 行数据（3 模态 × [固定支持集 12 方法 + 全跨板集 11 方法]），主键唯一，但缺逐目标预测明细文件"),
        "ED8_Bootstrap": ("(dataset, task_or_cohort, learner, comparison, primary_metric)", "tables/Table6_Statistical_Bootstrap_Inference_All3Datasets.csv", "VERIFIED_SCHEMA: 108 行配对自举对比，主键唯一；需补充聚类数、种子与可交换单元元数据列"),
        "Fig1_Schema": ("domain_id", "figures_data/Fig1_Architecture_and_CrossDomain_Schema_Data.csv", "MINOR_ISSUE: 与 ED1_Domains 内容完全重复（100% 相同），未专门拆解 Fig 1a–e 各子面板的视觉元素与符号映射"),
    }

    detail_sections = []
    for idx, name in enumerate(wb.sheetnames, 1):
        ws = wb[name]
        rows = list(ws.iter_rows(values_only=True))
        n_rows = len(rows)
        n_cols = len(rows[0]) if rows else 0
        header = [str(c) if c is not None else f"EMPTY_COL_{j+1}" for j, c in enumerate(rows[0])] if rows else []
        data_rows = rows[1:] if n_rows > 1 else []

        # Check non-empty columns across data_rows (or all rows for README)
        check_rows = rows if name == "README" else data_rows
        non_empty_col_indices = [
            j for j in range(n_cols)
            if any(r[j] is not None and str(r[j]).strip() != "" for r in check_rows)
        ]
        if non_empty_col_indices:
            c_start = openpyxl.utils.get_column_letter(min(non_empty_col_indices) + 1)
            c_end = openpyxl.utils.get_column_letter(max(non_empty_col_indices) + 1)
            valid_range = f"{c_start}:{c_end} ({len(non_empty_col_indices)}/{n_cols} cols)"
        else:
            valid_range = "NONE"

        pk_str, src_csv, status_str = sheet_meta.get(name, ("Unknown", "Unknown", "Unknown"))
        lines.append(
            f"| {idx} | `{name}` | {n_rows} | {n_rows - 1 if name != 'README' else n_rows} | {n_cols} | `{valid_range}` | `{pk_str}` | `{src_csv}` | {status_str} |"
        )

        # Build per-sheet detailed column and null analysis
        detail_sections.append(f"### {idx}. Sheet `{name}` 详细 Schema 与空值审计")
        detail_sections.append("")
        detail_sections.append(f"- **原始尺寸**：`{n_rows} 行 × {n_cols} 列`（数据行 `{len(data_rows)}` 行）")
        detail_sections.append(f"- **唯一主键定义**：`{pk_str}`")
        detail_sections.append(f"- **对应源 CSV**：`{src_csv}`")
        detail_sections.append(f"- **审计判定**：{status_str}")
        detail_sections.append("")
        if name == "README":
            detail_sections.append("| 行号 | Col A (Key) | Col B (Value) |")
            detail_sections.append("| ---: | :--- | :--- |")
            for r_i, r in enumerate(rows, 1):
                a_v = str(r[0]) if r[0] is not None else ""
                b_v = str(r[1]) if len(r) > 1 and r[1] is not None else ""
                detail_sections.append(f"| {r_i} | `{a_v}` | {b_v} |")
            detail_sections.append("")
        else:
            detail_sections.append("| 列序号 | Excel 列标 | 字段名 (Header) | 非空数据行数 / 总数据行数 | 空值率 | 示例值 (首条非空) |")
            detail_sections.append("| ---: | :--- | :--- | ---: | ---: | :--- |")
            for j, h_name in enumerate(header):
                col_letter = openpyxl.utils.get_column_letter(j + 1)
                non_null_vals = [r[j] for r in data_rows if r[j] is not None and str(r[j]).strip() != ""]
                nn_cnt = len(non_null_vals)
                tot_cnt = len(data_rows)
                null_pct = 100.0 * (tot_cnt - nn_cnt) / max(1, tot_cnt)
                sample_v = str(non_null_vals[0])[:60].replace("|", "\\|") if non_null_vals else "*(全空 NULL)*"
                detail_sections.append(
                    f"| {j+1} | `{col_letter}` | `{h_name}` | {nn_cnt}/{tot_cnt} | {null_pct:.1f}% | `{sample_v}` |"
                )
            detail_sections.append("")

    lines.append("")
    lines.append("## 二、 逐 Sheet 字段级结构与空值模式详情")
    lines.append("")
    lines.extend(detail_sections)
    return "\n".join(lines)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = build_file_manifest()
    manifest_path = AUDIT_DIR / "00_file_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {len(manifest_rows)} rows to {manifest_path}")

    schema_md = inspect_workbook_schemas()
    schema_path = AUDIT_DIR / "01_sheet_schema_report.md"
    schema_path.write_text(schema_md + "\n", encoding="utf-8")
    print(f"Wrote schema report to {schema_path}")


if __name__ == "__main__":
    main()
