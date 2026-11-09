#!/usr/bin/env python3
"""Source-grounded screening and curated graph export; no LLM calls or legacy edges.

Run from any directory. Screening is provisional, not a 1000-paper quality audit.
Only curated observations enter the case graphs. No cross-study causal edges.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = (Path(__file__).resolve().parents[4] if Path(__file__).resolve().parents[1].name == 'codes' else Path(__file__).resolve().parents[1])
VERSION = "source-grounded-multiscale-1.1"
PAGE = re.compile(r"<!--\s*PAGE\s+(\d+)\s*-->")
NEURO = re.compile(r"\b(brain|neural|neuronal?|neurons|neuro\w*|hippocamp\w*|cerebr\w*|cogniti\w*|memory|synap\w*|electroencephal\w*|EEG|sleep|headache|blood.brain|dementia|alzheimer\w*|parkinson\w*)\b", re.I)
FIELD = re.compile(r"electromagnetic|magnetic field|radiofrequency|radio.frequency|microwave|RF.EMF|ELF.MF|\bEMF\b|\bMHz\b|\bGHz\b", re.I)
SPECIES = {
    "mouse": re.compile(r"\b(mouse|mice|Mus musculus)\b", re.I),
    "rat": re.compile(r"\b(rats?|Rattus norvegicus)\b", re.I),
    "human": re.compile(r"\b(humans?|Homo sapiens|volunteers?|participants?|donors?)\b", re.I),
}
SCALES = {
    "micro": re.compile(r"\b(mRNA|protein|gene|expression|ROS|oxidative|apoptosis|viability|proliferation|DNA|comet|calcium|ATP|enzyme|cytokine|mitochondri\w*|sperm|motility|cell cycle)\b", re.I),
    "meso": re.compile(r"\b(synap\w*|dendrit\w*|spine[ s]*|endocytosis|histolog\w*|histopath\w*|EEG|electroencephal\w*|local.field|theta.gamma|potentiation|blood.brain.barrier|tissue morphology)\b", re.I),
    "macro": re.compile(r"\b(memory|cogniti\w*|behavio\w*|maze|escape latency|recognition|headache|sleep quality|pain threshold|fertility index|pregnan\w*|mating|motor performance)\b", re.I),
}


def norm(text):
    return re.sub(r"\s+", " ", str(text).replace("\u00ad", "")).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:24]


def pages(text):
    matches = list(PAGE.finditer(text))
    return {int(m[1]): text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
            for i, m in enumerate(matches)}


def locate(text, quote, page=None):
    """Never promote a match elsewhere to a pass at a claimed page."""
    if not quote or len(norm(quote)) < 12:
        return {"status": "missing_or_short_quote", "page": page}
    pg = pages(text)
    search = [(page, pg.get(page, ""))] if page is not None else sorted(pg.items())
    for number, body in search:
        if quote in body:
            return {"status": "literal_page_match", "page": number}
        if norm(quote) in norm(body):
            return {"status": "whitespace_soft_hyphen_page_match", "page": number}
    return {"status": "unlocated_at_claimed_page" if page is not None else "unlocated", "page": page}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: cell(v) for k, v in row.items()} for row in rows)


def cell(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    # Prevent spreadsheet formula interpretation of source text.
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def workbook(path, sheets):
    book = Workbook()
    book.remove(book.active)
    for name, rows in sheets.items():
        sheet = book.create_sheet(name[:31])
        keys = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
        sheet.append(keys)
        for row in rows:
            sheet.append([cell(row.get(k)) for k in keys])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for c in sheet[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="173A53")
        for i, key in enumerate(keys, 1):
            sheet.column_dimensions[get_column_letter(i)].width = 45 if any(t in key for t in ("quote", "finding", "limitations", "path")) else 25
        for row in sheet.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(vertical="top", wrap_text=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)


def candidate_title(data, text, fallback):
    prov = data.get("provenance") or {}
    title = prov.get("title") or (prov.get("meta") or {}).get("title")
    if title:
        return str(title), "legacy_metadata_unreviewed"
    # Do not silently call a filename or running header a verified article title.
    for line in text[:18000].splitlines():
        clean = re.sub(r"[#*]", "", line).strip()
        if line.startswith("#") and len(clean.split()) >= 8 and len(clean) < 400 and NEURO.search(clean):
            return clean, "source_heading_candidate_not_verified"
    return fallback.replace("_", " "), "filename_label_not_verified_title"


def screen(json_dir, target, out):
    manifest, candidate_rows, endpoints, errors = [], [], [], []
    statuses = Counter()
    seen_sources = {}
    seen_titles = {}
    neuro_curated = json.loads((ROOT / 'data/multiscale_release/neuro_curated.json').read_text())
    reviewed = {p['paper_id']: p for p in neuro_curated['papers']}
    for path in sorted(json_dir.glob("*.json")):
        if path.name.startswith("_"):
            manifest.append({"json_path": str(path), "record_kind": "auxiliary_excluded"})
            continue
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or not data.get("paper_id"):
                manifest.append({"json_path": str(path), "record_kind": "nonpaper_excluded"})
                continue
            pid = str(data["paper_id"])
            status = data.get("extraction_status", "unknown")
            statuses[status] += 1
            md = Path(data.get("markdown_path") or f"/data/AI4BioEM/markdowns/{path.stem}.md")
            row = {"paper_id": pid, "json_path": str(path), "json_sha256": digest(path),
                   "extraction_status": status, "source_path": str(md), "record_kind": "paper",
                   "human_review_status": "not_performed", "selection_status": "excluded"}
            manifest.append(row)
            if not md.is_file():
                row["exclusion_reason"] = "missing_markdown"
                continue
            text = md.read_text(encoding="utf-8")
            row["source_sha256"] = digest(md)
            duplicate = seen_sources.get(row["source_sha256"])
            seen_sources.setdefault(row["source_sha256"], pid)
            row["duplicate_source_of"] = duplicate
            # Primary study signal is a legacy screening label, not a validated diagnosis.
            primary = (data.get("triage") or {}).get("is_primary_experimental_study") is True
            title, title_status = candidate_title(data, text, path.stem)
            if pid in reviewed:
                title, title_status = reviewed[pid]['title'], 'assistant_source_review_not_independent_human'
            # Trim references; front-page mentions are retrieval aids, never outcomes.
            refs = re.search(r"(?im)^\s*#{0,5}\s*\**(?:\d+[. ]*)?references\**\s*$", text)
            body = text[:refs.start()] if refs else text
            front = body[:18000]
            eligible_text = NEURO.search(front) and FIELD.search(front)
            if not eligible_text:
                row["exclusion_reason"] = "no_front_text_neuro_and_emf_signal"
                continue
            row.update(title=title, title_status=title_status, primary_screen=primary)
            row["neuro_retrieval_terms"] = sorted(set(m[0].lower() for m in NEURO.finditer(front)))
            local_eps = []
            # Inventory only: never concatenate the JSON dump or raw responses into evidence.
            for entry in data.get("comparison_inventory") or []:
                metric = entry.get("endpoint_name")
                if not metric:
                    continue
                for evidence in entry.get("evidence") or []:
                    quote = evidence.get("quote") or ""
                    check = locate(body, quote, evidence.get("page"))
                    if not check["status"].endswith("page_match"):
                        continue
                    # Preserve a candidate label even when semantic binding is unresolved.
                    scale = [k for k, pattern in SCALES.items() if pattern.search(metric)]
                    local_eps.append({"paper_id": pid, "comparison_id": entry.get("comparison_id"),
                                      "endpoint_raw": metric, "scale_candidates": scale or ["unresolved"],
                                      "quote": quote, "page": check["page"], "location_status": check["status"],
                                      "semantic_status": "not_reviewed", "source_path": str(md),
                                      "source_sha256": row["source_sha256"]})
            # Species is only a provisional label with a source-located context quote.
            bio = data.get("biological") or {}
            sp_evidence = []
            for ev in bio.get("species__evidence") or []:
                quote = ev.get("quote") or ""
                check = locate(body, quote, ev.get("page"))
                if check["status"].endswith("page_match"):
                    for species, pattern in SPECIES.items():
                        if pattern.search(quote):
                            sp_evidence.append({"species": species, "quote": quote, "page": check["page"],
                                                "status": "source_located_context_not_semantic_review"})
            species = sorted({s["species"] for s in sp_evidence})
            unique_eps = {(e["endpoint_raw"], e["quote"], e["page"]): e for e in local_eps}
            local_eps = list(unique_eps.values())
            scale_set = sorted({s for e in local_eps for s in e["scale_candidates"]})
            topic_support = bool(NEURO.search(title)) or any(NEURO.search(e['endpoint_raw']) for e in local_eps)
            title_key = re.sub(r'[^a-z0-9]', '', title.lower())
            duplicate_title = seen_titles.get(title_key) if len(title_key) > 40 and title_status != 'filename_label_not_verified_title' else None
            if len(title_key) > 40 and title_status != 'filename_label_not_verified_title':
                seen_titles.setdefault(title_key, pid)
            row.update(species_raw=bio.get("species"), model_type_raw=bio.get("model_type"),
                       species_candidates=species or ["unresolved"], species_evidence=sp_evidence,
                       scale_candidates=scale_set or ["unresolved"], located_endpoint_quotes=len(local_eps),
                       endpoint_labels=len({e["endpoint_raw"] for e in local_eps}),
                       page_markers=len(pages(text)), legacy_exposure_type=(data.get("exposure") or {}).get("exposure_type"))
            row['topic_support'] = 'neural_title_or_inventory_metric' if topic_support else 'front_text_only_may_be_background'
            row['duplicate_title_of'] = duplicate_title
            row["selection_score"] = (20 * int(bool(NEURO.search(title))) + 10 * int(primary)
                                      + min(10, len(local_eps)) + 3 * len(species)
                                      + 2 * len(set(scale_set) - {"unresolved"}))
            row["selection_status"] = "screening_candidate"
            row["exclusion_reason"] = ("duplicate_source" if duplicate else
                                       "duplicate_normalized_title" if duplicate_title else
                                       "neuro_topic_not_supported_by_title_or_metric" if not topic_support else
                                       "legacy_nonprimary_or_failed" if status in {"skipped_non_primary", "failed"} else
                                       "primary_design_unconfirmed" if not primary else
                                       "no_located_inventory_evidence" if not local_eps else "")
            if pid in reviewed and not duplicate and not duplicate_title:
                row['exclusion_reason'] = ''
                row['selection_score'] += 100
                row['quality_status'] = 'assistant_source_reviewed_selective_observations_not_complete_validation'
                row['species_candidates'] = [s for s, pattern in SPECIES.items() if pattern.search(reviewed[pid]['species'])] or ['unresolved']
                row['scale_candidates'] = sorted({o['scale'] for o in reviewed[pid]['observations']})
            row["provisional_eligible"] = not bool(row["exclusion_reason"])
            row.setdefault("quality_status", "provisional_screening_only_not_high_quality_certified")
            candidate_rows.append(row)
            endpoints.extend(local_eps)
        except (ValueError, OSError, TypeError, AttributeError) as exc:
            errors.append({"json_path": str(path), "error": str(exc)})
    candidate_rows.sort(key=lambda r: (-int(r["provisional_eligible"]), -r["selection_score"], r["paper_id"]))
    # Up to target provisionally eligible records; never pad with failed/nonprimary papers.
    selected = [r for r in candidate_rows if r["provisional_eligible"]][:target]
    screening_1000 = candidate_rows[:target]
    selected_ids = {r["paper_id"] for r in selected}
    for row in selected:
        row["selection_status"] = "selected_provisional_neuro_pool"
    eps = [e for e in endpoints if e["paper_id"] in selected_ids]
    distributions = []
    for dimension, key in (("species", "species_candidates"), ("scale", "scale_candidates")):
        counts = Counter(label for row in selected for label in set(row[key]))
        distributions.extend({"dimension": dimension, "label": label, "paper_count": count,
                              "denominator_selected_papers": len(selected), "multi_label": True,
                              "status": "automatic_candidate_not_semantic_validation"} for label, count in sorted(counts.items()))
    counts = Counter(e["endpoint_raw"] for e in eps)
    endpoint_stats = [{"endpoint_raw": label, "located_candidate_rows": count,
                       "papers": len({e["paper_id"] for e in eps if e["endpoint_raw"] == label}),
                       "status": "raw_label_not_ontology_standardized"} for label, count in counts.most_common()]
    summary = {"version": VERSION, "json_files": len(list(json_dir.glob('*.json'))),
               "paper_records": sum(statuses.values()), "legacy_status_counts": dict(statuses),
               "screening_candidates": len(candidate_rows), "provisionally_eligible": sum(r['provisional_eligible'] for r in candidate_rows),
               "target": target, "selected_provisional_papers": len(selected), "shortfall": max(0, target - len(selected)),
               "ranked_screening_list": len(screening_1000),
               "independently_human_validated_papers": 0, "located_inventory_rows": len(eps),
               "selection": "Deterministic source-front-text retrieval, legacy primary-design screen, source-hash deduplication, located inventory quote; score uses title relevance, evidence availability and scale/species coverage, never result direction.",
               "limitations": ["Not a systematic neuro corpus or 1000-paper quality certification.",
                               "Front-text window 18000 characters; retrieval false positives/negatives unmeasured.",
                               "Legacy primary labels and inventory metrics require semantic review; a located quotation may belong to a secondary citation.",
                               "Species labels are source-located contexts; scale is a metric-name rule, both provisional and multilabel.",
                               "Identical source hashes and identical normalized candidate metadata titles deduplicated; DOI, fuzzy-title, cohort and study-family deduplication unresolved.",
                               "No paper-level dose is copied to observations. No result significance inferred from keywords."],
               "errors": errors}
    write_csv(out / "corpus_manifest.csv", manifest)
    write_csv(out / "neuro_screening_candidates.csv", candidate_rows)
    write_csv(out / "neuro_pool_selected.csv", selected)
    write_csv(out / "neuro_screening_1000.csv", screening_1000)
    write_csv(out / "neuro_candidate_endpoints.csv", eps)
    write_csv(out / "neuro_pool_distributions.csv", distributions)
    write_csv(out / "neuro_endpoint_distribution.csv", endpoint_stats)
    workbook(out / "neuro_literature_pool.xlsx", {"Screening_1000_Not_Validated": screening_1000, "Selected_Provisional": selected, "All_Candidates": candidate_rows,
              "Species_Scale_Counts": distributions, "Raw_Endpoint_Counts": endpoint_stats,
              "Located_Candidates": eps, "Definitions": [{"item": "limitations", "value": summary['limitations']} ]})
    return summary, selected_ids


def concept(metric):
    """Coarse retrieval concepts, never asserted exact ontology equivalence."""
    patterns = [
        ("ROS-associated measurement", r"\bROS\b|superoxide"),
        ("DNA damage-associated measurement", r"DNA|H2AX|comet"),
        ("Sperm motility", r"motility|sperm movement"),
        ("Sperm concentration/count", r"sperm (?:concentration|count)"),
        ("Memory-task performance", r"memory|latency|recognition"),
        ("Neural protein/transcript abundance", r"GluN|channel protein|BDNF|PHOX"),
        ("Synaptic/network measurement", r"synap|endocytosis|potentiation|EEG|theta.gamma"),
        ("Tissue histopathology", r"patholog|histolog"),
        ("Pregnancy-based fertility index", r"Fertility index"),
        ("Cell viability", r"viability"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, metric, re.I):
            return label
    return "Unmapped: " + metric


def text_match(a, b):
    """Text agreement is not scientific equivalence; missing stays unknown."""
    if not a or not b:
        return "unresolved_missing"
    return "same_text_not_equivalence" if norm(a) == norm(b) else "different_text_requires_review"


def audit_bridge(a, b, relation, reason):
    """Pair-specific audit, without inferring experiment identity or concordance."""
    within = a['paper_id'] == b['paper_id']
    sa = sorted(k for k, p in SPECIES.items() if p.search(a.get('species', '')))
    sb = sorted(k for k, p in SPECIES.items() if p.search(b.get('species', '')))
    species_match = ("unresolved" if len(sa) != 1 or len(sb) != 1 else
                     "same_species_candidate" if sa == sb else "different_species_candidate")
    refs_a = [f"{a['paper_id']}:{a['observation_id']}"]
    refs_b = [f"{b['paper_id']}:{b['observation_id']}"]
    refs_a += [f"{a['paper_id']}:{c}" for c in a.get('context_refs', [])]
    refs_b += [f"{b['paper_id']}:{c}" for c in b.get('context_refs', [])]
    row = {
        "id": stable('bridge-audit-v1', a['observation_id'], b['observation_id'], relation),
        "source": 'obs:' + a['observation_id'], "target": 'obs:' + b['observation_id'],
        "relation": relation, "source_paper": a['paper_id'], "target_paper": b['paper_id'],
        "rationale": reason, "rule_version": "bridge-audit-v1",
        "connection_basis": "paper_identity_and_different_scales" if within else "shared_coarse_retrieval_concept",
        "paper_relationship": "same_paper" if within else "cross_paper",
        "same_experiment": "unresolved" if within else "unresolved_cross_paper_overlap_not_excluded",
        "same_subjects": "unresolved", "shared_controls": "unresolved",
        "study_family_memberships": sorted({a['paper_id'], b['paper_id']}),
        "study_family_status": "unresolved_paper_ids_are_not_independent_family_ids",
        "species_match": species_match, "source_species_candidates": sa,
        "target_species_candidates": sb, "material_match": "unresolved_requires_observation_level_review",
        "dose_match": text_match(a.get('exposure_text'), b.get('exposure_text')),
        "internal_dose_compatibility": "not_established",
        "time_match": text_match(a.get('time_text'), b.get('time_text')),
        "assay_match": text_match(a.get('assay'), b.get('assay')),
        "endpoint_match": text_match(a.get('metric'), b.get('metric')),
        "cross_scale": a['scale'] != b['scale'],
        "species_context_differs": a.get('species') != b.get('species'),
        "species_context_differs_meaning": "description_difference_only_not_taxonomic_difference",
        "source_evidence_refs": refs_a, "target_evidence_refs": refs_b,
        "supporting_evidence": {"scope": "endpoint_reports_only_not_causal_bridge", "refs": refs_a + refs_b},
        "opposing_evidence": [], "opposing_evidence_search": "not_systematically_assessed_not_absent",
        "inconsistency_assessment": "not_adjudicated_no_direction_concordance_inferred",
        "limitations": list(dict.fromkeys(a.get('limitations', []) + b.get('limitations', []))),
        "human_adjudication_status": "not_performed", "layer": "candidate",
        "allowed_use": "source_navigation_and_hypothesis_specification_only",
        "prohibited_use": "causal_claim;effect_pooling;cross_species_prediction;safety_decision;training_label",
        "causal_claim": False, "transportability": "not_established",
        "dosimetry_compatibility": "not_established",
    }
    for prefix, ob in [('source', a), ('target', b)]:
        for field in ('species', 'metric', 'assay', 'comparison', 'exposure_text', 'time_text',
                      'finding', 'quote', 'page', 'source_line', 'source_line_end', 'source_sha256'):
            row[prefix + '_' + field] = ob.get(field)
    return row


def build_case(curated_path, out, basename, selected_ids):
    curated = json.loads(curated_path.read_text())
    paper_rows, observations, provenance, nodes, edges, gaps = [], [], [], [], [], []
    def node(identifier, label, kind, scale="", species=""):
        nodes.append({"id": identifier, "label": label, "kind": kind, "scale": scale, "species": species})
    def edge(source, target, relation, **extra):
        edges.append({"id": stable(source, target, relation, extra), "source": source, "target": target,
                      "relation": relation, "causal_claim": False, **extra})
    ids = set()
    for paper in curated["papers"]:
        pid = paper["paper_id"]
        source = Path(paper["source_path"])
        text = source.read_text()
        source_hash = digest(source)
        paper_rows.append({k: v for k, v in paper.items() if k != "observations"} | {
            "source_sha256": source_hash, "in_selected_neuro_pool": pid in selected_ids,
            "pool_role": "neuro_case_purposive" if "neuro" in basename else "separate_reproductive_case_pool",
            "json_path": str(Path('/data/AI4BioEM/jsons') / (pid + '.json')),
            "human_review_status": "not_performed"})
        node("paper:" + pid, paper["title"], "Paper", species=paper["species"])
        all_ev = [("species", {**paper.get("species_evidence", {}), "quote": paper["species_quote"]})]
        all_ev += [(e["id"], e) for e in paper.get("context_evidence", [])]
        all_ev += [(o["observation_id"], o) for o in paper["observations"]]
        for eid, evidence in all_ev:
            check = locate(text, evidence["quote"], evidence.get("page"))
            if not check["status"].endswith("page_match"):
                raise ValueError(f"Source quote fails: {pid} / {eid}: {check}")
            start, end = evidence.get("source_line"), evidence.get("source_line_end")
            if start and end and norm(evidence["quote"]) not in norm("\n".join(text.splitlines()[start-1:end])):
                raise ValueError(f"Source line range fails: {pid} / {eid}")
            provenance.append({"evidence_id": f"{pid}:{eid}", "paper_id": pid, "quote": evidence["quote"],
                               "page": evidence.get("page"), "source_line": start, "source_line_end": end,
                               "source_path": str(source), "source_sha256": source_hash,
                               "location_status": check["status"], "normalization": "whitespace_and_soft_hyphen_only",
                               "semantic_review": "assistant_source_review_not_independent_human",
                               "human_review_status": "not_performed"})
        for ob in paper["observations"]:
            oid = ob["observation_id"]
            if oid in ids:
                raise ValueError("Duplicate observation ID: " + oid)
            ids.add(oid)
            if ob["scale"] not in {"micro", "meso", "macro"}:
                raise ValueError("Invalid measurement scale")
            context_ids = {c['id'] for c in paper.get('context_evidence', [])}
            if set(ob.get('context_refs', [])) - context_ids:
                raise ValueError("Unresolved context reference: " + oid)
            observations.append(ob | {"paper_id": pid, "species": paper["species"],
                                      "source_path": str(source), "source_sha256": source_hash,
                                      "concept": concept(ob["metric"]), "human_review_status": "not_performed",
                                      "quantitative_training_eligible": False, "study_family_status": "unresolved"})
            node("obs:" + oid, ob["metric"], "Observation", ob["scale"], paper["species"])
            # Conditions remain unique to an observation, even if text looks similar.
            node("context:" + oid, ob.get("exposure_text") or "Unresolved exposure", "ObservationContext")
            edge("paper:" + pid, "obs:" + oid, "REPORTS_OBSERVATION", evidence_id=f"{pid}:{oid}")
            edge("context:" + oid, "obs:" + oid, "OBSERVED_UNDER", evidence_id=f"{pid}:{oid}")
            node("scale:" + ob["scale"], ob["scale"], "MeasurementScale", ob["scale"])
            edge("obs:" + oid, "scale:" + ob["scale"], "MEASURED_AT_SCALE")
            label = concept(ob["metric"])
            cid = "concept:" + stable(label)
            node(cid, label, "RetrievalConcept")
            edge("obs:" + oid, cid, "RELATED_MEASUREMENT_CONCEPT", mapping_status="author_grouping_not_exact_equivalence")
            for limitation in ob.get("limitations", []):
                gaps.append({"observation_id": oid, "paper_id": pid, "limitation": limitation})
    bridges = []
    for i, a in enumerate(observations):
        for b in observations[i+1:]:
            within = a['paper_id'] == b['paper_id']
            cross_scale = a['scale'] != b['scale']
            if within and cross_scale:
                relation = "WITHIN_STUDY_CROSS_SCALE_CO_REPORT"
                reason = "Same paper reports different measurement scales; not necessarily same experiment, animals, or dose; no mediation inferred."
            elif not within and a['concept'] == b['concept'] and not a['concept'].startswith('Unmapped'):
                relation = "CROSS_STUDY_RELATED_MEASUREMENT"
                reason = "Coarse retrieval grouping only; not assay equivalence, matched-dose replication, conserved mechanism, or transportability."
            else:
                continue
            bridge = audit_bridge(a, b, relation, reason)
            bridges.append(bridge)
            extra = {k: v for k, v in bridge.items() if k not in {'source', 'target', 'relation', 'id', 'causal_claim'}}
            edge(bridge['source'], bridge['target'], relation, audit_id=bridge['id'], **extra)
    nodes = list({n['id']: n for n in nodes}.values())
    known = {n['id'] for n in nodes}
    assert all(e['source'] in known and e['target'] in known for e in edges)
    counts = dict(Counter(o['scale'] for o in observations))
    stats = {"papers": len(paper_rows), "observations": len(observations), "scales": counts,
             "nodes": len(nodes), "edges": len(edges), "bridges": len(bridges), "source_anchors_checked": len(provenance),
             "in_selected_neuro_pool": sum(r['in_selected_neuro_pool'] for r in paper_rows),
             "independent_human_review": "not_performed", "cross_study_causal_edges": 0}
    definitions = [{"item": k, "value": v} for k, v in {
        "version": VERSION, "scope": "Purposive source-reviewed examples; non-exhaustive, not systematic synthesis or extraction benchmark.",
        "micro": "Molecular or cellular measurement, including sperm motility and molecular assays in tissue.",
        "meso": "Tissue architecture, synaptic structure/function or neural network measurement, including EEG.",
        "macro": "Whole-organism behavior/perception, actual mating/pregnancy outcome; not isolated sperm function.",
        "solid_edges": "Source-linked reporting/context relationships, not causal arrows between biological outcomes.",
        "dashed_edges": "Conceptual cross-study retrieval correspondence; no dose equivalence or causal transportability.",
        "source_review": "Assistant read local Methods/Results; not independent expert annotation or adjudication.",
        "numeric_use": "Quantitative claims may appear in source-reviewed prose; no harmonized quantitative training labels exported.",
        "dependencies": "Rows and endpoints within papers are dependent; study-family deduplication unresolved.",
    }.items()]
    tables = {"Papers": paper_rows, "Observations": observations, "Provenance": provenance,
              "Nodes": nodes, "Edges": edges, "Cross_Study_Bridges": bridges,
              "Uncertainties": gaps, "Definitions": definitions,
              "Summary": [{"item": k, "value": v} for k, v in stats.items()]}
    workbook(out / (basename + '.xlsx'), tables)
    case_dir = out / basename
    for name, rows in tables.items():
        write_csv(case_dir / (name.lower() + '.csv'), rows)
    (case_dir / 'graph.json').write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json-dir', type=Path, default=Path('/data/AI4BioEM/jsons'))
    parser.add_argument('--output', type=Path, default=ROOT / 'data/multiscale_release')
    parser.add_argument('--target', type=int, default=1000)
    args = parser.parse_args()
    if args.target < 1:
        parser.error('target must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    summary, selected = screen(args.json_dir, args.target, args.output)
    for name, basename in [('neuro', 'case1_multiscale_neuro_data'), ('repro', 'case2_cross_species_repro_data')]:
        summary[name + '_case'] = build_case(ROOT / f'data/multiscale_release/{name}_curated.json', args.output, basename, selected)
    summary['generated_at'] = datetime.now(timezone.utc).isoformat()
    summary['builder_sha256'] = digest(__file__)
    summary['curation_sha256'] = {name: digest(ROOT / f'data/multiscale_release/{name}_curated.json') for name in ('neuro', 'repro')}
    (args.output / 'release_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('limitations', 'errors')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()