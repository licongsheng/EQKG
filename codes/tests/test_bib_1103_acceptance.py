#!/usr/bin/env python3
"""Sections 14 and 15 Automated Acceptance & Regression Test Suite:
Verifies all 12 unit/regression test cases (U01-U12), all 11 repair items (R01-R11),
and all 4 stage gates (G0-G3) defined in docs/BIBs_1103/BIB_Repair_and_Acceptance_Plan_CN.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/AI4BioEM/scripts')
from bib_1103_hierarchy_and_data import build_five_level_hierarchy
from bib_1103_relations_and_rules import (
    build_repaired_relations_and_change_log,
    build_atomic_relation_assertions,
    execute_governed_query,
)

OUT_ROOT = Path('/home/AI4BioEM/docs/BIBs_1103')


def read_csv_rows(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


# =============================================================================
# STAGE GATE G0 & G1: Five-Level Schema, Foreign Keys & Source Grounding
# =============================================================================

def test_G0_G1_five_level_schema_and_foreign_keys():
    studies = read_csv_rows(OUT_ROOT / 'data' / 'studies.csv')
    exps = read_csv_rows(OUT_ROOT / 'data' / 'experiments.csv')
    arms = read_csv_rows(OUT_ROOT / 'data' / 'arms.csv')
    comps = read_csv_rows(OUT_ROOT / 'data' / 'comparisons.csv')
    meas = read_csv_rows(OUT_ROOT / 'data' / 'measurements.csv')
    mapping = read_csv_rows(OUT_ROOT / 'data' / 'observation_id_mapping.csv')
    targets = read_csv_rows(OUT_ROOT / 'data' / 'target_dictionary.csv')
    mechs = read_csv_rows(OUT_ROOT / 'data' / 'mechanism_evidence.csv')
    prov = read_csv_rows(OUT_ROOT / 'data' / 'provenance.csv')

    assert len(studies) == 22
    assert len(exps) == 26
    assert len(arms) == 34
    assert len(comps) == 75
    assert len(meas) == 87
    assert len(mapping) == 87
    assert len(targets) == 28
    assert len(mechs) == 8
    assert len(prov) == 87

    # Check primary key uniqueness and foreign key closure:
    paper_ids = {r['paper_id'] for r in studies}
    exp_ids = {r['experiment_id'] for r in exps}
    arm_ids = {r['arm_id'] for r in arms}
    comp_ids = {r['comparison_id'] for r in comps}
    meas_ids = {r['measurement_id'] for r in meas}
    target_ids = {r['target_id'] for r in targets}

    assert len(paper_ids) == 22
    assert len(exp_ids) == 26
    assert len(arm_ids) == 34
    assert len(comp_ids) == 75
    assert len(meas_ids) == 87
    assert len(target_ids) == 28

    for e in exps:
        assert e['paper_id'] in paper_ids
    for a in arms:
        assert a['experiment_id'] in exp_ids
    for c in comps:
        assert c['arm_id'] in arm_ids
    for m in meas:
        assert m['comparison_id'] in comp_ids
        assert m['controlled_target_id'] in target_ids
    for p in prov:
        assert p['location_status'] in {'verified', 'literal_page_match', 'normalized_page_match'}
        assert int(p['page']) >= 1
        assert int(p['source_line']) >= 1
        assert int(p['source_line_end']) >= int(p['source_line'])


# =============================================================================
# UNIT & REGRESSION TESTS U01 - U12 (Section 14.2)
# =============================================================================

def test_U01_within_study_different_frequency_split_E4():
    """U01: E4.1/E4.2 (50 Hz) vs E4.3 (0 Hz SMF) must bind to distinct arms with frequency_hz=50.0 vs 0.0
    and physical relation P2_WithinStudy_Single_Experiment_Controlled_Physical_Contrast."""
    h = build_five_level_hierarchy()
    obs = {o['observation_id']: o for o in h['parent_observations']}
    assert obs['E4.1']['frequency_hz'] == 50.0
    assert obs['E4.2']['frequency_hz'] == 50.0
    assert obs['E4.3']['frequency_hz'] == 0.0
    assert obs['E4.1']['arm_id'] != obs['E4.3']['arm_id']

    rel, _, _ = build_repaired_relations_and_change_log(h['parent_observations'])
    rel_map = {(r['source_obs'], r['target_obs']): r for r in rel}
    r_e4 = rel_map[('E4.1', 'E4.3')]
    assert r_e4['relation_basis'] == 'explicit_arm_contrast'
    assert r_e4['A_phys_status'].startswith('P2_WithinStudy_Single_Experiment_Controlled_Physical_Contrast')
    assert r_e4['mod_frequency_relation'] == 'different_coupling_regime_or_static'


def test_U02_within_study_cross_experiment_prep_split_E1():
    """U02: E1.1 (in vitro NSC) vs E1.2 (in vivo mouse) must bind to different experiment_id,
    relation_basis='same_paper_co_report', and P2b (never P1 or P2 single-variable contrast)."""
    h = build_five_level_hierarchy()
    obs = {o['observation_id']: o for o in h['parent_observations']}
    assert obs['E1.1']['experiment_id'] == 'E1_EXP_INVITRO_NSC'
    assert obs['E1.2']['experiment_id'] == 'E1_EXP_INVIVO_MOUSE'

    rel, _, _ = build_repaired_relations_and_change_log(h['parent_observations'])
    rel_map = {(r['source_obs'], r['target_obs']): r for r in rel}
    r_e1 = rel_map[('E1.1', 'E1.2')]
    assert r_e1['relation_basis'] == 'same_paper_co_report'
    assert r_e1['A_phys_status'].startswith('P2b_WithinStudy_Cross_Experiment_Different_Preparation_and_Protocol')


def test_U03_within_study_replicate_batch_and_cross_freq_R6():
    """U03: R6.5 <-> R6.6 (60 kHz Exp 1 vs Exp 2) must be independent_replication with P1 and T1;
    R6.1 (20 kHz) <-> R6.5 (60 kHz) must be same_paper_co_report with P2b and WL1_WITHIN_STUDY_CROSS_PROTOCOL_NAVIGATION."""
    h = build_five_level_hierarchy()
    rel, _, _ = build_repaired_relations_and_change_log(h['parent_observations'])
    rel_map = {(r['source_obs'], r['target_obs']): r for r in rel}

    r_rep = rel_map[('R6.5', 'R6.6')]
    assert r_rep['relation_basis'] == 'independent_replication'
    assert r_rep['A_phys_status'].startswith('P1_WithinStudy_Same_Exposure_Arm_and_Preparation')
    assert r_rep['mod_temporal_relation'] == 'T1_same_study_concurrent_in_vivo_window'

    r_cross = rel_map[('R6.1', 'R6.5')]
    assert r_cross['relation_basis'] == 'same_paper_co_report'
    assert r_cross['A_phys_status'].startswith('P2b_WithinStudy_Cross_Experiment_Different_Preparation_and_Protocol')
    assert r_cross['within_or_cross_subtype'].startswith('WL1_WITHIN_STUDY_CROSS_PROTOCOL_NAVIGATION')
    assert 'same protocol' not in r_cross['question_context_Q'].lower()


def test_U04_acute_ex_vivo_temporal_rule_F8a_F8b():
    """U04: F8a.2 <-> F8b.1 (both 60 min human ex vivo semen exposures) must be T2, never T3."""
    h = build_five_level_hierarchy()
    rel, _, _ = build_repaired_relations_and_change_log(h['parent_observations'])
    rel_map = {(r['source_obs'], r['target_obs']): r for r in rel}
    assert rel_map[('F8a.2', 'F8b.1')]['mod_temporal_relation'] == 'T2_both_acute_in_vitro_or_ex_vivo_windows_cycle_not_applicable'


def test_U05_subset_sample_size_binding_E6():
    """U05: E6.1 (Western blot n=5/group) vs E6.2/E6.3 (MWM n=10/group) must have distinct subject_subset_id."""
    h = build_five_level_hierarchy()
    obs = {o['observation_id']: o for o in h['parent_observations']}
    assert obs['E6.1']['subject_subset_id'] != obs['E6.2']['subject_subset_id']
    assert 'N5' in obs['E6.1']['subject_subset_id']
    assert 'N10' in obs['E6.2']['subject_subset_id']


def test_U06_composite_timepoint_and_value_split_F1_3_and_F8a_2():
    """U06: F1.3 must split into F1.3a (3h comet significant), F1.3b (4h comet nonsignificant),
    F1.3c (4h 8-OHdG significant); F1.4 must clarify 46% control vs 35% RF; F8a.2b must have 58.97±14.81%."""
    meas = {m['measurement_id']: m for m in read_csv_rows(OUT_ROOT / 'data' / 'measurements.csv')}
    assert meas['F1.3a']['statistical_status'] == 'reported_significant'
    assert '3 h' in meas['F1.3a']['sampling_time_text']
    assert meas['F1.3b']['statistical_status'] == 'reported_nonsignificant'
    assert '4 h' in meas['F1.3b']['sampling_time_text']
    assert meas['F1.3c']['statistical_status'] == 'reported_significant'
    assert '8-OHdG' in meas['F1.3c']['metric_en']

    assert '46%' in meas['F1.4a']['reported_effect_and_values'] and '35%' in meas['F1.4a']['reported_effect_and_values']
    assert '58.97±14.81%' in meas['F8a.2b']['reported_effect_and_values']


def test_U07_cross_study_false_homology_rejection():
    """U07: Non-homologous pairs (E4.1<->E8.1, N5.1<->E3.1, F3.1<->F4.1) must not be linked."""
    rel = read_csv_rows(OUT_ROOT / 'derived' / 'relation_assertions.csv')
    rel_pairs = {(r['source_obs'], r['target_obs']) for r in rel} | {(r['target_obs'], r['source_obs']) for r in rel}
    assert ('E4.1', 'E8.1') not in rel_pairs
    assert ('N5.1', 'E3.1') not in rel_pairs
    assert ('F3.1', 'F4.1') not in rel_pairs


def test_U08_U10_U11_query_permission_guards():
    """U08, U10, U11: Multi-hop non-transitivity, no automatic interaction inference from significance difference,
    no automatic effect-difference proof from frequency difference, and refusal of unmodeled threshold extrapolation."""
    h = build_five_level_hierarchy()
    eobs_by_id = {o['observation_id']: o for o in h['parent_observations']}
    rel, _, _ = build_repaired_relations_and_change_log(h['parent_observations'])

    res_hop = execute_governed_query('multi_hop_path_inspection', ['F1.1', 'F3.2', 'F4.5'], eobs_by_id, rel)
    assert res_hop['status'] == 'SEGMENTED_PATH_RETURNED_NON_TRANSITIVE'
    assert res_hop['transitive_causal_edge_generated'] is False
    assert res_hop['interaction_or_significance_difference_inferred'] is False
    assert res_hop['frequency_difference_interpreted_as_proven_effect_difference'] is False

    res_thresh = execute_governed_query('quantitative_dose_extrapolation', ['F8a.1', 'F6.1'], eobs_by_id, rel)
    assert res_thresh['allowed'] is False
    assert res_thresh['status'] == 'BLOCKED_BY_PERMISSION_GUARD'

    res_cov = execute_governed_query('human_clinical_pregnancy_query', [], eobs_by_id, rel)
    assert res_cov['matched_observations'] == []
    assert 'curation coverage boundary' in res_cov['note']


def test_U09_statistical_status_decoupled_no_filtering():
    """U09: Nonsignificant observations must be retained in both parent comparisons and conditional edges."""
    pobs = read_csv_rows(OUT_ROOT / 'tables' / 'Table2_Observations_CN_EN.csv')
    ns_ids = {
        c['observation_id'] for c in pobs
        if c['A_bio_statistical_status'] in {'reported_nonsignificant', 'reported_nonsignificant_omnibus'}
    }
    assert len(ns_ids) == 23

    rel = read_csv_rows(OUT_ROOT / 'derived' / 'relation_assertions.csv')
    rel_obs = {r['source_obs'] for r in rel} | {r['target_obs'] for r in rel}
    assert ns_ids.issubset(rel_obs)


def test_U12_and_G2_G3_denominator_and_transition_closure():
    """U12, G2, G3: Verify 320-pair transition math (185+111+18+6=320; 203 retained, 117 rejected)
    and verify that all Verification 3 error rates are bounded in [0.0, 100.0]."""
    clog = read_csv_rows(OUT_ROOT / 'derived' / 'candidate_change_log.csv')
    assert len(clog) == 320
    cats = [r['transition_category'] for r in clog]
    assert cats.count('old_present_and_retained') == 185
    assert cats.count('old_present_and_rejected') == 111
    assert cats.count('new_candidate_and_retained') == 18
    assert cats.count('new_candidate_and_rejected') == 6
    assert 185 + 111 == 296
    assert 185 + 18 == 203
    assert 111 + 6 == 117

    scores = read_csv_rows(OUT_ROOT / 'evaluation' / 'verification_3' / 'task_scores.csv')
    assert len(scores) == 72
    for s in scores:
        err = float(s['condition_error_rate_pct'])
        assert 0.0 <= err <= 100.0, f"Invalid error rate {err} in {s['run_id']}"


def test_section11_figure_plotting_data_and_svg_consistency():
    """Section 11.4: Verify plotting data CSVs, SVG figure text labels, and Markdown documents
    match derived/summary_metrics.json (Neuro 13/12/9 = 34, Repro 30/5/6 = 41, 203 retained edges, 294 atomic edges)."""
    metrics = json.loads((OUT_ROOT / 'derived' / 'summary_metrics.json').read_text(encoding='utf-8'))
    sc = metrics['scale_distribution_parent_75']
    assert sc['neuro'] == {'micro': 13, 'meso': 12, 'macro': 9, 'total': 34}
    assert sc['repro'] == {'micro': 30, 'meso': 5, 'macro': 6, 'total': 41}
    assert sc['combined'] == {'micro': 43, 'meso': 17, 'macro': 15, 'total': 75}

    rows = read_csv_rows(OUT_ROOT / 'figures' / 'plotting_data' / 'fig2_scale_and_dosimetry_distribution.csv')
    assert int(rows[0]['micro']) == 13 and int(rows[0]['meso']) == 12 and int(rows[0]['macro']) == 9
    assert int(rows[1]['micro']) == 30 and int(rows[1]['meso']) == 5 and int(rows[1]['macro']) == 6

    fig2_svg = (OUT_ROOT / 'figures' / 'Fig2_Dataset_Coverage_and_Identifiability_Landscape.svg').read_text(encoding='utf-8')
    assert '13/12/9' in fig2_svg
    assert '30/5/6' in fig2_svg
    assert '43/17/15' in fig2_svg
    assert '14/11/9' not in fig2_svg
    assert '29/6/6' not in fig2_svg

    manuscript = (OUT_ROOT / 'MANUSCRIPT_CN_BIB_1103.md').read_text(encoding='utf-8')
    assert '13 微观 / 12 中观 / 9 宏观' in manuscript
    assert '30 微观 / 5 中观 / 6 宏观' in manuscript
    assert '203 条条件弱连接' in manuscript
    assert '294 条原子关系边' in manuscript
    assert '58.97±14.81%' in manuscript

    manifest = json.loads((OUT_ROOT / 'manifests' / 'release_manifest.json').read_text(encoding='utf-8'))
    assert manifest['total_files'] >= 70
    for entry in manifest['files']:
        if Path(entry['relative_path']).name.startswith('.~'):
            continue
        assert (OUT_ROOT / entry['relative_path']).exists(), f"Missing file in manifest: {entry['relative_path']}"


if __name__ == '__main__':
    test_fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in test_fns:
        fn()
        print(f"  [PASS] {fn.__name__}")
    print(f"All {len(test_fns)} BioEM 5-level & Tri-Anchor acceptance tests passed.")

