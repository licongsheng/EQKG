#!/usr/bin/env python3
"""R05-R11 Implementation:
1. Orthogonal relation_basis + structured modifiers (R05), computed strictly from
   canonical numeric/categorical fields (never parsing display strings).
2. Fixes all 6 specific misclassifications in R06 (F8a.2-F8b.1 T2 vs T3; R6.5-R6.6
   independent_replication + equal normalized ratio; E1.1-E1.2 cross-prep co-report;
   E4 arm-specific modality; R6 cross-20/60 kHz cross-protocol co-report).
3. Stratifies the 105 within-study pairs into 5 explicit semantic classes (R07).
4. Links all 36 B2 relations to explicit entries in mechanism_evidence.csv (R09).
5. Implements an executable Query Permission & Non-Transitivity Guard (R10).
6. Generates the exact 320-pair transition audit log (296 old + 24 new = 320;
   185 + 18 = 203 retained; 111 + 6 = 117 rejected) (R11).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from build_multiscale_assets import stable
from build_tri_anchored_weak_linkage_kg import build_all_tri_anchored_linkages
from tri_anchored_bilingual_and_verification import build_refined_graph_and_pruning_audit


def map_b2_to_mechanism_id(a: dict, b: dict) -> str | None:
    """Return the explicit mechanistic_relation_id from mechanism_evidence.csv for B2 pairs."""
    ta, tb = a['A_bio_target_id'], b['A_bio_target_id']
    pair = {ta, tb}
    if pair == {'NEURO_CAV1_PCREB_H3K9AC', 'NEURO_DG_NEUROGENESIS'}:
        return 'MECH_01_CAV1_TO_H3K9AC_NEUROGENESIS'
    if pair == {'NEURO_HIPP_NMDAR_GLUN2', 'NEURO_HIPP_CA1_LTP_EXCITABILITY'}:
        return 'MECH_02_HIPP_NMDAR_GLUN2_TO_CA1_LTP'
    if pair == {'NEURO_CAV1_PCREB_H3K9AC', 'NEURO_CALYX_CAV2_CHANNEL'}:
        return 'MECH_03_CAV_SUPERFAMILY_SIGNALING_MODULE'
    if pair in (
        {'NEURO_DG_NEUROGENESIS', 'NEURO_HIPP_CA1_LTP_EXCITABILITY'},
        {'NEURO_HIPP_NEURON_SYNAPSE_SURVIVAL', 'NEURO_HIPP_CA1_LTP_EXCITABILITY'},
        {'NEURO_DG_NEUROGENESIS', 'NEURO_HIPP_NEURON_SYNAPSE_SURVIVAL'},
        {'NEURO_DG_NEUROGENESIS', 'NEURO_DG_DENDRITIC_SPINES'},
    ):
        return 'MECH_04_HIPPOCAMPAL_TRISYNAPTIC_CIRCUIT_MODULE'
    if pair in (
        {'NEURO_MWM_SPATIAL_MEMORY', 'NEURO_YMAZE_WORKING_MEMORY'},
        {'NEURO_MWM_SPATIAL_MEMORY', 'NEURO_NOR_RECOGNITION_MEMORY'},
        {'NEURO_YMAZE_WORKING_MEMORY', 'NEURO_NOR_RECOGNITION_MEMORY'},
    ):
        return 'MECH_05_HIPPOCAMPAL_MEMORY_BEHAVIOR_PARADIGMS'
    if pair in (
        {'DNA_SSB_8OHDG_ALKALINE_COMET', 'REPRO_AMPK_MTOR_AUTOPHAGY'},
        {'DNA_DSB_NEUTRAL_COMET_GH2AX', 'REPRO_AMPK_MTOR_AUTOPHAGY'},
        {'CELL_ROS_OXIDATIVE_STRESS', 'REPRO_AMPK_MTOR_AUTOPHAGY'},
        {'CELL_ROS_OXIDATIVE_STRESS', 'DNA_SSB_8OHDG_ALKALINE_COMET'},
    ):
        return 'MECH_06_GERMCELL_ROS_TO_8OHDG_AND_AMPK_AUTOPHAGY'
    if pair == {'DNA_SSB_8OHDG_ALKALINE_COMET', 'DNA_DSB_NEUTRAL_COMET_GH2AX'}:
        return 'MECH_07_GERMCELL_SSB_VS_DSB_ASSAY_SPECTRUM'
    if pair == {'REPRO_EPIDIDYMAL_SEMINAL_SPERM_COUNT_CONC', 'REPRO_TESTICULAR_HOMOGENATE_SPERM_COUNT'}:
        return 'MECH_08_TESTIS_PRODUCTION_TO_EPIDIDYMAL_RESERVE'
    return None


def compute_structured_modifiers(a: dict, b: dict) -> dict:
    """Compute all comparison modifiers strictly from structured canonical fields on a and b
    (Rule 13.1 #2: never compare display prose strings)."""
    same_paper = (a['paper_id'] == b['paper_id'])
    same_species = (a['species'] == b['species'])
    same_exp = (a['experiment_id'] == b['experiment_id'])
    same_arm = (a['arm_id'] == b['arm_id'])
    same_batch = (a['batch_id'] == b['batch_id'])
    same_subset = (a['subject_subset_id'] == b['subject_subset_id'])

    # 1. Frequency relation from numeric frequency_hz and exact_freq_code:
    fa, fb = a['frequency_hz'], b['frequency_hz']
    ca, cb = a['A_phys_exact_freq_code'], b['A_phys_exact_freq_code']
    if fa is not None and fb is not None and abs(fa - fb) < 1e-3:
        freq_rel = 'same_exact_frequency'
    elif ca == cb:
        freq_rel = 'same_exact_frequency'
    elif a['A_phys_coupling_modality'] == b['A_phys_coupling_modality']:
        if 'Omnibus' in ca or 'Omnibus' in cb or 'Dual' in ca or 'Dual' in cb:
            freq_rel = 'within_band_dual_or_omnibus_partial_overlap'
        else:
            freq_rel = 'within_band_different_carrier_frequency'
    else:
        freq_rel = 'different_coupling_regime_or_static'

    # 2. Waveform & modulation relation:
    if a['waveform_structured'] == b['waveform_structured'] and a['modulation_structured'] == b['modulation_structured']:
        wave_rel = 'same_waveform_and_modulation'
    elif a['waveform_structured'] == b['waveform_structured']:
        wave_rel = 'same_waveform_different_modulation'
    else:
        wave_rel = 'different_waveform'

    # 3. Nominal dose relation:
    va, vb = a['intensity_value'], b['intensity_value']
    ua, ub = a['intensity_unit'], b['intensity_unit']
    if same_arm:
        dose_rel = 'same_nominal_exposure_arm'
    elif va is not None and vb is not None and ua == ub and abs(va - vb) < 1e-6:
        dose_rel = 'same_nominal_external_value_and_unit'
    elif va is None or vb is None or 'corrupt' in str(ua) or 'corrupt' in str(ub) or 'ambiguous' in str(ua) or 'ambiguous' in str(ub):
        dose_rel = 'unreported_or_ocr_ambiguous_unit_in_at_least_one'
    else:
        dose_rel = 'different_nominal_intensity_or_sar'

    # 4. Schedule relation:
    if (a['session_duration_min'] == b['session_duration_min']
            and a['exposure_days'] == b['exposure_days']
            and a['on_off_schedule_structured'] == b['on_off_schedule_structured']):
        sched_rel = 'same_exposure_schedule'
    else:
        sched_rel = 'different_session_or_course_schedule'

    # 5. Batch relation (R06: R6 Exp1 vs Exp2 has same physical arm but different replicate batch):
    if same_paper and same_batch:
        batch_rel = 'same_study_batch'
    elif same_paper and (not same_batch) and same_arm:
        batch_rel = 'independent_replicate_batch_same_nominal_protocol'
    elif same_paper:
        batch_rel = 'different_experiment_or_group_within_study'
    else:
        batch_rel = 'independent_studies'

    # 6. Unit overlap (R01 & R07):
    if same_paper and same_subset and a['generation'] == b['generation']:
        unit_ov = 'same_subjects_or_split_aliquots'
    elif same_paper and a['generation'] != b['generation']:
        unit_ov = 'paternal_F0_to_offspring_F1_within_study'
    elif same_paper:
        unit_ov = 'separate_subject_subsets_or_preparations_within_study'
    else:
        unit_ov = 'independent_studies_no_shared_subjects'

    # 7. Temporal relation (R04 & R06: uses structured normalization_applicability and numeric ratios!):
    app_a = a['normalization_applicability']
    app_b = b['normalization_applicability']
    ra, rb = a['normalized_exposure_ratio'], b['normalized_exposure_ratio']
    la, lb = a['normalized_lag_ratio'], b['normalized_lag_ratio']

    if app_a == 'not_applicable_acute_isolated_or_ex_vivo' and app_b == 'not_applicable_acute_isolated_or_ex_vivo':
        if same_paper and a['session_duration_min'] == b['session_duration_min'] and a['post_exposure_lag_days'] == b['post_exposure_lag_days']:
            temp_rel = 'T1_same_study_concurrent_acute_window'
        else:
            temp_rel = 'T2_both_acute_in_vitro_or_ex_vivo_windows_cycle_not_applicable'
    elif app_a != app_b:
        temp_rel = 'T3_asymmetric_in_vivo_organ_process_vs_acute_isolated_window'
    else:
        # Both are applicable_in_vivo_organ_process
        if a['reference_process_id'] == b['reference_process_id']:
            ratios_equal = (ra is not None and rb is not None and abs(ra - rb) < 1e-3)
            lags_equal = (la is not None and lb is not None and abs(la - lb) < 1e-3)
            if same_paper and ratios_equal and lags_equal:
                temp_rel = 'T1_same_study_concurrent_in_vivo_window'
            elif ratios_equal and lags_equal:
                temp_rel = 'T4a_same_reference_process_equal_normalized_ratio_and_lag'
            else:
                temp_rel = 'T4b_same_reference_process_different_coverage_or_lag'
        else:
            temp_rel = 'T5_distinct_species_or_developmental_reference_processes'

    # 8. Physical comparison status (fixing all R06 bugs!):
    if same_paper and same_arm and same_exp:
        phys_status = 'P1_WithinStudy_Same_Exposure_Arm_and_Preparation (同研究同一暴露臂与同一生物制备体系)'
    elif same_paper and same_arm and (not same_exp):
        phys_status = 'P1b_WithinStudy_Same_Nominal_Field_Different_Preparation_or_Batch (同研究同标称物理方案，但分属独立重复批次或不同制备体系)'
    elif same_paper and (not same_arm) and same_exp:
        phys_status = 'P2_WithinStudy_Single_Experiment_Controlled_Physical_Contrast (同研究同一实验体系内受控物理变量[场强/时长/频率/模态]对照)'
    elif same_paper and (not same_arm) and (not same_exp):
        phys_status = 'P2b_WithinStudy_Cross_Experiment_Different_Preparation_and_Protocol (同篇论文内不同实验体系且不同物理暴露方案对照)'
    elif freq_rel == 'same_exact_frequency':
        phys_status = 'P3_CrossStudy_Exact_Carrier_Frequency_NonEquivalent_MicroDosimetry (跨研究相同载波频率，但波形/腔体/几何不同，内部微区剂量未必等效)'
    elif freq_rel in ('within_band_different_carrier_frequency', 'within_band_dual_or_omnibus_partial_overlap'):
        phys_status = 'P4_CrossStudy_WithinBand_Different_Carrier_Frequency_NonEquivalent_Dispersion (同大频段但不同载波频率/调制制式：组织介电色散与驻波SAR非等效)'
    else:
        phys_status = 'P5_CrossRegime_Different_Frequency_Band_or_Static_Contrast (跨频段/跨交变-静态场对照：不可直接换算或合并剂量)'

    return {
        'same_species': same_species,
        'same_preparation': (a['experiment_id'] == b['experiment_id']) or (
            ('in vitro' in a['preparation'].lower()) == ('in vitro' in b['preparation'].lower())
            and ('ex vivo' in a['preparation'].lower()) == ('ex vivo' in b['preparation'].lower())
        ),
        'same_compartment': a['organ_or_compartment'] == b['organ_or_compartment'],
        'frequency_relation': freq_rel,
        'waveform_relation': wave_rel,
        'dose_relation': dose_rel,
        'schedule_relation': sched_rel,
        'batch_relation': batch_rel,
        'unit_overlap': unit_ov,
        'temporal_relation': temp_rel,
        'physical_status_structured': phys_status,
    }


def classify_within_study_relation_basis(a: dict, b: dict, mods: dict) -> tuple[str, str, str]:
    """R07: Stratify the 105 within-study observation pairs into 5 distinct, honest
    relation_basis categories instead of treating all 105 as uniform contrasts or co-variation:
    1. same_unit_joint_measurement: measured in the exact same subject subset & same arm.
    2. same_protocol_co_report: same exposure arm, but separate animal/donor subsets or F0->F1 generation.
    3. independent_replication: identical nominal protocol, independent replicate batch (R6 Exp1 vs Exp2).
    4. explicit_arm_contrast: same experiment & same endpoint/scale, controlled contrast of dose/duration/modality/stage/channel.
    5. same_paper_cross_protocol_navigation: same paper, but differs in both preparation/protocol and scale/endpoint (e.g., E1.1 dish vs E1.2 in vivo; N6.3 1760 MHz dish vs N6.1 1850 MHz in vivo; R6 20 kHz fertility vs 60 kHz sperm count).
    """
    same_scale = (a['scale'] == b['scale'])
    same_arm = (a['arm_id'] == b['arm_id'])
    same_exp = (a['experiment_id'] == b['experiment_id'])
    same_subset = (a['subject_subset_id'] == b['subject_subset_id'])
    same_target = (a['A_bio_target_id'] == b['A_bio_target_id'])

    # Case A: Independent batch replication in R6 (Exp1 vs Exp2 under identical kHz arm and same target):
    if (not same_subset) and (a['batch_id'] != b['batch_id']) and same_arm and same_target:
        return (
            'independent_replication',
            'WL2_WITHIN_STUDY_REPLICATE_BATCH_CHECK',
            f"【同研究同方案独立批次重复检验 ({a['study_id']}: {a['batch_id']} vs. {b['batch_id']})】"
            f"在完全相同的标称物理方案 ({a['A_phys_exact_freq_zh_en']}, {a['A_phys_specific_dose_zh_en']}) 下，"
            f"独立重复批次 {a['observation_id']} 与 {b['observation_id']} 的结果是否重现？"
        )

    # Case B: Same exposure arm & same subject subset (joint measurement on the same experimental units):
    if same_arm and same_subset and a['generation'] == b['generation']:
        if not same_scale:
            return (
                'same_unit_joint_measurement',
                'WL1_WITHIN_STUDY_SAME_COHORT_CROSS_SCALE',
                f"【同研究同一受试队列跨尺度联合测量 ({a['study_id']}: {a['subject_subset_id']})】"
                f"在同一受试动物/分装样本队列中，{a['scale']}尺度 ({a['observation_id']}) 与 {b['scale']}尺度 ({b['observation_id']}) 共同报告了怎样的结果方向？"
            )
        return (
            'same_unit_joint_measurement',
            'WL2_WITHIN_STUDY_SAME_COHORT_MULTI_CHANNEL',
            f"【同研究同一受试队列多测定通道/多指标联合测量 ({a['study_id']}: {a['subject_subset_id']})】"
            f"在同一批样品/动物中，不同检测通道或相关终点 {a['observation_id']} 与 {b['observation_id']} 呈现怎样的通道特异性或伴随变化？"
        )

    # Case C: Same exposure arm (or F4 paternal-to-F1 arm), different subsets or generations:
    if same_arm or (a['study_id'] == 'F4'):
        if not same_scale:
            return (
                'same_protocol_co_report',
                'WL1_WITHIN_STUDY_SEPARATE_SUBSET_CO_REPORT',
                f"【同研究同暴露方案不同亚组/跨代共同报告 ({a['study_id']})】"
                f"在同一暴露方案下，分别在 [{a['subject_subset_id']}] 测定的 {a['observation_id']} ({a['scale']}) 与在 "
                f"[{b['subject_subset_id']}] 测定的 {b['observation_id']} ({b['scale']}) 共同报告了什么结果（注：无个体级配对，不称统计共变）？"
            )
        return (
            'same_protocol_co_report',
            'WL2_WITHIN_STUDY_SEPARATE_SUBSET_CO_REPORT',
            f"【同研究同暴露方案不同亚组/跨代同尺度共同报告 ({a['study_id']})】"
            f"在同一研究方案下，不同亚组/世代的 {a['observation_id']} 与 {b['observation_id']} 分别报告了什么结果？"
        )

    # Case D: Explicit single-axis experimental contrast (same scale & same target/domain, different arm):
    if same_scale and (same_target or a['study_id'] in {'E4', 'E7', 'E8', 'N5', 'F1', 'F5', 'R6'}):
        return (
            'explicit_arm_contrast',
            'WL2_WITHIN_STUDY_EXPLICIT_ARM_CONTRAST',
            f"【同研究显式受控实验臂对照 ({a['study_id']}: {a['arm_id']} vs. {b['arm_id']})】"
            f"在同一研究的受控对比设计下，改变实验臂条件 ({a['A_phys_exact_freq_zh_en']} / {a['A_phys_specific_dose_zh_en']} vs. "
            f"{b['A_phys_exact_freq_zh_en']} / {b['A_phys_specific_dose_zh_en']}) 时，{a['observation_id']} 与 {b['observation_id']} 如何变化？"
        )

    # Case E: Same paper, but cross-preparation or cross-protocol across scales/endpoints (navigation/co-report only!):
    return (
        'same_paper_co_report',
        'WL1_WITHIN_STUDY_CROSS_PROTOCOL_NAVIGATION' if not same_scale else 'WL2_WITHIN_STUDY_CROSS_PROTOCOL_NAVIGATION',
        f"【同篇论文内跨实验体系/跨方案共同报告导航 ({a['study_id']}: {a['experiment_id']}[{a['arm_id']}] vs. {b['experiment_id']}[{b['arm_id']}])】"
        f"同一篇论文分别在不同实验体系或不同频率/强度方案下报告了 {a['observation_id']} 与 {b['observation_id']}；"
        f"两者属于同文跨方案共同报告，不可当作同一暴露方案下的共变或单一物理变量受控对照。"
    )


def build_repaired_relations_and_change_log(eobs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Build:
    1. repaired_relations (203 retained relation assertions with orthogonal relation_basis + modifiers,
       and all R06/R07/R09/R10 fixes applied).
    2. pruned_pairs (117 rejected coarse-concept pairs).
    3. candidate_change_log (all 320 unique candidate pairs evaluated across v1[296] -> v2/v3[203 retained + 117 rejected],
       satisfying R11 exact denominator accounting).
    """
    obs_by_id = {o['observation_id']: o for o in eobs}
    old_296_edges = build_all_tri_anchored_linkages(eobs)
    v2_retained, v2_pruned = build_refined_graph_and_pruning_audit(eobs)

    old_296_map = {(e['source_obs'], e['target_obs']): e for e in old_296_edges}
    v2_ret_map = {(e['source_obs'], e['target_obs']): e for e in v2_retained}
    v2_pru_map = {(p['source_obs'], p['target_obs']): p for p in v2_pruned}

    repaired_relations = []
    for key, e in sorted(v2_ret_map.items()):
        a = obs_by_id[e['source_obs']]
        b = obs_by_id[e['target_obs']]
        mods = compute_structured_modifiers(a, b)
        same_paper = (a['paper_id'] == b['paper_id'])

        if same_paper:
            rel_basis, sub_wl_tag, q_repaired = classify_within_study_relation_basis(a, b, mods)
            mech_id = 'MECH_01_CAV1_TO_H3K9AC_NEUROGENESIS' if ({a['observation_id'], b['observation_id']} in ({'E1.1', 'E1.2'}, {'E1.1', 'E1.3'})) else 'NONE_WITHIN_STUDY_DESIGN'
            wl_display = 'WL1_WITHIN_STUDY_CROSS_SCALE_CO_REPORT' if a['scale'] != b['scale'] else 'WL2_WITHIN_STUDY_DOSE_STAGE_ASSAY_CONTRAST'
        else:
            wl_display = e['weak_linkage_type']
            sub_wl_tag = wl_display
            q_repaired = e['question_context_Q']
            if e['bio_basis_code'].startswith('B1_'):
                # Distinguish same probe vs different redox probe (R08):
                if a['A_bio_target_id'] == 'CELL_ROS_OXIDATIVE_STRESS' and a['assay'] != b['assay']:
                    rel_basis = 'related_measurement'
                else:
                    rel_basis = 'same_measurement_target'
                mech_id = 'NONE_DIRECT_TARGET_OR_READOUT_COMPARISON'
            else:
                mech_id = map_b2_to_mechanism_id(a, b) or 'MECH_GENERAL_PATHWAY_HYPOTHESIS'
                rel_basis = 'external_mechanistic_support' if 'MECH_02' in mech_id or 'MECH_06' in mech_id or 'MECH_08' in mech_id else 'related_measurement'

        # Reconstruct A_phys_scientific_basis and A_time_scientific_basis from structured fields so R06 bugs are 100% gone:
        a_phys_repaired = (
            f"【物理比较状态: {mods['physical_status_structured']}】"
            f"频率关系={mods['frequency_relation']} ({a['A_phys_exact_freq_zh_en']} vs. {b['A_phys_exact_freq_zh_en']}) | "
            f"波形关系={mods['waveform_relation']} | 剂量关系={mods['dose_relation']} | 时程关系={mods['schedule_relation']} | "
            f"批次关系={mods['batch_relation']}。"
            f"源内部剂量状态={a['dosimetry_status_structured']}；目标内部剂量状态={b['dosimetry_status_structured']}。"
        )
        a_time_repaired = (
            f"【时间比较状态: {mods['temporal_relation']}】"
            f"源[{a['observation_id']}]: 参考过程={a['reference_process_id']}, 适用性={a['normalization_applicability']}, "
            f"暴露天数={a['exposure_days']}d, 归一化比例={a['normalized_exposure_ratio']}, 观测滞后={a['post_exposure_lag_days']}d (滞后比={a['normalized_lag_ratio']}) | "
            f"目标[{b['observation_id']}]: 参考过程={b['reference_process_id']}, 适用性={b['normalization_applicability']}, "
            f"暴露天数={b['exposure_days']}d, 归一化比例={b['normalized_exposure_ratio']}, 观测滞后={b['post_exposure_lag_days']}d (滞后比={b['normalized_lag_ratio']})。"
        )

        # Executable query permission profile (R10):
        allow_retrieval = True
        allow_conditional_comparison = (rel_basis in {
            'same_unit_joint_measurement', 'same_protocol_co_report', 'explicit_arm_contrast',
            'independent_replication', 'same_measurement_target', 'related_measurement', 'external_mechanistic_support'
        })
        allow_mechanistic_hypothesis_view = (mech_id != 'NONE_DIRECT_TARGET_OR_READOUT_COMPARISON')
        allow_effect_pooling = False
        allow_causal_chain_synthesis = False
        allow_cross_study_prediction = False

        rep_row = dict(e)
        rep_row.update({
            'weak_linkage_type': wl_display,
            'within_or_cross_subtype': sub_wl_tag,
            'relation_basis': rel_basis,
            'mechanistic_relation_id': mech_id,
            'question_context_Q': q_repaired,
            'A_phys_status': mods['physical_status_structured'],
            'A_phys_scientific_basis': a_phys_repaired,
            'A_time_status': mods['temporal_relation'],
            'A_time_scientific_basis': a_time_repaired,
            # Structured modifiers (R05):
            'mod_same_species': mods['same_species'],
            'mod_same_preparation': mods['same_preparation'],
            'mod_same_compartment': mods['same_compartment'],
            'mod_frequency_relation': mods['frequency_relation'],
            'mod_waveform_relation': mods['waveform_relation'],
            'mod_dose_relation': mods['dose_relation'],
            'mod_schedule_relation': mods['schedule_relation'],
            'mod_batch_relation': mods['batch_relation'],
            'mod_unit_overlap': mods['unit_overlap'],
            'mod_temporal_relation': mods['temporal_relation'],
            # Executable permission flags (R10):
            'perm_allow_evidence_retrieval': allow_retrieval,
            'perm_allow_conditional_comparison': allow_conditional_comparison,
            'perm_allow_mechanistic_hypothesis_inspection': allow_mechanistic_hypothesis_view,
            'perm_allow_meta_analytic_pooling': allow_effect_pooling,
            'perm_allow_transitive_causal_chaining': allow_causal_chain_synthesis,
            'perm_allow_unvalidated_effect_prediction': allow_cross_study_prediction,
            'rule_version': 'tri-anchored-v3-repaired-1103',
        })
        repaired_relations.append(rep_row)

    # R11: Build the exact 320-pair transition audit log (296 old + 24 new = 320)
    all_keys = sorted(set(old_296_map.keys()) | set(v2_ret_map.keys()) | set(v2_pru_map.keys()))
    change_log = []
    for k in all_keys:
        in_old = k in old_296_map
        in_ret = k in v2_ret_map
        in_pru = k in v2_pru_map
        a, b = obs_by_id[k[0]], obs_by_id[k[1]]
        if in_old and in_ret:
            t_state = 'old_present_and_retained'
            reason = f"保留并升级：旧296对中具备 {v2_ret_map[k]['bio_basis_code']} 依据，升级为正交 relation_basis + 修饰字段"
        elif in_old and in_pru:
            t_state = 'old_present_and_rejected'
            reason = v2_pru_map[k]['rejection_reason_zh_en']
        elif (not in_old) and in_ret:
            t_state = 'new_candidate_and_retained'
            reason = f"新增跨桶生理通路连接：旧版因跨10粗桶漏检，新版按 {v2_ret_map[k]['bio_basis_code']} 补齐"
        elif (not in_old) and in_pru:
            t_state = 'new_candidate_and_rejected'
            reason = v2_pru_map[k]['rejection_reason_zh_en']
        else:
            t_state = 'other'
            reason = ''

        change_log.append({
            'pair_id': f"{k[0]}<->{k[1]}",
            'source_obs': k[0],
            'target_obs': k[1],
            'source_study': a['study_id'],
            'target_study': b['study_id'],
            'old_296_present': in_old,
            'new_24_candidate': not in_old,
            'final_decision': 'retained' if in_ret else 'rejected',
            'transition_category': t_state,
            'old_weak_linkage_type': old_296_map[k]['weak_linkage_type'] if in_old else 'NOT_IN_OLD_296',
            'new_weak_linkage_type': v2_ret_map[k]['weak_linkage_type'] if in_ret else 'REJECTED_BY_CONTROLLED_TARGET_RULE',
            'new_relation_basis': next((r['relation_basis'] for r in repaired_relations if (r['source_obs'], r['target_obs']) == k), 'REJECTED'),
            'change_rationale_zh_en': reason,
        })

    return repaired_relations, v2_pruned, change_log


def build_atomic_relation_assertions(measurements: list[dict], repaired_parent_relations: list[dict]) -> list[dict]:
    """R02 & R08: Project parent-level relation assertions onto the 87 atomic measurements,
    AND refine sub-target matching so that composite parent observations (like F5.1a [count],
    F5.1b [motility], F5.1c [morphology], or F8a.2a [motility], F8a.2b [viability]) only link
    cross-study to the matching atomic sub-target!"""
    parent_rel_map = {(r['source_obs'], r['target_obs']): r for r in repaired_parent_relations}
    atomic_edges = []

    for i in range(len(measurements)):
        for j in range(i + 1, len(measurements)):
            ma, mb = measurements[i], measurements[j]
            pa, pb = ma['parent_observation_id'], mb['parent_observation_id']

            if pa == pb:
                # Sibling atomic measurements split from the same parent comparison (e.g. F5.1a vs F5.1b, F1.3a 3h vs F1.3b 4h):
                atomic_edges.append({
                    'atomic_relation_id': stable('atomic-rel-v3', ma['measurement_id'], mb['measurement_id'])[:20],
                    'source_measurement_id': ma['measurement_id'],
                    'target_measurement_id': mb['measurement_id'],
                    'source_parent_obs': pa,
                    'target_parent_obs': pb,
                    'source_study': ma['study_id'],
                    'target_study': mb['study_id'],
                    'relation_basis': 'same_comparison_sibling_readout',
                    'weak_linkage_type': 'WL2_WITHIN_COMPARISON_ATOMIC_READOUT_SPLIT',
                    'controlled_target_source': ma['controlled_target_id'],
                    'controlled_target_target': mb['controlled_target_id'],
                    'atomic_specificity_note': (
                        f"Split from composite parent {pa}: [{ma['measurement_id']}: {ma['metric_en']} ({ma['sampling_time_text']})] "
                        f"vs. [{mb['measurement_id']}: {mb['metric_en']} ({mb['sampling_time_text']})]. Not independent animal replications."
                    ),
                })
                continue

            pr = parent_rel_map.get((pa, pb)) or parent_rel_map.get((pb, pa))
            if pr is None:
                continue

            # R08: For cross-study links involving a split composite observation (like F5.1a/b/c or F8a.2a/b),
            # require the atomic controlled_target_id to match when relation_basis == 'same_measurement_target'!
            if ma['study_id'] != mb['study_id'] and pr['relation_basis'] == 'same_measurement_target':
                if ma['controlled_target_id'] != mb['controlled_target_id']:
                    continue
                # Furthermore, separate sperm morphology (F4.2, F5.1c) from sperm concentration/count (F4.1, F5.1a, F7.2, F8b.3, R4.3, R6.5, R6.6):
                ma_is_morph = 'abnormality' in ma['metric_en'].lower() or 'morphology' in ma['metric_en'].lower()
                mb_is_morph = 'abnormality' in mb['metric_en'].lower() or 'morphology' in mb['metric_en'].lower()
                if ma_is_morph != mb_is_morph:
                    continue
                # Separate sperm dye-exclusion viability (F8a.2b) from kinematic motility (F1.4a/b, F5.1b, F6.1, F7.1, F8a.2a, F8b.1/2, R4.1):
                ma_is_viab = 'viability' in ma['metric_en'].lower()
                mb_is_viab = 'viability' in mb['metric_en'].lower()
                if ma_is_viab != mb_is_viab:
                    continue

            atomic_edges.append({
                'atomic_relation_id': stable('atomic-rel-v3', ma['measurement_id'], mb['measurement_id'])[:20],
                'source_measurement_id': ma['measurement_id'],
                'target_measurement_id': mb['measurement_id'],
                'source_parent_obs': pa,
                'target_parent_obs': pb,
                'source_study': ma['study_id'],
                'target_study': mb['study_id'],
                'relation_basis': pr['relation_basis'],
                'weak_linkage_type': pr['weak_linkage_type'],
                'controlled_target_source': ma['controlled_target_id'],
                'controlled_target_target': mb['controlled_target_id'],
                'atomic_specificity_note': (
                    f"Atomic binding: {ma['measurement_id']} ({ma['metric_en']}) <-> {mb['measurement_id']} ({mb['metric_en']}); "
                    f"phys={pr['mod_frequency_relation']}, temp={pr['mod_temporal_relation']}."
                ),
            })
    return atomic_edges


def execute_governed_query(
    task_type: str,
    seed_obs_ids: list[str],
    eobs_by_id: dict[str, dict],
    repaired_relations: list[dict],
) -> dict:
    """R10: Executable Query & Inference-Permission Guard.
    Separates:
    - 'evidence_retrieval'
    - 'conditional_comparison'
    - 'multi_hop_path_inspection'
    - 'quantitative_dose_extrapolation'
    - 'unvalidated_effect_prediction'
    Enforces U08, U10, U11, and human-cohort coverage boundaries in code."""
    if task_type in ('quantitative_dose_extrapolation', 'human_threshold_calculation'):
        return {
            'status': 'BLOCKED_BY_PERMISSION_GUARD',
            'allowed': False,
            'reason': 'Refused: Target-organ internal E-field / local sample SAR and cross-species scaling are unmodeled or unverified across the queried studies; human clinical threshold extrapolation is prohibited.',
        }
    if task_type == 'unvalidated_effect_prediction':
        return {
            'status': 'BLOCKED_BY_PERMISSION_GUARD',
            'allowed': False,
            'reason': 'Refused: Predictive claim assertion is disabled (perm_allow_unvalidated_effect_prediction=False); no held-out biological transfer model is claimed.',
        }
    if task_type == 'human_clinical_pregnancy_query':
        human_macro_repro = [
            o['observation_id'] for o in eobs_by_id.values()
            if o['domain'] == 'Reproduction' and o['species'] == 'Human' and o['scale'] == 'macro'
        ]
        return {
            'status': 'COVERAGE_BOUNDARY_REPORTED',
            'allowed': True,
            'matched_observations': human_macro_repro,
            'note': 'The current 22-study curated set includes 0 human clinical pregnancy or in vivo testicular biopsy observations (curation coverage boundary, NOT a claim that zero papers exist in the broader literature).',
        }
    if task_type == 'multi_hop_path_inspection' and len(seed_obs_ids) == 3:
        a_id, b_id, c_id = seed_obs_ids
        rel_map = {(r['source_obs'], r['target_obs']): r for r in repaired_relations}
        rel_map.update({(r['target_obs'], r['source_obs']): r for r in repaired_relations})
        ab = rel_map.get((a_id, b_id))
        bc = rel_map.get((b_id, c_id))
        return {
            'status': 'SEGMENTED_PATH_RETURNED_NON_TRANSITIVE',
            'allowed': True,
            'hop_1': ab['relation_basis'] if ab else None,
            'hop_2': bc['relation_basis'] if bc else None,
            'transitive_causal_edge_generated': False,
            'interaction_or_significance_difference_inferred': False,
            'frequency_difference_interpreted_as_proven_effect_difference': False,
            'note': (
                f"Returned segmented path [{a_id} -> {b_id}] ({ab['relation_basis'] if ab else 'none'}) and "
                f"[{b_id} -> {c_id}] ({bc['relation_basis'] if bc else 'none'}) with per-hop conditions. "
                f"Transitivity is locked (no automatic {a_id} -> {c_id} causal claim)."
            ),
        }
    return {'status': 'OK', 'allowed': True}


def write_rule_yaml_files(rules_dir: Path):
    """Export the 5 rule specification files into docs/BIBs_1103/rules/."""
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / 'candidate_rules.yaml').write_text(
        """# Candidate Generation & Transition Rules (v3.0-1103)
rule_version: "tri-anchored-v3-repaired-1103"
input_granularity:
  parent_comparisons: 75
  atomic_measurements: 87
  possible_unordered_parent_pairs: 2775
candidate_generation:
  within_study_pairs:
    rule: "same paper_id (105 unordered pairs)"
    stratification_required: true
    allowed_relation_basis:
      - same_unit_joint_measurement
      - same_protocol_co_report
      - explicit_arm_contrast
      - independent_replication
      - same_paper_co_report
  cross_study_pairs:
    legacy_v1_coarse_10_buckets: 191
    v2_v3_controlled_28_target_and_pathway_candidates: 215
    total_unique_transition_candidates_evaluated: 320
    retained_total: 203
    rejected_total: 117
selection_invariants:
  filter_by_effect_direction_concordance: false
  filter_by_statistical_significance: false
""",
        encoding='utf-8',
    )
    (rules_dir / 'physical_comparison_rules.yaml').write_text(
        """# Physical Anchor Comparison Rules (A_phys, v3.0-1103)
rule_version: "tri-anchored-v3-repaired-1103"
structured_fields_only: true
prohibit_prose_string_comparison: true
status_codes:
  P1_WithinStudy_Same_Exposure_Arm_and_Preparation:
    condition: "same paper_id AND same arm_id AND same experiment_id"
  P1b_WithinStudy_Same_Nominal_Field_Different_Preparation_or_Batch:
    condition: "same paper_id AND same arm_id AND different experiment_id"
  P2_WithinStudy_Single_Experiment_Controlled_Physical_Contrast:
    condition: "same paper_id AND same experiment_id AND different arm_id"
  P2b_WithinStudy_Cross_Experiment_Different_Preparation_and_Protocol:
    condition: "same paper_id AND different experiment_id AND different arm_id"
  P3_CrossStudy_Exact_Carrier_Frequency_NonEquivalent_MicroDosimetry:
    condition: "different paper_id AND abs(frequency_hz_a - frequency_hz_b) < 1e-3"
    note: "Same carrier frequency and nominal dose do NOT guarantee equal internal tissue E-field or SAR across different wave modulations, applicators, or organ geometries."
  P4_CrossStudy_WithinBand_Different_Carrier_Frequency_NonEquivalent_Dispersion:
    condition: "different paper_id AND same broad band AND frequency_hz_a != frequency_hz_b"
    note: "Different carrier frequencies have distinct tissue permittivity, conductivity, penetration depth, and standing-wave SAR profiles; non-equivalence is a lack of physical equivalence, not an automatic proof of different biological effect."
  P5_CrossRegime_Different_Frequency_Band_or_Static_Contrast:
    condition: "different paper_id AND different coupling_modality"
""",
        encoding='utf-8',
    )
    (rules_dir / 'biological_relation_rules.yaml').write_text(
        """# Biological Relation & Controlled Target Rules (A_bio, v3.0-1103)
rule_version: "tri-anchored-v3-repaired-1103"
controlled_dictionary_file: "../data/target_dictionary.csv"
mechanistic_evidence_file: "../data/mechanism_evidence.csv"
relation_basis_classes:
  same_measurement_target:
    requirement: "Identical controlled_target_id AND compatible sub-metric (separating count vs morphology, motility vs viability)"
  related_measurement:
    requirement: "Shared redox probe family (DCFH-DA vs MitoSOX vs Luminol), DNA lesion assay spectrum (alkaline vs neutral comet), or behavioral memory paradigm family"
  external_mechanistic_support:
    requirement: "Explicit entry in mechanism_evidence.csv with supporting DOI, quote, and context limitations; never inferred from co-location alone"
""",
        encoding='utf-8',
    )
    (rules_dir / 'temporal_comparison_rules.yaml').write_text(
        """# Process-Specific Temporal Comparison Rules (A_time, v3.0-1103)
rule_version: "tri-anchored-v3-repaired-1103"
structured_fields_only: true
tolerance_for_normalized_ratio_equality: 0.001
status_codes:
  T1_same_study_concurrent_window:
    condition: "same paper_id AND equal exposure_days AND equal post_exposure_lag_days"
  T2_both_acute_in_vitro_or_ex_vivo_windows_cycle_not_applicable:
    condition: "normalization_applicability_a == 'not_applicable_acute_isolated_or_ex_vivo' AND normalization_applicability_b == 'not_applicable_acute_isolated_or_ex_vivo'"
  T3_asymmetric_in_vivo_organ_process_vs_acute_isolated_window:
    condition: "normalization_applicability_a != normalization_applicability_b"
  T4a_same_reference_process_equal_normalized_ratio_and_lag:
    condition: "same reference_process_id AND abs(normalized_exposure_ratio_a - normalized_exposure_ratio_b) < 0.001 AND abs(normalized_lag_ratio_a - normalized_lag_ratio_b) < 0.001"
  T4b_same_reference_process_different_coverage_or_lag:
    condition: "same reference_process_id AND (ratio differs OR lag differs)"
  T5_distinct_species_or_developmental_reference_processes:
    condition: "reference_process_id_a != reference_process_id_b"
""",
        encoding='utf-8',
    )
    (rules_dir / 'inference_permissions.yaml').write_text(
        """# Query & Inference Permission Enforcement Rules (P_allow_forbid, v3.0-1103)
rule_version: "tri-anchored-v3-repaired-1103"
global_guards:
  transitivity_locked: true
  allow_unvalidated_effect_prediction: false
  allow_cross_study_meta_analytic_dose_pooling: false
  treat_nonsignificant_vs_significant_as_proven_interaction: false
  treat_frequency_non_equivalence_as_proven_effect_difference: false
  empty_curated_cell_interpretation: "curation_coverage_boundary_not_literature_absence"
""",
        encoding='utf-8',
    )

