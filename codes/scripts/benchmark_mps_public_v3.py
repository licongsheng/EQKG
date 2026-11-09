#!/usr/bin/env python3
"""Exploratory MPS cross-plate reference matching; never an inductive test.

Standalone: only this script's output directory is written. Native endpoint units
are encoded by the FOM keys, not inferred from undocumented synthesis fields.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import re
import time

import numpy as np

SQL_PATH = Path('/data/AI4BioEM/Caltech Materials Provenance Store_eighth/mps_eighth.sql')
OUT_DIR = Path('/home/AI4BioEM/data/caltech_mps_validation_v3')
ENDPOINTS = {'CA_FOMS_standard': 'I.A_ave', 'CP_FOMS_standard': 'E.V_ave',
             'CV_FOMS_standard': 'I.A_max'}
ANCHORS = ('physical', 'material', 'time')
SUBSETS = tuple('gate_' + ''.join(str(i) for i in range(3) if mask & (1 << i))
                for mask in range(8))
METHODS = ('mean', 'median', 'soft_knn_8', 'soft_kernel_0.25',
           'soft_kernel_1', 'count_matched_random', 'count_matched_knn') + SUBSETS
STRONG = ('mean', 'median', 'gate_0', 'soft_knn_8', 'soft_kernel_0.25', 'soft_kernel_1')
_GROUPS = {}
_CONFIG = {}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def copy_unescape(text):
    """PostgreSQL COPY TEXT: NULL recognized BEFORE backslash decoding."""
    if text == r'\N':
        return None
    if '\\' not in text:
        return text
    # Most JSON content is ordinary text: decode only escape tokens in C regex,
    # rather than iterating every character of the 3GB dump in Python.
    escapes = {'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t', 'v': '\v'}
    def replace(match):
        token = match.group()[1:]
        if token[0] in '01234567':
            return chr(int(token, 8))
        if token[0] == 'x' and len(token) > 1:
            return chr(int(token[1:], 16))
        return escapes.get(token, token)
    if text.endswith('\\') and len(re.search(r'\\+$', text).group()) % 2:
        raise ValueError('Trailing COPY escape')
    return re.sub(r'\\(?:[0-7]{1,3}|x[0-9a-fA-F]{1,2}|[^\n])', replace, text)


def copy_rows(path, wanted, audit=None):
    """Stream rows by column NAME, with exact field-count/schema validation."""
    table = None
    with Path(path).open(encoding='utf-8', errors='strict') as stream:
        for line in stream:
            if line.startswith('COPY '):
                m = re.fullmatch(r'COPY public_release\.(\w+) \((.+)\) FROM stdin;\n', line)
                table = m[1] if m and m[1] in wanted else None
                if table:
                    columns = tuple(c.strip().strip('"') for c in m[2].split(','))
                    if audit is not None:
                        audit.setdefault('schemas', {})[table] = columns
                continue
            if line.rstrip('\n') == r'\.':
                table = None
            elif table:
                raw = line.rstrip('\n').split('\t')
                if len(raw) != len(columns):
                    raise ValueError(f'{table}: field count {len(raw)} != {len(columns)}')
                yield table, dict(zip(columns, map(copy_unescape, raw)))


def json_field(row, name):
    value = row.get(name)
    return json.loads(value) if value is not None else None


def finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def known(value):
    if value is None or value == '' or value == 'unknown':
        return False
    if isinstance(value, (tuple, list)):
        return bool(value) and all(known(x) for x in value)
    if isinstance(value, dict):
        return bool(value) and all(known(x) for x in value.values())
    return True


def compatible(left, right, tolerance=None):
    if not known(left) or not known(right):
        return False
    if tolerance is None:
        return left == right
    a, b = finite(left), finite(right)
    return a is not None and b is not None and abs(a - b) <= tolerance


def parse_timestamp(value):
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def prior_history(events, measurement):
    """Only strictly earlier events. Never reinterpret ordering as time.

    One unplaceable linked event makes history unknown, even when the remaining
    events can be sorted. A future event is excluded, not a reason to include it.
    Equal-time synthesis events have no proven internal order and cause refusal.
    """
    stamp = parse_timestamp(measurement.get('timestamp'))
    if stamp is None:
        return None, 'measurement_timestamp_missing'
    prior = []
    for event in events:
        ts = parse_timestamp(event.get('timestamp'))
        if ts is None:
            return None, 'synthesis_timestamp_missing'
        if ts == stamp:
            return None, 'synthesis_measurement_tie'
        if ts < stamp:
            prior.append((ts, event))
    prior.sort(key=lambda x: x[0])
    if len({ts for ts, _ in prior}) != len(prior):
        return None, 'synthesis_tie'
    if not prior:
        return None, 'no_prior_synthesis'
    return [event for _, event in prior], 'timestamp_strict'


def history_features(history):
    if history is None:
        return None, None
    material, duration = [], []
    has_print = has_anneal = False
    for event in history:
        params = event['parameters']
        if event['type'] == 'print':
            has_print = True
            material.append(['print', params.get('method') or event['technique']])
        else:
            has_anneal = True
            recipe = params.get('recipe_text')
            steps = recipe.get('steps') if isinstance(recipe, dict) else None
            # A JSON object's multi-step key ordering is not a validated recipe order.
            step = next(iter(steps.items())) if isinstance(steps, dict) and len(steps) == 1 else None
            material.append(['anneal', finite(params.get('max_temperature')),
                             params.get('gas_string'), params.get('pressure')])
            duration.append([finite(params.get('soak_time')), list(step) if step else None])
    if not has_print or not has_anneal:
        return None, None
    return material, duration


def expand_links(sample_process, sppd, pda):
    """Preserve every link path, including duplicate/multiple analysis paths."""
    for row in sample_process:
        for link in sppd.get(row['id'], ()):
            for analysis_link in pda.get(link['process_data_id'], ()):
                yield row, link, analysis_link


def composition_of(row):
    for field in ('final_composition', 'initial_composition', 'inkjet_composition',
                  'platemap_inkjet_composition'):
        comp = json_field(row, field)
        if not isinstance(comp, dict) or not comp:
            continue
        vals = [(k, finite(v)) for k, v in comp.items()]
        if any(v is None or v < 0 for _, v in vals):
            continue
        # Exact native fractions (12 significant digits for JSON float noise), NOT 2-decimal bins.
        values = tuple(sorted((k, format(v, '.12g')) for k, v in vals if v > 0))
        if values:
            return values, field
    return None, None


def condition_features(modality, parameters, inputs, material, duration, loading, substrate):
    """(anchor, name, value, hard tolerance / fixed soft scale).

    Missing illumination stays missing, including CP. No 0/385 defaults.
    Synthesis quantities remain native, unspecified units; exact comparisons only.
    """
    p = parameters
    features = []
    def add(anchor, key, value, tolerance=None, scale=None):
        features.append((anchor, key, value, tolerance, scale))
    for key in ('solution_ph', 'electrolyte', 'concentration', 'ref_electrode_type',
                'solution_bubble_gas', 'toggle_value_illumination',
                'illumination_wavelength', 'illumination_intensity'):
        add(0, key, p.get(key))
    keys = {'CA_FOMS_standard': ('potential_vref',), 'CP_FOMS_standard': ('current_abs',),
            'CV_FOMS_standard': ('first_potential_vref', 'second_potential_vref')}[modality]
    for key in keys:
        value = inputs.get(key) if inputs.get(key) is not None else p.get(key)
        tolerance = 1e-6 if key == 'current_abs' else .01
        add(0, key, finite(value), tolerance, tolerance)
    add(1, 'ordered_synthesis', material)
    add(1, 'native_loading', loading, .05, .05)
    add(1, 'substrate', substrate)
    add(2, 'ordered_thermal_duration_and_recipe', duration)
    add(2, 'technique', p.get('_technique'))
    for key in (('potential_sweep_rate', 'num_potential_cycles')
                if modality == 'CV_FOMS_standard' else ('acquisition_time', 'acquisition_interval')):
        add(2, key, finite(p.get(key)))
    return features


def load_observations(path, out):
    audit = {'table_rows': Counter(), 'exclusions': Counter(), 'chronology': Counter(),
             'join': Counter(), 'raw_examples': {}, 'feature_missing': Counter()}
    details, processes, analyses, pda, sppd, samples, plates = {}, {}, {}, defaultdict(list), defaultdict(list), {}, {}
    memberships = defaultdict(set)
    tables = {'analysis', 'collection', 'collection__sample', 'process', 'process_detail',
              'process_data_analysis', 'sample_process_process_data', 'sample'}
    print('Loading named-column COPY tables', flush=True)
    for table, row in copy_rows(path, tables, audit):
        audit['table_rows'][table] += 1
        if table == 'analysis':
            if row['name'] in ENDPOINTS:
                output = json_field(row, 'output') or {}
                value = finite(output.get(ENDPOINTS[row['name']]))
                if row['name'] not in audit['raw_examples']:
                    audit['raw_examples'][row['name']] = row
                if value is None:
                    audit['exclusions']['missing_nonfinite_endpoint'] += 1
                else:
                    analyses[row['id']] = (row['name'], value, json_field(row, 'input') or {})
        elif table == 'process_detail':
            details[row['id']] = {'type': row['type'], 'technique': row['technique'],
                                  'parameters': json_field(row, 'parameters') or {}}
            if row['type'] in ('anneal', 'print'):
                audit['raw_examples'].setdefault(row['type'], row)
        elif table == 'process':
            processes[row['id']] = row
        elif table == 'process_data_analysis':
            pda[row['process_data_id']].append(row)
        elif table == 'sample_process_process_data':
            sppd[row['sample_process_id']].append(row)
        elif table == 'sample':
            samples[row['id']] = row
        elif table == 'collection':
            plates[row['id']] = {'label': row['label'], 'doi': row['doi'],
                                 'substrate': (json_field(row, 'details') or {}).get('substrate')}
        else:
            memberships[row['sample_id']].add(row['collection_id'])
    audit['join'].update(sppd_rows=sum(map(len, sppd.values())), sppd_keys=len(sppd),
                         sppd_extra_rows=sum(len(v) - 1 for v in sppd.values()),
                         sppd_multivalue_keys=sum(len(v) > 1 for v in sppd.values()),
                         pda_extra_rows=sum(len(v) - 1 for v in pda.values()),
                         multi_collection_samples=sum(len(v) > 1 for v in memberships.values()))
    histories = defaultdict(dict)
    sample_process = []
    for table, row in copy_rows(path, {'sample_process'}, audit):
        audit['table_rows'][table] += 1
        proc = processes.get(row['process_id'])
        detail = details.get(proc['process_detail_id']) if proc else None
        if not detail:
            audit['exclusions']['dangling_process_detail'] += 1
            continue
        if detail['type'] in ('print', 'anneal'):
            histories[row['sample_id']][proc['id']] = {**proc, **detail}
        if row['id'] in sppd:
            sample_process.append(row)
    observations = []
    history_cache = {}
    comp_cache = {}
    for row, link, alink in expand_links(sample_process, sppd, pda):
        analysis = analyses.get(alink['analysis_id'])
        if analysis is None:
            continue
        audit['join']['fom_link_paths'] += 1
        sid = row['sample_id']
        sample = samples.get(sid)
        if sample is None:
            audit['exclusions']['dangling_sample'] += 1
            continue
        if sid not in comp_cache:
            comp_cache[sid] = composition_of(sample)
        comp, comp_source = comp_cache[sid]
        if not comp:
            audit['exclusions']['missing_composition_paths'] += 1
            continue
        member = sorted(memberships.get(sid, ()))
        if not member:
            audit['exclusions']['missing_plate_paths'] += 1
            continue
        proc = processes[row['process_id']]
        detail = details[proc['process_detail_id']]
        cache_key = (sid, proc['id'])
        if cache_key not in history_cache:
            events = list(histories[sid].values())
            history, state = prior_history(events, proc)
            material, duration = history_features(history)
            prior_ids = [x['id'] for x in history] if history is not None else []
            history_cache[cache_key] = material, duration, state, prior_ids
            audit['chronology'][state] += 1
            stamp = parse_timestamp(proc['timestamp'])
            for event in events:
                ts = parse_timestamp(event['timestamp'])
                if ts and stamp:
                    audit['chronology']['future_synthesis_excluded'] += ts > stamp
                    a, b = finite(event['ordering']), finite(proc['ordering'])
                    if a is not None and b is not None:
                        audit['chronology']['ordering_vs_timestamp_disagreements'] += ((a < b) != (ts < stamp))
        material, duration, state, prior_ids = history_cache[cache_key]
        loading_dict = json_field(sample, 'platemap_inkjet_loading') or {}
        loads = [finite(v) for k, v in loading_dict.items() if k.endswith('.Loading') and not k.startswith('Tot')]
        loading = sum(loads) if loads and all(v is not None for v in loads) else None
        substrates = [plates[p]['substrate'] for p in member if p in plates]
        substrate = substrates[0] if substrates and all(x == substrates[0] for x in substrates) else None
        modality, value, inputs = analysis
        features = condition_features(modality, {**detail['parameters'], '_technique': detail['technique']},
                                      inputs, material, duration, loading, substrate)
        for _, name, val, _, _ in features:
            if not known(val):
                audit['feature_missing'][modality + ':' + name] += 1
        observation = {'id': ':'.join((row['id'], link['id'], alink['id'])), 'sample_id': sid,
                       'sample_process_id': row['id'], 'process_id': proc['id'],
                       'sppd_id': link['id'], 'process_data_id': link['process_data_id'],
                       'pda_id': alink['id'], 'analysis_id': alink['analysis_id'],
                       'plates': member, 'composition': comp, 'composition_source': comp_source,
                       'family': '-'.join(k for k, _ in comp), 'modality': modality,
                       'endpoint': ENDPOINTS[modality], 'y': value, 'features': features,
                       'timestamp': proc['timestamp'], 'ordering': proc['ordering'],
                       'chronology': state, 'prior_synthesis_process_ids': prior_ids}
        observations.append(observation)
    audit['join']['observations'] = len(observations)
    audit['join']['distinct_analysis_ids'] = len({o['analysis_id'] for o in observations})
    audit['join']['distinct_observation_ids'] = len({o['id'] for o in observations})
    audit['chronology']['policy'] = 'Strict native wall-clock timestamps; no global ordering fallback; ties/missing refuse; no ETL created_at.'
    audit['chronology']['recipe_policy'] = 'Only single-step recipe objects have unambiguous internal step order; multi-step missing.'
    write_json(out / 'join_chronology_audit.json', audit)
    write_json(out / 'plates.json', plates)
    print(f'Loaded {len(observations)} observation link paths', flush=True)
    return observations


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def source_eligible(target, source):
    return (target['id'] != source['id'] and target['sample_id'] != source['sample_id']
            and target['analysis_id'] != source['analysis_id']
            and target['process_data_id'] != source['process_data_id']
            and set(target['plates']).isdisjoint(source['plates'])
            and target['composition'] == source['composition']
            and target['modality'] == source['modality'])


def split_family(family):
    return 'dev' if int(hashlib.sha256(('mps-public-v3:' + family).encode()).hexdigest(), 16) % 2 == 0 else 'test'


def encoded_features(obs):
    """Categorical equality codes and numeric tolerance-scaled values; no y input."""
    array = np.zeros((len(obs), len(obs[0]['features'])), dtype=float)
    valid = np.zeros(array.shape, dtype=bool)
    for col, (_, _, _, tolerance, _) in enumerate(obs[0]['features']):
        codes = {}
        for i, o in enumerate(obs):
            value = o['features'][col][2]
            if not known(value):
                continue
            if tolerance is not None:
                number = finite(value)
                if number is None:
                    continue
                array[i, col] = number / tolerance
            else:
                key = canonical(value)
                array[i, col] = codes.setdefault(key, len(codes))
            valid[i, col] = True
    return array, valid


def process_group(key):
    obs = _GROUPS[key]
    out = Path(_CONFIG['output']) / 'predictions' / (digest(key) + '.jsonl.gz')
    values = np.array([o['y'] for o in obs])
    data, valid = encoded_features(obs)
    # Hash order is independent of targets' outcomes and condition features.
    ordered = sorted(range(len(obs)), key=lambda i: (digest([_CONFIG['seed'], obs[i]['id']]), obs[i]['id']))
    plate_pools = {}
    identity_indices = {field: defaultdict(set) for field in ('sample_id', 'analysis_id', 'process_data_id')}
    for j, o in enumerate(obs):
        for field, mapping in identity_indices.items():
            mapping[o[field]].add(j)
    cap = _CONFIG['reference_cap']
    audit = Counter(targets=len(obs))
    with gzip.open(out, 'wt', encoding='utf-8', compresslevel=3) as stream:
        for target_index, target in enumerate(obs):
            plate_key = tuple(target['plates'])
            if plate_key not in plate_pools:
                target_plates = set(plate_key)
                base = [j for j in ordered if target_plates.isdisjoint(obs[j]['plates'])]
                plate_pools[plate_key] = base, set(base)
            base, base_set = plate_pools[plate_key]
            excluded = set().union(*(mapping[target[field]] for field, mapping in identity_indices.items()))
            eligible_n = len(base) - len(base_set.intersection(excluded))
            candidates = (j for j in base if j not in excluded)
            refs = list(itertools.islice(candidates, cap)) if cap else list(candidates)
            audit['eligible_reference_edges'] += eligible_n
            audit['used_reference_edges'] += len(refs)
            audit['targets_reference_capped'] += len(refs) < eligible_n
            n = len(refs)
            preds = {name: None for name in METHODS}
            counts = {name: 0 for name in METHODS}
            full_ids = []
            if n:
                refs_array = np.array(refs)
                anchor_ok = np.ones((n, 3), dtype=bool)
                distance = np.zeros(n)
                for col, (anchor, _, _, tolerance, _) in enumerate(target['features']):
                    both = valid[target_index, col] & valid[refs_array, col]
                    delta = np.abs(data[target_index, col] - data[refs_array, col])
                    ok = both & ((delta <= 1 + 1e-12) if tolerance is not None else (delta == 0))
                    anchor_ok[:, anchor] &= ok
                    # Explicit missingness penalty = 1; never missing-compatible.
                    component = np.minimum(delta, 5) if tolerance is not None else (delta != 0).astype(float)
                    distance += np.where(both, component, 1)
                distance /= len(target['features'])
                ys = values[refs_array]
                rank = np.argsort(distance, kind='stable')
                preds['mean'], preds['median'] = float(ys.mean()), float(np.median(ys))
                counts['mean'] = counts['median'] = n
                knn = rank[:8]
                preds['soft_knn_8'] = float(ys[knn].mean())
                counts['soft_knn_8'] = len(knn)
                for bandwidth in (.25, 1):
                    name = f'soft_kernel_{bandwidth:g}'
                    weights = np.exp(-(distance - distance.min()) / bandwidth)
                    preds[name] = float(np.dot(weights, ys) / weights.sum())
                    counts[name] = n
                for mask, name in enumerate(SUBSETS):
                    anchors = [i for i in range(3) if mask & (1 << i)]
                    selected = np.all(anchor_ok[:, anchors], axis=1) if anchors else np.ones(n, dtype=bool)
                    count = int(selected.sum())
                    counts[name] = count
                    if count:
                        preds[name] = float(ys[selected].mean())
                    if mask == 7:
                        full_ids = [obs[refs[j]]['id'] for j in np.flatnonzero(selected)]
                full_n = counts['gate_012']
                if full_n:
                    preds['count_matched_random'] = float(ys[:full_n].mean())
                    preds['count_matched_knn'] = float(ys[rank[:full_n]].mean())
                    counts['count_matched_random'] = counts['count_matched_knn'] = full_n
            record = {'target_id': target['id'], 'y': target['y'], 'family': target['family'],
                      'plates': target['plates'], 'split': split_family(target['family']),
                      'modality': target['modality'], 'eligible_count': eligible_n,
                      'reference_count': n, 'predictions': preds, 'support_counts': counts,
                      'full_support_ids': full_ids}
            stream.write(canonical(record) + '\n')
    return dict(audit)


def metrics(y, prediction, plates=None, families=None):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(prediction) & np.isfinite(y)
    target_n = len(y)
    y, prediction = y[valid], prediction[valid]
    if not len(y):
        return {'n': 0, 'target_n': target_n, 'coverage': 0, 'mae': None, 'rmse': None,
                'target_mean_abs': None, 'nmae_pct': None}
    error = np.abs(y - prediction)
    scale = float(np.mean(np.abs(y)))
    result = {'n': len(y), 'target_n': target_n, 'coverage': len(y) / target_n,
              'mae': float(error.mean()), 'rmse': float(np.sqrt(np.mean(error ** 2))),
              'target_mean_abs': scale, 'nmae_pct': float(error.mean() / scale * 100) if scale > 0 else None}
    for name, labels in (('plate', plates), ('family', families)):
        if labels is not None:
            grouped = defaultdict(list)
            for err, label in zip(error, np.asarray(labels)[valid]):
                grouped[str(label)].append(float(err))
            result[f'macro_{name}_mae'] = float(np.mean([np.mean(v) for v in grouped.values()]))
            result[f'{name}_groups'] = len(grouped)
    return result


def cluster_bootstrap(difference, labels, seed=19471, repeats=1000):
    """Paired MAE(full)-MAE(comparator); negative favors full. Cluster resampling."""
    grouped = defaultdict(list)
    for diff, label in zip(difference, labels):
        grouped[str(label)].append(float(diff))
    groups = list(grouped.values())
    if not groups:
        return {'n': 0, 'clusters': 0, 'delta_mae': None, 'ci95': None}
    sums = np.array([sum(v) for v in groups])
    sizes = np.array([len(v) for v in groups])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        ids = rng.integers(0, len(groups), len(groups))
        estimates.append(float(sums[ids].sum() / sizes[ids].sum()))
    return {'n': int(sizes.sum()), 'clusters': len(groups),
            'delta_mae': float(sums.sum() / sizes.sum()),
            'ci95': [float(v) for v in np.quantile(estimates, [.025, .975])],
            'inferential': len(groups) >= 10,
            'warning': 'Exploratory; sparse clusters if <10; no multiplicity correction.'}


def summarize(out, bootstrap_repeats):
    buckets = defaultdict(list)
    for path in sorted((out / 'predictions').glob('*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                buckets[(r['split'], r['modality'])].append(r)
    summary = {'status': 'EXPLORATORY_PUBLIC_DATA_REUSED_REFERENCE_MATCHING_NOT_INDUCTIVE',
               'normalization': 'mean(abs(y_target)) on stated evaluation cohort, never predictions',
               'coverage_denominator': 'Every finite linked target with composition and plate, including unsupported targets',
               'dev_selection': {}, 'evaluations': {}, 'paired_comparisons': {}}
    selected = {}
    for modality in ENDPOINTS:
        rows = buckets.get(('dev', modality), [])
        common = [r for r in rows if r['predictions']['gate_012'] is not None]
        selection_rows = common or [r for r in rows if r['predictions']['mean'] is not None]
        scores = {}
        for method in STRONG:
            paired = [r for r in selection_rows if r['predictions'][method] is not None]
            # Selection requires identical target coverage; no selective advantage.
            if paired and len(paired) == len(selection_rows):
                scores[method] = float(np.mean([abs(r['y'] - r['predictions'][method]) for r in paired]))
        selected[modality] = min(scores, key=lambda x: (scores[x], x)) if scores else 'mean'
        summary['dev_selection'][modality] = {'selected': selected[modality], 'scores_mae': scores,
                                              'n': len(selection_rows),
                                              'cohort': 'full_supported' if common else 'unconditional_supported_no_full_dev'}
    per_group = {}
    for (split, modality), rows in sorted(buckets.items()):
        name = split + ':' + modality
        y = np.array([r['y'] for r in rows])
        plate = np.array(['|'.join(r['plates']) for r in rows])
        family = np.array([r['family'] for r in rows])
        predictions = {m: np.array([r['predictions'][m] if r['predictions'][m] is not None else np.nan for r in rows]) for m in METHODS}
        full = np.isfinite(predictions['gate_012'])
        summary['evaluations'][name] = {'all_targets': {}, 'fixed_full_supported_targets': {}}
        per_group[name] = {}
        for method, pred in predictions.items():
            summary['evaluations'][name]['all_targets'][method] = metrics(y, pred, plate, family)
            summary['evaluations'][name]['fixed_full_supported_targets'][method] = metrics(y[full], pred[full], plate[full], family[full])
            # Export actual group-level errors, not just macro averages.
            per_group[name][method] = {}
            for group_name, labels in (('plate', plate), ('family', family)):
                accum = defaultdict(list)
                for idx, label in enumerate(labels):
                    if np.isfinite(pred[idx]):
                        accum[str(label)].append(float(abs(y[idx] - pred[idx])))
                per_group[name][method][group_name] = {k: {'n': len(v), 'mae': sum(v) / len(v)} for k, v in accum.items()}
        comparisons = {}
        for comparator in dict.fromkeys((selected[modality], 'gate_01', 'gate_02', 'gate_12',
                                        'count_matched_random', 'count_matched_knn')):
            pair = full & np.isfinite(predictions[comparator])
            delta = np.abs(y[pair] - predictions['gate_012'][pair]) - np.abs(y[pair] - predictions[comparator][pair])
            comparisons[comparator] = {cluster: cluster_bootstrap(delta, labels[pair], repeats=bootstrap_repeats)
                                       for cluster, labels in (('plate_cluster', plate), ('family_cluster', family))}
        summary['paired_comparisons'][name] = comparisons
    write_json(out / 'metrics_by_plate_family.json', per_group)
    write_json(out / 'summary.json', summary)
    return summary


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sql', type=Path, default=SQL_PATH)
    parser.add_argument('--output', type=Path, default=OUT_DIR)
    parser.add_argument('--workers', type=int, default=min(8, max(1, len(os.sched_getaffinity(0)) // 4)))
    parser.add_argument('--reference-cap', type=int, default=128)
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.reference_cap < 0 or args.workers < 1:
        parser.error('reference-cap must be >=0; workers >=1')
    started = time.monotonic()
    out = args.output.resolve()
    if out != OUT_DIR:
        parser.error('Outputs restricted to new data/caltech_mps_validation_v3 directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'predictions').mkdir(exist_ok=True)
    if any((out / 'predictions').glob('*.jsonl.gz')):
        parser.error('Existing prediction shards found; refuse mixed/stale results')
    config = {'version': 3, 'output': str(out), 'seed': 19471, 'workers': min(args.workers, 8),
              'reference_cap': args.reference_cap, 'bootstrap_repeats': args.bootstrap_repeats,
              'endpoints': ENDPOINTS, 'methods': METHODS, 'anchors': ANCHORS,
              'split': 'sha256(mps-public-v3:element-family) parity, independent of counts and outcomes',
              'reference_policy': 'Same exact composition/modality; disjoint all plate memberships, sample, analysis, data IDs; ascending SHA256([seed,observation_id]); first cap after exclusions; all targets retained',
              'reference_manifest': 'observations.jsonl.gz + this script + config reconstruct every source pool and per-method support; predictions contain full support IDs and all counts',
              'deduplication': 'No scalar overwrites or link-path collapse; analysis IDs preserved and excluded across target/source',
              'soft_distance': 'Mean per-field categorical mismatch or tolerance-scaled numeric distance capped at 5; missing penalty 1; fixed scales, no outcome-trained features',
              'source_labels': 'Test-family cross-plate peer labels deliberately available; NOT held-out inductive prediction',
              'reuse': 'Public data and named test systems previously explored; all findings exploratory, not independent validation',
              'native_units': 'CA/CV keys encode A; CP E.V_ave encodes V, not Eta.V_ave; synthesis loading/soak/temp/wavelength units not guessed',
              'python': '/usr/bin/python3', 'sql': str(args.sql), 'sql_bytes': args.sql.stat().st_size,
              'sql_sha256': file_hash(args.sql), 'script_sha256': file_hash(Path(__file__)),
              'numpy_version': np.__version__, 'blas_threads': 1}
    write_json(out / 'config.json', config)
    observations = load_observations(args.sql, out)
    global _GROUPS, _CONFIG
    _CONFIG = config
    groups = defaultdict(list)
    families = defaultdict(Counter)
    with gzip.open(out / 'observations.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as stream:
        for o in observations:
            stream.write(canonical(o) + '\n')
            groups[(tuple(tuple(x) for x in o['composition']), o['modality'])].append(o)
            families[o['family']][o['modality']] += 1
    _GROUPS = dict(groups)
    write_json(out / 'split_membership.json', {f: {'split': split_family(f), 'counts': c} for f, c in sorted(families.items())})
    del observations
    tasks = sorted(groups, key=lambda k: (-len(groups[k]), k))
    totals = Counter()
    print(f'Running {len(tasks)} composition/modality tasks, {config["workers"]} fork workers', flush=True)
    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=multiprocessing.get_context('fork')) as pool:
        for i, counts in enumerate(pool.map(process_group, tasks, chunksize=1)):
            totals.update(counts)
            if i % 1000 == 0:
                print(f'Groups {i + 1}/{len(tasks)} elapsed {time.monotonic() - started:.1f}s', flush=True)
    write_json(out / 'reference_audit.json', totals)
    summary = summarize(out, args.bootstrap_repeats)
    files = [p for p in out.rglob('*') if p.is_file() and p.name != 'run_manifest.json']
    manifest = {'elapsed_seconds': time.monotonic() - started, 'targets': totals['targets'],
                'groups': len(groups), 'config_sha256': file_hash(out / 'config.json'),
                'files': {str(p.relative_to(out)): {'sha256': file_hash(p), 'bytes': p.stat().st_size} for p in sorted(files)}}
    write_json(out / 'run_manifest.json', manifest)
    print(canonical({'elapsed_seconds': manifest['elapsed_seconds'], 'references': totals,
                     'dev_selection': summary['dev_selection']}), flush=True)


if __name__ == '__main__':
    main()