"""Regression tests of semantics, not assertions that gating must win."""
import importlib.util
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_mps_public_v3.py'
SPEC = importlib.util.spec_from_file_location('mps_v3', SCRIPT)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class TestMPSV3(unittest.TestCase):
    def test_copy_null_and_escapes(self):
        self.assertIsNone(m.copy_unescape(r'\N'))
        self.assertEqual(m.copy_unescape(r'\\N'), r'\N')
        self.assertEqual(m.copy_unescape(r'a\tb\nc\\d\101\x42'), 'a\tb\nc\\dAB')
        raw = r'{"path":"C:\\\\temp\\\\x","text":"a\\nb"}'
        self.assertEqual(json.loads(m.copy_unescape(raw)), {'path': 'C:\\temp\\x', 'text': 'a\nb'})

    def test_copy_named_columns_not_positions(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'fixture.sql'
            path.write_text('COPY public_release.sample (label, id) FROM stdin;\nname\\tpart\tID\n\\.\n')
            self.assertEqual(list(m.copy_rows(path, {'sample'})), [('sample', {'label': 'name\tpart', 'id': 'ID'})])

    def test_multivalue_join_and_many_to_many(self):
        sp = [{'id': 'sp', 'sample_id': 's'}]
        links = {'sp': [{'id': 'l1', 'process_data_id': 'd1'}, {'id': 'l2', 'process_data_id': 'd2'}]}
        pda = {'d1': [{'analysis_id': 'a1'}, {'analysis_id': 'a2'}], 'd2': [{'analysis_id': 'a3'}]}
        expanded = list(m.expand_links(sp, links, pda))
        self.assertEqual(len(expanded), 3)
        self.assertEqual({x[2]['analysis_id'] for x in expanded}, {'a1', 'a2', 'a3'})

    def test_chronological_history_not_ordering(self):
        events = [{'id': 'late', 'timestamp': '2020-03-01', 'ordering': '0'},
                  {'id': 'early', 'timestamp': '2020-01-01', 'ordering': '999'},
                  {'id': 'middle', 'timestamp': '2020-01-15', 'ordering': '1000'}]
        history, state = m.prior_history(events, {'timestamp': '2020-02-01', 'ordering': '1'})
        self.assertEqual([x['id'] for x in history], ['early', 'middle'])
        self.assertEqual(state, 'timestamp_strict')
        self.assertIsNone(m.prior_history(events + [{'timestamp': None}], {'timestamp': '2020-02-01'})[0])
        self.assertIsNone(m.prior_history([events[1], events[1]], {'timestamp': '2020-02-01'})[0])

    def test_missing_never_equal_compatible(self):
        for value in (None, '', 'unknown', [], [None], {'soak': None}):
            self.assertFalse(m.compatible(value, value))
        self.assertTrue(m.compatible(0, 0))
        self.assertTrue(m.compatible(1., 1.005, .01))

    def test_disjoint_target_source(self):
        a = dict(id='a', sample_id='s1', analysis_id='a1', process_data_id='d1',
                 plates=['p1'], composition=[['Fe', '1']], modality='CA')
        b = dict(a, id='b', sample_id='s2', analysis_id='a2', process_data_id='d2', plates=['p2'])
        self.assertTrue(m.source_eligible(a, b))
        for change in ({'plates': ['p1', 'p2']}, {'sample_id': 's1'}, {'analysis_id': 'a1'}, {'process_data_id': 'd1'}):
            self.assertFalse(m.source_eligible(a, dict(b, **change)))

    def test_metrics_fixed_y_only_and_abstention(self):
        result = m.metrics([1., 3., 7.], [2., 1., np.nan], ['a', 'b', 'c'], ['x', 'x', 'y'])
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['target_n'], 3)
        self.assertAlmostEqual(result['mae'], 1.5)
        self.assertAlmostEqual(result['rmse'], np.sqrt(2.5))
        self.assertEqual(result['target_mean_abs'], 2.)
        self.assertEqual(result['nmae_pct'], 75.)
        self.assertEqual(m.metrics([1., 3.], [100., 100.])['target_mean_abs'], 2.)

    def test_modality_and_no_guessed_illumination(self):
        self.assertEqual(m.ENDPOINTS['CP_FOMS_standard'], 'E.V_ave')
        features = m.condition_features('CP_FOMS_standard', {'current_abs': 3e-5}, {}, None, None, None, None)
        values = {x[1]: x[2] for x in features}
        self.assertIsNone(values['toggle_value_illumination'])
        self.assertIsNone(values['illumination_wavelength'])
        self.assertEqual(values['current_abs'], 3e-5)

    def test_eight_subsets_leaveouts_and_split(self):
        self.assertEqual(len(set(m.SUBSETS)), 8)
        self.assertTrue({'gate_01', 'gate_02', 'gate_12', 'gate_012'} <= set(m.SUBSETS))
        self.assertEqual(m.split_family('Bi-Fe'), m.split_family('Bi-Fe'))
        ci = m.cluster_bootstrap([1., 1., 1.], ['a', 'b', 'c'], repeats=20)
        self.assertEqual(ci['ci95'], [1., 1.])
        self.assertFalse(ci['inferential'])

    def test_worker_preserves_targets_cap_and_support(self):
        observations = []
        for i in range(4):
            observations.append(dict(id=f'o{i}', sample_id=f's{i}', analysis_id=f'a{i}',
                                     process_data_id=f'd{i}', plates=[f'p{i}'],
                                     composition=[['Fe', '1']], modality='CA_FOMS_standard',
                                     family='Fe', y=float(i),
                                     features=[(a, f'feature{a}', 'known', None, None) for a in range(3)]))
        observations[-1]['features'][1] = (1, 'feature1', None, None, None)
        old_groups, old_config = m._GROUPS, m._CONFIG
        try:
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / 'predictions').mkdir()
                key = ('synthetic', 'CA')
                m._GROUPS = {key: observations}
                m._CONFIG = {'output': d, 'seed': 19471, 'reference_cap': 2}
                audit = m.process_group(key)
                with gzip.open(next((Path(d) / 'predictions').glob('*')), 'rt') as stream:
                    rows = [json.loads(line) for line in stream]
                self.assertEqual(len(rows), 4)
                self.assertEqual(audit['targets_reference_capped'], 4)
                for row in rows:
                    self.assertEqual(row['eligible_count'], 3)
                    self.assertEqual(row['reference_count'], 2)
                    self.assertNotIn(row['target_id'], row['full_support_ids'])
                    self.assertEqual(row['support_counts']['count_matched_knn'], row['support_counts']['gate_012'])
                missing = next(r for r in rows if r['target_id'] == 'o3')
                self.assertIsNone(missing['predictions']['gate_012'])
                self.assertIsNotNone(missing['predictions']['mean'])
        finally:
            m._GROUPS, m._CONFIG = old_groups, old_config


if __name__ == '__main__':
    unittest.main()