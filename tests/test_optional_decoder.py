import tempfile
import unittest
from unittest.mock import patch

import polars as pl
from upath import UPath

import run_capsule as pipeline


DECODER_COMPARISON = (
    ['is_aud_target', 'is_aud_rewarded', 'is_decoder_correct'],
    ['is_aud_target', 'is_vis_rewarded', 'is_decoder_correct'],
)


class OptionalDecoderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = UPath(self.tmp.name)
        with patch('sys.argv', ['test']):
            self.params = pipeline.Params(
                decoding_parquet_path=str(self.root / 'decoder.parquet'),
                decoder_areas_to_average=['MOs', 'SCm'],
                areas_to_process=['MOs'], min_units_across_sessions=0,
                datacube_version='v0.0.301',
            )
        self.trials = pl.DataFrame({
            'session_id': ['s', 's', 'missing'], 'trial_index': [0, 1, 0],
            'is_response': [True, False, True], 'grating_phase': [0., None, 0.5],
            'is_aud_rewarded': [True, True, False], 'is_vis_rewarded': [False, False, True],
        })

    def load_with_decoder_condition(self):
        with patch.object(pipeline, 'conditions_to_compare', [DECODER_COMPARISON]):
            return pipeline.load_decoder_results(self.params)

    def test_disabled_comparisons_do_not_touch_decoder_storage(self):
        with patch.object(pipeline.upath.UPath, 'exists', side_effect=AssertionError('Decoder access')):
            self.assertIsNone(pipeline.load_decoder_results(self.params))
            with patch.object(pipeline, 'conditions_to_compare', [DECODER_COMPARISON]):
                self.params.comparison_set = 'catch'
                self.assertIsNone(pipeline.load_decoder_results(self.params))
                self.params.comparison_set = 'context'
                self.params.cache_only = True
                self.assertIsNone(pipeline.load_decoder_results(self.params))

    def test_missing_decoder_file_logs_and_preserves_all_trials(self):
        with self.assertLogs(level='WARNING') as messages:
            decoder = self.load_with_decoder_condition()
        self.assertIsNone(decoder)
        self.assertIn('not found', messages.output[0])
        prepared = pipeline.prepare_trial_flags(self.trials, decoder)
        self.assertEqual(prepared.height, self.trials.height)
        for flag in ('is_decoder_correct', 'is_decoder_confident', 'is_decoder_incorrect'):
            self.assertEqual(prepared[flag].to_list(), [False] * 3)

    def test_partial_coverage_and_nan_predictions_remain_unavailable(self):
        pl.DataFrame({
            'session_id': ['s', 's'], 'trial_index': [0, 1],
            'MOs_predict_proba': [0.1, float('nan')],
            'SCm_predict_proba': [None, float('nan')],
        }).write_parquet(self.params.decoding_parquet_path)
        prepared = pipeline.prepare_trial_flags(self.trials, self.load_with_decoder_condition())
        self.assertEqual(prepared['session_id'].to_list(), ['s', 's', 'missing'])
        self.assertEqual(prepared['predict_proba'].to_list(), [0.1, None, None])
        self.assertEqual(prepared['is_decoder_correct'].to_list(), [True, False, False])
        self.assertEqual(prepared['is_decoder_confident'].to_list(), [True, False, False])
        self.assertEqual(prepared['is_decoder_incorrect'].to_list(), [False] * 3)

    def test_missing_area_columns_disable_decoder_without_changing_ensemble(self):
        pl.DataFrame({'session_id': ['s'], 'trial_index': [0], 'MOs_predict_proba': [0.1]}).write_parquet(
            self.params.decoding_parquet_path)
        with self.assertLogs(level='WARNING') as messages:
            self.assertIsNone(self.load_with_decoder_condition())
        self.assertIn('SCm_predict_proba', messages.output[0])

    def test_empty_decoder_file_is_optional(self):
        pl.DataFrame({name: [] for name in (
            'session_id', 'trial_index', 'MOs_predict_proba', 'SCm_predict_proba',
        )}).write_parquet(self.params.decoding_parquet_path)
        with self.assertLogs(level='WARNING') as messages:
            self.assertIsNone(self.load_with_decoder_condition())
        self.assertIn('empty', messages.output[0])

    def test_unrelated_read_errors_are_not_silently_swallowed(self):
        UPath(self.params.decoding_parquet_path).touch()
        with patch.object(pipeline.pl, 'read_parquet', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):
                self.load_with_decoder_condition()

    def test_main_reaches_analysis_without_decoder_file(self):
        inputs = self.root / 'inputs'
        inputs.mkdir()
        (inputs / 'MOs.parquet').touch()
        metadata = dict(align_to_col='stim_start_time', as_binarized_array=True,
                        pre=0.5, post=0.5, bin_size_s=0.001)
        with patch.object(pipeline, 'prepare_inputs', return_value=(inputs, metadata, self.trials, pl.DataFrame())), \
             patch.object(pipeline, 'sessions_with_good_fold_coverage', return_value=['s', 'missing']), \
             patch.object(pipeline, 'NEURAL_TRAJ_DIR', self.root / 'outputs'), \
             patch.object(pipeline, 'write_trajectory_separation_for_area') as writer:
            pipeline.main(self.params)
        prepared = writer.call_args.args[3]
        self.assertEqual(prepared.height, 3)
        self.assertEqual(prepared['is_decoder_correct'].to_list(), [False] * 3)

    def test_disabled_slots_keep_historical_output_ids(self):
        self.assertEqual(set(pipeline.integer_to_condition), {0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 14})
        visual = pipeline.integer_to_condition[3]
        self.assertIn('is_vis_target', visual[0])
        self.assertIn('is_correct_reject', visual[1])
        trials = pl.DataFrame({name: [False] for name in pipeline.condition_cols}).with_columns(
            session_id=pl.lit('s'), trial_index=pl.lit(0), block_index=pl.lit(0))
        spikes = trials.select('session_id', 'trial_index').with_columns(unit_id=pl.lit('u'))
        result = pl.DataFrame({'session_id': ['s'], 'traj_separation': [[0.]]})
        output = self.root / 'area_outputs'
        with patch.object(pipeline, 'compute_trajectory_separation_for_condition_pair', return_value=result):
            pipeline.write_trajectory_separation_for_population(spikes, 'MOs', output, self.params, {}, trials)
        ids = {int(path.stem.split('_')[-1]) for path in output.glob('*.parquet')}
        self.assertEqual(ids, set(pipeline.integer_to_condition))


if __name__ == '__main__':
    unittest.main()
