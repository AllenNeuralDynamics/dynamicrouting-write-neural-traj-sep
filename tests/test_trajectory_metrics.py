from __future__ import annotations

import unittest

import numpy as np

from trajectory_metrics import (
    compute_trajectory_metrics,
    fixed_block_folds,
    has_good_block_coverage,
    signed_rms,
    time_bin_centers,
    window_mask,
)


class FixedBlockFoldTests(unittest.TestCase):
    def test_accepts_both_alternation_orders(self) -> None:
        self.assertEqual(
            fixed_block_folds([0, 1, 2, 3, 4, 5], [1, 2, 1, 2, 1, 2]),
            (frozenset({0, 1, 2}), frozenset({3, 4, 5})),
        )
        self.assertIsNotNone(
            fixed_block_folds([1, 2, 3, 4, 5, 6], [2, 1, 2, 1, 2, 1])
        )

    def test_rejects_ragged_or_non_alternating_blocks(self) -> None:
        self.assertIsNone(
            fixed_block_folds([0, 1, 2, 4, 5, 6], [1, 2, 1, 2, 1, 2])
        )
        self.assertIsNone(
            fixed_block_folds([0, 1, 2, 3, 4, 5], [1, 2, 1, 1, 2, 1])
        )

    def test_good_block_coverage_requires_each_context_in_each_fold(self) -> None:
        blocks = [0, 1, 2, 3, 4, 5]
        contexts = [1, 2, 1, 2, 1, 2]
        self.assertTrue(has_good_block_coverage(blocks, contexts, {1, 2, 3, 4}))
        self.assertFalse(has_good_block_coverage(blocks, contexts, {0, 2, 3, 4}))
        self.assertFalse(has_good_block_coverage(blocks, contexts, {1, 2, 3, 5}))


class TimeWindowTests(unittest.TestCase):
    def test_bin_centers_and_half_open_window(self) -> None:
        time_s = time_bin_centers(1000, bin_size_s=0.001, pre_s=0.5)
        self.assertAlmostEqual(time_s[0], -0.4995)
        self.assertAlmostEqual(time_s[-1], 0.4995)
        self.assertEqual(window_mask(time_s, -0.5, -0.01).sum(), 490)
        self.assertEqual(window_mask(time_s, 0.0, 0.1).sum(), 100)


class TrajectoryMetricTests(unittest.TestCase):
    def test_baseline_correction_and_cross_fitted_axes(self) -> None:
        baseline = np.array([2.0, 0.0, 0.0])
        stimulus_modulation = np.array([4.0, 3.0, 0.0])
        delta = np.column_stack(
            [
                baseline,
                baseline,
                baseline + stimulus_modulation,
                baseline + stimulus_modulation,
            ]
        )
        baseline_mask = np.array([True, True, False, False])
        stimulus_mask = ~baseline_mask

        metrics = compute_trajectory_metrics(
            delta, delta, baseline_mask, stimulus_mask
        )

        np.testing.assert_allclose(metrics.raw_d2, [4.0, 4.0, 45.0, 45.0])
        np.testing.assert_allclose(
            metrics.baseline_corrected_d2, [0.0, 0.0, 25.0, 25.0]
        )
        self.assertEqual(metrics.baseline_d2, 4.0)
        np.testing.assert_allclose(
            metrics.baseline_axis_projection,
            [0.0, 0.0, 4.0 / np.sqrt(3), 4.0 / np.sqrt(3)],
        )
        np.testing.assert_allclose(
            metrics.orthogonal_stimulus_projection,
            [0.0, 0.0, 3.0 / np.sqrt(3), 3.0 / np.sqrt(3)],
        )
        self.assertAlmostEqual(metrics.baseline_axis_alignment, 1.0)
        self.assertAlmostEqual(metrics.orthogonal_axis_alignment, 1.0)

    def test_signed_rms_preserves_negative_cross_validated_values(self) -> None:
        np.testing.assert_allclose(
            signed_rms(np.array([-8.0, 0.0, 8.0]), n_units=2),
            [-2.0, 0.0, 2.0],
        )

    def test_independent_noise_does_not_create_a_distance_floor(self) -> None:
        rng = np.random.default_rng(1234)
        true_baseline = np.array([1.0, -0.5, 0.25, 0.0])
        true_evoked = np.array([0.5, 0.0, -0.25, 0.75])
        true_delta = np.column_stack([true_baseline, true_baseline + true_evoked])
        baseline_mask = np.array([True, False])
        stimulus_mask = np.array([False, True])

        estimates = []
        for _ in range(4000):
            fold_1 = true_delta + rng.normal(scale=0.5, size=true_delta.shape)
            fold_2 = true_delta + rng.normal(scale=0.5, size=true_delta.shape)
            estimates.append(
                compute_trajectory_metrics(
                    fold_1, fold_2, baseline_mask, stimulus_mask
                ).baseline_corrected_d2[1]
            )

        self.assertAlmostEqual(
            float(np.mean(estimates)), float(np.dot(true_evoked, true_evoked)), delta=0.05
        )


if __name__ == '__main__':
    unittest.main()
