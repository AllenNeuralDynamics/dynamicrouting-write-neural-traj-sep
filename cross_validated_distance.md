# Cross-validated trajectory separation

This document describes the distance estimator implemented by
`compute_trajectory_separation_for_condition_pair` in `code/run_capsule.py`.

## Why the ordinary distance is biased

At each timepoint, let the condition difference across units be

```text
delta(t) = mean_condition_1(t) - mean_condition_2(t).
```

The ordinary squared Euclidean distance, `delta(t) . delta(t)`, is positive even
when the two true condition means are identical. Finite-trial error is squared along
with the signal, producing a positive noise floor that depends on trial count.

The cross-validated estimator forms the condition difference independently in two
disjoint sets of blocks:

```text
d2_cv(t) = delta_fold_1(t) . delta_fold_2(t).
```

When fold errors are independent and zero-mean,

```text
E[d2_cv(t)] = ||true_delta(t)||^2.
```

Negative estimates are expected when the true distance is small. They must not be
clipped, because clipping recreates a positive noise floor.

## Fixed six-block folds

Every included session must contribute qualifying trials from six consecutive blocks,
with the two conditions alternating across blocks. The folds are fixed by time order:

```text
blocks:   0 1 2 | 3 4 5
pattern:  X Y X | Y X Y
fold 1:   0 1 2
fold 2:   3 4 5
```

This is valid whether condition 1 or condition 2 occurs first. Each condition's mean
block position equals the middle block of its fold, so a linear session drift cancels
within both condition contrasts. Blocks are weighted equally after trials are averaged
within block.

There is no combinatorial split search and no relaxed-balance parameter. A session is
omitted for a condition pair if that pair does not have at least one qualifying trial in
every block. This can happen for rare trial outcomes even when the recording itself has
all six task blocks.

The session selection still requires `is_good_behavior`. It also requires both fixed
folds to contain at least one good block from each context. By default, a good block has
`cross_modality_dprime >= 1.0` and `n_contingent_rewards >= 10`; both thresholds are
parameters. These criteria determine whether a session is eligible but do not remove
individual blocks. Thus all six blocks from a passing session participate when the
requested condition trials exist.

## Baseline correction

The primary output removes pre-stimulus condition separation before calculating the
cross-validated distance. Baseline correction is performed separately in each fold:

```text
b_f = mean over baseline timepoints of delta_f(t)
e_f(t) = delta_f(t) - b_f
d2_baseline_corrected(t) = e_fold_1(t) . e_fold_2(t).
```

This estimates the squared magnitude of the stimulus-related change in condition
separation, including changes parallel and orthogonal to the baseline context vector.
It is not equivalent to subtracting a scalar baseline from an already-computed distance.

Defaults are:

```text
baseline window:                [-0.5, -0.01) s
projection-axis stimulus window: [0, 0.1) s
stimulus onset:                  0 s
```

The 10 ms gap before onset prevents the centered smoothing kernel from leaking
post-onset activity into the baseline estimate. Window endpoints and PSTH timing are
recorded in the run sidecar JSON.

## Scaling

The unbiased quantity is the squared cross-validated inner product, `d2_cv`. For output,
it is converted to a signed RMS-per-unit scale:

```text
signed_rms(d2_cv) = sign(d2_cv) * sqrt(abs(d2_cv) / n_units).
```

This transform is easy to interpret and preserves negative estimates, but the nonlinear
signed square root is not itself an unbiased estimator of true Euclidean distance.

Binary 1 ms spike arrays are smoothed using a kernel normalized by both its number of
bins and the bin duration. Consequently condition differences and projection outputs
are expressed in Hz, rather than spikes/ms.

## Algorithm

For each session, area, and condition pair:

1. Assign trials to condition 1, condition 2, or neither.
2. Smooth each trial's binary spike train into a firing-rate trace in Hz.
3. Require six consecutive alternating blocks and the per-fold good-block coverage rule.
4. Retain all six blocks, then average trials within each unit and block.
5. In each fixed three-block fold, average block means within condition and calculate
   `condition_1 - condition_2`.
6. Keep units present in both conditions of both folds.
7. Compute raw and fold-specific baseline-corrected cross-validated distances.
8. Compute the cross-fitted projection analysis described in
   [`baseline_projection_analysis.md`](baseline_projection_analysis.md).

## Output

One parquet file is written per `(area, condition_pair_id)`, with one row per usable
session. `condition_pair_id` indexes `integer_id_to_condition_mapping` in the run's
sidecar JSON.

| Column | Meaning |
|---|---|
| `session_id` | Session identifier. |
| `traj_separation` | Primary baseline-corrected signed RMS distance at every timepoint. |
| `traj_separation_raw` | Raw signed RMS distance, including baseline separation. |
| `baseline_separation` | Cross-validated signed RMS magnitude of the baseline condition vector. |
| `baseline_axis_projection` | Baseline-corrected condition activity projected onto a cross-fitted baseline axis. |
| `orthogonal_stimulus_projection` | Baseline-corrected condition activity projected onto a cross-fitted stimulus axis orthogonal to baseline. |
| `baseline_axis_alignment` | Cosine alignment between baseline axes estimated in the two folds. |
| `orthogonal_axis_alignment` | Cosine alignment between orthogonal stimulus axes estimated in the two folds. |
| `n_units` | Units present in both conditions of both folds. |
| `fold_1_blocks` | First three block indices. |
| `fold_2_blocks` | Last three block indices. |
| `condition_pair_id` | Condition-pair index matching the filename and sidecar mapping. |

## Inference and limitations

Sessions are the independent unit of inference. Time-resolved tests must account for
temporal correlation and multiple comparisons.

The estimator removes the ordinary finite-sample distance floor under fold independence,
but block-to-block correlations can reduce that independence. The fixed split cancels a
linear drift; nonlinear session drift can remain. The geometry is ordinary, unwhitened
Euclidean geometry, so high-variance or high-rate units are not down-weighted.
