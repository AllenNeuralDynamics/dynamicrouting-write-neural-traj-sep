# Baseline and orthogonal stimulus projection analysis

This analysis gives a low-dimensional, hypothesis-driven description of the condition
difference. It asks two related questions:

1. Does the stimulus modulate the population along the context-separation direction
   already present before stimulus onset?
2. Does it produce a reproducible modulation in a new direction orthogonal to that
   baseline separation?

The implementation is split between `code/run_capsule.py`, which constructs condition
contrasts from trials and blocks, and `code/trajectory_metrics.py`, which performs the
numerical distance and projection calculations.

## Inputs

For each session, the six blocks are divided into two fixed, disjoint folds:

```text
fold 1 = first three blocks  (X Y X)
fold 2 = last three blocks   (Y X Y)
```

A session is eligible only when each fold contains at least one behaviorally good X
block and one behaviorally good Y block. This is an eligibility check only: after a
session passes, activity from all six blocks is retained in the analysis.

For each fold `f`, unit, and timepoint, the input is the condition-difference vector

```text
delta_f(t) = mean_condition_1,f(t) - mean_condition_2,f(t).
```

Trials are first averaged within blocks and blocks are then weighted equally. Only units
with both conditions in both folds are retained.

## Fold-specific baseline correction

The baseline context vector and baseline-corrected condition difference are

```text
b_f    = mean_t_in_baseline delta_f(t)
e_f(t) = delta_f(t) - b_f.
```

The default baseline window is `[-0.5, -0.01) s`. The end is kept 10 ms away from
stimulus onset to avoid leakage from the centered smoothing kernel.

## Axes

### Baseline context axis

Each fold's baseline axis is its normalized baseline condition difference:

```text
u_f = b_f / ||b_f||.
```

The sign is fixed by the condition ordering: condition 1 minus condition 2.

### Orthogonal stimulus axis

First average the baseline-corrected condition difference over a prespecified stimulus
window:

```text
m_f = mean_t_in_stimulus e_f(t).
```

The default axis-definition window is `[0, 0.1) s`. Remove its component along the
baseline axis and normalize the residual:

```text
r_f = m_f - (m_f . u_f) u_f
v_f = r_f / ||r_f||.
```

`v_f` is one targeted stimulus-modulation direction in the `(n_units - 1)`-dimensional
subspace orthogonal to `u_f`; it is not the entire orthogonal subspace.

Using `delta_f(t)` instead of `e_f(t)` to construct `v_f` would give the same result in
exact arithmetic, because the subtracted baseline vector is parallel to `u_f` and is
removed by the orthogonal projection. The code uses `e_f(t)` to keep the definition
explicitly tied to stimulus-related change.

## Cross-fitting

An axis must not be defined and evaluated on the same noisy fold. The analysis therefore
applies each fold's axes only to the other fold:

```text
x_2|1(t) = u_1 . e_2(t)       baseline axis learned in fold 1, tested in fold 2
x_1|2(t) = u_2 . e_1(t)       baseline axis learned in fold 2, tested in fold 1

y_2|1(t) = v_1 . e_2(t)       orthogonal axis learned in fold 1, tested in fold 2
y_1|2(t) = v_2 . e_1(t)       orthogonal axis learned in fold 2, tested in fold 1
```

The reported time courses average the two held-out directions:

```text
baseline_axis_projection(t) = 0.5 * [x_2|1(t) + x_1|2(t)] / sqrt(n_units)

orthogonal_stimulus_projection(t)
    = 0.5 * [y_2|1(t) + y_1|2(t)] / sqrt(n_units).
```

Dividing by `sqrt(n_units)` puts the projections on an RMS-per-unit scale. Positive
values mean that the independently estimated condition difference generalizes with the
condition-1-minus-condition-2 orientation learned in the other fold. Negative values
are retained and indicate reversed or non-generalizing structure.

Baseline subtraction remains necessary on the held-out fold. For example, `v_1` is
exactly orthogonal to `b_1`, but sampling variability means it need not be orthogonal to
the independently estimated `b_2`. Projecting `e_2(t) = delta_2(t) - b_2` prevents that
held-out baseline mismatch from being labeled stimulus modulation.

## Relationship to baseline-corrected distance

The baseline-corrected cross-validated distance is an omnibus measure:

```text
d2_baseline_corrected(t) = e_1(t) . e_2(t).
```

It includes reproducible stimulus-related changes in every population direction. The
two-axis analysis deliberately trades completeness for lower variance and clearer
interpretation:

- `baseline_axis_projection` measures reuse, amplification, or suppression of the
  baseline context direction.
- `orthogonal_stimulus_projection` measures the single stimulus-modulation direction
  learned after removing that baseline direction.
- Activity outside these two axes remains in the omnibus distance but not in the
  projection time courses.

The cross-fitted axes differ slightly between folds, so the two reported projection
time courses are not expected to reconstruct the full distance exactly.

## Reliability diagnostics

Two scalar cosine alignments are stored per session:

```text
baseline_axis_alignment   = u_1 . u_2
orthogonal_axis_alignment = v_1 . v_2.
```

Values near `1` indicate reproducible directions, values near `0` indicate unstable or
unrelated directions, and negative values indicate reversal across folds. If a baseline
or orthogonal residual has zero norm, its projection and alignment are stored as `NaN`.

Axis alignments should be shown or used as quality-control diagnostics rather than
silently selecting sessions based on the same data. In particular, an orthogonal axis
is not interpretable when baseline separation is too weak to define a stable baseline
direction.

## Output and interpretation

The projection analysis is written into the same per-area, per-condition-pair parquet
rows as the distance analysis:

| Column | Interpretation |
|---|---|
| `baseline_axis_projection` | Baseline-corrected condition modulation along a held-out baseline context axis. |
| `orthogonal_stimulus_projection` | Baseline-corrected condition modulation along a held-out orthogonal stimulus axis. |
| `baseline_axis_alignment` | Reproducibility of the baseline direction across block folds. |
| `orthogonal_axis_alignment` | Reproducibility of the orthogonal stimulus direction across block folds. |

Sessions remain the independent unit of inference. The stimulus window used to define
the orthogonal axis must be selected before examining its held-out time course. Changing
that window after viewing the result would turn a cross-fitted analysis into a
data-dependent selection procedure.
