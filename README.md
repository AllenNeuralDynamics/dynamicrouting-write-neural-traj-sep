# Cross-validated trajectory separation

This document describes the metric used by
[`compute_trajectory_separation_for_condition_pair`](run_capsule.py) to quantify how
separated two neural population trajectories are, and why it replaces the previous
label-shuffle null strategy.
## Analysis documentation

- [Cross-validated trajectory separation](cross_validated_distance.md)
- [Baseline and orthogonal stimulus projections](baseline_projection_analysis.md)

This template sets up a starting point for processing NWB files attached in a DynamicRouting "datacube" data asset.

## The problem with the old metric

For each session, condition pair, and timepoint `t`, the old code computed the
Euclidean distance between the two condition-**mean** trajectories, normalized to an
RMS-per-unit scale:

```
sep(t) = sqrt( (1/N_u) * sum_u ( mean_1[u,t] - mean_2[u,t] )^2 )
```

where `u` indexes units, `mean_c[u,t]` is the trial-averaged convolved firing rate for
condition `c`.

This estimator is **positively biased**. Even when the two conditions have identical
true means (`mu_1 = mu_2`), finite-trial sampling noise makes the empirical means
differ, so

```
E[ sep^2 ] = || mu_1 - mu_2 ||^2  +  sum_u ( var(mean_1[u,t]) + var(mean_2[u,t]) )
           = || mu_1 - mu_2 ||^2  +  (bias floor > 0)
```

The bias floor is always positive and grows as trial counts shrink, so raw distances
are not comparable across conditions/areas/sessions with different trial counts.

The old workaround estimated that floor by **shuffling condition labels** across trials
(`n_null_iterations = 100` per session) and recomputing the distance to build a null
distribution. This is expensive (100x the core computation per session) and only yields
a baseline to compare against rather than a corrected, zero-centered metric.

## The cross-validated distance

A cross-validated (crossnobis-style) distance removes the bias directly. Split the
trials into two **disjoint** folds, form the condition-difference vector in each fold,
and take their inner product over units (independently per timepoint):

```
delta_A(t) = mean_1^A(t) - mean_2^A(t)      (fold A)
delta_B(t) = mean_1^B(t) - mean_2^B(t)      (fold B)

d2_cv(t)   = delta_A(t) . delta_B(t)  =  sum_u  delta_A[u,t] * delta_B[u,t]
```

Because the sampling noise in fold A is independent of the noise in fold B, the
noise cross-terms vanish in expectation and

```
E[ d2_cv(t) ] = || mu_1(t) - mu_2(t) ||^2
```

This is **unbiased** for the true squared distance. Key consequences:

- It is **centered at zero** under the null (`mu_1 = mu_2`) and goes **negative** when
  the conditions do not truly separate — exactly the behavior the shuffle null was
  approximating. No null distribution is needed.
- **Do not clip negative values.** Clipping reintroduces the bias the metric exists to
  remove. Keep the sign.

### Output scaling

To stay on the same RMS-per-unit scale as the old metric (and remain interpretable as a
"distance"), we report the signed square root of the per-unit cross-validated distance:

```
traj_separation(t) = sign(d2_cv(t)) * sqrt( |d2_cv(t)| / N_u )
```

Only the sign and magnitude convention change; under the null this fluctuates around 0.

## Why folds must respect block structure

In this task the experimental design forces a specific fold structure:

- A session has **6 blocks of ~10 minutes each**.
- **Context alternates every block** (e.g. `A B A B A B`).
- The condition we compare across **is** the context (e.g. aud-rewarded vs vis-rewarded
  for the same stimulus).

Therefore **the two conditions never co-occur within a block** — every block contributes
trials to exactly one of the two conditions. Two problems follow:

1. **You cannot cross-validate within a block.** The resampling unit must be the block,
   not the trial.
2. **Slow drift is confounded with context.** Engagement/arousal drift over the session
   means trials cluster by block (positive intra-block correlation). Two failure modes:
   - A naive i.i.d. bias correction (`s^2/N`) or random single-trial leave-one-out
     treats trials as independent. With block clustering the effective N is closer to
     the number of blocks than the number of trials, so the noise floor is
     **under-corrected** and a positive bias leaks through.
   - Because context alternates, every A-block sits at an *earlier* position than its
     B-block partner. A monotonic drift therefore adds a consistent A-vs-B offset that
     is **real signal in the data but not the context effect we want** — a confound, not
     just noise.

So folds must (a) be **disjoint at the block level** to remove the noise floor, and
(b) be **balanced in time** so a linear drift contributes equally to both folds and
cancels.

## Balanced-split cross-validation

Model each block's response as `m = mu_context + gamma * t` (per-unit linear drift slope
`gamma`, block time `t`). For a fold, define its **centroid offset**

```
c = (mean position of cond-1 blocks) - (mean position of cond-2 blocks)
```

The fold difference vector carries a drift term proportional to `c`:

```
delta_fold = (mu_1 - mu_2) + c * gamma
```

For the cross-validated product `delta_train . delta_test`, expanding gives a linear
drift bias proportional to `(c_train + c_test)`. It vanishes iff

```
c_train + c_test = 0      (equal-and-opposite centroid offsets)
```

The fully-balanced special case `c_train = c_test = 0` leaves no residual at all; this is
the natural **`A B A` | `B A B` triple split** of an intact 6-block session:

```
blocks:   1   2   3 | 4   5   6
context:  A   B   A | B   A   B

fold train = {1,2,3}:  cond-1 = {1,3} (centroid 2),  cond-2 = {2} (centroid 2)  -> c=0
fold test  = {4,5,6}:  cond-1 = {5}   (centroid 5),  cond-2 = {4,6} (centroid 5) -> c=0
```

Both difference vectors are linear-drift-free estimates of `mu_1 - mu_2` computed from
disjoint blocks, so their inner product is both noise-floor-unbiased **and** immune to
linear drift, in a single estimator with no shuffle loop. This works regardless of
whether the session starts on context A or B (you always get `XYX | YXY`).

### Ragged blocks

Blocks are dropped before this stage by the good-block filter (`cross_modality_dprime`
and `n_contingent_rewards` thresholds in `run_capsule.py`), so the clean 6-block layout
is the best case, not the only case. The general criterion (`c_train + c_test = 0`,
each fold with at least one block per condition, folds disjoint) handles whatever blocks
survive. Feasibility depends on the surviving block **positions**, not just the count:
dropping an *interior* block can flip an infeasible set into a feasible one, because it
changes which context leads in time.

Enumerating all subsets of a full session (context alternating, odd positions = A, even
= B), only **7 of the 63** non-empty subsets admit a drift-cancelling split:

| # blocks | Surviving blocks   | Pattern (time order) | Chosen split          |
|----------|--------------------|----------------------|-----------------------|
| 6        | `{1,2,3,4,5,6}`    | ABABAB               | `{1,2,3} \| {4,5,6}`  |
| 5        | `{1,2,3,4,5}`      | ABABA                | `{1,2,5} \| {3,4}`    |
| 5        | `{2,3,4,5,6}`      | BABAB                | `{2,3,6} \| {4,5}`    |
| 5        | `{1,2,4,5,6}`      | ABBAB                | `{1,2}   \| {4,5}`    |
| 5        | `{1,2,3,5,6}`      | ABAAB                | `{2,3}   \| {5,6}`    |
| 4        | `{1,2,4,5}`        | ABBA                 | `{1,2}   \| {4,5}`    |
| 4        | `{2,3,5,6}`        | BAAB                 | `{2,3}   \| {5,6}`    |

Everything else is infeasible — all subsets of <= 3 blocks, and 12 of the 15 four-block
subsets. Reading off the pattern:

- **Minimum is 4 blocks**, and only the two *palindromic* patterns `ABBA` (`{1,2,4,5}`)
  and `BAAB` (`{2,3,5,6}`): A- and B-centroids coincide, so the `{early} | {late}` fold
  pair gives offsets `-1 / +1`.
- **5 blocks = drop exactly one block.** Feasible iff you drop block **1, 3, 4, or 6**.
  Dropping block **2 or 5 breaks it** — that leaves both B-blocks (or both A-blocks)
  stuck on one side of the session, so no fold can lean the other way.
- **6 blocks** always works (the `ABA | BAB` triple split).
- The feasible set is symmetric under time-reversal (1<->6, 2<->5, 3<->4). For a session
  that *starts on B* (`BABABA`), swap the A/B labels; the feasibility pattern is identical.

Two things to keep in mind reading the table: it treats blocks as equally spaced with
equal weight (block-center coordinate), so the real per-condition trial-time centroids
shift the exact offsets slightly but not the feasibility structure; and in practice you
do not choose which blocks survive (the good-block filter does), so this is a lookup for
"given which blocks passed, is the session usable" — exactly what `find_balanced_split`
computes per session. Infeasible sessions are **omitted**, and the count is printed per
area/condition pair (`omitted N/M sessions ...`) so the data loss is visible.

### Selection rule

`find_balanced_split(blocks, tolerance)` enumerates all `{unused, train, test}`
assignments of the available blocks (at most `3^6` = 729), keeps those with
`|c_train + c_test| <= tolerance`, and picks the one that

1. minimizes the residual imbalance `|c_train + c_test|` (first-order drift bias), then
2. minimizes the second-order term `|c_train * c_test|` (this reduces to "prefer the
   fully-balanced `c=0` split" when the imbalance is 0), then
3. maximizes the number of trials used.

It returns `(train_blocks, test_blocks, c_train, c_test)`, or `None` when no split within
`tolerance` exists.

### Relaxing the balance: the `balance_tolerance` parameter

`Params.balance_tolerance` sets the `tolerance` above (block-position units). Default
`0.0` enforces exact linear-drift cancellation (`c_train + c_test = 0`). Raising it
admits more ragged-block sessions at the cost of a residual first-order drift bias
proportional to `c_train + c_test`.

The catch, from enumerating the block grid: the minimum achievable imbalance is **either
0 or >= 2** — there is nothing in between (blocks are equally spaced, so the next-best
split is a full two block-positions off). Consequently:

- any `balance_tolerance` in `(0, 2)` admits **no** new sessions;
- `balance_tolerance >= 2` keeps the 6 additional four/five-block subsets (`ABAB`-type),
  but every one of them carries a `~2` block-position (`~20 min`) uncancelled drift
  offset. That is a cliff, not a gentle loosening.

A sign-only criterion ("A before B in one fold, B before A in the other") is **not**
used and is not equivalent: it does not bound the residual magnitude, and for `ABAB` the
opposite-sign split has the *same* imbalance of 2 as the same-sign split, so it buys
nothing. The magnitude `|c_train + c_test|`, not the sign pattern, is what controls the
bias.

Every kept session records its chosen `c_train`, `c_test`, and
`balance_residual = c_train + c_test` in the output (see below), and the run logs how
many sessions used a non-zero residual and the max `|residual|`. Recommended workflow:
run once at `balance_tolerance=0.0` and once at `2.0`, then use the `balance_residual`
column to check whether the extra (residual-2) sessions move your conclusions before
trusting them. If the drift is as mild as expected, they will not.

## Algorithm summary

Per session, per condition pair:

1. Convolve and bin spikes per trial; carry `block_index` through.
2. Require at least `min_units_per_session_area` unique units in the area/session
   (default 10).
3. Build a per-block `(block_index, condition_id, n_trials)` table.
4. `find_balanced_split(blocks, params.balance_tolerance)` ->
   `(train_blocks, test_blocks, c_train, c_test)`, or omit the session.
5. Compute per-unit block means, then each fold's condition-difference vector
   `delta = mean-over-blocks(cond_1) - mean-over-blocks(cond_2)` (equal weight per block,
   which keeps the centroid algebra above exact).
6. `d2_cv(t) = sum_u delta_train[u,t] * delta_test[u,t]` (units present in both folds).
7. `traj_separation(t) = sign(d2_cv) * sqrt(|d2_cv| / N_u)`.

Output is **one row per session** with `traj_separation`, `n_units`, the chosen
`train_blocks` / `test_blocks`, and the balance audit columns `c_train`, `c_test`, and
`balance_residual` (`= c_train + c_test`; `0` means exact linear-drift cancellation).
There are no null rows.

## Inference

With a zero-centered, unbiased per-session metric, significance no longer comes from a
per-session shuffle null. Treat each session's signed `traj_separation(t)` as one
observation and aggregate **across sessions** — e.g. mean +/- SEM across sessions and a
test against 0 (separation is present where the across-session distribution sits
reliably above zero). Aggregating across sessions is also the statistically cleaner unit
of inference than the within-session shuffle it replaces.

## Caveats and limitations

- **Linear drift only.** The balanced split cancels a linear trend exactly; quadratic
  curvature in the drift survives. This is usually negligible over a session; removing it
  would require explicit detrending (deliberately not used here).
- **Single fold per session.** The split is symmetric (`train . test == test . train`),
  so there is effectively one estimate per session. If per-session variance matters, one
  could average over several valid balanced splits; currently a single best split is
  used.
- **Session omission.** Drift cancellation requires near-complete block alternation, so
  ragged-block sessions are dropped. Monitor the printed omission counts; if too lossy, a
  fallback (e.g. plain block-disjoint CV without the drift-cancellation requirement, at
  the cost of a residual drift confound) can be added for the affected sessions.
- **Plain Euclidean, not whitened.** The inner product is unweighted across units, so
  correlated neural noise is not down-weighted. Accounting for it would require whitening
  by the noise covariance (Mahalanobis / true crossnobis) — an orthogonal extension that
  applies equally to any of these estimators.
- **Per-timepoint independence.** The metric is computed independently per timepoint;
  temporal correlations do not bias the point estimate but should be accounted for in any
  inference that treats timepoints as independent samples.
