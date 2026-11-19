# Catch–target trajectory separation and paired T25

The pipeline can calculate stimulus-versus-catch separation using the same
baseline-corrected, cross-validated signed RMS metric as context separation.
The default `comparison_set='context'` runs the existing condition list.
`comparison_set='catch'` runs the four new comparisons only; `'all'` runs both
and supplies the inputs for paired latency analysis. Existing numeric condition
IDs are unchanged.

## Comparisons

| Output name | Target trials | Catch trials | Context reference ID |
| --- | --- | --- | --- |
| `catch_visual_unrewarded` | Visual target, auditory context, correct rejection | Auditory context, no response | 3 |
| `catch_visual_rewarded` | Visual target, visual context, hit | Visual context, no response | 3 |
| `catch_auditory_unrewarded` | Auditory target, visual context, correct rejection | Visual context, no response | 0 |
| `catch_auditory_rewarded` | Auditory target, auditory context, hit | Auditory context, no response | 0 |

The target is condition 1 and catch is condition 2. A no-response catch is
defined as `is_catch & ~is_response`; it does not require `is_correct_reject`
to be true on catch trials. Target outcome filters match the existing context
comparison. The nonrewarded-context comparisons are the primary sensory timing
measure; rewarded-context contrasts can also contain response preparation and
reward-related activity. No additional instruction-trial or response-time
exclusions are introduced.

## Estimator and coverage

Each session must have six consecutive blocks with alternating rewarded contexts.
Folds are determined from the full session: the first three blocks and the last
three blocks. Within the selected context, every one of its three blocks must
contain both qualifying target and catch trials. Each fold therefore contains
one or two blocks of that context. Missing blocks are not silently removed or
used to redefine folds.

Within each block, trials are averaged separately for target and catch. Block
means receive equal weight within each fold, regardless of trial counts. Because
both conditions use the same blocks and weights, subtracting their fold means
equals averaging their within-block differences. The two independent fold
differences enter the existing baseline correction and cross-product calculation.
Negative cross-validated values are retained through signed RMS conversion.

`min_trials_per_condition_block` defaults to 1 and applies to the new mode only.
Increase it if desired; it is a coverage threshold, not a guarantee of precision.
Every retained unit must satisfy it in every selected block for both conditions.
The per-session unit threshold is checked again after this filtering.

New files record the actual `unit_ids`, fold blocks, and the unique session trial
counts by block and condition. These counts are not multiplied by the number of
units. All newly computed context outputs also record `unit_ids` so matched
population checks are possible. Existing saved files without those IDs must be
recomputed for paired latency analysis.

## Run trajectory calculation

In the capsule environment, with its usual trial/performance inputs and access
to the source PSTHs:

```bash
python code/run_capsule.py \
  --input_dir_name 2026-08-20 \
  --output_dir_name catch_target_25ms \
  --areas_to_process '["MRN", "SCm", "MOs", "CP"]' \
  --comparison_set all \
  --conv_kernel_s 0.025 \
  --decoder_areas_to_average '[]'
```

The input is `s3://aind-scratch-data/dynamic-routing/psths/2026-08-20`;
its sidecar specifies 1 ms stimulus-aligned spike arrays from -500 to +500 ms,
using datacube `v0.0.289`. The 25 ms centered kernel
matches the preprocessing described in the existing backward-T25 analysis; both
new and reference distances are recomputed with these settings. No additional
smoothing is applied by the latency script. Disabling decoder loading is sufficient
for the catch contrasts and reference IDs 0 and 3; decoder-filtered condition files
will be empty. Leave the decoder setting at its default if those extra comparisons
are also wanted and the decoder input is available.

As with the existing runner, `parameters.json` in the working directory takes
precedence over CLI values. Set these options there instead if that file is in use.
Existing session/performance and unit eligibility thresholds still apply.
Choose a fresh output name for the paired analysis. Catch analysis manifests
reject changed settings with `skip_existing=True`, and reject reusing a directory
of old outputs without provenance. Setting `skip_existing=False` permits
recomputation.

Output area files live under
`s3://aind-scratch-data/dynamic-routing/neural_trajectory_separation_all_conditions/catch_target_25ms/`:

```text
catch_analysis.json
SCm_0.parquet
SCm_3.parquet
SCm_catch_visual_unrewarded.parquet
SCm_catch_visual_rewarded.parquet
SCm_catch_auditory_unrewarded.parquet
SCm_catch_auditory_rewarded.parquet
...
```

Custom unit groups work with the same options. Their manifest and parquet files
are in `unit_groups/<name>/` inside the output run.

For PSTHs already loaded in Python, join trial flags including `block_index`,
both rewarded-context flags, and `is_no_response`, then call:

```python
from run_capsule import compute_trajectory_separation_for_condition_pair

result = compute_trajectory_separation_for_condition_pair(
    psth_df,
    ['is_vis_target', 'is_aud_rewarded', 'is_correct_reject'],
    ['is_catch', 'is_aud_rewarded', 'is_no_response'],
    params, psth_params,
    comparison_mode='within_block',
    context_column='is_aud_rewarded',
)
```

Pass the full session's PSTH table, including blocks from the other context:
the function needs it to establish the original six-block structure. The direct
calculation does not apply the capsule's session/performance filter. To write all
configured comparisons and their manifest from loaded data, use
`write_trajectory_separation_for_population(psths, name, upath.UPath(output_dir),
params, psth_params, trials)` with the prepared eligible trial table. This also
works with a local output directory.

The writers require finite nominal `stim_start_time` timestamps for catch trials
and at least some catch PSTHs in the population input. If catches were omitted
from the source PSTHs, those PSTHs must be regenerated. The timestamp check does
not establish that the task's catch timestamps are semantically aligned; that
still depends on the source dataset. Decoder annotation uses a left join, so
trials absent from decoder outputs remain available to comparisons that do not
require decoder predictions. Missing predictions pass neither correct nor
incorrect decoder filters. Grating-phase flags also work without decoder input.

## Run paired T25 locally

From the repository root:

```bash
.venv/bin/python dynamicrouting-write-neural-traj-sep/code/catch_latency_analysis.py \
  --trajectory-dir s3://aind-scratch-data/dynamic-routing/neural_trajectory_separation_all_conditions/catch_target_25ms \
  --output-dir results/catch_target_25ms \
  --populations MRN SCm MOs CP \
  --n-bootstrap 10000 \
  --seed 20261118
```

`--trajectory-dir` also accepts a local folder. For a custom unit group, point at
its `unit_groups/<name>/` folder and pass its name as `--populations`.
The source manifest must specify `comparison_set='all'`. The existing T25
definition requires 1 ms bins and coverage of at least -100 to +150 ms.

For each population/comparison, the runner intersects stimulus and context session
cohorts, requires identical unit identities, and excludes malformed or nonfinite
traces. Excluded sessions and reasons are saved in `session_audit.csv`; comparisons
with no matched sessions are listed in `unavailable_comparisons.csv`. If no pairs
survive, the runner stops after writing these audits. Regenerate mismatched
populations with consistent unit coverage rather than comparing different units.

Sessions are averaged within mouse, using the datacube session-ID prefix as mouse
identity. Point estimates weight mice equally. One bootstrap resamples the union
of mice across all requested comparisons; a mouse carries the same multiplicity
in every signal. Each draw fits the mean trace and recomputes its baseline,
reference maximum, and backward T25. It does not average individual-mouse latencies.
At least two mice are required in the union; inspect the reported per-comparison
mouse counts when interpreting uncertainty for sparse cohorts.

The spline settings are unchanged: fixed penalty 1,000 in millisecond coordinates,
fit interval -99.5 to 149.5 ms, integral baseline -99.5 to -25 ms, and reference
maximum in 0–100 ms. T25 is the last upward 25% crossing before that maximum.
Negative crossings remain in `crossing_ms`, with `rise_ms=0` and a censoring flag.
Failed or empty bootstrap draws remain NaN with explicit statuses.

Outputs include:

- `overview.csv`: stimulus and context T25, status, and unadjusted 95% intervals.
- `delays.csv`: **context separation T25 minus catch–target T25**, with paired
  intervals and valid/failed draw counts. Positive values mean context separation
  reaches its own 25% amplitude later.
- `delay_draws.csv`, `bootstrap.npz`: paired differences, all timing measurements,
  statuses, and shared mouse multiplicities.
- `inputs.npz`, `metadata.json`: matched mouse traces, cohorts, settings, seed,
  and source parquet hashes.
- `fitted_curves.csv`, `latencies.png`, `latencies.pdf`: fitted means and summaries.

Intervals condition on finite paired draws and retain the existing zero-boundary
convention. They are not a dedicated censored-data analysis or corrected multiple
comparison test. T25 measures development relative to each trace's reference
amplitude; it is not the first biological response or a transmission delay.
