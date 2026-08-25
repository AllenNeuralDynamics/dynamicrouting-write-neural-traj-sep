from __future__ import annotations

import json
import sys
import time

import polars as pl
import polars_ds as pds
import polars_vec_ops as vec
import numpy as np
import pydantic_settings
import pydantic
import upath

import utils
from trajectory_metrics import (
    compute_condition_projections,
    compute_trajectory_metrics,
    fixed_block_folds,
    has_good_block_coverage,
    normalize_by_baseline_rate,
    signed_rms,
    time_bin_centers,
    window_mask,
)

DATACUBE_VERSION = 'v0.0.289'
PSTH_DIR = upath.UPath('s3://aind-scratch-data/dynamic-routing/psths')
NEURAL_TRAJ_DIR = upath.UPath('s3://aind-scratch-data/dynamic-routing/neural_trajectory_separation_all_conditions')
decoding_parquet_path = '/root/capsule/data/all_trials_with_predict_proba.parquet'

# Minimum number of unique neurons required for a session/area to be included.
MIN_UNITS_PER_SESSION_AREA = 10

conditions_to_compare = (

    # aud targets
    [['is_aud_target', 'is_aud_rewarded', 'is_hit'], ['is_aud_target', 'is_vis_rewarded', 'is_correct_reject']],
    [['is_aud_target', 'is_aud_rewarded', 'is_hit'], ['is_aud_target', 'is_vis_rewarded', 'is_false_alarm']],
    [['is_aud_target', 'is_aud_rewarded', 'is_hit', 'is_decoder_correct', 'is_decoder_confident'], ['is_aud_target', 'is_vis_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident']],


    # vis targets
    [['is_vis_target', 'is_vis_rewarded', 'is_hit'], ['is_vis_target', 'is_aud_rewarded', 'is_correct_reject']],
    [['is_vis_target', 'is_vis_rewarded', 'is_hit'], ['is_vis_target', 'is_aud_rewarded', 'is_false_alarm']],
    [['is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_decoder_correct', 'is_decoder_confident'], ['is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident']],

    [['is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_grating_phase_half'], ['is_vis_target', 'is_aud_rewarded', 'is_correct_reject', 'is_grating_phase_half']],
    [['is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_grating_phase_half'], ['is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_grating_phase_half']],
    [['is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_decoder_correct', 'is_decoder_confident', 'is_grating_phase_half'], ['is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident', 'is_grating_phase_half']],

    [['is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_grating_phase_zero'], ['is_vis_target', 'is_aud_rewarded', 'is_correct_reject', 'is_grating_phase_zero']],
    [['is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_grating_phase_zero'], ['is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_grating_phase_zero']],
    [['is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_decoder_correct', 'is_decoder_confident', 'is_grating_phase_zero'], ['is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident', 'is_grating_phase_zero']],


    # aud nontargets
    [['is_aud_nontarget', 'is_aud_rewarded', 'is_correct_reject'], ['is_aud_nontarget', 'is_vis_rewarded', 'is_correct_reject']],


    # vis nontargets
    [['is_vis_nontarget', 'is_grating_phase_zero', 'is_vis_rewarded', 'is_correct_reject'], ['is_vis_nontarget', 'is_grating_phase_zero', 'is_aud_rewarded', 'is_correct_reject']],
    [['is_vis_nontarget', 'is_grating_phase_half', 'is_vis_rewarded', 'is_correct_reject'], ['is_vis_nontarget', 'is_grating_phase_half', 'is_aud_rewarded', 'is_correct_reject']],    
)

    
condition_cols = set()
for conds in conditions_to_compare:
    for cond in conds:
        condition_cols.update(cond)
condition_cols = sorted(condition_cols)

#Create mapping of null_condition_pairs to integers for easy storage and lookup
integer_to_condition = {i: cond for i, cond in enumerate(conditions_to_compare)}


class Params(pydantic_settings.BaseSettings):
    input_dir_name: str
    output_dir_name: str = pydantic.Field('test', exclude=True)
    skip_existing: bool = pydantic.Field(True, exclude=True)
    areas_to_process: list[str] | None = pydantic.Field(None, exclude=True)

    conv_kernel_s: float = 0.01
    decoder_areas_to_average: list[str] = pydantic.Field(default_factory=lambda: sorted(['ACAd', 'AId', 'AIp', 'FRP', 'ILA', 'MOs', 'MOp', 'ORBl', 'ORBvl', 'PL', 'SSp', 'SSs', 'MRN', 'SCm', 'CP']))
    baseline_start_s: float = -0.5
    baseline_end_s: float = -0.01
    baseline_rate_floor_hz: float = pydantic.Field(1.0, gt=0)
    projection_stimulus_start_s: float = 0.0
    projection_stimulus_end_s: float = 0.1
    good_block_dprime_threshold: float = 1.0
    good_block_min_contingent_rewards: int = 10
    min_units_per_session_area: int = pydantic.Field(
        MIN_UNITS_PER_SESSION_AREA, exclude=True
    )
    min_units_across_sessions: int = pydantic.Field(500, exclude=True)
    integer_id_to_condition_mapping: dict[int, list[list[str], list[str]]] = pydantic.Field(default_factory=lambda: integer_to_condition)

    # set the priority of the input sources:
    @classmethod  
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        *args,
        **kwargs,
    ):
        # instantiating the class will use arguments passed directly, or provided via the command line/app panel
        # the order of the sources below defines the priority (highest to lowest):
        # - for each field in the class, the first source that contains a value will be used
        return (
            init_settings,
            pydantic_settings.sources.JsonConfigSettingsSource(settings_cls, json_file='parameters.json'),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )


def sessions_with_good_fold_coverage(
    trials: pl.DataFrame, performance: pl.DataFrame, params: Params
) -> list[str]:
    """Return sessions with a good block of each context in both fixed folds.

    This is a session-level eligibility check. It does not remove any trials or blocks
    from sessions that pass.
    """
    good_blocks_by_session: dict[str, set[int]] = {}
    good_block_rows = (
        performance
        .filter(
            pl.col('cross_modality_dprime') >= params.good_block_dprime_threshold,
            pl.col('n_contingent_rewards')
            >= params.good_block_min_contingent_rewards,
        )
        .select('session_id', 'block_index')
        .unique()
    )
    for session_id, block_index in good_block_rows.iter_rows():
        good_blocks_by_session.setdefault(str(session_id), set()).add(int(block_index))

    block_contexts = (
        trials
        .select('session_id', 'block_index', 'is_aud_rewarded', 'is_vis_rewarded')
        .unique()
        .sort('session_id', 'block_index')
    )
    eligible_sessions = []
    for session_blocks in block_contexts.partition_by('session_id', maintain_order=True):
        session_id = str(session_blocks['session_id'][0])
        block_indices = [int(v) for v in session_blocks['block_index']]
        context_ids = []
        for is_aud, is_vis in session_blocks.select(
            'is_aud_rewarded', 'is_vis_rewarded'
        ).iter_rows():
            if bool(is_aud) == bool(is_vis):
                context_ids = []
                break
            context_ids.append(1 if is_aud else 2)
        if context_ids and has_good_block_coverage(
            block_indices,
            context_ids,
            good_blocks_by_session.get(session_id, set()),
        ):
            eligible_sessions.append(session_id)
    return eligible_sessions


def compute_trajectory_separation_for_condition_pair(
    area_psth_df: pl.DataFrame,
    condition_1: list[str],
    condition_2: list[str],
    params: Params,
    psth_params: dict,
) -> pl.DataFrame:
    """Compute distance and projection metrics for one area/condition pair."""
    condition_1_expr = pl.all_horizontal([pl.col(c) for c in condition_1])
    condition_2_expr = pl.all_horizontal([pl.col(c) for c in condition_2])

    bin_size_s = float(psth_params['bin_size_s'])
    kernel_bins_float = params.conv_kernel_s / bin_size_s
    kernel_bins = round(kernel_bins_float)
    if kernel_bins < 1 or not np.isclose(kernel_bins_float, kernel_bins):
        raise ValueError(
            f"conv_kernel_s ({params.conv_kernel_s}) must be a positive integer "
            f"multiple of bin_size_s ({bin_size_s})"
        )
    # Binary spike counts convolved with this kernel are expressed in spikes/s (Hz).
    convolution_kernel = np.full(
        kernel_bins, 1.0 / (kernel_bins * bin_size_s), dtype=np.float64
    )

    empty_schema = {
        'session_id': pl.String,
        # Primary output: baseline-corrected cross-validated signed RMS distance.
        'traj_separation': pl.List(pl.Float64),
        'traj_separation_rate_normalized': pl.List(pl.Float64),
        'traj_separation_raw': pl.List(pl.Float64),
        'baseline_separation': pl.Float64,
        'median_baseline_rate_hz': pl.Float64,
        'n_units_below_rate_floor': pl.Int64,
        'baseline_axis_projection': pl.List(pl.Float64),
        'baseline_axis_projection_condition_1': pl.List(pl.Float64),
        'baseline_axis_projection_condition_2': pl.List(pl.Float64),
        'orthogonal_stimulus_projection': pl.List(pl.Float64),
        'baseline_axis_alignment': pl.Float64,
        'orthogonal_axis_alignment': pl.Float64,
        'n_units': pl.Int64,
        'fold_1_blocks': pl.List(pl.Int64),
        'fold_2_blocks': pl.List(pl.Int64),
    }

    results = []
    n_omitted = 0
    session_unit_counts = (
        area_psth_df
        .select('session_id', 'unit_id')
        .unique()
        .group_by('session_id')
        .agg(pl.col('unit_id').n_unique().alias('n_area_units'))
    )
    session_unit_count_by_id = dict(session_unit_counts.iter_rows())
    session_list = area_psth_df['session_id'].unique().sort()
    for isess, session in enumerate(session_list):
        print(f"\rSession: {isess + 1} of {len(session_list)}", end="", flush=True)
        if (
            params.min_units_per_session_area
            and session_unit_count_by_id.get(session, 0)
            < params.min_units_per_session_area
        ):
            n_omitted += 1
            continue
        binned = (
            area_psth_df
            .filter(pl.col('session_id') == session)
            .with_columns(
                pl.when(condition_1_expr).then(pl.lit(1))
                .when(condition_2_expr).then(pl.lit(2))
                .otherwise(pl.lit(None))
                .alias('condition_id')
            )
            .drop_nulls(subset=['binarized_spike_times', 'condition_id'])
            .with_columns(pl.col('binarized_spike_times').cast(pl.List(pl.Float32)))
            .select(
                'unit_id', 'condition_id', 'binarized_spike_times', 'trial_index',
                'block_index', 'session_id'
            )
            .explode('binarized_spike_times')
            .group_by(
                'unit_id', 'trial_index', 'condition_id', 'block_index', 'session_id',
                maintain_order=True,
            )
            .agg(
                pds.convolve(
                    x='binarized_spike_times',
                    kernel=convolution_kernel,
                    mode='same',
                    method='direct',
                ).alias('firing_rate')
            )
            .sort('unit_id', 'trial_index')
        )

        if binned.height == 0:
            n_omitted += 1
            continue

        block_info = (
            binned
            .select('block_index', 'condition_id', 'trial_index')
            .unique()
            .group_by('block_index', 'condition_id')
            .agg(pl.len().alias('n_trials'))
            .sort('block_index')
        )
        block_indices = [int(v) for v in block_info['block_index']]
        condition_ids = [int(v) for v in block_info['condition_id']]
        folds = fixed_block_folds(block_indices, condition_ids)
        if folds is None:
            n_omitted += 1
            continue

        # For XYX | YXY, each condition's block centroid equals the middle block of
        # its fold.  The fixed split therefore uses every block and cancels linear drift.
        fold_1_blocks, fold_2_blocks = folds
        block_means = (
            binned
            .group_by('unit_id', 'block_index', 'condition_id')
            .agg(vec.mean('firing_rate').alias('block_mean'))
        )

        def fold_condition_means(
            fold_blocks: frozenset[int], fold_name: str
        ) -> pl.DataFrame:
            return (
                block_means
                .filter(pl.col('block_index').is_in(list(fold_blocks)))
                .group_by('unit_id', 'condition_id')
                .agg(vec.mean('block_mean').alias('condition_mean'))
                .pivot(on='condition_id', values='condition_mean')
                .drop_nulls(subset=['1', '2'])
                .select(
                    'unit_id',
                    pl.col('1').alias(f'condition_1_{fold_name}'),
                    pl.col('2').alias(f'condition_2_{fold_name}'),
                    pl.col('1').sub('2').alias(f'delta_{fold_name}'),
                )
            )

        joined = (
            fold_condition_means(fold_1_blocks, 'fold_1')
            .join(
                fold_condition_means(fold_2_blocks, 'fold_2'),
                on='unit_id',
                how='inner',
            )
            .sort('unit_id')
        )
        if joined.height == 0:
            n_omitted += 1
            continue

        delta_fold_1 = np.asarray(joined['delta_fold_1'].to_list(), dtype=np.float64)
        delta_fold_2 = np.asarray(joined['delta_fold_2'].to_list(), dtype=np.float64)
        if delta_fold_1.ndim != 2 or delta_fold_1.shape != delta_fold_2.shape:
            raise ValueError(f"invalid fold arrays for session {session}")

        time_s = time_bin_centers(
            n_timepoints=delta_fold_1.shape[1],
            bin_size_s=bin_size_s,
            pre_s=float(psth_params['pre']),
        )
        baseline_mask = window_mask(
            time_s, params.baseline_start_s, params.baseline_end_s
        )
        stimulus_mask = window_mask(
            time_s,
            params.projection_stimulus_start_s,
            params.projection_stimulus_end_s,
        )
        metrics = compute_trajectory_metrics(
            delta_fold_1, delta_fold_2, baseline_mask, stimulus_mask
        )

        # Estimate one operating-rate scale per unit from the qualifying trials in
        # all six blocks. Trial means are formed within block above, so averaging
        # these baseline rates gives every block equal weight and pools conditions.
        block_baseline_rates_by_unit: dict[object, list[float]] = {}
        for unit_id, block_mean in block_means.select(
            'unit_id', 'block_mean'
        ).iter_rows():
            block_trace = np.asarray(block_mean, dtype=np.float64)
            if block_trace.ndim != 1 or block_trace.size != baseline_mask.size:
                raise ValueError(
                    f"invalid block mean for session {session}, unit {unit_id}"
                )
            block_baseline_rates_by_unit.setdefault(unit_id, []).append(
                float(block_trace[baseline_mask].mean())
            )
        baseline_rate_hz = np.asarray(
            [
                np.mean(block_baseline_rates_by_unit[unit_id])
                for unit_id in joined['unit_id'].to_list()
            ],
            dtype=np.float64,
        )
        normalized_delta_fold_1, normalized_delta_fold_2, _ = (
            normalize_by_baseline_rate(
                delta_fold_1,
                delta_fold_2,
                baseline_rate_hz,
                params.baseline_rate_floor_hz,
            )
        )
        normalized_metrics = compute_trajectory_metrics(
            normalized_delta_fold_1,
            normalized_delta_fold_2,
            baseline_mask,
            stimulus_mask,
        )
        condition_projections = compute_condition_projections(
            np.asarray(joined['condition_1_fold_1'].to_list(), dtype=np.float64),
            np.asarray(joined['condition_2_fold_1'].to_list(), dtype=np.float64),
            np.asarray(joined['condition_1_fold_2'].to_list(), dtype=np.float64),
            np.asarray(joined['condition_2_fold_2'].to_list(), dtype=np.float64),
            baseline_mask,
        )
        n_units = joined.height

        results.append(
            pl.DataFrame(
                {
                    'session_id': [session],
                    'traj_separation': [
                        signed_rms(metrics.baseline_corrected_d2, n_units).tolist()
                    ],
                    'traj_separation_rate_normalized': [
                        signed_rms(
                            normalized_metrics.baseline_corrected_d2, n_units
                        ).tolist()
                    ],
                    'traj_separation_raw': [
                        signed_rms(metrics.raw_d2, n_units).tolist()
                    ],
                    'baseline_separation': [
                        signed_rms(metrics.baseline_d2, n_units)
                    ],
                    'median_baseline_rate_hz': [
                        float(np.median(baseline_rate_hz))
                    ],
                    'n_units_below_rate_floor': [
                        int(np.sum(baseline_rate_hz < params.baseline_rate_floor_hz))
                    ],
                    'baseline_axis_projection': [
                        metrics.baseline_axis_projection.tolist()
                    ],
                    'baseline_axis_projection_condition_1': [
                        condition_projections.condition_1.tolist()
                    ],
                    'baseline_axis_projection_condition_2': [
                        condition_projections.condition_2.tolist()
                    ],
                    'orthogonal_stimulus_projection': [
                        metrics.orthogonal_stimulus_projection.tolist()
                    ],
                    'baseline_axis_alignment': [metrics.baseline_axis_alignment],
                    'orthogonal_axis_alignment': [metrics.orthogonal_axis_alignment],
                    'n_units': [n_units],
                    'fold_1_blocks': [sorted(fold_1_blocks)],
                    'fold_2_blocks': [sorted(fold_2_blocks)],
                },
                schema=empty_schema,
            )
        )

    if n_omitted:
        print(
            f"\n  omitted {n_omitted}/{len(session_list)} sessions; each condition "
            "pair requires at least "
            f"{params.min_units_per_session_area} area units and qualifying trials "
            "in all six consecutive alternating blocks"
        )
    return pl.concat(results, how='diagonal') if results else pl.DataFrame(schema=empty_schema)


def write_trajectory_separation_for_area(
    area: str, params: Params, psth_params: dict, trials: pl.DataFrame
):
    psth_dir = PSTH_DIR / params.input_dir_name
    psth_path = psth_dir / f"{area}.parquet"
    area_traj_directory = NEURAL_TRAJ_DIR / params.output_dir_name

    area_psths = pl.read_parquet(psth_path.as_posix())

    ### identify trials columns that are missing from psths df and must be added
    cols_to_add = (set(condition_cols) | {'block_index'}) - set(area_psths.columns)
    cols_to_add = list(cols_to_add) + ['session_id', 'trial_index']

    ### join with trials
    area_psths = (
        area_psths
        .join(trials.select(cols_to_add), on=['session_id', 'trial_index'])
    )
    
    def get_parquet_path(condition_id) -> upath.UPath:
        return area_traj_directory / f"{area}_{condition_id}.parquet"

    for icond, cond_pair in enumerate(conditions_to_compare):
        if (path := get_parquet_path(icond)).exists() and params.skip_existing:
            print(f"\nSkipping: {path.as_posix()} already exists.")
            continue

        condition_1, condition_2 = cond_pair
        traj_df = compute_trajectory_separation_for_condition_pair(
            area_psths, condition_1, condition_2, params, psth_params
        )
        traj_df = traj_df.with_columns(pl.lit(icond).alias('condition_pair_id'))
        print(f"\nWriting {path.as_posix()}")
        (
            traj_df
            .sort('session_id')
            .write_parquet(path.as_posix())  # one row per session (cross-validated estimate)
        )


if __name__ == "__main__":

    if len(sys.argv) == 1:
        params = Params(
            input_dir_name='2026-01-06',
            output_dir_name='test',
            skip_existing=False,
            areas_to_process=['MRN',],
        )
    else:
        params = Params()
    print(params)

    psth_root = PSTH_DIR / params.input_dir_name
    
    if not psth_root.with_suffix('.json').exists():
        raise FileNotFoundError(f"No valid PSTH parameter file found in {psth_root}")

    if not params.areas_to_process:
        areas = [d.stem for d in (psth_root).glob('*.parquet')]
    else:
        areas = params.areas_to_process

    if len(areas) == 0:
        raise FileNotFoundError(f"No valid PSTH areas found in {psth_root}")

    psth_params_json = json.loads((PSTH_DIR / f'{params.input_dir_name}.json').read_text())
    required_psth_params = {
        'align_to_col', 'pre', 'post', 'bin_size_s', 'as_binarized_array'
    }
    missing_psth_params = required_psth_params - psth_params_json.keys()
    if missing_psth_params:
        raise ValueError(f"PSTH sidecar is missing fields: {sorted(missing_psth_params)}")
    if (
        psth_params_json['align_to_col'] != 'stim_start_time'
        or not psth_params_json['as_binarized_array']
    ):
        raise ValueError(
            "analysis requires stimulus-aligned PSTHs stored as binarized spike arrays"
        )
    available_start_s = -float(psth_params_json['pre'])
    available_end_s = float(psth_params_json['post'])
    if not (
        available_start_s <= params.baseline_start_s
        < params.baseline_end_s <= 0
        <= params.projection_stimulus_start_s
        < params.projection_stimulus_end_s <= available_end_s
    ):
        raise ValueError(
            "baseline and projection stimulus windows must lie within the PSTH: "
            f"available=[{available_start_s}, {available_end_s}], "
            f"baseline=[{params.baseline_start_s}, {params.baseline_end_s}), "
            "projection_stimulus="
            f"[{params.projection_stimulus_start_s}, "
            f"{params.projection_stimulus_end_s})"
        )
    traj_params_json_path = NEURAL_TRAJ_DIR / f'{params.output_dir_name}.json'
    # if traj_params_json_path.exists() and params.output_dir_name != 'test':
    #     existing_params = json.loads(traj_params_json_path.read_text())
    #     current_params = psth_params_json | params.model_dump()
    #     existing_condition_id_map = existing_params.pop('integer_id_to_condition_mapping')
    #     current_condition_id_map = current_params.pop('integer_id_to_condition_mapping')
    #     if existing_params != current_params:
    #         raise ValueError(f"Params file already exists and does not match current params:\n{existing_params=}\n{current_params=}.\nDelete the data dir and params.json on S3 if you want to update parameters (or encode time in dir path)")
    #     for k, v in existing_condition_id_map.items():
    #         k = int(k) # keys must be stored as strings in json, but originally created as ints
    #         if k not in current_condition_id_map:
    #             raise LookupError(f"A previously-used condtion ({v!r}) is missing from the current integer-id mapping/list. Please restore previous mapping and append new conditions")
    #         if current_condition_id_map[k] != v:
    #             raise ValueError(f"Condition ID {k} was previously {v!r}, but has been changed to {current_condition_id_map[k]} - please restore previous value!")
    #         # otherwise, new conditions are ok
    # else:
    traj_params_json_path.write_text(json.dumps(psth_params_json | params.model_dump(), indent=4))
    
    # get filtered trials table
    # use table from future datacube version with grating phase info: 
    assert psth_params_json['intervals_table'] == 'trials' and psth_params_json['datacube_version'] == DATACUBE_VERSION
    trials = pl.read_parquet(f'/root/capsule/data/dynamicrouting_datacube_{DATACUBE_VERSION}/consolidated/trials.parquet')
    
    session_table = pl.read_parquet(f'/root/capsule/data/dynamicrouting_datacube_{DATACUBE_VERSION}/session_table.parquet')
    good_behavior_sessions = session_table.filter(pl.col('is_good_behavior'))['session_id'].to_list()

    # Get latest
    url = "https://raw.githubusercontent.com/allenneuraldynamics/dr-datacube/main/assets/datacube_sessions.csv"
    session_ids = pl.read_csv(url).filter(
        pl.col("is_behavior_pass") & (pl.col("session_type") == "brainwide")
        )["session_id"].to_list()

    sessions_to_analyze = (
        utils.get_df('session')
        .filter(
            #pl.col('keywords').list.contains('production'),
            ~pl.col('keywords').list.contains('templeton'),
            ~pl.col('keywords').list.contains('injection_perturbation'),
            ~pl.col('keywords').list.contains('injection_control'),
            ~pl.col('keywords').list.contains('opto_perturbation'),
            ~pl.col('keywords').list.contains('opto_control'),
            ~pl.col('keywords').list.contains('issues'),
            ~pl.col('keywords').list.contains('naive'),
            ~pl.col('keywords').list.contains('context_naive'),
            pl.col('session_id').is_in(good_behavior_sessions),
        )
    )
    trials = trials.filter(pl.col('session_id').is_in(sessions_to_analyze['session_id'].implode()))

    n_sessions_before_block_filter = trials['session_id'].n_unique()
    eligible_sessions = sessions_with_good_fold_coverage(
        trials, utils.get_df('performance'), params
    )
    trials = trials.filter(pl.col('session_id').is_in(eligible_sessions))
    print(
        "Good-block fold coverage: retained "
        f"{len(eligible_sessions)}/{n_sessions_before_block_filter} sessions"
    )

    if params.decoder_areas_to_average:
        decoder_area_cols = [f"{a}_predict_proba" for a in params.decoder_areas_to_average]

        decoding_df = (
            pl.read_parquet(decoding_parquet_path)
            .with_columns(pl.mean_horizontal(decoder_area_cols).alias('predict_proba'))
            .with_columns(
                predict_proba_quintile=pl.col('predict_proba').cut([0.2, 0.4, 0.6, 0.8], include_breaks=False)
            )
            .select('session_id', 'trial_index', 'predict_proba', 'predict_proba_quintile')
        )
        trials = (
            trials
            .join(decoding_df, on=['trial_index', 'session_id'], how='inner')
            .with_columns(
                is_decoder_confident=pl.col('predict_proba').sub(0.5).abs().gt(0.1),
                is_decoder_correct = ((pl.col('predict_proba')<0.5)&(pl.col('is_aud_rewarded'))) | ((pl.col('predict_proba')>0.5)&(pl.col('is_vis_rewarded'))),
                is_grating_phase_zero=pl.col('grating_phase').eq(0),
                is_grating_phase_half=pl.col('grating_phase').eq(0.5),
            ) 
            .with_columns(
                is_decoder_incorrect=~pl.col('is_decoder_correct'),
            )
        )


    for i, area in enumerate(areas):
        print(f"{i+1}/{len(areas)} | {area}")
        psth_path = psth_root / f"{area}.parquet"
        if not psth_path.exists():
            raise FileNotFoundError(f"PSTH path does not exist: {psth_path}")
        if (m := params.min_units_across_sessions):
            n_units = (
                pl.scan_parquet(psth_path.as_posix())
                .select('session_id', 'unit_id')
                .unique('unit_id')
                .collect()
                .join(trials.select('session_id'), on='session_id', how='inner')
                .select(pl.col('unit_id').n_unique())
                .item()
            )
            if n_units < m:
                print(f"Skipping {area}: {n_units} units found across sessions (min required is {m})")
                continue
        t0 = time.time()
        write_trajectory_separation_for_area(area, params, psth_params_json, trials)
        print(f"Finished {area} in {time.time() - t0:.1f} s")

    print(f"\nAll finished")
