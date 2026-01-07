import itertools
import json
from typing import Iterable

import lazynwb
import polars as pl
import polars_ds as pds
import polars_vec_ops as vec
import numpy as np
import pydantic_settings
import pydantic
import tqdm
import upath

import utils

PSTH_DIR = upath.UPath('s3://aind-scratch-data/dynamic-routing/psths')
NEURAL_TRAJ_DIR = upath.UPath('s3://aind-scratch-data/dynamic-routing/neural_trajectory_separation_all_conditions')
decoding_parquet_path = '/root/capsule/data/all_trials_with_predict_proba.parquet'

conditions_to_compare = (

    # aud targets
    (('is_aud_target', 'is_aud_rewarded', 'is_hit'), ('is_aud_target', 'is_vis_rewarded', 'is_correct_reject')),
    (('is_aud_target', 'is_aud_rewarded', 'is_hit'), ('is_aud_target', 'is_vis_rewarded', 'is_false_alarm')),
    (('is_aud_target', 'is_aud_rewarded', 'is_hit', 'is_decoder_correct', 'is_decoder_confident'), ('is_aud_target', 'is_vis_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident')),


    # vis targets
    (('is_vis_target', 'is_vis_rewarded', 'is_hit'), ('is_vis_target', 'is_aud_rewarded', 'is_correct_reject')),
    (('is_vis_target', 'is_vis_rewarded', 'is_hit'), ('is_vis_target', 'is_aud_rewarded', 'is_false_alarm')),
    (('is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_decoder_correct', 'is_decoder_confident'), ('is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident')),

    (('is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_grating_phase_half'), ('is_vis_target', 'is_aud_rewarded', 'is_correct_reject', 'is_grating_phase_half')),
    (('is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_grating_phase_half'), ('is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_grating_phase_half')),
    (('is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_decoder_correct', 'is_decoder_confident', 'is_grating_phase_half'), ('is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident', 'is_grating_phase_half')),

    (('is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_grating_phase_zero'), ('is_vis_target', 'is_aud_rewarded', 'is_correct_reject', 'is_grating_phase_zero')),
    (('is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_grating_phase_zero'), ('is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_grating_phase_zero')),
    (('is_vis_target', 'is_vis_rewarded', 'is_hit', 'is_decoder_correct', 'is_decoder_confident', 'is_grating_phase_zero'), ('is_vis_target', 'is_aud_rewarded', 'is_false_alarm', 'is_decoder_correct', 'is_decoder_confident', 'is_grating_phase_zero')),


    # aud nontargets
    (('is_aud_nontarget', 'is_aud_rewarded', 'is_correct_reject'), ('is_aud_nontarget', 'is_vis_rewarded', 'is_correct_reject')),


    # vis nontargets
    (('is_vis_nontarget', 'is_grating_phase_zero', 'is_vis_rewarded', 'is_correct_reject'), ('is_vis_nontarget', 'is_grating_phase_zero', 'is_aud_rewarded', 'is_correct_reject')),
    (('is_vis_nontarget', 'is_grating_phase_half', 'is_vis_rewarded', 'is_correct_reject'), ('is_vis_nontarget', 'is_grating_phase_half', 'is_aud_rewarded', 'is_correct_reject')),    
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
    include_only_good_blocks: bool = True
    good_block_dprime_threshold: float = 1.0
    include_good_blocks_in_bad_sessions: bool = False
    min_units_across_sessions: int = pydantic.Field(500, exclude=True)
    n_null_iterations: int = 100
    integer_id_to_condition_mapping: dict[int, tuple[tuple[str,...], tuple[str,...]]] = pydantic.Field(default_factory=lambda: integer_to_condition)
    # n_resample_iterations: int = 100

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

units = utils.get_df('units')


def get_condition_id(integer_id_to_condition: dict[int, list[str] | list[list[str]]], search_input: list[str] | list[list[str]]) -> int:
    """Get ID for a given set of col names representing a condition filter, or for a list of such
    sets. Order of col names within a set does not matter.

    >>> integer_id_to_condition = json.loads(upath.UPath('s3://aind-scratch-data/dynamic-routing/psths/2025-12-18_10ms_good-blocks_good-sessions.json').read_text())['integer_id_to_condition_mapping']
    >>> get_condition_id(integer_id_to_condition, ['is_vis_target', 'is_vis_rewarded', 'is_hit'])
    7
    >>> get_condition_id(integer_id_to_condition, ['is_vis_target', 'is_vis_rewarded', 'is_false_alarm'])
    LookupError: Condition matching [['is_vis_target', 'is_vis_rewarded', 'is_false_alarm']] not found in mapping.
    
    """
    if isinstance(search_input[0], str):
        search_input = [list(search_input)]
    is_null_condition = len(search_input) == 2
    if not is_null_condition:
        for integer_id, condition in integer_id_to_condition.items():
            if isinstance(condition[0], list):
                continue
            if set(condition) == set(search_input[0]):
                return int(integer_id)
        else:
            raise LookupError(f'Condition matching {search_input!r} not found in mapping.')
    else:
        if set(search_input[0]).issubset(set(search_input[1])) or set(search_input[1]).issubset(set(search_input[0])):
            raise ValueError(f"Skipping null condition {search_input} because one condition is a subset of the other and nulls cannot be computed.")

        for integer_id, null_condition_pair in integer_id_to_condition.items():
            if len(null_condition_pair) != 2:
                continue
            if all(
                set(null_condition) == set(search_inner)
                for null_condition, search_inner in zip(null_condition_pair, search_input)
            ):
                return int(integer_id)
        else:
            raise LookupError(f'Condition matching {search_input!r} not found in mapping.')


def compute_trajectory_separation_for_condition_pair(area_psth_df, condition_1, condition_2):

    if isinstance(condition_1[0], str):
        condition_1 = [pl.col(c) for c in condition_1]
    if isinstance(condition_2[0], str):
        condition_2 = [pl.col(c) for c in condition_2]

    conv_kernel_length = int(params.conv_kernel_s * 1000)  # in ms

    trajs = []
    null_trajs = []
    session_list = area_psth_df['session_id'].unique().sort()
    for isess, session in enumerate(session_list):
        print(f"\rIteration: {isess} of {len(session_list)}", end="", flush=True)    
        session_df = area_psth_df.filter(pl.col('session_id')==session)
        binned = (
            session_df
            .with_columns([
                pl.when(condition_1).then(pl.lit(1))
                .when(condition_2).then(pl.lit(2))
                .otherwise(pl.lit(None))
                .alias('condition_id')
            ])
            .drop_nulls(subset=['binarized_spike_times', 'condition_id'])
            .with_columns(pl.col('binarized_spike_times').cast(pl.List(pl.Float32)))
            .select('unit_id', 'condition_id', 'binarized_spike_times', 'trial_index', 'session_id')
            .explode('binarized_spike_times')
            .group_by('unit_id', 'trial_index', 'condition_id', 'session_id', maintain_order=True)
            .agg(
                pds.convolve(
                    x='binarized_spike_times',
                    kernel=np.ones(conv_kernel_length) * (1/conv_kernel_length),
                    mode="same",
                    method="direct"
                ).alias("binarized_spike_times_convolved")
            )
        ).sort(by=['unit_id', 'trial_index'])

        if len(binned) == 0:
            continue

        if binned['condition_id'].n_unique() < 2:
            continue

        traj = (
            binned
            .group_by('unit_id', 'condition_id', 'session_id')
            .agg(vec.mean('binarized_spike_times_convolved'))
            .pivot(on='condition_id', values='binarized_spike_times_convolved')
            .drop_nulls()
            .with_columns(
                pl.col(str(1)).sub(str(2)).list.eval(pl.element().pow(2)).alias('diff^2')
            )
            .group_by('session_id')
            .agg(
                pl.all(),
                # pl.lit(f"{condition_id_1}_vs_{condition_id_2}").alias('description'),
                vec.sum('diff^2').list.eval(pl.element().sqrt()).truediv(pl.col('unit_id').count().sqrt()).cast(pl.List(pl.Float64)).alias('traj_separation'),
                # ^ cast ensures compat with any list[null] 
            )
            .drop('diff^2', '1', '2')
        )

        trajs.append(traj)

        ### NULL TRAJECTORIES ###
        starting_seed = isess * params.n_null_iterations
        n_null_iterations = params.n_null_iterations
        for null_iteration in range(n_null_iterations):
            null_traj = (
                binned
                .group_by('unit_id')
                .agg(
                    pl.all()
                )
                .with_columns(
                    pl.col('condition_id').list.sample(n=pl.col('condition_id').list.len(), shuffle=True, with_replacement=False,seed=starting_seed+null_iteration),)
                .explode(
                    'condition_id', 'session_id', 'binarized_spike_times_convolved', 'trial_index'
                )
                .group_by('unit_id', 'condition_id', 'session_id')
                .agg(vec.mean('binarized_spike_times_convolved'))
                .pivot(on='condition_id', values='binarized_spike_times_convolved')
                .drop_nulls()
                .with_columns(
                    pl.col(str(1)).sub(str(2)).list.eval(pl.element().pow(2)).alias('diff^2')
                )
                .group_by('session_id')
                .agg(
                    pl.all().exclude('diff^2', '1', '2'),
                    vec.sum('diff^2').list.eval(pl.element().sqrt()).truediv(pl.col('unit_id').count().sqrt()).cast(pl.List(pl.Float64)).alias('traj_separation'),
                    # ^ cast ensures compat with any list[null] 
                )
                .drop('diff^2', '1', '2', strict=False)
            )
            null_trajs.extend([null_traj.with_columns(pl.lit(null_iteration).alias('null_iteration'))])
    
    return pl.concat(trajs + null_trajs, how='diagonal')


def write_trajectory_separation_for_area(area: str, params: Params, trials: pl.DataFrame):
    psth_dir = PSTH_DIR / params.input_dir_name
    psth_path = psth_dir / f"{area}.parquet"
    params_path = PSTH_DIR / f"{params.input_dir_name}.json"
    area_traj_directory = NEURAL_TRAJ_DIR / params.output_dir_name / area

    area_psths = pl.read_parquet(psth_path.as_posix())

    ### identify trials columns that are missing from psths df and must be added
    cols_to_add = set(condition_cols) - set(area_psths.columns)
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
            print(f"Skipping stim {stim} with condition id {icond} because file already exists.")
            continue

        condition_1, condition_2 = cond_pair
        traj_df = compute_trajectory_separation_for_condition_pair(area_psths, condition_1, condition_2)
        traj_df = traj_df.with_columns(pl.lit(icond).alias('condition_pair_id'))
        print(f"Writing {path.as_posix()}")
        traj_df.write_parquet(path.as_posix())


def sessionwise_trajectory_distances(lf: pl.LazyFrame, condition_id_1: int, condition_id_2: int, group_by: str | Iterable[str] | None = None, streaming: bool = True) -> pl.DataFrame:
    if isinstance(lf, pl.DataFrame):
        streaming = False
    lf = lf.lazy()
    if group_by is None:
        group_by = []
    elif isinstance(group_by, str):
        group_by = [group_by]
    group_by = tuple(group_by)
    df = (
            lf
            # .filter(pl.col('condition_id').is_in([condition_id_1, condition_id_2]))
            .filter((pl.col('condition_id')==condition_id_1) | (pl.col('condition_id')==condition_id_2))
            .group_by('unit_id', 'condition_id', *group_by)
            .agg(vec.mean('psth'))
            .collect(engine='streaming' if streaming else 'auto')
        )
    
    if df['condition_id'].n_unique() < 2:
        raise ValueError(f"Not enough unique condition_ids found in data for {condition_id_1} vs {condition_id_2}")

    return (
        df
        .pivot(on='condition_id', values='psth')
        .drop_nulls()
        .with_columns(
            pl.col(str(condition_id_1)).sub(str(condition_id_2)).list.eval(pl.element().pow(2)).alias('diff^2')
        )
        .group_by(*group_by or ['unit_id'])
        .agg(
            pl.all(),
            pl.lit(f"{condition_id_1}_vs_{condition_id_2}").alias('description'),
            vec.sum('diff^2').list.eval(pl.element().sqrt()).truediv(pl.col('unit_id').count().sqrt()).cast(pl.List(pl.Float64)).alias('traj_separation'),
            # ^ cast ensures compat with any list[null] 
        )
        .drop('diff^2', str(condition_id_1), str(condition_id_2))
    )

def sessionwise_null_trajectory_distances(lf: pl.LazyFrame, null_condition_id:int, group_by: str | Iterable[str] | None = None, streaming: bool = True) -> pl.DataFrame:
    if isinstance(lf, pl.DataFrame):
        streaming = False
    lf = lf.lazy()
    if group_by is None:
        group_by = []
    elif isinstance(group_by, str):
        group_by = [group_by]
    group_by = tuple(group_by)
    return (
        lf.lazy()
        .filter(pl.col('null_condition_pair_id')==null_condition_id)
        .group_by('unit_id', 'null_condition_index', *group_by)
        .agg(vec.mean('psth')) # should only be one psth)
        .collect(engine='streaming' if streaming else 'auto')
        .pivot(on='null_condition_index', values='psth')
        .drop_nulls()
        .with_columns(
            pl.col(str(1)).sub(str(2)).list.eval(pl.element().pow(2)).alias('diff^2')
        )
        .group_by(*group_by or ['unit_id'])
        .agg(
            pl.all(),
            vec.sum('diff^2').list.eval(pl.element().sqrt()).truediv(pl.col('unit_id').count().sqrt()).cast(pl.List(pl.Float64)).alias('traj_separation'),
            # ^ cast ensures compat with any list[null] 
        )
        .with_columns(null_condition_pair_id=pl.lit(null_condition_id))
        .drop('diff^2', str(1), str(2))
    )

def write_trajectories_for_area_old(area: str, params: Params):
    psth_dir = PSTH_DIR / params.name
    psth_path = psth_dir / f"{area}.parquet"
    params_path = PSTH_DIR / f"{params.name}.json"
    area_traj_directory = NEURAL_TRAJ_DIR / params.name / area
    area_lf = None

    integer_id_to_condition = json.loads(params_path.read_text())['integer_id_to_condition_mapping']

    def get_parquet_path(condition_id) -> upath.UPath:
        return area_traj_directory / f"{area}_null_pair_id_{condition_id}.parquet"

    unique_stims = ['is_vis_target', 'is_aud_target', 'is_vis_nontarget', 'is_aud_nontarget']
    for stim in unique_stims:
        
        null_stim_conditions = [cond for cond in integer_id_to_condition.values() if isinstance(cond[0], list) and stim in cond[0]]

        for null_stim_condition_combo in null_stim_conditions:

            #grab condition ids
            try:
                null_condition_id = get_condition_id(integer_id_to_condition, null_stim_condition_combo)
            except ValueError as e:
                print(f"{e!r}")
                continue

            stim_condition_ids = [get_condition_id(integer_id_to_condition, null_stim_condition) for null_stim_condition in null_stim_condition_combo]

            if (path := get_parquet_path(null_condition_id)).exists() and params.skip_existing:
                print(f"Skipping stim {stim} with null condition {null_stim_condition_combo} because file already exists.")
                continue

            #make trajectories
            if area_lf is None:
                area_lf = pl.scan_parquet(psth_path.as_posix())

            try:
                traj = sessionwise_trajectory_distances(area_lf, condition_id_1=stim_condition_ids[0], condition_id_2=stim_condition_ids[1], group_by='session_id')
                traj = traj.with_columns(pl.lit(null_condition_id).alias('null_condition_pair_id'))
                null_traj = sessionwise_null_trajectory_distances(area_lf, null_condition_id=null_condition_id, group_by=['session_id', 'null_iteration'])
                
            except ValueError as e:
                print(f"Skipping stim {stim} with null condition {null_stim_condition_combo} due to error: {e}")
                continue

            else:
                combined = pl.concat([traj, null_traj], how='diagonal').sort(['null_iteration', 'session_id'])
                combined.write_parquet((area_traj_directory / f"{area}_null_pair_id_{null_condition_id}.parquet").as_posix())


def write_neural_trajectories(psth_dir: upath.UPath, params: Params) -> None:
    root_dir = NEURAL_TRAJ_DIR / psth_dir.name

    # write full set of trajectory separation data for each area
    for psth_path in psth_dir.glob('*.parquet'):
        area = psth_path.stem
        all_traj_sep_path = root_dir / f"{area}.parquet"
        if params.skip_existing and all_traj_sep_path.exists():
            print(f'Skipping {area}: parquet already on S3')
            continue

        lf = pl.scan_parquet(psth_path.as_posix())

        def resample_units(lf: pl.LazyFrame, seed: int) -> pl.LazyFrame:
            return (
                lf
                .sort('unit_id') # sorting and maintaining order critical to ensure same unit sample for each context
                .group_by('session_id', 'context_state', maintain_order=True)
                .agg(pl.all().sample(fraction=1, with_replacement=True, seed=seed))
                .explode(pl.all().exclude('session_id', 'context_state'))
            )
        null_iter = pl.col('null_iteration').is_null()
        named_lfs = {
            'actual': lf.filter(null_iter),
            'null': lf.filter(~null_iter),
            'resampled units': lf.filter(null_iter),
        }

        vis_hit_expr = pl.col('is_vis_target') & pl.col('is_hit')
        aud_hit_expr = pl.col('is_aud_target') & pl.col('is_hit')
        vis_confident_false_alarm_expr =  pl.col('is_vis_target') & pl.col('is_false_alarm') & pl.col('predict_proba').is_in(["(-inf, 0.2]", "(0.2, 0.4]"])
        aud_confident_false_alarm_expr =  pl.col('is_aud_target') & pl.col('is_false_alarm') & pl.col('predict_proba').is_in(["(0.6, 0.8]", "(0.8, inf]"])

        name_df_context_pair: list[tuple[str, pl.DataFrame, tuple[str, str]]] = []
        for name, named_lf in named_lfs.items():
            for label_1, context_1, label_2, context_2 in [
                ('vis_hit', vis_hit_expr, 'vis_fa', vis_confident_false_alarm_expr), 
                ('aud_hit', aud_hit_expr, 'aud_fa', aud_confident_false_alarm_expr),
            ]:
                print(f"Processing: {area} | {name} trajectories | {label_1} vs {label_2}")
                n = params.n_resample_iterations if name == 'resampled units' else 1
                if name == 'resampled units':
                    # fetch df to avoid reading 100 times
                    named_lf = named_lf.collect().lazy()
                named_lf = named_lf.with_columns(pl.when(context_1).then(pl.lit(label_1)).when(context_2).then(pl.lit(label_2)).alias('context_state'))
                for i in range(n):
                    if name == 'resampled units':
                        named_lf = named_lf.pipe(resample_units, seed=i)
                    df = sessionwise_trajectory_distances(named_lf, label_1=label_1, label_2=label_2, group_by=['session_id', 'null_iteration'], streaming=True)
                    name_df_context_pair.append((name, df, (label_1, label_2)))

#! TODO FIX FILTERING FOR NULL

        # calculate average null for each session:
        null_avgs = (
            pl.concat([df for name, df, _ in name_df_context_pair if name == 'null'])
            .group_by('session_id', 'description')
            .agg(
                vec.avg('traj_separation').alias('avg_null_traj_separation'),
            )
        )

        # store other dfs, with an additional null subtracted column
        dfs: list[pl.DataFrame] = []
        for name, df, _ in name_df_context_pair:
            if name == 'null':
                continue
            dfs.append(
                df
                .drop('null_iteration')
                .join(null_avgs, on=['session_id', 'description'], how='inner')
                .with_columns(
                    null_subtracted_traj_separation=pl.col('traj_separation') - pl.col('avg_null_traj_separation'),
                )
            )

        print(f"Writing {all_traj_sep_path.as_posix()}")
        (
            pl.concat(dfs)
            .with_columns(pl.lit(area).alias('area'))
        ).write_parquet(all_traj_sep_path.as_posix())


if __name__ == "__main__":

    params = Params()

    if params.input_dir_name == 'test':
        params = Params(
            name='2026-01-06',
            skip_existing=False,
            areas_to_process=['MRN',],
            n_null_iterations=10,
        )

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
    traj_params_json_path = NEURAL_TRAJ_DIR / f'{params.output_dir_name}.json'
    if traj_params_json_path.exists() and params.output_dir_name != 'test':
        existing_params = json.loads(traj_params_json_path.read_text())
        existing_condition_id_map = existing_params.pop('integer_id_to_condition_mapping')
        current_params = psth_params_json | params.model_dump().pop('integer_id_to_condition_mapping')
        if existing_params != current_params:
            raise ValueError(f"Params file already exists and does not match current params:\n{existing_params=}\n{current_params=}.\nDelete the data dir and params.json on S3 if you want to update parameters (or encode time in dir path)")
        for k, v in existing_condition_id_map.items():
            if k not in params.integer_id_to_condition_mapping:
                raise LookupError(f"A previously-used condtion ({v!r}) is missing from the current integer-id mapping/list. Please restore previous mapping and append new conditions")
            if params.integer_id_to_condition_mapping[k] != v:
                raise ValueError(f"Condition ID {k} was previously {v!r}, but has been changed to {params.integer_id_to_condition_mapping[k]} - please restore previous value!")
            # otherwise, new conditions are ok
    else:
        traj_params_json_path.write_text(json.dumps(psth_params_json | params.model_dump(), indent=4))
    
    # get filtered trials table
    # use table from future datacube version with grating phase info: 
    assert psth_params_json['intervals_table'] == 'trials' and psth_params_json['datacube_version'] == 'v0.0.274'
    trials = pl.read_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.274/consolidated/trials.parquet')
    
    session_table = pl.read_parquet('/root/capsule/data/dynamicrouting_datacube_v0.0.272/session_table.parquet')
    good_behavior_sessions = session_table.filter(pl.col('is_good_behavior'))['session_id'].to_list()
    sessions_to_analyze = (
        utils.get_df('session')
        .filter(pl.col('keywords').list.contains('production'),
            ~pl.col('keywords').list.contains('templeton'),
            ~pl.col('keywords').list.contains('injection_perturbation'),
            ~pl.col('keywords').list.contains('injection_control'),
            ~pl.col('keywords').list.contains('opto_perturbation'),
            ~pl.col('keywords').list.contains('opto_control'),
            ~pl.col('keywords').list.contains('issues'),
            ~pl.col('keywords').list.contains('naive'),
            ~pl.col('keywords').list.contains('context_naive'),
            pl.lit(True) if (params.include_good_blocks_in_bad_sessions and params.include_only_good_blocks) else pl.col('session_id').is_in(good_behavior_sessions),
        )
    )
    trials = trials.filter(pl.col('session_id').is_in(sessions_to_analyze['session_id'].implode()))

    if params.include_only_good_blocks:
        trials = (
            trials
            .join(
                (
                    utils.get_df('performance')
                    # .with_columns(pl.col('_nwb_path').str.split('/').list.get(-1).str.strip_suffix('.nwb').alias('session_id'))
                    .filter(
                        pl.col('cross_modality_dprime') >= params.good_block_dprime_threshold,
                        pl.col('n_contingent_rewards') >= 10,
                    )
                ),
                on=['session_id', 'block_index'], 
                how='semi',
            )
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
        write_trajectories_for_area(area, params)

    print(f"All finished")