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


class Params(pydantic_settings.BaseSettings):
    name: str = pydantic.Field(None, exclude=True)
    skip_existing: bool = pydantic.Field(True, exclude=True)
    areas: list[str] | None = pydantic.Field(None, exclude=True)
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
            .filter(pl.col('condition_id').is_in([condition_id_1, condition_id_2]))
            .group_by('unit_id', 'condition_id', *group_by)
            .agg(pl.col('psth').first()) # should only be one psth)
            .collect(engine='streaming' if streaming else 'auto')
        )
    
    if df['condition_id'].n_unique() < 2:
        raise ValueError(f"Not enough unique condition_ids found in data for {condition_id_1} vs {condition_id_2}")

    return (
        df
        .pivot(on='condition_id', values='psth')
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
        .agg(pl.col('psth').first()) # should only be one psth)
        .collect(engine='auto')
        .pivot(on='null_condition_index', values='psth')
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


def write_trajectories_for_area(area: str, params: Params):
    psth_dir = PSTH_DIR / params.name / area
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
                area_lf = pl.scan_parquet(psth_dir.as_posix() + '/')

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
    # if params.name:
    #     psth_dirs = [PSTH_DIR / params.name]
    #     if not psth_dirs[0].exists():
    #         raise FileNotFoundError(f"PSTH directory does not exist: {psth_dirs[0]}")
    # else:
    #     psth_dirs = list(d for d in PSTH_DIR.glob('*') if d.is_dir() if d.with_suffix('.json').exists())
    #     if not psth_dirs:
    #         raise FileNotFoundError(f"No valid PSTH directories found in {PSTH_DIR}")

    psth_root = PSTH_DIR / params.name
    
    if not psth_root.with_suffix('.json').exists():
        raise FileNotFoundError(f"No valid PSTH parameter file found in {psth_root}")

    if params.areas:
        areas = params.areas
    else:
        areas = [d.stem for d in (psth_root).glob('*') if d.is_dir()]

    if len(areas) == 0:
        raise FileNotFoundError(f"No valid PSTH areas found in {psth_root}")


    psth_params_json = json.loads((PSTH_DIR / f'{params.name}.json').read_text())
    traj_params_json_path = NEURAL_TRAJ_DIR / f'{params.name}.json'
    if traj_params_json_path.exists():
        existing_params = json.loads(traj_params_json_path.read_text())
        if existing_params != psth_params | params.model_dump():
            raise ValueError(f"Params file already exists and does not match current params:\n{existing_params=}\n{params.model_dump()=}.\nDelete the data dir and params.json on S3 if you want to update parameters (or encode time in dir path)")
    else:
        traj_params_json_path.write_text(json.dumps(psth_params_json | params.model_dump(), indent=4))
    
    for i, area in enumerate(areas):
        print(f"{i+1}/{len(areas)} | Processing PSTHs in {area}")
        psth_dir = psth_root / area
        if not psth_dir.exists():
            raise FileNotFoundError(f"PSTH directory does not exist: {psth_dir}")
        write_trajectories_for_area(area, params)

    print(f"All finished")