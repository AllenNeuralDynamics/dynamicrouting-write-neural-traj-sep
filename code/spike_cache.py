"""Build and reuse trial-aligned spike arrays in the shared PSTH cache."""
from __future__ import annotations

import ast
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, asdict
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen

import dr_datacube
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from upath import UPath

logger = logging.getLogger(__name__)
PSTH_DIR = UPath('s3://aind-scratch-data/dynamic-routing/psths')
RELEASE_SETTINGS_URL = (
    'https://raw.githubusercontent.com/AllenNeuralDynamics/'
    'dr-datacube/main/src/dr_datacube/settings.py'
)
SCHEMA = pa.schema([
    ('session_id', pa.string()), ('unit_id', pa.string()),
    ('trial_index', pa.int64()), ('binarized_spike_times', pa.list_(pa.bool_())),
    ('is_observed', pa.bool_()),
])
# Preserve the superior-colliculus populations in the existing area cache.
AREA_ALIASES = {**dict.fromkeys(('SCig', 'SCiw', 'SCdg', 'SCdw'), 'SCm'),
                **dict.fromkeys(('SCop', 'SCzo', 'SCsg'), 'SCs')}


def normalize_version(version: str) -> str:
    version = version if version.startswith('v') else f'v{version}'
    if not re.fullmatch(r'v\d+\.\d+\.\d+(?:[.-][A-Za-z0-9]+)*', version):
        raise ValueError(f'Invalid datacube version: {version!r}')
    return version


def latest_datacube_version() -> str:
    """Read the release promoted by dr-datacube, independently of our lockfile.

    dr-datacube currently publishes its release as DatacubeConfig.version rather
    than via a latest-release API. Parse that literal; never execute remote code
    or silently fall back to a potentially stale installed default.
    """
    with urlopen(RELEASE_SETTINGS_URL, timeout=30) as response:
        tree = ast.parse(response.read().decode('utf-8'))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'DatacubeConfig':
            for field in node.body:
                if (isinstance(field, ast.AnnAssign)
                        and isinstance(field.target, ast.Name)
                        and field.target.id == 'version'):
                    return normalize_version(ast.literal_eval(field.value))
    raise RuntimeError('Cannot resolve the published dr-datacube release; pin --datacube_version')


@dataclass(frozen=True)
class SpikeSettings:
    datacube_version: str
    pre: float = 0.5
    post: float = 0.5
    bin_size_s: float = 0.001
    intervals_table: str = 'trials'
    align_to_col: str = 'stim_start_time'
    default_qc_only: bool = True
    as_spike_count: bool = False
    as_binarized_array: bool = True
    spike_col: str = 'binarized_spike_times'

    def __post_init__(self):
        object.__setattr__(self, 'datacube_version', normalize_version(self.datacube_version))
        values = (self.pre, self.post, self.bin_size_s)
        if not all(np.isfinite(values)) or self.pre < 0 or self.post <= 0 or self.bin_size_s <= 0:
            raise ValueError('PSTH windows and bin size must be finite and valid')
        n = (self.pre + self.post) / self.bin_size_s
        if not np.isclose(n, round(n), rtol=0, atol=1e-8):
            raise ValueError('PSTH duration must be an integer number of bins')
        if (self.intervals_table != 'trials' or self.align_to_col != 'stim_start_time'
                or not self.as_binarized_array or self.as_spike_count
                or self.spike_col != 'binarized_spike_times'):
            raise ValueError('Only stimulus-aligned binary trial arrays are supported')

    @property
    def n_bins(self):
        return round((self.pre + self.post) / self.bin_size_s)

    @property
    def cache_name(self):
        return f'{self.datacube_version}_spikes_{digest(asdict(self))[:12]}'


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sidecar(root: UPath) -> UPath:
    # with_suffix would strip part of a version such as v0.0.301.
    return root.parent / f'{root.name}.json'


def read_settings(root: UPath) -> SpikeSettings:
    data = json.loads(sidecar(root).read_text())
    return SpikeSettings(**{key: data[key] for key in SpikeSettings.__dataclass_fields__ if key in data})


def binary_spikes(spikes, starts, settings: SpikeSettings) -> np.ndarray:
    """Binary occupancy in half-open bins [left, right), including silent trials."""
    spikes, starts = np.asarray(spikes, dtype=float), np.asarray(starts, dtype=float)
    if spikes.ndim != 1 or starts.ndim != 1 or not np.isfinite(spikes).all() or not np.isfinite(starts).all():
        raise ValueError('Spike and alignment times must be finite one-dimensional arrays')
    if np.any(np.diff(spikes) < 0):
        spikes = np.sort(spikes)
    edges = starts[:, None] + (np.arange(settings.n_bins + 1) * settings.bin_size_s - settings.pre)
    return np.diff(np.searchsorted(spikes, edges, side='left'), axis=1) > 0


def _publish(local: Path, destination: UPath):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # S3 publishes a PUT/multipart object only on successful close. Completion
    # metadata is published afterwards, so failed uploads are never reusable.
    with local.open('rb') as source, destination.open('wb') as target:
        shutil.copyfileobj(source, target)


def _write_json(path: UPath, value):
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / 'metadata.json'
        local.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        _publish(local, path)


def _fingerprint(settings, units, trials):
    return digest({
        'builder_version': 1, 'settings': asdict(settings),
        'units': units.select('session_id', 'unit_id', 'obs_intervals').sort('session_id', 'unit_id').rows(),
        'trials': trials.select('session_id', 'trial_index', 'stim_start_time')
        .sort('session_id', 'trial_index').rows(),
    })


def _complete(path: UPath, fingerprint: str) -> bool:
    marker = sidecar(path)
    if not path.exists() or not marker.exists():
        return False
    try:
        metadata = json.loads(marker.read_text())
        if metadata.get('fingerprint') != fingerprint:
            return False
        # Detect a truncated/replaced object without reading its spike arrays.
        with path.open('rb') as stream:
            parquet = pq.ParquetFile(stream)
            return (parquet.metadata.num_rows == metadata['n_rows']
                    and parquet.schema_arrow == SCHEMA)
    except (json.JSONDecodeError, KeyError, pa.ArrowInvalid):
        return False


def _write_session(path, units, trials, settings, fingerprint):
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / 'spikes.parquet'
        with pq.ParquetWriter(local, SCHEMA, compression='zstd') as writer:
            for unit in units.iter_rows(named=True):
                for batch in trials.iter_slices(128):
                    starts = batch['stim_start_time'].to_numpy()
                    values = binary_spikes(unit['spike_times'], starts, settings)
                    observed = observed_windows(unit['obs_intervals'], starts, settings)
                    arrays = pa.ListArray.from_arrays(
                        np.arange(len(batch) + 1, dtype=np.int32) * settings.n_bins,
                        pa.array(values.ravel()),
                        mask=pa.array(~observed),
                    )
                    writer.write_table(pa.Table.from_arrays([
                        pa.array([unit['session_id']] * len(batch)),
                        pa.array([unit['unit_id']] * len(batch)),
                        pa.array(batch['trial_index'].to_numpy(), type=pa.int64()), arrays, pa.array(observed),
                    ], schema=SCHEMA))
        _publish(local, path)
    _write_json(sidecar(path), {'fingerprint': fingerprint, 'n_rows': len(units) * len(trials)})


def _combine_area(path, shards, fingerprint):
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / 'area.parquet'
        n_rows = 0
        with pq.ParquetWriter(local, SCHEMA, compression='zstd') as writer:
            for shard in shards:
                with shard.open('rb') as stream:
                    for batch in pq.ParquetFile(stream).iter_batches(batch_size=1024):
                        writer.write_batch(batch)
                        n_rows += batch.num_rows
        _publish(local, path)
    _write_json(sidecar(path), {'fingerprint': fingerprint, 'n_rows': n_rows})


def observed_windows(intervals, starts, settings):
    """Unrecorded time must remain missing, rather than becoming zero firing."""
    if intervals is None:
        raise ValueError('Missing unit observation intervals')
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    if not np.isfinite(intervals).all() or np.any(intervals[:, 1] < intervals[:, 0]):
        raise ValueError('Invalid unit observation intervals')
    return np.any((starts[:, None] - settings.pre >= intervals[:, 0])
                  & (starts[:, None] + settings.post <= intervals[:, 1]), axis=1)


def load_source_tables():
    """Use one release for trials, performance, unit QC, and subsequent spikes.

    Caller holds dr_datacube.config.override. Its default published session
    selection preserves the existing behavior-passing brainwide cohort.
    """
    trials = dr_datacube.get_lf('trials').collect()
    performance = dr_datacube.get_lf('performance').collect()
    units = dr_datacube.get_lf('unit_metrics').select(
        'session_id', 'unit_id', 'structure', 'default_qc', 'obs_intervals',
    ).collect()
    return trials, performance, units


def _legacy_covers(root, areas, units, trials):
    """Check all unit/trial keys before adopting an older date-named cache."""
    for area in areas:
        path = root / f'{area}.parquet'
        if not path.exists():
            return False
        expected_units = units.filter(pl.col('structure') == area).select('session_id', 'unit_id')
        expected_trials = trials.select('session_id', 'trial_index')
        expected_rows = (expected_units.group_by('session_id').len(name='units')
                         .join(expected_trials.group_by('session_id').len(name='trials'), on='session_id')
                         .select((pl.col('units').cast(pl.Int64) * pl.col('trials')).sum()).item())
        with path.open('rb') as stream:
            parquet = pq.ParquetFile(stream)
            if parquet.metadata.num_rows < expected_rows:
                return False
            if 'binarized_spike_times' not in parquet.schema_arrow.names:
                return False
            keys = pl.from_arrow(parquet.read(columns=['session_id', 'unit_id', 'trial_index']))
        keys = (keys.join(expected_units, on=['session_id', 'unit_id'], how='semi')
                .join(expected_trials, on=['session_id', 'trial_index'], how='semi'))
        if keys.height != expected_rows or keys.is_duplicated().any():
            return False
    return True


@dataclass
class SessionJob:
    session_id: str
    units: pl.DataFrame
    trials: pl.DataFrame
    # Only unfinished area shards; spike arrays never pass through the parent.
    areas: list[tuple[str, str]]


def _build_session(job: SessionJob, settings, root_uri, storage_options, datacube_config):
    root = UPath(root_uri, **storage_options)
    # Spawned processes do not inherit the parent's ContextVar overrides.
    with dr_datacube.config.override(**datacube_config):
        logger.info('Binning %s: %d units, %d trials', job.session_id, len(job.units), len(job.trials))
        # dr-datacube uses spike-inclusive parquet where available, otherwise NWB.
        spikes = (dr_datacube.get_lf('units', session_id=job.session_id)
                  .filter(pl.col('unit_id').is_in(job.units['unit_id'].to_list()))
                  .select('session_id', 'unit_id', 'spike_times').collect())
        if (spikes['unit_id'].n_unique() != len(job.units) or len(spikes) != len(job.units)
                or spikes['spike_times'].null_count()):
            raise ValueError(f'Missing or duplicate spike trains in session {job.session_id}')
        for area, fingerprint in job.areas:
            area_units = job.units.filter(pl.col('structure') == area)
            selected = spikes.join(area_units.select('unit_id', 'obs_intervals'), on='unit_id', how='inner')
            shard = root / '_sessions' / job.session_id / f'{area}.parquet'
            _write_session(shard, selected, job.trials, settings, fingerprint)


def _initialize_session_worker():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


def _build_sessions(jobs, settings, root, session_workers):
    if not jobs:
        return
    # Public override() snapshots the active configuration, not the potentially
    # different module-level default, including storage/authentication options.
    with dr_datacube.config.override() as active_config:
        config = active_config.model_dump()
    args = (settings, str(root), dict(root.storage_options), config)
    workers = min(session_workers, len(jobs))
    logger.info('Building %d sessions with %d worker process(es)', len(jobs), workers)
    if workers == 1:
        for job in jobs:
            _build_session(job, *args)
        return

    # Forking an initialized Polars/S3 client can deadlock. Spawn also keeps
    # worker memory and filesystem clients independent on macOS and Linux.
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_initialize_session_worker)
    in_flight = {}
    remaining = iter(jobs)
    try:
        for _ in range(workers):
            job = next(remaining)
            in_flight[pool.submit(_build_session, job, *args)] = job.session_id
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            # Check every completed result before scheduling additional work.
            for future in done:
                session = in_flight.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f'Spike cache build failed for session {session}') from exc
                logger.info('Completed session %s', session)
            for _ in done:
                job = next(remaining, None)
                if job is not None:
                    in_flight[pool.submit(_build_session, job, *args)] = job.session_id
    finally:
        # Running sessions finish their checkpoints; queued work is cancelled.
        # Exceptions propagate before any area file is assembled or published.
        pool.shutdown(wait=True, cancel_futures=True)


def ensure_spike_cache(settings: SpikeSettings, trials: pl.DataFrame, units: pl.DataFrame,
                       *, cache_dir=PSTH_DIR, input_dir_name=None, areas=None,
                       unit_groups=None, session_workers: int = 1) -> UPath:
    """Ensure complete area files; checkpoints survive interrupted builds.

    Caller must hold the matching dr_datacube configuration while this runs.
    Existing explicitly named legacy caches remain read-only.
    Set session_workers > 1 for spawned processes; call from a guarded script
    entry point (if __name__ == '__main__') when using the Python API.
    """
    if isinstance(session_workers, bool) or not isinstance(session_workers, int) or session_workers < 1:
        raise ValueError('session_workers must be a positive integer')
    cache_dir = UPath(cache_dir)
    if input_dir_name and (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', input_dir_name)):
        raise ValueError('input_dir_name must be a single cache directory name')
    units = units.filter(pl.col('default_qc')) if settings.default_qc_only else units
    units = units.with_columns(pl.col('structure').replace(AREA_ALIASES))
    units = units.drop_nulls(['structure']).join(trials.select('session_id').unique(), on='session_id', how='semi')
    if units.select('unit_id').is_duplicated().any():
        raise ValueError('Duplicate unit IDs in source metrics')
    if trials.select('session_id', 'trial_index').is_duplicated().any():
        raise ValueError('Duplicate trial IDs in source table')
    requested = set(units['structure'].unique()) if areas is None else set(areas)
    group_ids = {u for ids in (unit_groups or {}).values() for u in ids}
    missing = group_ids - set(units['unit_id'])
    if missing:
        raise ValueError(f'Unit IDs absent from eligible source units: {sorted(missing)}')
    requested.update(units.filter(pl.col('unit_id').is_in(sorted(group_ids)))['structure'])
    unknown = requested - set(units['structure'])
    if unknown or not requested:
        raise ValueError(f'No eligible units for requested areas: {sorted(unknown or requested)}')
    for area in requested:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_. -]*', area):
            raise ValueError(f'Invalid area name: {area!r}')
    units = units.filter(pl.col('structure').is_in(sorted(requested)))
    trials = trials.join(units.select('session_id').unique(), on='session_id', how='semi').sort('session_id', 'trial_index')
    if trials['stim_start_time'].null_count() or not trials['stim_start_time'].is_finite().all():
        raise ValueError('Every trial, including catch trials, must have a finite stim_start_time')

    if not input_dir_name and cache_dir.exists():
        for metadata_path in sorted(cache_dir.glob('*.json')):
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('builder_version') or any(metadata.get(k) != v for k, v in asdict(settings).items()):
                continue
            candidate = cache_dir / metadata_path.name[:-5]
            if _legacy_covers(candidate, requested, units, trials):
                logger.info('Reusing matching legacy spike cache %s', candidate)
                return candidate
    root = cache_dir / (input_dir_name or settings.cache_name)
    if sidecar(root).exists() and read_settings(root) != settings:
        raise ValueError(f'PSTH settings mismatch at {root}; choose a new cache name')
    # A managed cache records its builder; old date-named caches are left intact.
    if sidecar(root).exists() and 'builder_version' not in json.loads(sidecar(root).read_text()):
        for area in requested:
            if not (root / f'{area}.parquet').exists():
                raise FileNotFoundError(f'Legacy cache is missing {area}; omit input_dir_name to build a new cache')
        logger.info('Using explicitly selected legacy cache %s', root)
        return root

    root.mkdir(parents=True, exist_ok=True)
    if not sidecar(root).exists():
        _write_json(sidecar(root), {**asdict(settings), 'builder_version': 1})
    pending = {}
    for area in sorted(requested):
        area_units = units.filter(pl.col('structure') == area)
        area_trials = trials.join(area_units.select('session_id').unique(), on='session_id', how='semi')
        fingerprint = _fingerprint(settings, area_units, area_trials)
        if not _complete(root / f'{area}.parquet', fingerprint):
            pending[area] = fingerprint
    if not pending:
        logger.info('Reusing complete spike cache %s', root)
        return root

    shards = {area: [] for area in pending}
    jobs = []
    pending_units = units.filter(pl.col('structure').is_in(list(pending)))
    for session_units in pending_units.sort('session_id', 'unit_id').partition_by('session_id', maintain_order=True):
        session = session_units['session_id'][0]
        session_trials = trials.filter(pl.col('session_id') == session)
        tasks = []
        for area_units in session_units.partition_by('structure'):
            area = area_units['structure'][0]
            shard = root / '_sessions' / session / f'{area}.parquet'
            shards[area].append(shard)
            fingerprint = _fingerprint(settings, area_units, session_trials)
            if not _complete(shard, fingerprint):
                tasks.append((area, fingerprint))
        if not tasks:
            continue
        job_units = session_units.filter(pl.col('structure').is_in([area for area, _ in tasks]))
        jobs.append(SessionJob(session, job_units, session_trials, tasks))
    _build_sessions(jobs, settings, root, session_workers)
    for area, fingerprint in pending.items():
        logger.info('Publishing %s/%s.parquet', root, area)
        _combine_area(root / f'{area}.parquet', shards[area], fingerprint)
    return root
