"""Paired catch–target and context-separation T25 from a trajectory output run.

Run with --help for CLI usage. No additional smoothing is applied to the saved
separation traces. Session cohorts and unit identities are matched before mouse
averaging; all comparisons share one union-mouse bootstrap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.interpolate import PPoly
import upath

from latency_full_dataset import draw_mouse_counts, interval
from latency_spline_analysis import build_polynomials, measure_polynomial, NUMERIC_FIELDS
from trajectory_metrics import time_bin_centers


def match_sessions(stimulus: pl.DataFrame, reference: pl.DataFrame, time: np.ndarray,
                   *, key: str) -> tuple[list[dict], list[dict]]:
    """Require identical unit populations and complete finite traces in each pair."""
    required = {'session_id', 'unit_ids', 'n_units', 'traj_separation'}
    for label, frame in [('stimulus', stimulus), ('context', reference)]:
        if required - set(frame.columns):
            raise ValueError(f'{key}: {label} output lacks {sorted(required - set(frame.columns))}; '
                             'recompute with the updated trajectory pipeline')
        if frame['session_id'].n_unique() != frame.height:
            raise ValueError(f'{key}: duplicate session rows in {label} output')
    stim = {r['session_id']: r for r in stimulus.iter_rows(named=True)}
    ref = {r['session_id']: r for r in reference.iter_rows(named=True)}
    records, audit = [], []
    for session in sorted(stim.keys() | ref.keys()):
        reason = 'included'
        a, b = stim.get(session), ref.get(session)
        if a is None or b is None:
            reason = 'missing_stimulus' if a is None else 'missing_context'
        elif (not a['unit_ids'] or not b['unit_ids']
              or len(set(a['unit_ids'])) != a['n_units']
              or len(set(b['unit_ids'])) != b['n_units']
              or set(a['unit_ids']) != set(b['unit_ids'])):
            reason = 'unit_population_mismatch'
        else:
            curves = [np.asarray(r['traj_separation'], dtype=float) for r in [a, b]]
            if any(y.shape != time.shape or not np.isfinite(y).all() for y in curves):
                reason = 'invalid_trace'
        mouse = str(session).split('_', 1)[0]
        audit.append(dict(pair=key, session_id=session, mouse_id=mouse, status=reason))
        if reason == 'included':
            records.append(dict(session_id=session, mouse_id=mouse,
                                stimulus=curves[0], context=curves[1]))
    traces = []
    if records:
        mice = sorted({r['mouse_id'] for r in records})
        for signal in ['stimulus', 'context']:
            traces.append(dict(key=f'{key}_{signal}', pair=key, signal=signal, time=time,
                               mouse_ids=mice, n_sessions=len(records),
                               session_ids=[r['session_id'] for r in records],
                               mice=np.stack([np.mean([r[signal] for r in records if r['mouse_id'] == m], axis=0)
                                              for m in mice])))
    return traces, audit


def bootstrap_latencies(traces: list[dict], *, n_bootstrap=10000, seed=20261118):
    """Fit resampled mean traces, retaining censored estimates and failed draws."""
    if not traces:
        raise ValueError('No matched sessions available for latency analysis')
    ids, counts = draw_mouse_counts([m for d in traces for m in d['mouse_ids']], n_bootstrap, seed)
    shape = (n_bootstrap, len(traces))
    result = {field: np.full(shape, np.nan) for field in NUMERIC_FIELDS}
    result.update(statuses=np.full(shape, 'not_fitted', dtype='U40'),
                  counts=counts, mouse_ids=np.array(ids), keys=np.array([d['key'] for d in traces]),
                  sampled_mice=np.zeros(shape, dtype=np.int64),
                  unique_mice=np.zeros(shape, dtype=np.int64), seed=np.array(seed))
    overview, fitted = [], []
    for column, d in enumerate(traces):
        knots, coeffs = build_polynomials(d['time'] * 1000, d['mice'])
        point_poly = PPoly.construct_fast(np.ascontiguousarray(coeffs.mean(axis=0)), knots, extrapolate=False)
        point = measure_polynomial(point_poly)
        selected = counts[:, [ids.index(m) for m in d['mouse_ids']]]
        sampled = selected.sum(axis=1)
        result['sampled_mice'][:, column] = sampled
        result['unique_mice'][:, column] = (selected > 0).sum(axis=1)
        weights = np.divide(selected, sampled[:, None], out=np.zeros_like(selected, dtype=float),
                            where=sampled[:, None] > 0)
        for start in range(0, n_bootstrap, 250):
            coefficients = (weights[start:start+250] @ coeffs.reshape(len(coeffs), -1)).reshape(-1, *coeffs.shape[1:])
            for offset, c in enumerate(coefficients):
                draw = start + offset
                if sampled[draw] == 0:
                    result['statuses'][draw, column] = 'no_sampled_mice'
                    continue
                poly = PPoly.construct_fast(np.ascontiguousarray(c), knots, extrapolate=False)
                measurement = measure_polynomial(poly)
                for field in NUMERIC_FIELDS:
                    result[field][draw, column] = measurement[field]
                result['statuses'][draw, column] = measurement['status']
        values = result['rise_ms'][:, column]
        ci = interval(values)
        overview.append(dict(key=d['key'], pair=d['pair'], signal=d['signal'],
                             n_mice=len(d['mouse_ids']), n_sessions=d['n_sessions'], **point,
                             ci_low_ms=ci[0], ci_high_ms=ci[1],
                             n_valid=int(np.isfinite(values).sum()), n_failed=int((~np.isfinite(values)).sum()),
                             n_left_censored=int((result['statuses'][:, column] == 'left_censored_at_zero').sum())))
        grid = np.linspace(-99.5, 149.5, 2491)
        fitted.append(pd.DataFrame(dict(key=d['key'], time_ms=grid, fitted_hz=point_poly(grid))))
    overview = pd.DataFrame(overview)
    delays, delay_draws = [], {}
    for pair in dict.fromkeys(d['pair'] for d in traces):
        a, b = [next(i for i, d in enumerate(traces) if d['pair'] == pair and d['signal'] == s)
                for s in ['stimulus', 'context']]
        if traces[a]['mouse_ids'] != traces[b]['mouse_ids'] or traces[a]['session_ids'] != traces[b]['session_ids']:
            raise ValueError(f'{pair}: stimulus and context cohorts must be identical')
        diff = result['rise_ms'][:, b] - result['rise_ms'][:, a]
        ci = interval(diff)
        delay_draws[pair] = diff
        delays.append(dict(pair=pair, n_mice=len(traces[a]['mouse_ids']), n_sessions=traces[a]['n_sessions'],
                           delay_ms=overview.iloc[b].rise_ms - overview.iloc[a].rise_ms,
                           ci_low_ms=ci[0], ci_high_ms=ci[1],
                           n_valid=int(np.isfinite(diff).sum()), n_failed=int((~np.isfinite(diff)).sum())))
    return dict(overview=overview, delays=pd.DataFrame(delays),
                delay_draws=pd.DataFrame(delay_draws), fitted_curves=pd.concat(fitted, ignore_index=True)), result


def run(trajectory_dir, output_dir, populations, *, n_bootstrap=10000, seed=20261118):
    root, output = upath.UPath(trajectory_dir), Path(output_dir)
    manifest_path = root / 'catch_analysis.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['params']['comparison_set'] != 'all':
        raise ValueError('Paired latency requires comparison_set=all in a fresh trajectory output directory')
    psth = manifest['psth_params']
    if not np.isclose(psth['bin_size_s'], .001, rtol=0, atol=1e-12):
        raise ValueError('The existing spline T25 definition requires 1 ms PSTH bins')
    if psth['pre'] < .1 or psth['post'] < .15:
        raise ValueError('Spline T25 requires PSTHs covering at least -100 to +150 ms')
    n_timepoints = round((psth['pre'] + psth['post']) / psth['bin_size_s'])
    time = time_bin_centers(n_timepoints, psth['bin_size_s'], psth['pre'])
    traces, audits, sources, unavailable = [], [], {}, []
    for population in populations:
        for name, spec in manifest['comparisons'].items():
            key = f'{population}_{name}'
            stim_path = root / f'{population}_{name}.parquet'
            ref_path = root / f'{population}_{spec["reference_condition_id"]}.parquet'
            for path in [stim_path, ref_path]:
                if not path.exists():
                    raise FileNotFoundError(f'Missing trajectory output: {path}')
                if str(path) not in sources:
                    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            paired, audit = match_sessions(pl.read_parquet(str(stim_path)), pl.read_parquet(str(ref_path)), time, key=key)
            traces.extend(paired)
            audits.extend(audit)
            if not paired:
                unavailable.append(dict(pair=key, reason='no_matched_sessions'))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audits, columns=['pair', 'session_id', 'mouse_id', 'status']).to_csv(output / 'session_audit.csv', index=False)
    pd.DataFrame(unavailable, columns=['pair', 'reason']).to_csv(output / 'unavailable_comparisons.csv', index=False)
    tables, result = bootstrap_latencies(traces, n_bootstrap=n_bootstrap, seed=seed)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    np.savez_compressed(output / 'bootstrap.npz', **result)
    np.savez_compressed(output / 'inputs.npz', **{f'{d["key"]}_{field}': d[field]
                                                for d in traces for field in ['time', 'mice']})
    metadata = dict(source_manifest=manifest, source_sha256=sources, n_bootstrap=n_bootstrap, seed=seed,
                    traces=[{k: v for k, v in d.items() if k not in {'time', 'mice'}} for d in traces],
                    method='Backward peak-connected spline T25; lambda=1000 in milliseconds; '
                           'fit -99.5 to 149.5 ms; baseline -99.5 to -25 ms; peak 0 to 100 ms',
                    preprocessing='Saved baseline-corrected signed RMS separation; no additional smoothing',
                    sampling='Sessions averaged within mouse, equal mouse weights; one shared union-mouse bootstrap',
                    mouse_identity='Prefix before first underscore in datacube session_id',
                    delay='Context separation T25 minus catch–target separation T25',
                    intervals='Unadjusted percentile 95%, conditional on finite paired estimates',
                    boundary='Negative crossings retained in crossing_ms; rise_ms=0 and left_censored_at_zero. '
                             'Failed/empty draws retained as NaN with status.',
                    interpretation='Response development relative to each trace amplitude, not first biological onset')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    make_figure(tables, output)
    print(tables['delays'].to_string(index=False))
    return tables


def make_figure(tables, output):
    from matplotlib import pyplot as plt
    delays, overview = tables['delays'], tables['overview']
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, .4 * len(delays) + 1.5)), layout='constrained')
    for j, row in enumerate(delays.itertuples()):
        for signal, offset, color in [('stimulus', -.12, '#217f9c'), ('context', .12, '#c56a22')]:
            r = overview[(overview.pair == row.pair) & (overview.signal == signal)].iloc[0]
            axes[0].plot([r.ci_low_ms, r.ci_high_ms], [j+offset]*2, color=color)
            axes[0].plot(r.rise_ms, j+offset, 'o', color=color, label=signal if j == 0 else None)
        axes[1].plot([row.ci_low_ms, row.ci_high_ms], [j]*2, color='.3')
        axes[1].plot(row.delay_ms, j, 'o', color='.3')
    for ax in axes:
        ax.set(yticks=range(len(delays)), yticklabels=delays.pair.str.replace('_', ' '))
        ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_xlabel('Backward T25 (ms)')
    axes[0].legend(frameon=False)
    axes[1].set_xlabel('Context − catch–target T25 (ms)')
    axes[1].axvline(0, color='.5', ls=':')
    fig.suptitle('Matched sessions and units; paired mouse bootstrap\nUnadjusted 95% intervals')
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'latencies.{suffix}', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory-dir', required=True, help='Local/S3 folder containing catch_analysis.json and parquets')
    parser.add_argument('--output-dir', required=True, help='Local output directory')
    parser.add_argument('--populations', nargs='+', required=True, help='Area names, or a group name when pointing at its folder')
    parser.add_argument('--n-bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261118)
    run(**vars(parser.parse_args()))
