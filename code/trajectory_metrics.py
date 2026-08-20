"""Numerical helpers for cross-validated trajectory analyses.

The functions in this module operate on condition-difference matrices with shape
``(n_units, n_timepoints)``.  Keeping the numerical definitions independent of the
Polars/S3 pipeline makes the estimators easy to test on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Cross-validated distances and cross-fitted projection time courses."""

    raw_d2: FloatArray
    baseline_corrected_d2: FloatArray
    baseline_d2: float
    baseline_axis_projection: FloatArray
    orthogonal_stimulus_projection: FloatArray
    baseline_axis_alignment: float
    orthogonal_axis_alignment: float


def fixed_block_folds(
    block_indices: list[int], condition_ids: list[int]
) -> tuple[frozenset[int], frozenset[int]] | None:
    """Return the fixed ``XYX | YXY`` split for a complete six-block session."""
    if len(block_indices) != 6 or len(condition_ids) != 6:
        return None
    if len(set(block_indices)) != 6 or set(condition_ids) != {1, 2}:
        return None
    if any(b - a != 1 for a, b in zip(block_indices, block_indices[1:])):
        return None
    if any(a == b for a, b in zip(condition_ids, condition_ids[1:])):
        return None
    return frozenset(block_indices[:3]), frozenset(block_indices[3:])


def has_good_block_coverage(
    block_indices: list[int],
    context_ids: list[int],
    good_block_indices: set[int] | frozenset[int],
) -> bool:
    """Check for at least one good block of each context in each fixed fold."""
    folds = fixed_block_folds(block_indices, context_ids)
    if folds is None:
        return False
    block_to_context = dict(zip(block_indices, context_ids))
    return all(
        {
            block_to_context[block]
            for block in fold
            if block in good_block_indices
        }
        == {1, 2}
        for fold in folds
    )


def time_bin_centers(n_timepoints: int, bin_size_s: float, pre_s: float) -> FloatArray:
    """Return stimulus-relative bin centers for a ``[-pre_s, ...]`` PSTH."""
    if n_timepoints <= 0:
        raise ValueError("n_timepoints must be positive")
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be positive")
    if pre_s < 0:
        raise ValueError("pre_s must be non-negative")
    return -pre_s + (np.arange(n_timepoints, dtype=np.float64) + 0.5) * bin_size_s


def window_mask(time_s: FloatArray, start_s: float, end_s: float) -> BoolArray:
    """Return a half-open ``[start_s, end_s)`` time-window mask."""
    if end_s <= start_s:
        raise ValueError(f"window end must exceed start: [{start_s}, {end_s})")
    mask = (time_s >= start_s) & (time_s < end_s)
    if not np.any(mask):
        raise ValueError(f"window [{start_s}, {end_s}) contains no PSTH bins")
    return mask


def signed_rms(d2: FloatArray | float, n_units: int) -> FloatArray | float:
    """Convert a squared cross-validated inner product to signed RMS/unit scale."""
    if n_units <= 0:
        raise ValueError("n_units must be positive")
    values = np.asarray(d2, dtype=np.float64)
    result = np.sign(values) * np.sqrt(np.abs(values) / n_units)
    return float(result) if result.ndim == 0 else result


def compute_trajectory_metrics(
    delta_fold_1: FloatArray,
    delta_fold_2: FloatArray,
    baseline_mask: BoolArray,
    stimulus_mask: BoolArray,
) -> TrajectoryMetrics:
    """Compute distance and projection metrics from two independent block folds.

    ``delta_fold_*`` is condition 1 minus condition 2 for every unit and timepoint.
    Baseline correction is performed separately in each fold.  Projection axes are
    learned in one fold and evaluated in the other, then the two directions are
    averaged.  Projection time courses are divided by ``sqrt(n_units)`` so they use
    the same RMS-per-unit convention as the distance output.
    """
    fold_1 = np.asarray(delta_fold_1, dtype=np.float64)
    fold_2 = np.asarray(delta_fold_2, dtype=np.float64)
    if fold_1.ndim != 2 or fold_2.ndim != 2:
        raise ValueError("fold difference arrays must have shape (units, timepoints)")
    if fold_1.shape != fold_2.shape:
        raise ValueError(f"fold shapes differ: {fold_1.shape} != {fold_2.shape}")
    if fold_1.shape[0] == 0 or fold_1.shape[1] == 0:
        raise ValueError("fold difference arrays cannot be empty")
    if not np.all(np.isfinite(fold_1)) or not np.all(np.isfinite(fold_2)):
        raise ValueError("fold difference arrays must contain only finite values")

    baseline_mask = _validate_mask(baseline_mask, fold_1.shape[1], "baseline")
    stimulus_mask = _validate_mask(stimulus_mask, fold_1.shape[1], "stimulus")

    baseline_1 = fold_1[:, baseline_mask].mean(axis=1)
    baseline_2 = fold_2[:, baseline_mask].mean(axis=1)
    corrected_1 = fold_1 - baseline_1[:, None]
    corrected_2 = fold_2 - baseline_2[:, None]

    raw_d2 = np.einsum("ut,ut->t", fold_1, fold_2)
    corrected_d2 = np.einsum("ut,ut->t", corrected_1, corrected_2)
    baseline_d2 = float(np.dot(baseline_1, baseline_2))

    baseline_axis_1 = _unit_vector(baseline_1)
    baseline_axis_2 = _unit_vector(baseline_2)

    stimulus_modulation_1 = corrected_1[:, stimulus_mask].mean(axis=1)
    stimulus_modulation_2 = corrected_2[:, stimulus_mask].mean(axis=1)
    orthogonal_axis_1 = _orthogonal_unit_vector(
        stimulus_modulation_1, baseline_axis_1
    )
    orthogonal_axis_2 = _orthogonal_unit_vector(
        stimulus_modulation_2, baseline_axis_2
    )

    scale = np.sqrt(fold_1.shape[0])
    baseline_projection = _cross_fitted_projection(
        corrected_1, corrected_2, baseline_axis_1, baseline_axis_2
    ) / scale
    orthogonal_projection = _cross_fitted_projection(
        corrected_1, corrected_2, orthogonal_axis_1, orthogonal_axis_2
    ) / scale

    return TrajectoryMetrics(
        raw_d2=raw_d2,
        baseline_corrected_d2=corrected_d2,
        baseline_d2=baseline_d2,
        baseline_axis_projection=baseline_projection,
        orthogonal_stimulus_projection=orthogonal_projection,
        baseline_axis_alignment=_axis_alignment(baseline_axis_1, baseline_axis_2),
        orthogonal_axis_alignment=_axis_alignment(
            orthogonal_axis_1, orthogonal_axis_2
        ),
    )


def _validate_mask(mask: BoolArray, n_timepoints: int, name: str) -> BoolArray:
    result = np.asarray(mask, dtype=np.bool_)
    if result.ndim != 1 or result.size != n_timepoints:
        raise ValueError(
            f"{name} mask must be one-dimensional with {n_timepoints} elements"
        )
    if not np.any(result):
        raise ValueError(f"{name} mask must select at least one timepoint")
    return result


def _unit_vector(vector: FloatArray) -> FloatArray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        return None
    return vector / norm


def _orthogonal_unit_vector(
    vector: FloatArray, reference_axis: FloatArray | None
) -> FloatArray | None:
    if reference_axis is None:
        return None
    residual = vector - np.dot(vector, reference_axis) * reference_axis
    return _unit_vector(residual)


def _cross_fitted_projection(
    corrected_fold_1: FloatArray,
    corrected_fold_2: FloatArray,
    axis_fold_1: FloatArray | None,
    axis_fold_2: FloatArray | None,
) -> FloatArray:
    if axis_fold_1 is None or axis_fold_2 is None:
        return np.full(corrected_fold_1.shape[1], np.nan, dtype=np.float64)
    # Each axis is evaluated only on the independent, held-out block fold.
    fold_2_on_axis_1 = axis_fold_1 @ corrected_fold_2
    fold_1_on_axis_2 = axis_fold_2 @ corrected_fold_1
    return 0.5 * (fold_2_on_axis_1 + fold_1_on_axis_2)


def _axis_alignment(axis_1: FloatArray | None, axis_2: FloatArray | None) -> float:
    if axis_1 is None or axis_2 is None:
        return float("nan")
    return float(np.dot(axis_1, axis_2))
