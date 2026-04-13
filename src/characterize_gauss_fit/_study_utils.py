################################################################################
# characterize_gauss_fit/_study_utils.py
################################################################################

"""Shared utilities for study modules in characterize_gauss_fit.

Not part of the public API; prefixed with ``_`` to indicate internal use.
"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

from characterize_gauss_fit.config import FittingConfig
from characterize_gauss_fit.trial import (
    BACKGROUND_TYPE_NONE,
    TrialResult,
    TrialSpec,
)


def offset_tag(offset_y: float, offset_x: float) -> str:
    """Return a filename-safe tag encoding a (offset_y, offset_x) pair.

    Decimal points are replaced with 'p' and negative signs with 'm', so
    ``(0.25, 0.0)`` becomes ``'oy0p25_ox0p00'``.

    Parameters:
        offset_y: Y component of the offset (fractional pixels).
        offset_x: X component of the offset (fractional pixels).

    Returns:
        A short ASCII string safe for use in file and directory names.
    """
    def _fmt(v: float) -> str:
        return f'{v:.2f}'.replace('.', 'p').replace('-', 'm')

    return f'oy{_fmt(offset_y)}_ox{_fmt(offset_x)}'


def make_spec(
    *,
    sigma_y: float,
    sigma_x: float,
    angle: float,
    offset_y: float,
    offset_x: float,
    scale: float,
    base: float,
    box_size: int,
    fitting: FittingConfig,
    fit_sigma_y: float | None = None,
    fit_sigma_x: float | None = None,
    fit_angle: float | None = None,
    background_type: str = BACKGROUND_TYPE_NONE,
    background_amplitude: float = 0.0,
    noise_rms: float = 0.0,
    hot_pixel_count: int = 0,
    hot_pixel_amplitude: float = 0.0,
    rng_seed: int = 0,
) -> TrialSpec:
    """Construct a :class:`~trial.TrialSpec` from explicit keyword arguments.

    Fills in all fields, pulling fitting parameters from the provided
    :class:`~config.FittingConfig`. Study modules call this helper to avoid
    repeating the long field list.

    Parameters:
        sigma_y: True PSF sigma in the Y direction (pixels).
        sigma_x: True PSF sigma in the X direction (pixels).
        angle: True PSF rotation angle (radians).
        offset_y: Sub-pixel offset of the PSF centre from the image centre (Y).
        offset_x: Sub-pixel offset of the PSF centre from the image centre (X).
        scale: PSF amplitude scale factor.
        base: Additive base level on the clean PSF image.
        box_size: Side length of the square image patch (must be odd).
        fitting: Fitting configuration for this trial.
        fit_sigma_y: Fixed fitter sigma_y (``None`` = float during fitting).
        fit_sigma_x: Fixed fitter sigma_x (``None`` = float during fitting).
        fit_angle: Fixed fitter angle (``None`` = float during fitting).
        background_type: Type of injected background.
        background_amplitude: Background amplitude as fraction of PSF peak.
        noise_rms: Additive Gaussian noise standard deviation.
        hot_pixel_count: Number of hot pixels to inject.
        hot_pixel_amplitude: Hot pixel amplitude as a multiple of PSF peak.
        rng_seed: Seed for the NumPy random number generator.

    Returns:
        A fully populated :class:`~trial.TrialSpec`.
    """
    return TrialSpec(
        sigma_y=sigma_y,
        sigma_x=sigma_x,
        angle=angle,
        offset_y=offset_y,
        offset_x=offset_x,
        scale=scale,
        base=base,
        box_size=box_size,
        fit_sigma_y=fit_sigma_y,
        fit_sigma_x=fit_sigma_x,
        fit_angle=fit_angle,
        background_type=background_type,
        background_amplitude=background_amplitude,
        noise_rms=noise_rms,
        hot_pixel_count=hot_pixel_count,
        hot_pixel_amplitude=hot_pixel_amplitude,
        bkgnd_degree=fitting.bkgnd_degree,
        bkgnd_ignore_center=fitting.bkgnd_ignore_center,
        bkgnd_num_sigma=fitting.bkgnd_num_sigma,
        num_sigma=fitting.num_sigma,
        max_bad_frac=fitting.max_bad_frac,
        allow_nonzero_base=fitting.allow_nonzero_base,
        use_angular_params=fitting.use_angular_params,
        tolerance=fitting.tolerance,
        search_limit=fitting.search_limit,
        scale_limit=fitting.scale_limit,
        rng_seed=rng_seed,
    )


def progress_callback(study_name: str) -> Any:
    """Return a progress-printing callback for the executor.

    The returned callable prints a single updating line to stderr showing
    completed / total trial counts.

    Parameters:
        study_name: Name of the running study (included in the output line).

    Returns:
        A ``Callable[[int, int], None]`` suitable for
        :func:`~executor.run_trials`.
    """

    def _callback(completed: int, total: int) -> None:
        if total > 0:
            pct = 100 * completed // total
            msg = f'\r  {study_name}: {completed}/{total} trials ({pct}%)   '
        else:
            msg = f'\r  {study_name}: {completed}/0 trials   '
        print(msg, end='', flush=True, file=sys.stderr)
        if completed == total:
            print(file=sys.stderr)

    return _callback


def collect_metric(
    results: list[TrialResult],
    metric: str,
) -> npt.NDArray[np.float64]:
    """Extract a named metric from a list of results as a float array.

    Unconverged trials produce NaN for all metrics.

    Parameters:
        results: List of :class:`~trial.TrialResult` objects.
        metric: Name of the :class:`~trial.TrialResult` field to extract.

    Returns:
        A 1-D float64 array of length ``len(results)``.

    Raises:
        AttributeError: If ``metric`` is not a field of :class:`~trial.TrialResult`.
    """
    values = [
        float('nan') if getattr(r, metric) is None else float(getattr(r, metric))
        for r in results
    ]
    return np.array(values, dtype=np.float64)


def safe_nanmean(arr: npt.NDArray[np.float64]) -> float:
    """Return the nanmean of ``arr``, or NaN if all values are NaN.

    Parameters:
        arr: Input float array.

    Returns:
        The mean of finite values, or ``float('nan')`` if none are finite.
    """
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return float('nan')
    return float(np.mean(finite))


def safe_nanstd(arr: npt.NDArray[np.float64]) -> float:
    """Return the nanstd (ddof=1) of ``arr``, or NaN if fewer than 2 finite values.

    Parameters:
        arr: Input float array.

    Returns:
        The standard deviation of finite values, or ``float('nan')`` if too few.
    """
    finite = arr[np.isfinite(arr)]
    if len(finite) < 2:
        return float('nan')
    return float(np.std(finite, ddof=1))


def build_groups_by_keys(
    specs: list[TrialSpec],
    results: list[TrialResult],
    key_funcs: list[Any],
) -> list[dict[str, Any]]:
    """Group results by a tuple of key functions, building JSON-summary groups.

    Parameters:
        specs: Ordered list of :class:`~trial.TrialSpec` objects.
        results: Accepted for API symmetry with
            :func:`~output.write_json_summary` but not used here; grouping
            is based solely on ``specs`` and ``key_funcs``.
        key_funcs: List of ``(label, callable)`` pairs where the callable
            takes a :class:`~trial.TrialSpec` and returns the group key value.

    Returns:
        A list of group dicts suitable for :func:`~output.write_json_summary`.
    """
    from collections import defaultdict

    group_map: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, spec in enumerate(specs):
        key = tuple(fn(spec) for _, fn in key_funcs)
        group_map[key].append(idx)

    def _sort_key(item: tuple[tuple[Any, ...], list[int]]) -> tuple[Any, ...]:
        """Convert a group key to a sortable tuple, replacing None with a sentinel."""
        return tuple((0, v) if v is not None else (1, '') for v in item[0])

    groups: list[dict[str, Any]] = []
    for key, indices in sorted(group_map.items(), key=_sort_key):
        g: dict[str, Any] = {}
        for (label, _), val in zip(key_funcs, key, strict=False):
            g[label] = val
        g['indices'] = indices
        groups.append(g)

    return groups


def ensure_study_dir(output_dir: pathlib.Path, study_name: str) -> pathlib.Path:
    """Create and return the study output subdirectory.

    Parameters:
        output_dir: Root output directory.
        study_name: Study name (used as subdirectory name).

    Returns:
        Path to the created subdirectory.
    """
    study_dir = output_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    return study_dir


def recovery_fraction(
    results: list[TrialResult],
    *,
    delta: float,
) -> float:
    """Compute the fraction of trials where the position error is within delta/2.

    A trial is considered a "successful recovery" if the fitter converged and
    the Euclidean position error is less than half the injected offset delta.
    This metric is more informative than mean error alone for Study 3.

    Parameters:
        results: List of :class:`~trial.TrialResult` objects.
        delta: The injected offset magnitude used as the success threshold
            denominator.

    Returns:
        Fraction in [0, 1]. Returns ``float('nan')`` if ``results`` is empty.
    """
    if len(results) == 0:
        return float('nan')
    threshold = delta / 2.0
    successes = sum(
        1 for r in results if r.converged and math.isfinite(r.pos_err) and r.pos_err < threshold
    )
    return successes / len(results)
