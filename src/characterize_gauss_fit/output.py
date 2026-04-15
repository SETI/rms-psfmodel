################################################################################
# characterize_gauss_fit/output.py
################################################################################

"""CSV and JSON output writers for characterize_gauss_fit study results.

Each study writes its results to a subdirectory of the configured output
directory. This module provides two writers:

- :func:`write_csv` -- one row per trial, all input parameters and all result
  metrics, directly loadable by pandas or any data-analysis tool.
- :func:`write_json_summary` -- aggregate statistics per parameter group plus
  overall convergence rates and the exact config used, suitable for AI analysis.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
from typing import Any

import numpy as np

from characterize_gauss_fit.trial import TrialResult, TrialSpec

# Column order for the CSV file. Non-applicable columns use empty string.
_CSV_COLUMNS: list[str] = [
    # Study-level context
    'study',
    'rng_seed',
    # Input: geometry
    'box_size',
    'sigma_y_true',
    'sigma_x_true',
    'angle_true',
    'offset_y_true',
    'offset_x_true',
    'scale_true',
    'base',
    # Input: fitter construction
    'fit_sigma_y',
    'fit_sigma_x',
    'fit_angle',
    # Input: background / noise
    'background_type',
    'background_amplitude',
    'noise_rms',
    # Input: hot pixels
    'num_hot_pixels',
    'hot_pixel_amplitude',
    # Input: fitting kwargs
    'bkgnd_degree',
    'num_sigma',
    'bkgnd_num_sigma',
    'bkgnd_ignore_center_y',
    'bkgnd_ignore_center_x',
    'max_bad_frac',
    'allow_nonzero_base',
    'use_angular_params',
    'tolerance',
    'search_limit_lo',
    'search_limit_hi',
    'scale_limit',
    # Outcome
    'converged',
    # Position errors
    'pos_err_y',
    'pos_err_x',
    'pos_err',
    # Fitted values
    'sigma_y_fit',
    'sigma_x_fit',
    'angle_fit',
    'scale_fit',
    # Parameter errors
    'sigma_y_err',
    'sigma_x_err',
    'angle_err',
    'scale_err',
]


def _float_cell(value: float | None) -> str:
    """Format a float value for a CSV cell.

    Parameters:
        value: The float to format, or ``None`` for a not-applicable field.

    Returns:
        The formatted string. NaN is written as ``"NaN"``, infinities as
        ``"Inf"`` / ``"-Inf"``, and None as ``""``.
    """
    if value is None:
        return ''
    if math.isnan(value):
        return 'NaN'
    if math.isinf(value):
        return 'Inf' if value > 0 else '-Inf'
    return repr(float(value))


def _result_row(
    study: str,
    spec: TrialSpec,
    result: TrialResult,
) -> dict[str, str]:
    """Build a CSV row dict from a spec and its result.

    Parameters:
        study: The study name string (e.g. ``'box_vs_sigma'``).
        spec: The :class:`~trial.TrialSpec` that was executed.
        result: The :class:`~trial.TrialResult` from the trial.

    Returns:
        A ``dict[str, str]`` mapping CSV column names to string values.
    """
    row: dict[str, str] = dict.fromkeys(_CSV_COLUMNS, '')
    row['study'] = study
    row['rng_seed'] = str(spec.rng_seed)
    row['box_size'] = str(spec.box_size)
    row['sigma_y_true'] = repr(spec.sigma_y)
    row['sigma_x_true'] = repr(spec.sigma_x)
    row['angle_true'] = repr(spec.angle)
    row['offset_y_true'] = repr(spec.offset_y)
    row['offset_x_true'] = repr(spec.offset_x)
    row['scale_true'] = repr(spec.scale)
    row['base'] = repr(spec.base)
    row['fit_sigma_y'] = _float_cell(spec.fit_sigma_y)
    row['fit_sigma_x'] = _float_cell(spec.fit_sigma_x)
    row['fit_angle'] = _float_cell(spec.fit_angle)
    row['background_type'] = spec.background_type
    row['background_amplitude'] = repr(spec.background_amplitude)
    row['noise_rms'] = repr(spec.noise_rms)
    row['num_hot_pixels'] = str(spec.hot_pixel_count)
    row['hot_pixel_amplitude'] = repr(spec.hot_pixel_amplitude)
    row['bkgnd_degree'] = '' if spec.bkgnd_degree is None else str(spec.bkgnd_degree)
    row['num_sigma'] = _float_cell(spec.num_sigma)
    row['bkgnd_num_sigma'] = _float_cell(spec.bkgnd_num_sigma)
    row['bkgnd_ignore_center_y'] = str(spec.bkgnd_ignore_center[0])
    row['bkgnd_ignore_center_x'] = str(spec.bkgnd_ignore_center[1])
    row['max_bad_frac'] = repr(spec.max_bad_frac)
    row['allow_nonzero_base'] = 'true' if spec.allow_nonzero_base else 'false'
    row['use_angular_params'] = 'true' if spec.use_angular_params else 'false'
    row['tolerance'] = repr(spec.tolerance)
    row['search_limit_lo'] = repr(spec.search_limit[0])
    row['search_limit_hi'] = repr(spec.search_limit[1])
    row['scale_limit'] = repr(spec.scale_limit)
    row['converged'] = 'true' if result.converged else 'false'
    row['pos_err_y'] = _float_cell(result.pos_err_y)
    row['pos_err_x'] = _float_cell(result.pos_err_x)
    row['pos_err'] = _float_cell(result.pos_err)
    row['sigma_y_fit'] = _float_cell(result.sigma_y_fit)
    row['sigma_x_fit'] = _float_cell(result.sigma_x_fit)
    row['angle_fit'] = _float_cell(result.angle_fit)
    row['scale_fit'] = _float_cell(result.scale_fit)
    row['sigma_y_err'] = _float_cell(result.sigma_y_err)
    row['sigma_x_err'] = _float_cell(result.sigma_x_err)
    row['angle_err'] = _float_cell(result.angle_err)
    row['scale_err'] = _float_cell(result.scale_err)
    return row


def write_csv(
    output_dir: pathlib.Path,
    study: str,
    specs: list[TrialSpec],
    results: list[TrialResult],
) -> pathlib.Path:
    """Write per-trial results to a CSV file.

    Creates ``{output_dir}/{study}/trials.csv`` with one row per trial.
    All input parameters and all result metrics are included. Non-applicable
    fields (e.g. ``sigma_y_fit`` when sigma_y was fixed) are written as empty
    strings. NaN values are written as ``"NaN"``.

    Parameters:
        output_dir: Root output directory.
        study: Study name (used as subdirectory).
        specs: List of :class:`~trial.TrialSpec` objects (same order as ``results``).
        results: List of :class:`~trial.TrialResult` objects.

    Returns:
        Path to the written CSV file.

    Raises:
        ValueError: If ``specs`` and ``results`` have different lengths.
    """
    if len(specs) != len(results):
        raise ValueError(
            f'specs ({len(specs)}) and results ({len(results)}) must have the same length'
        )
    study_dir = output_dir / study
    study_dir.mkdir(parents=True, exist_ok=True)
    csv_path = study_dir / 'trials.csv'

    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for spec, result in zip(specs, results, strict=True):
            writer.writerow(_result_row(study, spec, result))

    return csv_path


def _safe_mean(values: list[float]) -> float | None:
    """Compute the mean of a list of finite floats, or None if the list is empty.

    Parameters:
        values: A list of float values. NaN and infinite values are excluded.

    Returns:
        The mean of finite values, or ``None`` if no finite values exist.
    """
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) == 0:
        return None
    return float(np.mean(finite))


def _safe_std(values: list[float]) -> float | None:
    """Compute the std dev of a list of finite floats, or None if too few values.

    Parameters:
        values: A list of float values. NaN and infinite values are excluded.

    Returns:
        The standard deviation, or ``None`` if fewer than two finite values exist.
    """
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return None
    return float(np.std(finite, ddof=1))


def _aggregate_results(results: list[TrialResult]) -> dict[str, Any]:
    """Compute aggregate statistics over a list of results.

    Parameters:
        results: List of :class:`~trial.TrialResult` objects to aggregate.

    Returns:
        A dict of aggregate statistics including counts and per-metric
        mean and std values.
    """
    n_total = len(results)
    converged = [r for r in results if r.converged]
    n_converged = len(converged)

    pos_errs = [r.pos_err for r in converged if math.isfinite(r.pos_err)]
    sigma_y_errs = [
        r.sigma_y_err
        for r in converged
        if r.sigma_y_err is not None and math.isfinite(r.sigma_y_err)
    ]
    sigma_x_errs = [
        r.sigma_x_err
        for r in converged
        if r.sigma_x_err is not None and math.isfinite(r.sigma_x_err)
    ]
    angle_errs = [
        r.angle_err for r in converged if r.angle_err is not None and math.isfinite(r.angle_err)
    ]
    scale_errs = [r.scale_err for r in converged if math.isfinite(r.scale_err)]

    return {
        'n_trials': n_total,
        'n_converged': n_converged,
        'convergence_rate': n_converged / n_total if n_total > 0 else None,
        'pos_err_mean': _safe_mean(pos_errs),
        'pos_err_std': _safe_std(pos_errs),
        'sigma_y_err_mean': _safe_mean(sigma_y_errs),
        'sigma_y_err_std': _safe_std(sigma_y_errs),
        'sigma_x_err_mean': _safe_mean(sigma_x_errs),
        'sigma_x_err_std': _safe_std(sigma_x_errs),
        'angle_err_mean': _safe_mean(angle_errs),
        'angle_err_std': _safe_std(angle_errs),
        'scale_err_mean': _safe_mean(scale_errs),
        'scale_err_std': _safe_std(scale_errs),
    }


def _sanitize_json(obj: Any) -> Any:
    """Recursively replace non-finite floats with None for JSON safety.

    :func:`json.dump` with ``allow_nan=True`` emits non-standard tokens
    (``NaN``, ``Infinity``, ``-Infinity``) that many JSON parsers reject.
    This function replaces such values with ``None`` (serialised as ``null``).

    Parameters:
        obj: Any JSON-serialisable Python object (dict, list, float, etc.).

    Returns:
        A new object with the same structure but with non-finite floats
        replaced by ``None``.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


def write_json_summary(
    output_dir: pathlib.Path,
    study: str,
    specs: list[TrialSpec],
    results: list[TrialResult],
    *,
    groups: list[dict[str, Any]],
    config_used: dict[str, Any],
) -> pathlib.Path:
    """Write an aggregate JSON summary for a study.

    Creates ``{output_dir}/{study}/summary.json`` with overall statistics,
    per-group breakdowns, and the exact configuration used for reproducibility.

    Parameters:
        output_dir: Root output directory.
        study: Study name (used as subdirectory).
        specs: All :class:`~trial.TrialSpec` objects for the study.
        results: All :class:`~trial.TrialResult` objects (same order as ``specs``).
        groups: A list of dicts, each describing one parameter group. Each dict
            must contain a ``'indices'`` key with the list of result indices
            belonging to that group (these are removed before writing). All other
            keys are written verbatim as group labels.
        config_used: The serialised configuration dict (from
            :func:`~config.config_to_dict`).

    Returns:
        Path to the written JSON file.

    Raises:
        ValueError: If ``specs`` and ``results`` have different lengths.
    """
    if len(specs) != len(results):
        raise ValueError(
            f'specs ({len(specs)}) and results ({len(results)}) must have the same length'
        )
    study_dir = output_dir / study
    study_dir.mkdir(parents=True, exist_ok=True)
    json_path = study_dir / 'summary.json'

    overall = _aggregate_results(results)

    group_summaries: list[dict[str, Any]] = []
    for group in groups:
        indices: list[int] = group['indices']
        group_results = [results[i] for i in indices]
        agg = _aggregate_results(group_results)
        # Write label keys (everything except 'indices').
        summary: dict[str, Any] = {k: v for k, v in group.items() if k != 'indices'}
        summary.update(agg)
        group_summaries.append(summary)

    payload: dict[str, Any] = {
        'study': study,
        'total_trials': overall['n_trials'],
        'converged_trials': overall['n_converged'],
        'convergence_rate': overall['convergence_rate'],
        'overall': overall,
        'groups': group_summaries,
        'config_used': config_used,
    }

    with json_path.open('w', encoding='utf-8') as fh:
        json.dump(_sanitize_json(payload), fh, indent=2)

    return json_path
