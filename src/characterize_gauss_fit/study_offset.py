################################################################################
# characterize_gauss_fit/study_offset.py
################################################################################

"""Study 2: Subpixel offset bias.

Explores how the fractional pixel position of the PSF centre introduces
systematic bias in the recovered position.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from characterize_gauss_fit import _study_utils as utils
from characterize_gauss_fit.config import Config, config_to_dict
from characterize_gauss_fit.executor import run_trials
from characterize_gauss_fit.output import write_csv, write_json_summary
from characterize_gauss_fit.plotting import plot_heatmap, plot_line_with_bands, save_figure
from characterize_gauss_fit.trial import TrialResult, TrialSpec

_LOG = logging.getLogger(__name__)
_STUDY_NAME = 'subpixel_offset'

# Shared grouping key definitions for JSON summary and _write_outputs.
_GROUP_KEYS: list[tuple[str, Callable[[TrialSpec], float]]] = [
    ('sigma', lambda s: s.sigma_y),
    ('offset_y', lambda s: round(s.offset_y, 6)),
    ('offset_x', lambda s: round(s.offset_x, 6)),
]


def build_specs(cfg: Config) -> list[TrialSpec]:
    """Build trial specs for Study 2.

    Generates a 2-D grid of (offset_y, offset_x) combinations for each sigma.
    Sigma is left to float during fitting.

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A list of :class:`~trial.TrialSpec` objects, ordered by
        [sigma, offset_y, offset_x].
    """
    study = cfg.studies.subpixel_offset
    offsets = np.linspace(study.offset_range[0], study.offset_range[1], study.offset_steps)
    specs: list[TrialSpec] = []
    seed = 2000  # seed range 2000+ reserved for this study to avoid RNG seed collisions
    for sigma in study.sigmas:
        for oy in offsets:
            for ox in offsets:
                specs.append(
                    utils.make_spec(
                        sigma_y=sigma,
                        sigma_x=sigma,
                        angle=study.angle,
                        offset_y=float(oy),
                        offset_x=float(ox),
                        scale=cfg.generation.scale,
                        base=cfg.generation.base,
                        box_size=study.box_size,
                        fitting=study.fitting,
                        fit_angle=0.0,
                        rng_seed=seed,
                    )
                )
                seed += 1
    return specs


def run(cfg: Config, *, num_workers: int = 1) -> None:
    """Execute Study 2 and write all outputs.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        num_workers: Number of parallel worker processes.
    """
    study = cfg.studies.subpixel_offset
    if not study.enabled:
        _LOG.info('Study %s is disabled; skipping.', _STUDY_NAME)
        return

    specs = build_specs(cfg)
    _LOG.info('Study %s: %d trials', _STUDY_NAME, len(specs))

    results = run_trials(
        specs,
        num_workers=num_workers,
        progress_callback=utils.progress_callback(_STUDY_NAME),
    )

    study_dir = utils.ensure_study_dir(cfg.output_dir, _STUDY_NAME)
    _write_outputs(cfg, specs, results, study_dir)


def _write_outputs(
    cfg: Config,
    specs: list[TrialSpec],
    results: list[TrialResult],
    study_dir: pathlib.Path,
) -> None:
    """Write CSV, JSON, and PNG outputs for Study 2.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        specs: All trial specifications.
        results: All trial results.
        study_dir: Output subdirectory.
    """
    study = cfg.studies.subpixel_offset
    scale = cfg.generation.scale
    offsets = list(np.linspace(study.offset_range[0], study.offset_range[1], study.offset_steps))
    n_off = study.offset_steps
    sigmas = study.sigmas

    offset_labels = [f'{v:.2f}' for v in offsets]

    # Heatmaps: one set per sigma value — Euclidean error plus Y and X axes.
    step = n_off * n_off  # trials per sigma
    for s_idx, sigma in enumerate(sigmas):
        slice_results = results[s_idx * step : (s_idx + 1) * step]
        grid = np.full((n_off, n_off), float('nan'))
        grid_y = np.full((n_off, n_off), float('nan'))
        grid_x = np.full((n_off, n_off), float('nan'))
        fail_mask = np.zeros((n_off, n_off), dtype=bool)
        for oy_idx in range(n_off):
            for ox_idx in range(n_off):
                r = slice_results[oy_idx * n_off + ox_idx]
                if not r.converged:
                    fail_mask[oy_idx, ox_idx] = True
                else:
                    grid[oy_idx, ox_idx] = r.pos_err
                    grid_y[oy_idx, ox_idx] = abs(r.pos_err_y)
                    grid_x[oy_idx, ox_idx] = abs(r.pos_err_x)

        for hmap, metric_label, fname in [
            (grid, 'Position error (Euclidean)', f'pos_err_sigma{sigma:.1f}.png'),
            (grid_y, '|pos_err_y|', f'pos_err_y_sigma{sigma:.1f}.png'),
            (grid_x, '|pos_err_x|', f'pos_err_x_sigma{sigma:.1f}.png'),
        ]:
            fig = plot_heatmap(
                hmap,
                offset_labels,
                offset_labels,
                title=f'{metric_label} vs. offset (sigma={sigma:.1f})',
                xlabel='offset_x (pixels)',
                ylabel='offset_y (pixels)',
                cbar_label='log10(error)',
                log_scale=True,
                mask=fail_mask,
                note=(
                    f'PSF: sigma_y = sigma_x = {sigma:.2g} px (fixed for this heatmap);'
                    f' angle = {study.angle:.0f}\u00b0 (fixed); box_size = {study.box_size} px;'
                    f' scale = {scale:.2g}; one noiseless trial per cell\n'
                    'Offset: Y and X each swept over a'
                    f' {study.offset_steps}\u00d7{study.offset_steps}'
                    f' grid from {study.offset_range[0]:.2f} to {study.offset_range[1]:.2f} px'
                    ' (the two heatmap axes)\n'
                    'Background / noise: none injected\n'
                    'Fitting: sigma_y and sigma_x float freely; angle fixed at 0\u00b0;'
                    ' no background subtraction'
                ),
            )
            save_figure(fig, study_dir, f'{_STUDY_NAME}_{fname}')

    # Line plots: error vs offset_x at fixed offset_y (midpoint row), and
    # error vs offset_y at fixed offset_x (midpoint column).
    # mid_idx selects the midpoint of the configured offset range (n_off // 2).
    mid_idx = n_off // 2
    x_arr = np.array(offsets)

    for metric_attr, metric_label, vary_axis in [
        ('pos_err', 'Position error (Euclidean, pixels)', 'offset_x'),
        ('pos_err_y', '|pos_err_y| (pixels)', 'offset_x'),
        ('pos_err_x', '|pos_err_x| (pixels)', 'offset_x'),
        ('pos_err', 'Position error (Euclidean, pixels)', 'offset_y'),
        ('pos_err_y', '|pos_err_y| (pixels)', 'offset_y'),
        ('pos_err_x', '|pos_err_x| (pixels)', 'offset_y'),
    ]:
        y_means: list[npt.NDArray[np.float64]] = []
        y_stds: list[npt.NDArray[np.float64]] = []
        line_labels: list[str] = []

        for s_idx, sigma in enumerate(sigmas):
            slice_results = results[s_idx * step : (s_idx + 1) * step]
            row: list[float] = []
            for idx in range(n_off):
                if vary_axis == 'offset_x':
                    r = slice_results[mid_idx * n_off + idx]
                else:
                    r = slice_results[idx * n_off + mid_idx]
                val = getattr(r, metric_attr) if r.converged else None
                row.append(float('nan') if val is None else abs(float(val)))
            y_means.append(np.array(row, dtype=np.float64))
            y_stds.append(np.zeros(n_off, dtype=np.float64))
            line_labels.append(f'sigma={sigma:.1f}')

        fixed_val = offsets[mid_idx]
        fixed_axis = 'offset_y' if vary_axis == 'offset_x' else 'offset_x'
        safe_metric = metric_attr.replace('_', '')
        fname_line = f'{safe_metric}_vs_{vary_axis}.png'
        fig = plot_line_with_bands(
            x_arr,
            y_means,
            y_stds,
            labels=line_labels,
            title=f'{metric_label} vs. {vary_axis} at {fixed_axis}={fixed_val:.2f}',
            xlabel=f'{vary_axis} (pixels)',
            ylabel=metric_label,
            log_y=True,
            note=(
                f'PSF: sigma_y = sigma_x = sigma (series label); angle = {study.angle:.0f}\u00b0'
                f' (fixed); box_size = {study.box_size} px; scale = {scale:.2g};'
                ' one noiseless trial per point\n'
                f'Offset: the fixed axis is held at its midpoint ({offsets[mid_idx]:.2f} px);'
                ' the swept axis is the x-axis\n'
                'Background / noise: none injected\n'
                'Fitting: sigma_y and sigma_x float freely; angle fixed at 0\u00b0;'
                ' no background subtraction'
            ),
        )
        save_figure(fig, study_dir, f'{_STUDY_NAME}_{fname_line}')

    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)

    groups = utils.build_groups_by_keys(
        specs,
        results,
        _GROUP_KEYS,
    )
    write_json_summary(
        cfg.output_dir,
        _STUDY_NAME,
        specs,
        results,
        groups=groups,
        config_used=config_to_dict(cfg),
    )
    _LOG.info('Study %s outputs written to %s', _STUDY_NAME, study_dir)
