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
from typing import Any

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
    seed = 2000
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
    offsets = list(np.linspace(study.offset_range[0], study.offset_range[1], study.offset_steps))
    n_off = study.offset_steps
    sigmas = study.sigmas

    offset_labels = [f'{v:.2f}' for v in offsets]

    # Heatmaps: one per sigma value.
    step = n_off * n_off  # trials per sigma
    for s_idx, sigma in enumerate(sigmas):
        slice_results = results[s_idx * step : (s_idx + 1) * step]
        grid = np.full((n_off, n_off), float('nan'))
        fail_mask = np.zeros((n_off, n_off), dtype=bool)
        for oy_idx in range(n_off):
            for ox_idx in range(n_off):
                r = slice_results[oy_idx * n_off + ox_idx]
                if not r.converged:
                    fail_mask[oy_idx, ox_idx] = True
                else:
                    grid[oy_idx, ox_idx] = r.pos_err

        fig = plot_heatmap(
            grid,
            offset_labels,
            offset_labels,
            title=f'Position error vs. offset (sigma={sigma:.1f})',
            xlabel='offset_x (pixels)',
            ylabel='offset_y (pixels)',
            cbar_label='log10(pos error)',
            log_scale=True,
            mask=fail_mask,
        )
        save_figure(fig, study_dir, f'pos_err_sigma{sigma:.1f}.png')

    # Line plot: pos error vs offset_x at offset_y=0.25 for all sigmas.
    # Use the midpoint of the offset range as the fixed offset_y row.
    mid_oy_idx = n_off // 2
    x_arr = np.array(offsets)
    y_means: list[npt.NDArray[np.float64]] = []
    y_stds: list[npt.NDArray[np.float64]] = []
    line_labels: list[str] = []

    for s_idx, sigma in enumerate(sigmas):
        slice_results = results[s_idx * step : (s_idx + 1) * step]
        row: list[float] = []
        for ox_idx in range(n_off):
            r = slice_results[mid_oy_idx * n_off + ox_idx]
            row.append(r.pos_err if r.converged else float('nan'))
        arr = np.array(row, dtype=np.float64)
        y_means.append(arr)
        y_stds.append(np.zeros_like(arr))  # single trial; no std
        line_labels.append(f'sigma={sigma:.1f}')

    mid_oy = offsets[mid_oy_idx]
    fig = plot_line_with_bands(
        x_arr,
        y_means,
        y_stds,
        labels=line_labels,
        title=f'Position error vs. offset_x at offset_y={mid_oy:.2f}',
        xlabel='offset_x (pixels)',
        ylabel='Position error (pixels)',
        log_y=True,
    )
    save_figure(fig, study_dir, 'pos_err_vs_offset_x.png')

    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)

    groups = utils.build_groups_by_keys(
        specs,
        results,
        [
            ('sigma', lambda s: s.sigma_y),
            ('offset_y', lambda s: round(s.offset_y, 6)),
            ('offset_x', lambda s: round(s.offset_x, 6)),
        ],
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


def build_json_groups(specs: list[TrialSpec], results: list[TrialResult]) -> list[dict[str, Any]]:
    """Build JSON summary groups for Study 2.

    Parameters:
        specs: Trial specifications.
        results: Trial results.

    Returns:
        List of group dicts for :func:`~output.write_json_summary`.
    """
    return utils.build_groups_by_keys(
        specs,
        results,
        [
            ('sigma', lambda s: s.sigma_y),
            ('offset_y', lambda s: round(s.offset_y, 6)),
            ('offset_x', lambda s: round(s.offset_x, 6)),
        ],
    )
