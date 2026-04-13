################################################################################
# characterize_gauss_fit/study_box_sigma.py
################################################################################

"""Study 1: Box size vs. PSF sigma.

Explores how the subimage size relative to the PSF width affects position,
sigma, and scale recovery accuracy.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np

from characterize_gauss_fit import _study_utils as utils
from characterize_gauss_fit.config import Config, config_to_dict
from characterize_gauss_fit.executor import run_trials
from characterize_gauss_fit.output import write_csv, write_json_summary
from characterize_gauss_fit.plotting import plot_heatmap, save_figure
from characterize_gauss_fit.trial import TrialResult, TrialSpec

_LOG = logging.getLogger(__name__)
_STUDY_NAME = 'box_vs_sigma'


def build_specs(cfg: Config) -> list[TrialSpec]:
    """Build the full list of :class:`~trial.TrialSpec` objects for Study 1.

    Each spec represents one (box_size, sigma) combination with fixed offset
    and no background or noise, with sigma left to float during fitting.

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A list of :class:`~trial.TrialSpec` objects.
    """
    study = cfg.studies.box_vs_sigma
    specs: list[TrialSpec] = []
    seed = 1000
    for box_size in study.box_sizes:
        for sigma in study.sigmas:
            specs.append(
                utils.make_spec(
                    sigma_y=sigma,
                    sigma_x=sigma,
                    angle=study.angle,
                    offset_y=study.offset[0],
                    offset_x=study.offset[1],
                    scale=study.scale,
                    base=cfg.generation.base,
                    box_size=box_size,
                    fitting=study.fitting,
                    # sigma left to float (fit_sigma_y/x = None)
                    fit_angle=0.0,  # axis-aligned; no need to fit angle
                    rng_seed=seed,
                )
            )
            seed += 1
    return specs


def run(cfg: Config, *, num_workers: int = 1) -> None:
    """Execute Study 1 and write all outputs.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        num_workers: Number of parallel worker processes.
    """
    study = cfg.studies.box_vs_sigma
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
    """Write CSV, JSON, and PNG outputs for Study 1.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        specs: All trial specifications.
        results: All trial results (same order as ``specs``).
        study_dir: Output subdirectory for this study.
    """
    study = cfg.studies.box_vs_sigma
    box_sizes = study.box_sizes
    sigmas = study.sigmas

    # Build 2-D arrays indexed by [box_size_idx, sigma_idx].
    n_box = len(box_sizes)
    n_sigma = len(sigmas)
    pos_err_grid = np.full((n_box, n_sigma), float('nan'))
    pos_err_y_grid = np.full((n_box, n_sigma), float('nan'))
    pos_err_x_grid = np.full((n_box, n_sigma), float('nan'))
    sigma_y_err_grid = np.full((n_box, n_sigma), float('nan'))
    sigma_x_err_grid = np.full((n_box, n_sigma), float('nan'))
    scale_err_grid = np.full((n_box, n_sigma), float('nan'))
    fail_mask = np.zeros((n_box, n_sigma), dtype=bool)

    spec_idx = 0
    for b_idx in range(n_box):
        for s_idx in range(n_sigma):
            r = results[spec_idx]
            if not r.converged:
                fail_mask[b_idx, s_idx] = True
            else:
                pos_err_grid[b_idx, s_idx] = r.pos_err
                pos_err_y_grid[b_idx, s_idx] = abs(r.pos_err_y)
                pos_err_x_grid[b_idx, s_idx] = abs(r.pos_err_x)
                if r.sigma_y_err is not None:
                    sigma_y_err_grid[b_idx, s_idx] = abs(r.sigma_y_err)
                if r.sigma_x_err is not None:
                    sigma_x_err_grid[b_idx, s_idx] = abs(r.sigma_x_err)
                if np.isfinite(r.scale_err):
                    scale_err_grid[b_idx, s_idx] = abs(r.scale_err)
            spec_idx += 1

    x_labels = [f'{s:.2g}' for s in sigmas]
    y_labels = [str(b) for b in box_sizes]

    for data, title, filename in [
        (pos_err_grid, 'Position error (Euclidean) vs. box size and sigma', 'pos_err.png'),
        (pos_err_y_grid, '|pos_err_y| vs. box size and sigma', 'pos_err_y.png'),
        (pos_err_x_grid, '|pos_err_x| vs. box size and sigma', 'pos_err_x.png'),
        (
            sigma_y_err_grid,
            'Relative |sigma_y| error vs. box size and sigma',
            'sigma_y_err.png',
        ),
        (
            sigma_x_err_grid,
            'Relative |sigma_x| error vs. box size and sigma',
            'sigma_x_err.png',
        ),
        (scale_err_grid, 'Relative |scale| error vs. box size and sigma', 'scale_err.png'),
    ]:
        fig = plot_heatmap(
            data,
            x_labels,
            y_labels,
            title=title,
            xlabel='Sigma (pixels)',
            ylabel='Box size (pixels)',
            cbar_label='log10(error)',
            log_scale=True,
            mask=fail_mask,
        )
        save_figure(fig, study_dir, filename)

    # CSV
    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)

    # JSON summary: group by (box_size, sigma)
    groups = build_json_groups(specs, results)
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
    """Build JSON summary groups for Study 1, keyed by (box_size, sigma_y).

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
            ('box_size', lambda s: s.box_size),
            ('sigma', lambda s: s.sigma_y),
        ],
    )
