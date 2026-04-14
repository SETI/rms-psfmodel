################################################################################
# characterize_gauss_fit/study_box_sigma.py
################################################################################

"""Study 1: Box size vs. PSF sigma.

Explores how the subimage size relative to the PSF width affects position,
sigma, and scale recovery accuracy.  The study is repeated for each configured
subpixel offset so that offset-sensitivity can be compared visually.
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

    Each spec represents one (offset, box_size, sigma) combination with no
    background or noise, with sigma left to float during fitting.  Specs are
    ordered by offset first, then box_size, then sigma so that
    :func:`_write_outputs` can slice them by offset slice.

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A list of :class:`~trial.TrialSpec` objects.
    """
    study = cfg.studies.box_vs_sigma
    specs: list[TrialSpec] = []
    seed = 1000
    for offset_y, offset_x in study.offsets:
        for box_size in study.box_sizes:
            for sigma in study.sigmas:
                specs.append(
                    utils.make_spec(
                        sigma_y=sigma,
                        sigma_x=sigma,
                        angle=study.angle,
                        offset_y=offset_y,
                        offset_x=offset_x,
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

    One set of six heatmaps is produced for each configured offset pair.
    Each filename includes the offset tag so files from different offsets do
    not collide.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        specs: All trial specifications.
        results: All trial results (same order as ``specs``).
        study_dir: Output subdirectory for this study.
    """
    study = cfg.studies.box_vs_sigma
    box_sizes = study.box_sizes
    sigmas = study.sigmas
    offsets = study.offsets

    n_box = len(box_sizes)
    n_sigma = len(sigmas)
    n_per_offset = n_box * n_sigma

    x_labels = [f'{s:.2g}' for s in sigmas]
    y_labels = [str(b) for b in box_sizes]

    for off_idx, (offset_y, offset_x) in enumerate(offsets):
        tag = utils.offset_tag(offset_y, offset_x)
        slice_results = results[off_idx * n_per_offset : (off_idx + 1) * n_per_offset]

        pos_err_grid = np.full((n_box, n_sigma), float('nan'))
        pos_err_y_grid = np.full((n_box, n_sigma), float('nan'))
        pos_err_x_grid = np.full((n_box, n_sigma), float('nan'))
        sigma_y_err_grid = np.full((n_box, n_sigma), float('nan'))
        sigma_x_err_grid = np.full((n_box, n_sigma), float('nan'))
        scale_err_grid = np.full((n_box, n_sigma), float('nan'))
        fail_mask = np.zeros((n_box, n_sigma), dtype=bool)

        idx = 0
        for b_idx in range(n_box):
            for s_idx in range(n_sigma):
                r = slice_results[idx]
                if not r.converged:
                    fail_mask[b_idx, s_idx] = True
                else:
                    pos_err_grid[b_idx, s_idx] = r.pos_err
                    pos_err_y_grid[b_idx, s_idx] = abs(r.pos_err_y)
                    pos_err_x_grid[b_idx, s_idx] = abs(r.pos_err_x)
                    if r.sigma_y_err is not None and np.isfinite(r.sigma_y_err):
                        sigma_y_err_grid[b_idx, s_idx] = abs(r.sigma_y_err)
                    if r.sigma_x_err is not None and np.isfinite(r.sigma_x_err):
                        sigma_x_err_grid[b_idx, s_idx] = abs(r.sigma_x_err)
                    if np.isfinite(r.scale_err):
                        scale_err_grid[b_idx, s_idx] = abs(r.scale_err)
                idx += 1

        offset_str = f'offset ({offset_y:+.2f}, {offset_x:+.2f})'
        plot_note = (
            f'PSF: sigma_y = sigma_x = sigma (x-axis); angle = {study.angle:.0f}\u00b0 (fixed,'
            f' axis-aligned); scale = {study.scale:.2g}; one noiseless trial per cell\n'
            f'Offset: Y = {offset_y:+.2f}, X = {offset_x:+.2f} px from pixel centre'
            f' (fixed; one heatmap produced per offset pair)\n'
            f'Background / noise: none injected; image is clean Gaussian pixel integrals only\n'
            f'Fitting: sigma_y and sigma_x float freely; angle fixed at 0\u00b0;'
            f' no background subtraction'
        )
        for data, metric_title, metric_key in [
            (pos_err_grid,   'Position error (Euclidean)',    'pos_err'),
            (pos_err_y_grid, '|pos_err_y|',                  'pos_err_y'),
            (pos_err_x_grid, '|pos_err_x|',                  'pos_err_x'),
            (sigma_y_err_grid, 'Relative |sigma_y| error',   'sigma_y_err'),
            (sigma_x_err_grid, 'Relative |sigma_x| error',   'sigma_x_err'),
            (scale_err_grid, 'Relative |scale| error',       'scale_err'),
        ]:
            fig = plot_heatmap(
                data,
                x_labels,
                y_labels,
                title=f'{metric_title} vs. box size and sigma  [{offset_str}]',
                xlabel='Sigma (pixels)',
                ylabel='Box size (pixels)',
                cbar_label='log10(error)',
                log_scale=True,
                mask=fail_mask,
                note=plot_note,
            )
            save_figure(fig, study_dir, f'{_STUDY_NAME}_{metric_key}_{tag}.png')

    # CSV and JSON (all offsets together)
    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)
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
    """Build JSON summary groups for Study 1, keyed by (offset_y, offset_x, box_size, sigma_y).

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
            ('offset_y', lambda s: s.offset_y),
            ('offset_x', lambda s: s.offset_x),
            ('box_size', lambda s: s.box_size),
            ('sigma',    lambda s: s.sigma_y),
        ],
    )
