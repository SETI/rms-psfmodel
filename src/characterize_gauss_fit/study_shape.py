################################################################################
# characterize_gauss_fit/study_shape.py
################################################################################

"""Study 4: Sigma asymmetry and angle recovery.

Explores how well elongated (asymmetric) and rotated PSFs are recovered across
a grid of sigma ratios and rotation angles.
"""

from __future__ import annotations

import logging
import math
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
_STUDY_NAME = 'sigma_asymmetry_angle'

# When sigma_ratio is within this tolerance of 1.0, the PSF is effectively
# circular and angle is degenerate. Angle error is not meaningful in this case.
_CIRCULAR_TOLERANCE = 1e-3


def build_specs(cfg: Config) -> list[TrialSpec]:
    """Build trial specs for Study 4.

    For each (sigma_ratio, angle, sigma_x) combination, creates a spec with
    all PSF parameters left to float.

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A list of :class:`~trial.TrialSpec` objects ordered by
        [sigma_x, sigma_ratio, angle].
    """
    study = cfg.studies.sigma_asymmetry_angle
    angles = np.linspace(0.0, math.pi, study.angle_steps)
    specs: list[TrialSpec] = []
    seed = 4000
    for sigma_x in study.sigma_x_values:
        for ratio in study.sigma_ratios:
            sigma_y = sigma_x * ratio
            for angle in angles:
                specs.append(
                    utils.make_spec(
                        sigma_y=sigma_y,
                        sigma_x=sigma_x,
                        angle=float(angle),
                        offset_y=study.offset[0],
                        offset_x=study.offset[1],
                        scale=cfg.generation.scale,
                        base=cfg.generation.base,
                        box_size=study.box_size,
                        fitting=study.fitting,
                        # All parameters float.
                        rng_seed=seed,
                    )
                )
                seed += 1
    return specs


def run(cfg: Config, *, num_workers: int = 1) -> None:
    """Execute Study 4 and write all outputs.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        num_workers: Number of parallel worker processes.
    """
    study = cfg.studies.sigma_asymmetry_angle
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
    """Write CSV, JSON, and PNG outputs for Study 4.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        specs: All trial specifications.
        results: All trial results.
        study_dir: Output subdirectory.
    """
    study = cfg.studies.sigma_asymmetry_angle
    scale = cfg.generation.scale
    angles = list(np.linspace(0.0, math.pi, study.angle_steps))
    n_angles = len(angles)
    n_ratios = len(study.sigma_ratios)
    n_sigma_x = len(study.sigma_x_values)

    angle_labels = [f'{math.degrees(a):.0f}\u00b0' for a in angles]
    ratio_labels = [f'{r:.2f}' for r in study.sigma_ratios]

    trials_per_sigma_x = n_ratios * n_angles

    for sx_idx, sigma_x in enumerate(study.sigma_x_values):
        base_idx = sx_idx * trials_per_sigma_x
        pos_err_grid = np.full((n_ratios, n_angles), float('nan'))
        pos_err_y_grid = np.full((n_ratios, n_angles), float('nan'))
        pos_err_x_grid = np.full((n_ratios, n_angles), float('nan'))
        angle_err_grid = np.full((n_ratios, n_angles), float('nan'))
        sigma_y_err_grid = np.full((n_ratios, n_angles), float('nan'))
        fail_mask = np.zeros((n_ratios, n_angles), dtype=bool)

        for r_idx, ratio in enumerate(study.sigma_ratios):
            is_circular = abs(ratio - 1.0) < _CIRCULAR_TOLERANCE
            for a_idx in range(n_angles):
                result = results[base_idx + r_idx * n_angles + a_idx]
                if not result.converged:
                    fail_mask[r_idx, a_idx] = True
                    continue
                pos_err_grid[r_idx, a_idx] = result.pos_err
                pos_err_y_grid[r_idx, a_idx] = abs(result.pos_err_y)
                pos_err_x_grid[r_idx, a_idx] = abs(result.pos_err_x)
                if result.sigma_y_err is not None:
                    sigma_y_err_grid[r_idx, a_idx] = abs(result.sigma_y_err)
                if not is_circular and result.angle_err is not None:
                    err_deg = math.degrees(result.angle_err) % 180.0
                    # Reduce modulo 90° to account for the sigma-swap equivalence:
                    # (sigma_y, sigma_x, theta) == (sigma_x, sigma_y, theta+90°).
                    # When both sigmas float the optimizer may land on either form,
                    # making the raw error jump to ~90° for a geometrically perfect
                    # fit.  min(e, 180-e) maps the [0,180) range into [0,90] as it
                    # should be, and the modulo ensures negative or >180 values are
                    # handled correctly before the min.
                    angle_err_grid[r_idx, a_idx] = min(err_deg, 180.0 - err_deg)

        label = f'sigma_x={sigma_x:.1f}'
        for data, metric_title, fname_prefix, cbar in [
            (pos_err_grid, f'Position error (Euclidean) -- {label}', 'pos_err', 'log10(pos error)'),
            (pos_err_y_grid, f'|pos_err_y| -- {label}', 'pos_err_y', 'log10(|pos_err_y|)'),
            (pos_err_x_grid, f'|pos_err_x| -- {label}', 'pos_err_x', 'log10(|pos_err_x|)'),
            (
                angle_err_grid,
                f'Angle error (\u00b0, mod 90\u00b0) -- {label}',
                'angle_err',
                'Angle error (\u00b0, mod 90\u00b0)',
            ),
            (sigma_y_err_grid, f'Rel sigma_y error -- {label}', 'sigma_y_err', 'log10(rel error)'),
        ]:
            use_log = 'pos' in fname_prefix or 'sigma' in fname_prefix
            fig = plot_heatmap(
                data,
                angle_labels,
                ratio_labels,
                title=metric_title,
                xlabel='Angle (degrees)',
                ylabel='sigma_y / sigma_x ratio',
                cbar_label=cbar,
                log_scale=use_log,
                mask=fail_mask,
                note=(
                    f'PSF: sigma_x = {sigma_x:.2g} px (this panel); sigma_y = ratio \u00d7 sigma_x'
                    f' (y-axis); angle rotates 0\u2013180\u00b0 (x-axis); scale = {scale:.2g};'
                    f' one noiseless trial per cell\n'
                    f'Offset: Y = {study.offset[0]:+.2f}, X = {study.offset[1]:+.2f} px'
                    f' from pixel centre (fixed for all cells)\n'
                    f'Background / noise: none injected\n'
                    f'Fitting: sigma_y, sigma_x, and angle ALL float freely;'
                    f' no background subtraction\n'
                    f'Angle error is reduced mod 90\u00b0: (sigma_y,sigma_x,\u03b8) \u2261'
                    f' (sigma_x,sigma_y,\u03b8+90\u00b0) so raw error \u2248 90\u00b0 means'
                    f' a perfect fit in the swapped-sigma form'
                ),
            )
            save_figure(fig, study_dir, f'{_STUDY_NAME}_{fname_prefix}_sx{sigma_x:.1f}.png')

    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)

    groups: list[dict[str, Any]] = utils.build_groups_by_keys(
        specs,
        results,
        [
            ('sigma_x', lambda s: s.sigma_x),
            (
                'sigma_ratio',
                lambda s: round(s.sigma_y / s.sigma_x, 4) if s.sigma_x != 0.0 else None,
            ),
            ('angle_true', lambda s: round(s.angle, 4)),
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
    _LOG.info(
        'Study %s outputs written to %s (%d sigma_x panels)',
        _STUDY_NAME,
        study_dir,
        n_sigma_x,
    )
