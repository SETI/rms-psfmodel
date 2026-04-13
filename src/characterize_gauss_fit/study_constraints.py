################################################################################
# characterize_gauss_fit/study_constraints.py
################################################################################

"""Study 5: Constraint modes (fixed vs floating PSF parameters).

Tests how fixing sigma or angle (correctly or with error) affects the accuracy
of position, scale, sigma, and angle recovery. All four metrics are reported.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import pathlib
from typing import Any

import numpy as np

from characterize_gauss_fit import _study_utils as utils
from characterize_gauss_fit.config import Config, config_to_dict
from characterize_gauss_fit.executor import run_trials
from characterize_gauss_fit.output import write_csv, write_json_summary
from characterize_gauss_fit.plotting import plot_constraint_summary, save_figure
from characterize_gauss_fit.trial import TrialResult, TrialSpec

_LOG = logging.getLogger(__name__)
_STUDY_NAME = 'constraint_modes'


@dataclasses.dataclass(frozen=True)
class ConstraintMode:
    """A single fitter constraint configuration for Study 5.

    Specifies which PSF parameters are fixed and what values they are fixed at.

    When ``sigma_is_fraction`` is ``True``, ``fit_sigma_y`` and ``fit_sigma_x``
    are interpreted as *error fractions* of the true sigma: ``0.0`` means fix
    at the true value, positive means fix at ``true_sigma * (1 + fraction)``.
    When ``sigma_is_fraction`` is ``False``, ``fit_sigma_y`` and ``fit_sigma_x``
    are actual sigma values (or ``None`` meaning float during fitting).

    For ``fit_angle``: ``None`` = float, ``0.0`` = fix at true angle, other
    value = fix at ``true_angle + fit_angle`` radians.
    """

    label: str
    fit_sigma_y: float | None
    fit_sigma_x: float | None
    fit_angle: float | None
    sigma_is_fraction: bool = False


def _build_modes(cfg: Config) -> list[ConstraintMode]:
    """Build the list of constraint modes from the study configuration.

    Modes cover the full spectrum from all-floating to all-fixed-with-error.

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A list of :class:`ConstraintMode` objects.
    """
    study = cfg.studies.constraint_modes
    modes: list[ConstraintMode] = []

    # All parameters float.
    modes.append(ConstraintMode('all_float', None, None, None))

    # Sigma fixed at true value (error_frac=0) and with each error fraction.
    # Angle is left to float in all sigma-fixed modes.
    for frac in study.sigma_error_fractions:
        if frac == 0.0:
            label = 'sigma_fixed_correct'
        else:
            label = f'sigma_fixed_err{int(frac * 100):d}pct'
        modes.append(ConstraintMode(label, frac, frac, None, sigma_is_fraction=True))

    # Angle fixed at true value, sigma floats.
    modes.append(ConstraintMode('angle_fixed_correct', None, None, 0.0))
    modes.append(
        ConstraintMode(
            'angle_fixed_error',
            None,
            None,
            study.angle_error_rad,
        )
    )

    # All fixed at correct values.
    modes.append(ConstraintMode('all_fixed_correct', 0.0, 0.0, 0.0, sigma_is_fraction=True))

    # All fixed with combined errors (only when sigma_error_fractions is non-empty).
    if study.sigma_error_fractions:
        max_frac = max(study.sigma_error_fractions)
        modes.append(
            ConstraintMode(
                'all_fixed_errors',
                max_frac,
                max_frac,
                study.angle_error_rad,
                sigma_is_fraction=True,
            )
        )

    return modes


def build_specs(cfg: Config) -> tuple[list[TrialSpec], list[ConstraintMode]]:
    """Build trial specs for Study 5 and return the mode list for reference.

    Each (mode, shape) combination produces one trial. The fixed sigma/angle
    values are computed from the true shape values and the mode's error fractions.

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A tuple of ``(specs, modes)`` where ``specs`` is the ordered trial list
        and ``modes`` is the constraint mode list used to construct them.
    """
    study = cfg.studies.constraint_modes
    modes = _build_modes(cfg)
    specs: list[TrialSpec] = []
    seed = 5000

    for mode in modes:
        for shape in study.psf_shapes:
            true_sigma_y, true_sigma_x = shape.sigma
            true_angle = shape.angle

            # Resolve fixed sigma values from mode fractions.
            if mode.fit_sigma_y is None:
                fit_sigma_y: float | None = None
            elif mode.sigma_is_fraction:
                fit_sigma_y = true_sigma_y * (1.0 + float(mode.fit_sigma_y))
            else:
                fit_sigma_y = mode.fit_sigma_y

            if mode.fit_sigma_x is None:
                fit_sigma_x: float | None = None
            elif mode.sigma_is_fraction:
                fit_sigma_x = true_sigma_x * (1.0 + float(mode.fit_sigma_x))
            else:
                fit_sigma_x = mode.fit_sigma_x

            # Resolve fixed angle values from mode.
            if mode.fit_angle is None:
                fit_angle: float | None = None
            elif mode.fit_angle == 0.0:
                fit_angle = true_angle
            else:
                fit_angle = true_angle + float(mode.fit_angle)

            specs.append(
                utils.make_spec(
                    sigma_y=true_sigma_y,
                    sigma_x=true_sigma_x,
                    angle=true_angle,
                    offset_y=study.offset[0],
                    offset_x=study.offset[1],
                    scale=study.scale,
                    base=cfg.generation.base,
                    box_size=study.box_size,
                    fitting=study.fitting,
                    fit_sigma_y=fit_sigma_y,
                    fit_sigma_x=fit_sigma_x,
                    fit_angle=fit_angle,
                    rng_seed=seed,
                )
            )
            seed += 1

    return specs, modes


def run(cfg: Config, *, num_workers: int = 1) -> None:
    """Execute Study 5 and write all outputs.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        num_workers: Number of parallel worker processes.
    """
    study = cfg.studies.constraint_modes
    if not study.enabled:
        _LOG.info('Study %s is disabled; skipping.', _STUDY_NAME)
        return

    specs, modes = build_specs(cfg)
    _LOG.info('Study %s: %d trials', _STUDY_NAME, len(specs))

    results = run_trials(
        specs,
        num_workers=num_workers,
        progress_callback=utils.progress_callback(_STUDY_NAME),
    )

    study_dir = utils.ensure_study_dir(cfg.output_dir, _STUDY_NAME)
    _write_outputs(cfg, specs, results, study_dir, modes)


def _write_outputs(
    cfg: Config,
    specs: list[TrialSpec],
    results: list[TrialResult],
    study_dir: pathlib.Path,
    modes: list[ConstraintMode],
) -> None:
    """Write CSV, JSON, and PNG outputs for Study 5.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        specs: All trial specifications.
        results: All trial results.
        study_dir: Output subdirectory.
        modes: Constraint modes used to generate specs.
    """
    study = cfg.studies.constraint_modes
    n_shapes = len(study.psf_shapes)
    n_modes = len(modes)

    shape_labels = [
        f's=({s.sigma[0]:.1f},{s.sigma[1]:.1f}),a={s.angle:.2f}' for s in study.psf_shapes
    ]
    mode_labels = [m.label for m in modes]

    # Build (n_modes, n_shapes) arrays for each metric.
    pos_err_vals = np.full((n_modes, n_shapes), float('nan'))
    pos_err_y_vals = np.full((n_modes, n_shapes), float('nan'))
    pos_err_x_vals = np.full((n_modes, n_shapes), float('nan'))
    scale_err_vals = np.full((n_modes, n_shapes), float('nan'))
    sigma_y_err_vals = np.full((n_modes, n_shapes), float('nan'))
    angle_err_vals = np.full((n_modes, n_shapes), float('nan'))

    for m_idx in range(n_modes):
        for s_idx in range(n_shapes):
            result = results[m_idx * n_shapes + s_idx]
            if result.converged:
                pos_err_vals[m_idx, s_idx] = result.pos_err
                pos_err_y_vals[m_idx, s_idx] = abs(result.pos_err_y)
                pos_err_x_vals[m_idx, s_idx] = abs(result.pos_err_x)
                if np.isfinite(result.scale_err):
                    scale_err_vals[m_idx, s_idx] = abs(result.scale_err)
                if result.sigma_y_err is not None:
                    sigma_y_err_vals[m_idx, s_idx] = abs(result.sigma_y_err)
                if result.angle_err is not None and np.isfinite(result.angle_err):
                    angle_err_vals[m_idx, s_idx] = math.degrees(result.angle_err)

    fig = plot_constraint_summary(
        mode_labels,
        shape_labels,
        pos_err_vals,
        pos_err_y_vals,
        pos_err_x_vals,
        scale_err_vals,
        sigma_y_err_vals,
        angle_err_vals,
        title='Effect of parameter constraints on fitting accuracy',
        note=(
            f'box={study.box_size}, '
            f'offset=({study.offset[0]:+.2f},{study.offset[1]:+.2f}), '
            f'noiseless, scale={study.scale:.0f}'
        ),
    )
    save_figure(fig, study_dir, f'{_STUDY_NAME}_summary.png')

    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)

    # Precompute position index for O(1) grouping lambda access.
    spec_index: dict[int, int] = {id(s): i for i, s in enumerate(specs)}
    groups: list[dict[str, Any]] = utils.build_groups_by_keys(
        specs,
        results,
        [
            ('constraint_mode_idx', lambda s: spec_index[id(s)] // n_shapes),
            ('psf_shape_idx', lambda s: spec_index[id(s)] % n_shapes),
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
