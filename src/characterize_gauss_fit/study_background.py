################################################################################
# characterize_gauss_fit/study_background.py
################################################################################

"""Study 6: Background conditions and modeling.

Explores how different injected background types and fitting model choices
interact to affect position, sigma, and scale recovery accuracy.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np

from characterize_gauss_fit import _study_utils as utils
from characterize_gauss_fit.config import Config, FittingConfig, config_to_dict
from characterize_gauss_fit.executor import run_trials
from characterize_gauss_fit.output import write_csv, write_json_summary
from characterize_gauss_fit.plotting import plot_heatmap, save_figure
from characterize_gauss_fit.trial import BACKGROUND_TYPE_NONE, TrialResult, TrialSpec

_LOG = logging.getLogger(__name__)
_STUDY_NAME = 'background'


def _bkgnd_fitting(base_fitting: FittingConfig, degree: int | None) -> FittingConfig:
    """Return a copy of ``base_fitting`` with ``bkgnd_degree`` overridden.

    Parameters:
        base_fitting: The base fitting configuration to copy from.
        degree: The background polynomial degree to set (``None`` disables).

    Returns:
        A new :class:`~config.FittingConfig` with the degree overridden.
    """
    import dataclasses

    return dataclasses.replace(base_fitting, bkgnd_degree=degree)


def build_specs(cfg: Config) -> tuple[list[TrialSpec], list[int | None]]:
    """Build trial specs for Study 6 and return the fitting-degree list.

    Iterates over background types, fitting degrees, and ignore-center sizes.
    For each combination that involves noise (``noisy_constant``), the base
    fitting configuration is used unchanged. For all others, background
    amplitude variants are included.

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A tuple ``(specs, fit_degrees)`` where ``fit_degrees`` is the list of
        ``bkgnd_degree`` values used per spec (same order as ``specs``).
    """
    study = cfg.studies.background

    # Build the list of fitting degrees to test.
    fit_degrees: list[int | None] = list(study.bkgnd_degrees)
    if study.bkgnd_degrees_with_null:
        fit_degrees = [None, *fit_degrees]

    specs: list[TrialSpec] = []
    spec_fit_degrees: list[int | None] = []
    seed = 6000

    for bkgnd_type in study.background_types:
        for amplitude in study.background_amplitudes:
            for fit_degree in fit_degrees:
                for ignore_center in study.bkgnd_ignore_centers:
                    fitting = _bkgnd_fitting(study.fitting, fit_degree)
                    import dataclasses

                    fitting = dataclasses.replace(
                        fitting,
                        bkgnd_ignore_center=ignore_center,
                        bkgnd_degree=fit_degree,
                    )
                    # For backgrounds that are 'none', amplitude doesn't matter
                    # but we still use it to keep the spec structure uniform.
                    inj_amplitude = 0.0 if bkgnd_type == BACKGROUND_TYPE_NONE else amplitude
                    specs.append(
                        utils.make_spec(
                            sigma_y=study.sigma[0],
                            sigma_x=study.sigma[1],
                            angle=0.0,
                            offset_y=study.offset[0],
                            offset_x=study.offset[1],
                            scale=cfg.generation.scale,
                            base=cfg.generation.base,
                            box_size=study.box_size,
                            fitting=fitting,
                            fit_angle=0.0,
                            background_type=bkgnd_type,
                            background_amplitude=inj_amplitude,
                            rng_seed=seed,
                        )
                    )
                    spec_fit_degrees.append(fit_degree)
                    seed += 1

    return specs, spec_fit_degrees


def run(cfg: Config, *, num_workers: int = 1) -> None:
    """Execute Study 6 and write all outputs.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        num_workers: Number of parallel worker processes.
    """
    study = cfg.studies.background
    if not study.enabled:
        _LOG.info('Study %s is disabled; skipping.', _STUDY_NAME)
        return

    specs, _fit_degrees = build_specs(cfg)
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
    """Write CSV, JSON, and PNG outputs for Study 6.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        specs: All trial specifications.
        results: All trial results.
        study_dir: Output subdirectory.
    """
    study = cfg.studies.background
    fit_degrees: list[int | None] = list(study.bkgnd_degrees)
    if study.bkgnd_degrees_with_null:
        fit_degrees = [None, *fit_degrees]

    bkgnd_types = study.background_types
    amplitudes = study.background_amplitudes
    ignore_centers = study.bkgnd_ignore_centers

    degree_labels = ['null' if d is None else str(d) for d in fit_degrees]
    type_labels = bkgnd_types

    # For each (amplitude, ignore_center) combination, create one heatmap panel.
    n_amp = len(amplitudes)
    n_ic = len(ignore_centers)

    for amp_idx, amplitude in enumerate(amplitudes):
        for ic_idx, ignore_center in enumerate(ignore_centers):
            grid = np.full((len(bkgnd_types), len(fit_degrees)), float('nan'))
            fail_mask = np.zeros_like(grid, dtype=bool)

            for bt_idx, bkgnd_type in enumerate(bkgnd_types):
                for fd_idx, fit_degree in enumerate(fit_degrees):
                    # Find the matching spec index.
                    inj_amplitude = 0.0 if bkgnd_type == BACKGROUND_TYPE_NONE else amplitude
                    for _i, (spec, result) in enumerate(zip(specs, results, strict=False)):
                        if (
                            spec.background_type == bkgnd_type
                            and abs(spec.background_amplitude - inj_amplitude) < 1e-12
                            and spec.bkgnd_degree == fit_degree
                            and spec.bkgnd_ignore_center == tuple(ignore_center)
                        ):
                            if not result.converged:
                                fail_mask[bt_idx, fd_idx] = True
                            else:
                                grid[bt_idx, fd_idx] = result.pos_err
                            break

            ic_str = f'{ignore_center[0]}x{ignore_center[1]}'
            fig = plot_heatmap(
                grid,
                degree_labels,
                type_labels,
                title=f'Position error -- amp={amplitude:.2f}, ignore={ic_str}',
                xlabel='Fitting bkgnd_degree',
                ylabel='Injected background type',
                cbar_label='log10(pos error)',
                log_scale=True,
                mask=fail_mask,
            )
            save_figure(
                fig,
                study_dir,
                f'pos_err_amp{amp_idx}_ic{ic_idx}.png',
            )

    _ = n_amp
    _ = n_ic

    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)

    groups: list[dict[str, Any]] = utils.build_groups_by_keys(
        specs,
        results,
        [
            ('background_type', lambda s: s.background_type),
            ('background_amplitude', lambda s: s.background_amplitude),
            ('bkgnd_degree', lambda s: s.bkgnd_degree),
            ('bkgnd_ignore_center', lambda s: s.bkgnd_ignore_center),
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
