################################################################################
# characterize_gauss_fit/study_background.py
################################################################################

"""Study 6: Background conditions and modeling.

Explores how different injected background types and fitting model choices
interact to affect position, sigma, and scale recovery accuracy.  The study
is repeated for each configured subpixel offset so that offset-sensitivity can
be compared visually alongside background effects.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any

import numpy as np

from characterize_gauss_fit import _study_utils as utils
from characterize_gauss_fit.config import (
    Config,
    StudyBackgroundConfig,
    config_to_dict,
)
from characterize_gauss_fit.executor import run_trials
from characterize_gauss_fit.output import write_csv, write_json_summary
from characterize_gauss_fit.plotting import plot_heatmap, save_figure
from characterize_gauss_fit.trial import BACKGROUND_TYPE_NONE, TrialResult, TrialSpec

_LOG = logging.getLogger(__name__)
_STUDY_NAME = 'background'

# Type alias for the lookup dict key used in _write_outputs.
_BkgndKey = tuple[float, float, str, float, int | None, tuple[int, int]]


def _compute_fit_degrees(study: StudyBackgroundConfig) -> list[int | None]:
    """Build the ordered list of fitting-degree values for Study 6.

    Parameters:
        study: The study configuration.

    Returns:
        The list of ``bkgnd_degree`` values to iterate over, with ``None``
        prepended when ``bkgnd_degrees_with_null`` is enabled.
    """
    degrees: list[int | None] = list(study.bkgnd_degrees)
    if study.bkgnd_degrees_with_null:
        degrees = [None, *degrees]
    return degrees


def build_specs(cfg: Config) -> tuple[list[TrialSpec], list[int | None]]:
    """Build trial specs for Study 6 and return the fitting-degree list.

    Iterates over offsets, background types, fitting degrees, and ignore-center
    sizes.  Specs are ordered by offset first so that :func:`_write_outputs`
    can slice them by offset.

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A tuple ``(specs, fit_degrees)`` where ``fit_degrees`` is the list of
        ``bkgnd_degree`` values used per spec (same order as ``specs``).
    """
    study = cfg.studies.background

    fit_degrees = _compute_fit_degrees(study)

    specs: list[TrialSpec] = []
    spec_fit_degrees: list[int | None] = []
    seed = 6000

    for offset_y, offset_x in study.offsets:
        for bkgnd_type in study.background_types:
            for amplitude in study.background_amplitudes:
                for fit_degree in fit_degrees:
                    for ignore_center in study.bkgnd_ignore_centers:
                        fitting = dataclasses.replace(
                            study.fitting,
                            bkgnd_degree=fit_degree,
                            bkgnd_ignore_center=ignore_center,
                        )
                        inj_amplitude = 0.0 if bkgnd_type == BACKGROUND_TYPE_NONE else amplitude
                        specs.append(
                            utils.make_spec(
                                sigma_y=study.sigma[0],
                                sigma_x=study.sigma[1],
                                angle=0.0,
                                offset_y=offset_y,
                                offset_x=offset_x,
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

    One set of heatmaps is produced for each configured (offset, amplitude,
    ignore_center) combination.  Each filename includes the offset tag so files
    from different offsets do not collide.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        specs: All trial specifications.
        results: All trial results.
        study_dir: Output subdirectory.
    """
    study = cfg.studies.background
    scale = cfg.generation.scale
    fit_degrees = _compute_fit_degrees(study)

    bkgnd_types = study.background_types
    amplitudes = study.background_amplitudes
    ignore_centers = study.bkgnd_ignore_centers

    degree_labels = ['null' if d is None else str(d) for d in fit_degrees]
    type_labels = bkgnd_types

    # Build O(1) lookup: (offset_y, offset_x, bkgnd_type, amplitude,
    #                     fit_degree, ignore_center) -> TrialResult.
    lookup: dict[_BkgndKey, TrialResult] = {}
    for spec, result in zip(specs, results, strict=True):
        key: _BkgndKey = (
            spec.offset_y,
            spec.offset_x,
            spec.background_type,
            spec.background_amplitude,
            spec.bkgnd_degree,
            spec.bkgnd_ignore_center,
        )
        lookup[key] = result

    for offset_y, offset_x in study.offsets:
        tag = utils.offset_tag(offset_y, offset_x)
        offset_str = f'offset ({offset_y:+.2f}, {offset_x:+.2f})'

        for amp_idx, amplitude in enumerate(amplitudes):
            for ic_idx, ignore_center in enumerate(ignore_centers):
                grid = np.full((len(bkgnd_types), len(fit_degrees)), float('nan'))
                grid_y = np.full((len(bkgnd_types), len(fit_degrees)), float('nan'))
                grid_x = np.full((len(bkgnd_types), len(fit_degrees)), float('nan'))
                fail_mask = np.zeros_like(grid, dtype=bool)

                for bt_idx, bkgnd_type in enumerate(bkgnd_types):
                    inj_amplitude = 0.0 if bkgnd_type == BACKGROUND_TYPE_NONE else amplitude
                    for fd_idx, fit_degree in enumerate(fit_degrees):
                        ic_key: tuple[int, int] = (ignore_center[0], ignore_center[1])
                        found: TrialResult | None = lookup.get(
                            (offset_y, offset_x, bkgnd_type, inj_amplitude, fit_degree, ic_key)
                        )
                        if found is None:
                            continue
                        if not found.converged:
                            fail_mask[bt_idx, fd_idx] = True
                        else:
                            grid[bt_idx, fd_idx] = found.pos_err
                            grid_y[bt_idx, fd_idx] = abs(found.pos_err_y)
                            grid_x[bt_idx, fd_idx] = abs(found.pos_err_x)

                ic_str = f'{ignore_center[0]}x{ignore_center[1]}'
                bkgnd_note = (
                    f'PSF: sigma_y = {study.sigma[0]:.1f}, sigma_x = {study.sigma[1]:.1f} px'
                    f' (fixed); angle = 0\u00b0 (fixed); box_size = {study.box_size} px;'
                    f' scale = {scale:.2g}; one noiseless trial per cell\n'
                    f'Offset: Y = {offset_y:+.2f}, X = {offset_x:+.2f} px from pixel centre'
                    f' (fixed; one heatmap set per offset pair)\n'
                    f'Background: type on y-axis; amplitude = {amplitude:.2g} \u00d7 PSF peak'
                    f' (see title); no Gaussian detector noise added\n'
                    f'Fitting: sigma_y and sigma_x float freely; angle fixed at 0\u00b0;'
                    f' bkgnd_degree on x-axis (null = no subtraction);'
                    f' bkgnd_ignore_center = {ic_str} (see title)'
                )
                for hmap, metric_label, fsuffix in [
                    (grid, 'Position error (Euclidean)', ''),
                    (grid_y, '|pos_err_y|', '_y'),
                    (grid_x, '|pos_err_x|', '_x'),
                ]:
                    fig = plot_heatmap(
                        hmap,
                        degree_labels,
                        type_labels,
                        title=(
                            f'{metric_label} -- amp={amplitude:.2f}, '
                            f'ignore={ic_str}  [{offset_str}]'
                        ),
                        xlabel='Fitting bkgnd_degree',
                        ylabel='Injected background type',
                        cbar_label='log10(pos error)',
                        log_scale=True,
                        mask=fail_mask,
                        note=bkgnd_note,
                    )
                    save_figure(
                        fig,
                        study_dir,
                        f'{_STUDY_NAME}_pos_err{fsuffix}_amp{amp_idx}_ic{ic_idx}_{tag}.png',
                    )

    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)

    groups: list[dict[str, Any]] = utils.build_groups_by_keys(
        specs,
        results,
        [
            ('offset_y', lambda s: s.offset_y),
            ('offset_x', lambda s: s.offset_x),
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
