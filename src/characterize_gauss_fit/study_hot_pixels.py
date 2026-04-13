################################################################################
# characterize_gauss_fit/study_hot_pixels.py
################################################################################

"""Study 8: Hot pixel rejection.

Varies the number, amplitude, and sigma-rejection threshold for hot pixels to
measure how effectively the PSF fitter's bad-pixel rejection works.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt

from characterize_gauss_fit import _study_utils as utils
from characterize_gauss_fit.config import Config, config_to_dict
from characterize_gauss_fit.executor import run_trials
from characterize_gauss_fit.output import write_csv, write_json_summary
from characterize_gauss_fit.plotting import plot_line_with_bands, save_figure
from characterize_gauss_fit.trial import TrialResult, TrialSpec

_LOG = logging.getLogger(__name__)
_STUDY_NAME = 'hot_pixel_rejection'


def build_specs(cfg: Config) -> list[TrialSpec]:
    """Build trial specs for Study 8.

    Iterates over (num_hot_pixels, num_sigma, hot_amplitude) combinations.
    For each, generates ``noise_samples`` trials with randomised hot-pixel
    positions (different RNG seeds).

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A list of :class:`~trial.TrialSpec` objects ordered by
        [num_hot, num_sigma_val, hot_amp, trial_idx].
    """
    study = cfg.studies.hot_pixel_rejection
    scale = cfg.generation.scale
    noise_rms = scale / study.snr

    # Build num_sigma list including null if requested.
    num_sigma_list: list[float | None] = []
    if study.num_sigma_with_null:
        num_sigma_list.append(None)
    num_sigma_list.extend(study.num_sigma_values)

    specs: list[TrialSpec] = []
    seed = 8000

    for n_hot in study.num_hot_pixels:
        for ns_val in num_sigma_list:
            for hot_amp in study.hot_amplitudes:
                fitting = dataclasses.replace(study.fitting, num_sigma=ns_val)
                for _ in range(study.noise_samples):
                    specs.append(
                        utils.make_spec(
                            sigma_y=study.sigma[0],
                            sigma_x=study.sigma[1],
                            angle=0.0,
                            offset_y=study.offset[0],
                            offset_x=study.offset[1],
                            scale=scale,
                            base=cfg.generation.base,
                            box_size=study.box_size,
                            fitting=fitting,
                            fit_angle=0.0,
                            noise_rms=noise_rms,
                            hot_pixel_count=n_hot,
                            hot_pixel_amplitude=hot_amp,
                            rng_seed=seed,
                        )
                    )
                    seed += 1

    return specs


def run(cfg: Config, *, num_workers: int = 1) -> None:
    """Execute Study 8 and write all outputs.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        num_workers: Number of parallel worker processes.
    """
    study = cfg.studies.hot_pixel_rejection
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
    """Write CSV, JSON, and PNG outputs for Study 8.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        specs: All trial specifications.
        results: All trial results.
        study_dir: Output subdirectory.
    """
    study = cfg.studies.hot_pixel_rejection

    num_sigma_list: list[float | None] = []
    if study.num_sigma_with_null:
        num_sigma_list.append(None)
    num_sigma_list.extend(study.num_sigma_values)

    n_hot_list = study.num_hot_pixels
    hot_amps = study.hot_amplitudes

    x_arr = np.array(n_hot_list, dtype=float)
    x_labels = [str(n) for n in n_hot_list]
    ns_labels = ['no_rejection' if ns is None else f'num_sigma={ns:.0f}' for ns in num_sigma_list]

    n_ns = len(num_sigma_list)
    n_hot = len(n_hot_list)

    for ha_idx, hot_amp in enumerate(hot_amps):
        y_means: list[npt.NDArray[np.float64]] = []
        y_stds: list[npt.NDArray[np.float64]] = []

        for ns_val in num_sigma_list:
            means: list[float] = []
            stds: list[float] = []
            for n_hot_count in n_hot_list:
                # Find matching results.
                bucket: list[TrialResult] = []
                for spec, result in zip(specs, results, strict=False):
                    if (
                        spec.hot_pixel_count == n_hot_count
                        and spec.num_sigma == ns_val
                        and abs(spec.hot_pixel_amplitude - hot_amp) < 1e-9
                    ):
                        bucket.append(result)
                arr = utils.collect_metric(bucket, 'pos_err')
                means.append(utils.safe_nanmean(arr))
                stds.append(utils.safe_nanstd(arr))
            y_means.append(np.array(means))
            y_stds.append(np.array(stds))

        fig = plot_line_with_bands(
            x_arr,
            y_means,
            y_stds,
            labels=ns_labels,
            title=f'Position error vs. hot pixels -- amplitude={hot_amp:.0f}x peak',
            xlabel='Number of hot pixels',
            ylabel='Mean position error (pixels)',
            log_y=True,
        )
        save_figure(fig, study_dir, f'pos_err_hotamp{ha_idx}.png')

    _ = n_hot
    _ = n_ns
    _ = x_labels

    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)

    groups: list[dict[str, Any]] = utils.build_groups_by_keys(
        specs,
        results,
        [
            ('num_hot_pixels', lambda s: s.hot_pixel_count),
            ('num_sigma', lambda s: s.num_sigma),
            ('hot_amplitude', lambda s: s.hot_pixel_amplitude),
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
