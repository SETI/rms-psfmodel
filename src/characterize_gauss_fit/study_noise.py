################################################################################
# characterize_gauss_fit/study_noise.py
################################################################################

"""Study 7: Noise sensitivity (SNR sweep).

Sweeps a log-spaced range of signal-to-noise ratios and measures position,
sigma, and scale recovery accuracy as a function of SNR and PSF sigma.
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt

from characterize_gauss_fit import _study_utils as utils
from characterize_gauss_fit.config import Config, StudyNoiseSensitivityConfig, config_to_dict
from characterize_gauss_fit.executor import run_trials
from characterize_gauss_fit.output import write_csv, write_json_summary
from characterize_gauss_fit.plotting import plot_line_with_bands, save_figure
from characterize_gauss_fit.trial import TrialResult, TrialSpec

_LOG = logging.getLogger(__name__)
_STUDY_NAME = 'noise_sensitivity'


def _compute_snr_values(study: StudyNoiseSensitivityConfig) -> list[float]:
    """Compute the log-spaced SNR values for Study 7.

    Parameters:
        study: The study configuration.

    Returns:
        A list of SNR values in log-spaced order.
    """
    log_lo, log_hi = study.snr_log_range
    return list(np.logspace(log_lo, log_hi, study.snr_steps))


def build_specs(cfg: Config) -> list[TrialSpec]:
    """Build trial specs for Study 7.

    For each (SNR, sigma) combination, generates ``noise_samples`` trials
    with random offsets and different noise realisations.

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A list of :class:`~trial.TrialSpec` objects ordered by
        [snr, sigma, trial_idx].
    """
    study = cfg.studies.noise_sensitivity
    scale = cfg.generation.scale

    snr_values = _compute_snr_values(study)

    specs: list[TrialSpec] = []
    rng = np.random.default_rng(7000)
    seed_counter = 7000

    for snr in snr_values:
        noise_rms = utils.snr_to_noise_rms(snr, scale)
        for sigma in study.sigmas:
            for _ in range(study.noise_samples):
                # Randomise offset uniformly in [-0.5, 0.5].
                oy = float(rng.uniform(-0.5, 0.5))
                ox = float(rng.uniform(-0.5, 0.5))
                specs.append(
                    utils.make_spec(
                        sigma_y=sigma,
                        sigma_x=sigma,
                        angle=0.0,
                        offset_y=oy,
                        offset_x=ox,
                        scale=scale,
                        base=cfg.generation.base,
                        box_size=study.box_size,
                        fitting=study.fitting,
                        fit_angle=0.0,
                        noise_rms=noise_rms,
                        rng_seed=seed_counter,
                    )
                )
                seed_counter += 1

    return specs


def run(cfg: Config, *, num_workers: int = 1) -> None:
    """Execute Study 7 and write all outputs.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        num_workers: Number of parallel worker processes.
    """
    study = cfg.studies.noise_sensitivity
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
    """Write CSV, JSON, and PNG outputs for Study 7.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        specs: All trial specifications.
        results: All trial results.
        study_dir: Output subdirectory.
    """
    study = cfg.studies.noise_sensitivity
    scale = cfg.generation.scale

    snr_values = _compute_snr_values(study)
    n_snr = len(snr_values)
    n_sigma = len(study.sigmas)
    n_samples = study.noise_samples

    snr_arr = np.array(snr_values)
    trials_per_snr_sigma = n_samples

    def _collect_metric_grid(
        metric: str,
        *,
        abs_values: bool = False,
    ) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]:
        """Compute per-SNR mean and std for one metric, one array per sigma.

        Parameters:
            metric: Attribute name on :class:`~trial.TrialResult`.
            abs_values: If ``True``, apply ``abs`` to each sample before
                computing mean and std so that the error bands reflect the
                magnitude distribution rather than signed-value distribution.
        """
        # Defensive: check the spec ordering assumption holds.
        expected_total = n_snr * n_sigma * n_samples
        if len(results) != expected_total:
            raise RuntimeError(
                f'Expected {expected_total} results for noise_sensitivity but got {len(results)}'
            )
        all_means: list[npt.NDArray[np.float64]] = []
        all_stds: list[npt.NDArray[np.float64]] = []
        for s_idx in range(n_sigma):
            means: list[float] = []
            stds: list[float] = []
            for snr_idx in range(n_snr):
                start = snr_idx * n_sigma * n_samples + s_idx * n_samples
                bucket = results[start : start + trials_per_snr_sigma]
                arr = utils.collect_metric(bucket, metric)
                if abs_values:
                    arr = np.abs(arr)
                means.append(utils.safe_nanmean(arr))
                stds.append(utils.safe_nanstd(arr))
            all_means.append(np.array(means))
            all_stds.append(np.array(stds))
        return all_means, all_stds

    sigma_labels = [f'sigma={s:.1f}' for s in study.sigmas]

    for metric, ylabel, fname in [
        ('pos_err', 'Position error, Euclidean (pixels)', 'pos_err_vs_snr.png'),
        ('pos_err_y', '|pos_err_y| (pixels)', 'pos_err_y_vs_snr.png'),
        ('pos_err_x', '|pos_err_x| (pixels)', 'pos_err_x_vs_snr.png'),
        ('sigma_y_err', 'Relative |sigma_y| error', 'sigma_y_err_vs_snr.png'),
        ('sigma_x_err', 'Relative |sigma_x| error', 'sigma_x_err_vs_snr.png'),
        ('scale_err', 'Relative |scale| error', 'scale_err_vs_snr.png'),
    ]:
        # pos_err is a Euclidean distance (always non-negative); abs is not needed.
        means, stds = _collect_metric_grid(metric, abs_values=(metric != 'pos_err'))
        fig = plot_line_with_bands(
            snr_arr,
            means,
            stds,
            labels=sigma_labels,
            title=f'{ylabel} vs. SNR',
            xlabel='SNR (peak / noise RMS)',
            ylabel=ylabel,
            log_x=True,
            log_y=True,
            note=(
                'PSF: sigma_y = sigma_x = sigma (see series label); angle = 0\u00b0 (fixed);'
                f' box_size = {study.box_size} px; scale = {scale:.2g}\n'
                'Offset: Y and X each drawn independently from Uniform[\u22120.5, +0.5] px'
                ' per trial (different subpixel position every trial)\n'
                'Noise: Gaussian, noise_rms = scale / SNR (x-axis);'
                f' {study.noise_samples} independent trials per (sigma, SNR) point\n'
                'Fitting: sigma_y and sigma_x float freely; angle fixed at 0\u00b0;'
                ' no background subtraction'
            ),
        )
        save_figure(fig, study_dir, f'{_STUDY_NAME}_{fname}')

    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)

    groups: list[dict[str, Any]] = utils.build_groups_by_keys(
        specs,
        results,
        [
            ('snr', lambda s: round(scale / s.noise_rms, 1) if s.noise_rms > 0 else math.inf),
            ('sigma', lambda s: s.sigma_y),
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
