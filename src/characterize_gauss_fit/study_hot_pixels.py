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


def _make_num_sigma_list(
    num_sigma_with_null: bool, num_sigma_values: list[float]
) -> list[float | None]:
    """Build the ordered num_sigma list, prepending None when requested.

    Parameters:
        num_sigma_with_null: If ``True``, prepend ``None`` (no rejection).
        num_sigma_values: The configured threshold values.

    Returns:
        Ordered list of thresholds including ``None`` when requested.
    """
    result: list[float | None] = []
    if num_sigma_with_null:
        result.append(None)
    result.extend(num_sigma_values)
    return result


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
    noise_rms = utils.snr_to_noise_rms(study.snr, scale)

    # Build num_sigma list including null if requested.
    num_sigma_list = _make_num_sigma_list(study.num_sigma_with_null, study.num_sigma_values)

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


def _convergence_means(
    num_sigma_list: list[float | None],
    n_hot_list: list[int],
    hot_amp: float,
    bucket_map: dict[tuple[int, float | None, float], list[TrialResult]],
) -> list[npt.NDArray[np.float64]]:
    """Compute per-series convergence fractions for a given hot amplitude.

    Parameters:
        num_sigma_list: Ordered list of num_sigma thresholds tested.
        n_hot_list: Ordered list of hot-pixel counts tested.
        hot_amp: The hot-pixel amplitude being examined.
        bucket_map: Pre-built lookup mapping (n_hot, num_sigma, hot_amp)
            to a list of :class:`~trial.TrialResult` objects.

    Returns:
        A list of 1-D float64 arrays, one per ``num_sigma_list`` entry.
        Each element is the mean convergence fraction across the
        ``n_hot_list`` axis (1.0 = all converged, 0.0 = all failed).
    """
    series: list[npt.NDArray[np.float64]] = []
    for ns_val in num_sigma_list:
        fracs: list[float] = []
        for n_hot in n_hot_list:
            bucket = bucket_map.get((n_hot, ns_val, round(hot_amp, 9)), [])
            if len(bucket) == 0:
                fracs.append(float('nan'))
            else:
                fracs.append(float(np.mean([float(r.converged) for r in bucket])))
        series.append(np.array(fracs, dtype=np.float64))
    return series


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
    scale = cfg.generation.scale
    noise_rms_val = utils.snr_to_noise_rms(study.snr, scale)

    num_sigma_list = _make_num_sigma_list(study.num_sigma_with_null, study.num_sigma_values)

    n_hot_list = study.num_hot_pixels
    hot_amps = study.hot_amplitudes

    x_arr = np.array(n_hot_list, dtype=float)
    ns_labels = ['no_rejection' if ns is None else f'num_sigma={ns:.0f}' for ns in num_sigma_list]

    # Build O(1) lookup: (hot_pixel_count, num_sigma, hot_pixel_amplitude) -> bucket.
    bucket_map: dict[tuple[int, float | None, float], list[TrialResult]] = {}
    for spec, result in zip(specs, results, strict=True):
        key: tuple[int, float | None, float] = (
            spec.hot_pixel_count,
            spec.num_sigma,
            round(spec.hot_pixel_amplitude, 9),
        )
        if key not in bucket_map:
            bucket_map[key] = []
        bucket_map[key].append(result)

    for ha_idx, hot_amp in enumerate(hot_amps):
        plot_note = (
            f'PSF: sigma_y = {study.sigma[0]:.1f}, sigma_x = {study.sigma[1]:.1f} px (fixed);'
            f' angle = 0\u00b0 (fixed); box_size = {study.box_size} px; scale = {scale:.2g}\n'
            f'Offset: Y = {study.offset[0]:+.2f}, X = {study.offset[1]:+.2f} px from pixel'
            ' centre (fixed for all trials)\n'
            f'Noise: Gaussian, noise_rms = {noise_rms_val:.3g} (SNR = {study.snr:.0f});'
            f' {study.noise_samples} independent trials per condition;'
            ' hot-pixel positions randomized per trial\n'
            'Fitting: sigma_y and sigma_x float freely; angle fixed at 0\u00b0;'
            ' num_sigma rejection = series label (see legend);'
            f' hot-pixel amplitude = {hot_amp:.0f}\u00d7 PSF peak (see title)'
        )
        # --- Convergence-rate plot -------------------------------------------
        conv_means = _convergence_means(num_sigma_list, n_hot_list, hot_amp, bucket_map)
        conv_stds = [np.zeros_like(m) for m in conv_means]  # deterministic fraction
        fig = plot_line_with_bands(
            x_arr,
            conv_means,
            conv_stds,
            labels=ns_labels,
            title=f'Convergence fraction vs. hot pixels -- amplitude={hot_amp:.0f}x peak',
            xlabel='Number of hot pixels',
            ylabel='Fraction of trials converged',
            log_y=False,
            note=plot_note,
        )
        save_figure(
            fig,
            study_dir,
            f'{_STUDY_NAME}_convergence_hotamp{ha_idx}.png',
        )

        # --- Position-error plots --------------------------------------------
        for metric_attr, metric_label, fname_prefix in [
            ('pos_err', 'Mean position error, Euclidean (pixels)', 'pos_err'),
            ('pos_err_y', 'Mean |pos_err_y| (pixels)', 'pos_err_y'),
            ('pos_err_x', 'Mean |pos_err_x| (pixels)', 'pos_err_x'),
        ]:
            y_means: list[npt.NDArray[np.float64]] = []
            y_stds: list[npt.NDArray[np.float64]] = []
            for ns_val, _ns_label in zip(num_sigma_list, ns_labels, strict=True):
                means: list[float] = []
                stds: list[float] = []
                for n_hot_count in n_hot_list:
                    bucket = bucket_map.get((n_hot_count, ns_val, round(hot_amp, 9)), [])
                    arr = np.abs(utils.collect_metric(bucket, metric_attr))
                    means.append(utils.safe_nanmean(arr))
                    stds.append(utils.safe_nanstd(arr))
                y_means.append(np.array(means))
                y_stds.append(np.array(stds))

            fig = plot_line_with_bands(
                x_arr,
                y_means,
                y_stds,
                labels=ns_labels,
                title=(
                    f'{metric_label} vs. hot pixels -- amplitude={hot_amp:.0f}x peak'
                    '\n(missing lines = 0 % convergence; see companion convergence plot)'
                ),
                xlabel='Number of hot pixels',
                ylabel=metric_label,
                log_y=True,
                note=plot_note,
            )
            save_figure(fig, study_dir, f'{_STUDY_NAME}_{fname_prefix}_hotamp{ha_idx}.png')

    write_csv(cfg.output_dir, _STUDY_NAME, specs=specs, results=results)

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
        specs=specs,
        results=results,
        groups=groups,
        config_used=config_to_dict(cfg),
    )
    _LOG.info('Study %s outputs written to %s', _STUDY_NAME, study_dir)
