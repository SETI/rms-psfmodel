################################################################################
# characterize_gauss_fit/study_min_offset.py
################################################################################

"""Study 3: Minimum detectable position offset.

Measures how small a sub-pixel offset delta can be reliably recovered as a
function of PSF sigma and noise level. Reports both mean position error and a
recovery fraction metric.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt

from characterize_gauss_fit import _study_utils as utils
from characterize_gauss_fit.config import Config, StudyMinDetectableOffsetConfig, config_to_dict
from characterize_gauss_fit.executor import run_trials
from characterize_gauss_fit.output import write_csv, write_json_summary
from characterize_gauss_fit.plotting import (
    plot_line_with_bands,
    plot_recovery_fraction_heatmap,
    save_figure,
)
from characterize_gauss_fit.trial import TrialResult, TrialSpec

_LOG = logging.getLogger(__name__)
_STUDY_NAME = 'min_detectable_offset'

# A string sentinel used in CSV/JSON to label the noiseless condition.
_NOISELESS_LABEL = 'noiseless'


def _build_conditions(
    study: StudyMinDetectableOffsetConfig, scale: float
) -> list[tuple[float, str]]:
    """Build the list of noise conditions for Study 3.

    Parameters:
        study: The study configuration.
        scale: PSF amplitude scale factor.

    Returns:
        List of ``(noise_rms, label)`` pairs ordered as noiseless-first then
        per-SNR.
    """
    conditions: list[tuple[float, str]] = []
    if study.include_noiseless:
        conditions.append((0.0, _NOISELESS_LABEL))
    for snr_val in study.snr_values:
        conditions.append((utils.snr_to_noise_rms(snr_val, scale), f'snr_{snr_val:.0f}'))
    return conditions


def build_specs(cfg: Config) -> list[TrialSpec]:
    """Build trial specs for Study 3.

    For each (delta, sigma) pair and each noise condition (noiseless + each
    SNR), generates ``noise_samples`` trials with different random seeds
    (1 trial for noiseless).

    Parameters:
        cfg: The active :class:`~config.Config` instance.

    Returns:
        A flat list of :class:`~trial.TrialSpec` objects.
    """
    study = cfg.studies.min_detectable_offset
    scale = cfg.generation.scale
    specs: list[TrialSpec] = []
    seed = 3000

    # Noise conditions: (noise_rms, snr_label) pairs.
    conditions = _build_conditions(study, scale)

    for delta in study.delta_offsets:
        for sigma in study.sigmas:
            for noise_rms, _snr_label in conditions:
                n_trials = 1 if noise_rms == 0.0 else study.noise_samples
                for _ in range(n_trials):
                    specs.append(
                        utils.make_spec(
                            sigma_y=sigma,
                            sigma_x=sigma,
                            angle=0.0,
                            # Inject the offset purely in X for simplicity.
                            offset_y=0.0,
                            offset_x=delta,
                            scale=scale,
                            base=cfg.generation.base,
                            box_size=study.box_size,
                            fitting=study.fitting,
                            fit_angle=0.0,
                            noise_rms=noise_rms,
                            rng_seed=seed,
                        )
                    )
                    seed += 1

    return specs


def run(cfg: Config, *, num_workers: int = 1) -> None:
    """Execute Study 3 and write all outputs.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        num_workers: Number of parallel worker processes.
    """
    study = cfg.studies.min_detectable_offset
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
    """Write CSV, JSON, and PNG outputs for Study 3.

    Parameters:
        cfg: The active :class:`~config.Config` instance.
        specs: All trial specifications.
        results: All trial results.
        study_dir: Output subdirectory.
    """
    study = cfg.studies.min_detectable_offset
    scale = cfg.generation.scale
    deltas = study.delta_offsets
    sigmas = study.sigmas

    conditions = _build_conditions(study, scale)

    # Map (delta, sigma, condition_label) -> list of TrialResult.
    # Slicing by index is safe here because build_specs generates specs in the
    # same deterministic order (delta, sigma, condition, trial_idx) and
    # run_trials preserves spec ordering.
    result_map: dict[tuple[float, float, str], list[TrialResult]] = {}
    idx = 0
    for delta in deltas:
        for sigma in sigmas:
            for noise_rms, cond_label in conditions:
                n_trials = 1 if noise_rms == 0.0 else study.noise_samples
                bucket_key = (delta, sigma, cond_label)
                result_map[bucket_key] = results[idx : idx + n_trials]
                idx += n_trials
    if idx != len(results):
        raise RuntimeError(
            f'result_map bucketing consumed {idx} results but {len(results)} were returned'
        )

    x_arr = np.array(deltas)
    delta_labels = [f'{d:.3g}' for d in deltas]
    sigma_labels = [f'{s:.2g}' for s in sigmas]

    # Line plots: one per condition, three metrics (Euclidean, Y-axis, X-axis).
    for noise_rms, cond_label in conditions:
        noise_desc = (
            'noiseless (1 trial per point -- numerical precision floor)'
            if noise_rms == 0.0
            else (
                f'Gaussian noise, noise_rms = {noise_rms:.3g}'
                f' (SNR = scale / noise_rms = {scale / noise_rms:.0f});'
                f' {study.noise_samples} independent trials per point'
            )
        )
        for metric_attr, metric_label, fname_prefix in [
            ('pos_err', 'Mean position error (Euclidean, pixels)', 'pos_err'),
            ('pos_err_y', 'Mean |pos_err_y| (pixels)', 'pos_err_y'),
            ('pos_err_x', 'Mean |pos_err_x| (pixels)', 'pos_err_x'),
        ]:
            y_means: list[npt.NDArray[np.float64]] = []
            y_stds: list[npt.NDArray[np.float64]] = []
            line_labels: list[str] = []

            for sigma in sigmas:
                means_per_delta: list[float] = []
                stds_per_delta: list[float] = []
                for delta in deltas:
                    bucket = result_map[(delta, sigma, cond_label)]
                    errs = np.array(
                        [abs(float(getattr(r, metric_attr))) for r in bucket if r.converged],
                        dtype=np.float64,
                    )
                    errs = errs[np.isfinite(errs)]
                    if len(errs) == 0:
                        means_per_delta.append(float('nan'))
                        stds_per_delta.append(float('nan'))
                    else:
                        means_per_delta.append(float(np.mean(errs)))
                        stds_per_delta.append(float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0)
                y_means.append(np.array(means_per_delta))
                y_stds.append(np.array(stds_per_delta))
                line_labels.append(f'sigma={sigma:.2g}')

            fig = plot_line_with_bands(
                x_arr,
                y_means,
                y_stds,
                labels=line_labels,
                title=f'Min detectable offset ({metric_label}) -- {cond_label}',
                xlabel='Injected X offset (pixels)',
                ylabel=metric_label,
                log_x=True,
                log_y=True,
                note=(
                    'PSF: sigma_y = sigma_x = sigma (see series label); angle = 0\u00b0 (fixed);'
                    f' box_size = {study.box_size} px; scale = {scale:.2g}\n'
                    'Offset: Y = 0 px (fixed); X = delta (x-axis only); all positional offset'
                    ' is injected in the X direction\n'
                    f'Noise: {noise_desc}\n'
                    'Fitting: sigma_y and sigma_x float freely; angle fixed at 0\u00b0;'
                    ' no background subtraction'
                ),
            )
            save_figure(fig, study_dir, f'{_STUDY_NAME}_{fname_prefix}_{cond_label}.png')

    # Recovery fraction heatmap: one per SNR condition (skip noiseless).
    for noise_rms, cond_label in conditions:
        if noise_rms == 0.0:
            continue
        rec_grid = np.zeros((len(sigmas), len(deltas)))
        for s_idx, sigma in enumerate(sigmas):
            for d_idx, delta in enumerate(deltas):
                bucket = result_map[(delta, sigma, cond_label)]
                rec_grid[s_idx, d_idx] = utils.recovery_fraction(bucket, delta=delta)

        fig = plot_recovery_fraction_heatmap(
            rec_grid,
            delta_labels,
            sigma_labels,
            title=(f'Recovery fraction (pos_err < delta/2) -- {cond_label}'),
            xlabel='Injected X offset (delta, pixels)',
            ylabel='Sigma (pixels)',
            note=(
                'PSF: sigma_y = sigma_x = sigma (y-axis); angle = 0\u00b0 (fixed);'
                f' box_size = {study.box_size} px; scale = {scale:.2g}\n'
                'Offset: Y = 0 px (fixed); X = delta (x-axis); all offset in X only\n'
                f'Noise: Gaussian, noise_rms = {noise_rms:.3g}'
                f' (SNR = {scale / noise_rms:.0f}); {study.noise_samples} trials per cell\n'
                'Fitting: sigma_y and sigma_x float freely; angle fixed at 0\u00b0;'
                ' no background subtraction\n'
                'Recovery = fraction of trials where Euclidean pos_err < delta / 2'
            ),
        )
        save_figure(fig, study_dir, f'{_STUDY_NAME}_recovery_{cond_label}.png')

    write_csv(cfg.output_dir, _STUDY_NAME, specs, results)

    groups = _build_json_groups(specs, results, conditions)
    write_json_summary(
        cfg.output_dir,
        _STUDY_NAME,
        specs,
        results,
        groups=groups,
        config_used=config_to_dict(cfg),
    )
    _LOG.info('Study %s outputs written to %s', _STUDY_NAME, study_dir)


def _build_json_groups(
    specs: list[TrialSpec],
    results: list[TrialResult],
    conditions: list[tuple[float, str]],
) -> list[dict[str, Any]]:
    """Build JSON summary groups for Study 3.

    Groups by (delta_offset, sigma, noise_condition).

    Parameters:
        specs: Trial specifications.
        results: Trial results.
        conditions: List of ``(noise_rms, label)`` pairs.

    Returns:
        List of group dicts for :func:`~output.write_json_summary`.
    """
    condition_map: dict[float, str] = dict(conditions)

    def _cond_label(spec: TrialSpec) -> str:
        label = condition_map.get(spec.noise_rms)
        return label if label is not None else f'noise_{spec.noise_rms:.3g}'

    return utils.build_groups_by_keys(
        specs,
        results,
        [
            ('delta_offset', lambda s: s.offset_x),
            ('sigma', lambda s: s.sigma_y),
            ('noise_condition', _cond_label),
        ],
    )
