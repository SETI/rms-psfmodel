################################################################################
# characterize_gauss_fit/executor.py
################################################################################

"""Sequential and multiprocess trial dispatch for characterize_gauss_fit.

Provides :func:`run_trials`, which executes a list of :class:`~trial.TrialSpec`
objects either sequentially in the calling process or in parallel using
:class:`concurrent.futures.ProcessPoolExecutor`.

Worker isolation: :class:`~psfmodel.gaussian.GaussianPSF` objects are
constructed inside each worker from the plain-data fields of
:class:`~trial.TrialSpec`. No complex objects cross process boundaries.
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable

from characterize_gauss_fit.trial import TrialResult, TrialSpec, run_trial

_LOG = logging.getLogger(__name__)


def _safe_run_trial(spec: TrialSpec) -> TrialResult:
    """Run a single trial, converting any exception into a failed result.

    This wrapper is used by worker processes so that an unexpected exception
    in one trial does not crash the entire pool.

    Parameters:
        spec: The trial to execute.

    Returns:
        A :class:`~trial.TrialResult` with ``converged=False`` and NaN errors
        if any exception occurs, otherwise the normal result.
    """
    try:
        return run_trial(spec)
    except Exception:
        _LOG.exception(
            'Unexpected error in trial (sigma_y=%s, box_size=%s)',
            spec.sigma_y,
            spec.box_size,
        )
        return TrialResult(
            converged=False,
            sigma_y_true=spec.sigma_y,
            sigma_x_true=spec.sigma_x,
            angle_true=spec.angle,
            scale_true=spec.scale,
            offset_y_true=spec.offset_y,
            offset_x_true=spec.offset_x,
            pos_err_y=float('nan'),
            pos_err_x=float('nan'),
            pos_err=float('nan'),
            sigma_y_fit=None,
            sigma_x_fit=None,
            angle_fit=None,
            scale_fit=float('nan'),
            sigma_y_err=None,
            sigma_x_err=None,
            angle_err=None,
            scale_err=float('nan'),
        )


def run_trials(
    trial_specs: list[TrialSpec],
    *,
    num_workers: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[TrialResult]:
    """Execute a list of trials sequentially or in parallel.

    Parameters:
        trial_specs: List of :class:`~trial.TrialSpec` objects to execute.
        num_workers: Number of parallel worker processes. ``1`` runs all trials
            sequentially in the calling process with no multiprocessing overhead.
            Values ``>1`` use :class:`concurrent.futures.ProcessPoolExecutor`.
        progress_callback: Optional callable ``(completed, total)`` invoked after
            each trial result is collected, useful for progress display.

    Returns:
        A list of :class:`~trial.TrialResult` objects in the same order as
        ``trial_specs``.
    """
    total = len(trial_specs)
    results: list[TrialResult] = []

    if num_workers == 1:
        for idx, spec in enumerate(trial_specs):
            results.append(_safe_run_trial(spec))
            if progress_callback is not None:
                progress_callback(idx + 1, total)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_safe_run_trial, spec): i for i, spec in enumerate(trial_specs)}
            # Collect in submission order to preserve determinism.
            ordered: list[TrialResult | None] = [None] * total
            for n_done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                idx = futures[future]
                ordered[idx] = future.result()
                if progress_callback is not None:
                    progress_callback(n_done, total)
        # All futures completed; ordered contains no None entries.
        results = [r for r in ordered if r is not None]

    return results
