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

Chunking: to reduce IPC and pickle overhead, specs are grouped into batches
before dispatch.  Each worker executes a full batch per ``submit()`` call.
The batch size is ``ceil(total / (num_workers * _CHUNK_MULTIPLIER))``.  A
multiplier of 4 gives 4x over-subscription, balancing load across workers
while keeping the number of round-trips low.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import multiprocessing
from collections.abc import Callable

from characterize_gauss_fit.trial import TrialResult, TrialSpec, run_trial

# Number of chunks per worker.  Higher values improve load balancing at the
# cost of more (but still far fewer than one-per-trial) IPC round-trips.
_CHUNK_MULTIPLIER = 4

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


def _run_trial_batch(specs: list[TrialSpec]) -> list[TrialResult]:
    """Execute a batch of trials inside a single worker call.

    Running multiple trials per ``submit()`` call amortises the per-call
    pickle and IPC overhead across the whole batch, which dramatically
    reduces the per-trial overhead compared to submitting one trial at a time.

    Parameters:
        specs: Ordered list of :class:`~trial.TrialSpec` objects to execute.

    Returns:
        Results in the same order as ``specs``.
    """
    return [_safe_run_trial(spec) for spec in specs]


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
            Values ``>1`` use :class:`concurrent.futures.ProcessPoolExecutor`
            with the ``spawn`` start method to avoid fork-related crashes.
            Trials are grouped into chunks of size
            ``ceil(total / (num_workers * _CHUNK_MULTIPLIER))`` so that each
            worker executes many trials per ``submit()`` call, reducing IPC and
            pickle overhead and making speedup more linear with CPU count.
        progress_callback: Optional callable ``(completed, total)`` invoked
            after each chunk of results is collected, useful for progress
            display.

    Returns:
        A list of :class:`~trial.TrialResult` objects in the same order as
        ``trial_specs``.
    """
    total = len(trial_specs)
    results: list[TrialResult] = []

    if not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError(f'num_workers must be a positive integer, got {num_workers!r}')

    if num_workers == 1:
        for idx, spec in enumerate(trial_specs):
            results.append(_safe_run_trial(spec))
            if progress_callback is not None:
                progress_callback(idx + 1, total)
    else:
        # Use the 'spawn' start method so worker processes begin as fresh
        # Python interpreters.  The default 'fork' method on Linux copies
        # the parent's BLAS/OpenMP thread-pool state into the child without
        # actually transferring the threads, causing a segfault during worker
        # cleanup when those phantom pools are torn down.
        mp_ctx = multiprocessing.get_context('spawn')

        # Group specs into batches so each worker handles multiple trials per
        # submit() call.  _CHUNK_MULTIPLIER chunks per worker gives 4x
        # over-subscription for load balancing.
        chunk_size = max(1, math.ceil(total / (num_workers * _CHUNK_MULTIPLIER)))
        chunks: list[tuple[int, list[TrialSpec]]] = []
        start = 0
        while start < total:
            end = min(start + chunk_size, total)
            chunks.append((start, trial_specs[start:end]))
            start = end

        ordered: list[TrialResult | None] = [None] * total
        n_done = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers, mp_context=mp_ctx
        ) as pool:
            futures: dict[concurrent.futures.Future[list[TrialResult]], int] = {
                pool.submit(_run_trial_batch, chunk): chunk_start for chunk_start, chunk in chunks
            }
            for future in concurrent.futures.as_completed(futures):
                chunk_start = futures[future]
                chunk_results = future.result()
                for j, result in enumerate(chunk_results):
                    ordered[chunk_start + j] = result
                n_done += len(chunk_results)
                if progress_callback is not None:
                    progress_callback(n_done, total)

        # All futures completed; ordered contains no None entries.
        if not all(r is not None for r in ordered):
            raise RuntimeError('Internal error: some futures did not produce a result')
        results = [r for r in ordered if r is not None]

    return results
