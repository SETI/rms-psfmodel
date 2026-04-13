################################################################################
# characterize_gauss_fit/main.py
################################################################################

"""Command-line entry point for characterize_gauss_fit.

Parses arguments, loads configuration, and dispatches requested studies.
Exit code is 0 on success, 1 if any study raises an unhandled exception.
"""

from __future__ import annotations

import argparse
import importlib.resources
import logging
import pathlib
import shutil
import sys
import time
from collections.abc import Callable

from characterize_gauss_fit import (
    study_background,
    study_box_sigma,
    study_constraints,
    study_hot_pixels,
    study_min_offset,
    study_noise,
    study_offset,
    study_shape,
)
from characterize_gauss_fit.config import STUDY_NAMES, load_config

_LOG = logging.getLogger(__name__)

# Type alias for a study run function.
_StudyRunner = Callable[..., None]

# Registry: study name -> run function.
_STUDY_REGISTRY: dict[str, _StudyRunner] = {
    'box_vs_sigma': study_box_sigma.run,
    'subpixel_offset': study_offset.run,
    'min_detectable_offset': study_min_offset.run,
    'sigma_asymmetry_angle': study_shape.run,
    'constraint_modes': study_constraints.run,
    'background': study_background.run,
    'noise_sensitivity': study_noise.run,
    'hot_pixel_rejection': study_hot_pixels.run,
}

# Ensure STUDY_NAMES (from config) and _STUDY_REGISTRY stay in sync.
assert set(STUDY_NAMES) == set(_STUDY_REGISTRY), (
    f'STUDY_NAMES and _STUDY_REGISTRY are out of sync: '
    f'{set(STUDY_NAMES).symmetric_difference(set(_STUDY_REGISTRY))}'
)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog='characterize_gauss_fit',
        description=(
            'Characterize Gaussian PSF fitting accuracy across a configurable '
            'parameter space. Produces PNG plots, CSV tables, and JSON summaries.'
        ),
    )
    parser.add_argument(
        '--config',
        metavar='FILE',
        type=pathlib.Path,
        default=None,
        help='Path to a YAML override file merged onto built-in defaults.',
    )
    parser.add_argument(
        '--study',
        metavar='NAME',
        action='append',
        dest='studies',
        default=None,
        help=(
            'Run only this study (repeatable). Default: all enabled studies. '
            f'Available names: {", ".join(STUDY_NAMES)}'
        ),
    )
    parser.add_argument(
        '--output-dir',
        metavar='DIR',
        type=pathlib.Path,
        default=None,
        help='Override the output directory from the config file.',
    )
    parser.add_argument(
        '--num-workers',
        metavar='N',
        type=int,
        default=None,
        help=(
            'Number of parallel worker processes. '
            'Default: None (resolved from config file). '
            '1 = sequential in the main process; '
            '>1 = concurrent.futures.ProcessPoolExecutor.'
        ),
    )
    parser.add_argument(
        '--list-studies',
        action='store_true',
        help='Print available study names and exit.',
    )
    parser.add_argument(
        '--copy-default-config-to',
        metavar='FILE',
        type=pathlib.Path,
        default=None,
        help=(
            'Write the built-in default configuration to FILE and exit. '
            'No studies are run.'
        ),
    )
    parser.add_argument(
        '--copy-test-config-to',
        metavar='FILE',
        type=pathlib.Path,
        default=None,
        help=(
            'Write the built-in reduced-grid test configuration to FILE and exit. '
            'No studies are run.'
        ),
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable DEBUG-level logging.',
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure the root logger for CLI use.

    Parameters:
        verbose: If ``True``, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr,
    )


def _copy_bundled_config(source_name: str, dest: pathlib.Path) -> None:
    """Copy a bundled YAML config file to a user-supplied path.

    Parameters:
        source_name: Filename of the bundled resource (e.g. ``'defaults.yaml'``).
        dest: Destination path supplied by the user.

    Raises:
        SystemExit: If the destination already exists or the copy fails.
    """
    if dest.exists():
        print(f'ERROR: destination already exists: {dest}', file=sys.stderr)
        sys.exit(1)
    try:
        pkg = importlib.resources.files('characterize_gauss_fit')
        src_ref = pkg.joinpath(source_name)
        with importlib.resources.as_file(src_ref) as src_path:
            shutil.copy(src_path, dest)
        print(f'Written: {dest}')
    except OSError as exc:
        print(f'ERROR copying config: {exc}', file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Entry point for the ``characterize_gauss_fit`` command.

    Parses command-line arguments, loads configuration, and runs all requested
    studies in order. Prints timing information to stderr. Exits with code 1
    if any study fails with an unhandled exception.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_studies:
        print('Available studies:')
        for name in STUDY_NAMES:
            print(f'  {name}')
        return

    if args.copy_default_config_to is not None:
        _copy_bundled_config('defaults.yaml', args.copy_default_config_to)
        return

    if args.copy_test_config_to is not None:
        _copy_bundled_config('test_config.yaml', args.copy_test_config_to)
        return

    _configure_logging(args.verbose)

    try:
        cfg = load_config(
            path=args.config,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f'ERROR loading config: {exc}', file=sys.stderr)
        sys.exit(1)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    requested_names: list[str] = args.studies if args.studies is not None else STUDY_NAMES

    # Validate requested study names.
    for name in requested_names:
        if name not in _STUDY_REGISTRY:
            print(
                f'ERROR: unknown study "{name}". Run with --list-studies to see available names.',
                file=sys.stderr,
            )
            sys.exit(1)

    num_workers = cfg.num_workers
    failed: list[str] = []
    overall_start = time.monotonic()

    for name in requested_names:
        run_fn = _STUDY_REGISTRY[name]
        print(f'[{name}] Starting...', file=sys.stderr)
        t0 = time.monotonic()
        try:
            run_fn(cfg, num_workers=num_workers)
        except Exception:
            _LOG.exception('Study %s failed with an unhandled exception.', name)
            failed.append(name)
        elapsed = time.monotonic() - t0
        status = 'FAILED' if name in failed else 'done'
        print(f'[{name}] {status} ({elapsed:.1f}s)', file=sys.stderr)

    total = time.monotonic() - overall_start
    if failed:
        print(
            f'\nCompleted with {len(failed)} failed studies. '
            f'Total time: {total:.1f}s. Output: {cfg.output_dir}',
            file=sys.stderr,
        )
        print(f'FAILED studies: {", ".join(failed)}', file=sys.stderr)
        sys.exit(1)
    else:
        print(
            f'\nAll studies complete. Total time: {total:.1f}s. Output: {cfg.output_dir}',
            file=sys.stderr,
        )
