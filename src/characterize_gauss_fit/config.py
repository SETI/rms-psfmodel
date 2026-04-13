################################################################################
# characterize_gauss_fit/config.py
################################################################################

"""YAML configuration loading and typed dataclasses for characterize_gauss_fit.

Loads ``defaults.yaml`` from the package, deep-merges an optional user-supplied
override file, and exposes a typed ``Config`` object for use by study modules.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.resources
import math
import pathlib
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FittingConfig:
    """Parameters forwarded to :meth:`~psfmodel.PSF.find_position` for every trial."""

    bkgnd_degree: int | None
    bkgnd_ignore_center: tuple[int, int]
    bkgnd_num_sigma: float | None
    num_sigma: float | None
    max_bad_frac: float
    allow_nonzero_base: bool
    use_angular_params: bool
    tolerance: float
    search_limit: tuple[float, float]
    scale_limit: float


@dataclasses.dataclass
class GenerationConfig:
    """Parameters controlling synthetic PSF image generation."""

    scale: float
    base: float


@dataclasses.dataclass
class PsfShapeConfig:
    """A single PSF shape specification for the constraint-modes study."""

    sigma: tuple[float, float]
    angle: float


@dataclasses.dataclass
class StudyBoxVsSigmaConfig:
    """Configuration for Study 1: box size vs. sigma."""

    enabled: bool
    box_sizes: list[int]
    sigmas: list[float]
    offset: tuple[float, float]
    angle: float
    scale: float
    fitting: FittingConfig


@dataclasses.dataclass
class StudySubpixelOffsetConfig:
    """Configuration for Study 2: subpixel offset bias."""

    enabled: bool
    offset_steps: int
    offset_range: tuple[float, float]
    sigmas: list[float]
    box_size: int
    angle: float
    fitting: FittingConfig


@dataclasses.dataclass
class StudyMinDetectableOffsetConfig:
    """Configuration for Study 3: minimum detectable offset."""

    enabled: bool
    delta_offsets: list[float]
    sigmas: list[float]
    box_size: int
    noise_samples: int
    snr_values: list[float]
    include_noiseless: bool
    fitting: FittingConfig


@dataclasses.dataclass
class StudySigmaAsymmetryAngleConfig:
    """Configuration for Study 4: sigma asymmetry and angle recovery."""

    enabled: bool
    sigma_ratios: list[float]
    angle_steps: int
    sigma_x_values: list[float]
    box_size: int
    offset: tuple[float, float]
    fitting: FittingConfig


@dataclasses.dataclass
class StudyConstraintModesConfig:
    """Configuration for Study 5: constraint modes."""

    enabled: bool
    sigma_error_fractions: list[float]
    angle_error_rad: float
    psf_shapes: list[PsfShapeConfig]
    box_size: int
    offset: tuple[float, float]
    scale: float
    fitting: FittingConfig


@dataclasses.dataclass
class StudyBackgroundConfig:
    """Configuration for Study 6: background conditions."""

    enabled: bool
    background_amplitudes: list[float]
    bkgnd_degrees: list[int]
    bkgnd_degrees_with_null: bool
    bkgnd_ignore_centers: list[tuple[int, int]]
    background_types: list[str]
    box_size: int
    sigma: tuple[float, float]
    offset: tuple[float, float]
    fitting: FittingConfig


@dataclasses.dataclass
class StudyNoiseSensitivityConfig:
    """Configuration for Study 7: noise sensitivity."""

    enabled: bool
    snr_log_range: tuple[float, float]
    snr_steps: int
    sigmas: list[float]
    noise_samples: int
    box_size: int
    fitting: FittingConfig


@dataclasses.dataclass
class StudyHotPixelRejectionConfig:
    """Configuration for Study 8: hot pixel rejection."""

    enabled: bool
    num_hot_pixels: list[int]
    num_sigma_values: list[float]
    num_sigma_with_null: bool
    hot_amplitudes: list[float]
    noise_samples: int
    snr: float
    box_size: int
    sigma: tuple[float, float]
    offset: tuple[float, float]
    fitting: FittingConfig


@dataclasses.dataclass
class StudiesConfig:
    """Container for all per-study configurations."""

    box_vs_sigma: StudyBoxVsSigmaConfig
    subpixel_offset: StudySubpixelOffsetConfig
    min_detectable_offset: StudyMinDetectableOffsetConfig
    sigma_asymmetry_angle: StudySigmaAsymmetryAngleConfig
    constraint_modes: StudyConstraintModesConfig
    background: StudyBackgroundConfig
    noise_sensitivity: StudyNoiseSensitivityConfig
    hot_pixel_rejection: StudyHotPixelRejectionConfig


@dataclasses.dataclass
class Config:
    """Top-level configuration object for the entire run."""

    output_dir: pathlib.Path
    num_workers: int
    noise_samples: int
    fitting: FittingConfig
    generation: GenerationConfig
    studies: StudiesConfig


# ---------------------------------------------------------------------------
# YAML loading and merging
# ---------------------------------------------------------------------------

STUDY_NAMES: list[str] = [
    'box_vs_sigma',
    'subpixel_offset',
    'min_detectable_offset',
    'sigma_asymmetry_angle',
    'constraint_modes',
    'background',
    'noise_sensitivity',
    'hot_pixel_rejection',
]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict that is ``base`` deep-merged with ``override``.

    Scalar values and lists in ``override`` replace those in ``base`` entirely.
    Nested dicts are merged recursively.

    Parameters:
        base: The base dictionary (from defaults.yaml).
        override: The user-supplied override dictionary.

    Returns:
        A new merged dictionary. Neither input is mutated.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_raw(path: pathlib.Path | None) -> dict[str, Any]:
    """Load and deep-merge defaults.yaml with an optional user override file.

    Parameters:
        path: Optional path to a user-supplied YAML override file. Pass ``None``
            to use only the built-in defaults.

    Returns:
        A merged dictionary of all configuration values.

    Raises:
        FileNotFoundError: If ``path`` is provided but does not exist.
        ValueError: If the YAML file cannot be parsed.
    """
    pkg_ref = importlib.resources.files('characterize_gauss_fit').joinpath('defaults.yaml')
    with (
        importlib.resources.as_file(pkg_ref) as defaults_path,
        defaults_path.open('r', encoding='utf-8') as fh,
    ):
        raw: dict[str, Any] = yaml.safe_load(fh)

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f'Config file not found: {path}')
        with path.open('r', encoding='utf-8') as fh:
            try:
                user_raw: dict[str, Any] = yaml.safe_load(fh) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f'Failed to parse config file {path}: {exc}') from exc
        raw = _deep_merge(raw, user_raw)

    return raw


def _parse_fitting(raw: dict[str, Any]) -> FittingConfig:
    """Parse a fitting-section dict into a :class:`FittingConfig`.

    Parameters:
        raw: Dict from the ``fitting`` YAML section.

    Returns:
        A populated :class:`FittingConfig`.

    Raises:
        ValueError: If required keys are missing or values are out of range.
    """
    bkgnd_ic = raw['bkgnd_ignore_center']
    search_lim = raw['search_limit']
    return FittingConfig(
        bkgnd_degree=raw['bkgnd_degree'],
        bkgnd_ignore_center=(int(bkgnd_ic[0]), int(bkgnd_ic[1])),
        bkgnd_num_sigma=raw['bkgnd_num_sigma'],
        num_sigma=raw['num_sigma'],
        max_bad_frac=float(raw['max_bad_frac']),
        allow_nonzero_base=bool(raw['allow_nonzero_base']),
        use_angular_params=bool(raw['use_angular_params']),
        tolerance=float(raw['tolerance']),
        search_limit=(float(search_lim[0]), float(search_lim[1])),
        scale_limit=float(raw['scale_limit']),
    )


def _resolve_fitting(global_fitting: dict[str, Any], study_raw: dict[str, Any]) -> FittingConfig:
    """Merge the global fitting defaults with optional per-study overrides.

    Parameters:
        global_fitting: The top-level ``fitting`` section dict.
        study_raw: The study-specific dict (may contain a ``fitting`` sub-dict).

    Returns:
        A :class:`FittingConfig` reflecting any per-study overrides.
    """
    if 'fitting' in study_raw:
        merged = _deep_merge(global_fitting, study_raw['fitting'])
    else:
        merged = global_fitting
    return _parse_fitting(merged)


def _parse_two_floats(value: list[Any]) -> tuple[float, float]:
    """Parse a two-element list into a tuple of floats.

    Parameters:
        value: A list with exactly two numeric elements.

    Returns:
        A ``(float, float)`` tuple.

    Raises:
        ValueError: If ``value`` does not have exactly two elements.
    """
    if len(value) != 2:
        raise ValueError(f'Expected a list of 2 values, got {len(value)}')
    return (float(value[0]), float(value[1]))


def _parse_two_ints(value: list[Any]) -> tuple[int, int]:
    """Parse a two-element list into a tuple of ints.

    Parameters:
        value: A list with exactly two numeric elements.

    Returns:
        An ``(int, int)`` tuple.

    Raises:
        ValueError: If ``value`` does not have exactly two elements.
    """
    if len(value) != 2:
        raise ValueError(f'Expected a list of 2 values, got {len(value)}')
    return (int(value[0]), int(value[1]))


def _build_config(raw: dict[str, Any]) -> Config:
    """Convert a merged raw dict into a fully typed :class:`Config`.

    Parameters:
        raw: Merged configuration dictionary.

    Returns:
        A populated :class:`Config` instance.

    Raises:
        ValueError: If any required field is missing or invalid.
        KeyError: If a required YAML key is absent.
    """
    global_fitting = raw['fitting']
    gen = raw['generation']
    studies = raw['studies']

    def study_fitting(name: str) -> FittingConfig:
        return _resolve_fitting(global_fitting, studies[name])

    # Study 4: build angle list from steps
    sa_raw = studies['sigma_asymmetry_angle']
    angle_steps: int = int(sa_raw['angle_steps'])
    if angle_steps < 2:
        raise ValueError(f'angle_steps must be >= 2, got {angle_steps}')

    # Study 6: parse list of ignore-center pairs
    bg_raw = studies['background']
    ignore_centers: list[tuple[int, int]] = [
        _parse_two_ints(ic) for ic in bg_raw['bkgnd_ignore_centers']
    ]

    # Study 5: parse psf_shapes list
    cm_raw = studies['constraint_modes']
    psf_shapes: list[PsfShapeConfig] = [
        PsfShapeConfig(
            sigma=_parse_two_floats(s['sigma']),
            angle=float(s['angle']),
        )
        for s in cm_raw['psf_shapes']
    ]

    return Config(
        output_dir=pathlib.Path(raw['output_dir']),
        num_workers=int(raw['num_workers']),
        noise_samples=int(raw['noise_samples']),
        fitting=_parse_fitting(global_fitting),
        generation=GenerationConfig(
            scale=float(gen['scale']),
            base=float(gen['base']),
        ),
        studies=StudiesConfig(
            box_vs_sigma=StudyBoxVsSigmaConfig(
                enabled=bool(studies['box_vs_sigma']['enabled']),
                box_sizes=[int(x) for x in studies['box_vs_sigma']['box_sizes']],
                sigmas=[float(x) for x in studies['box_vs_sigma']['sigmas']],
                offset=_parse_two_floats(studies['box_vs_sigma']['offset']),
                angle=float(studies['box_vs_sigma']['angle']),
                scale=float(studies['box_vs_sigma']['scale']),
                fitting=study_fitting('box_vs_sigma'),
            ),
            subpixel_offset=StudySubpixelOffsetConfig(
                enabled=bool(studies['subpixel_offset']['enabled']),
                offset_steps=int(studies['subpixel_offset']['offset_steps']),
                offset_range=_parse_two_floats(studies['subpixel_offset']['offset_range']),
                sigmas=[float(x) for x in studies['subpixel_offset']['sigmas']],
                box_size=int(studies['subpixel_offset']['box_size']),
                angle=float(studies['subpixel_offset']['angle']),
                fitting=study_fitting('subpixel_offset'),
            ),
            min_detectable_offset=StudyMinDetectableOffsetConfig(
                enabled=bool(studies['min_detectable_offset']['enabled']),
                delta_offsets=[float(x) for x in studies['min_detectable_offset']['delta_offsets']],
                sigmas=[float(x) for x in studies['min_detectable_offset']['sigmas']],
                box_size=int(studies['min_detectable_offset']['box_size']),
                noise_samples=int(studies['min_detectable_offset']['noise_samples']),
                snr_values=[float(x) for x in studies['min_detectable_offset']['snr_values']],
                include_noiseless=bool(studies['min_detectable_offset']['include_noiseless']),
                fitting=study_fitting('min_detectable_offset'),
            ),
            sigma_asymmetry_angle=StudySigmaAsymmetryAngleConfig(
                enabled=bool(sa_raw['enabled']),
                sigma_ratios=[float(x) for x in sa_raw['sigma_ratios']],
                angle_steps=angle_steps,
                sigma_x_values=[float(x) for x in sa_raw['sigma_x_values']],
                box_size=int(sa_raw['box_size']),
                offset=_parse_two_floats(sa_raw['offset']),
                fitting=study_fitting('sigma_asymmetry_angle'),
            ),
            constraint_modes=StudyConstraintModesConfig(
                enabled=bool(cm_raw['enabled']),
                sigma_error_fractions=[float(x) for x in cm_raw['sigma_error_fractions']],
                angle_error_rad=float(cm_raw['angle_error_rad']),
                psf_shapes=psf_shapes,
                box_size=int(cm_raw['box_size']),
                offset=_parse_two_floats(cm_raw['offset']),
                scale=float(cm_raw['scale']),
                fitting=study_fitting('constraint_modes'),
            ),
            background=StudyBackgroundConfig(
                enabled=bool(bg_raw['enabled']),
                background_amplitudes=[float(x) for x in bg_raw['background_amplitudes']],
                bkgnd_degrees=[int(x) for x in bg_raw['bkgnd_degrees']],
                bkgnd_degrees_with_null=bool(bg_raw['bkgnd_degrees_with_null']),
                bkgnd_ignore_centers=ignore_centers,
                background_types=[str(x) for x in bg_raw['background_types']],
                box_size=int(bg_raw['box_size']),
                sigma=_parse_two_floats(bg_raw['sigma']),
                offset=_parse_two_floats(bg_raw['offset']),
                fitting=study_fitting('background'),
            ),
            noise_sensitivity=StudyNoiseSensitivityConfig(
                enabled=bool(studies['noise_sensitivity']['enabled']),
                snr_log_range=_parse_two_floats(studies['noise_sensitivity']['snr_log_range']),
                snr_steps=int(studies['noise_sensitivity']['snr_steps']),
                sigmas=[float(x) for x in studies['noise_sensitivity']['sigmas']],
                noise_samples=int(studies['noise_sensitivity']['noise_samples']),
                box_size=int(studies['noise_sensitivity']['box_size']),
                fitting=study_fitting('noise_sensitivity'),
            ),
            hot_pixel_rejection=StudyHotPixelRejectionConfig(
                enabled=bool(studies['hot_pixel_rejection']['enabled']),
                num_hot_pixels=[int(x) for x in studies['hot_pixel_rejection']['num_hot_pixels']],
                num_sigma_values=[
                    float(x) for x in studies['hot_pixel_rejection']['num_sigma_values']
                ],
                num_sigma_with_null=bool(studies['hot_pixel_rejection']['num_sigma_with_null']),
                hot_amplitudes=[float(x) for x in studies['hot_pixel_rejection']['hot_amplitudes']],
                noise_samples=int(studies['hot_pixel_rejection']['noise_samples']),
                snr=float(studies['hot_pixel_rejection']['snr']),
                box_size=int(studies['hot_pixel_rejection']['box_size']),
                sigma=_parse_two_floats(studies['hot_pixel_rejection']['sigma']),
                offset=_parse_two_floats(studies['hot_pixel_rejection']['offset']),
                fitting=study_fitting('hot_pixel_rejection'),
            ),
        ),
    )


def load_config(
    path: pathlib.Path | None = None,
    *,
    output_dir: pathlib.Path | None = None,
    num_workers: int | None = None,
) -> Config:
    """Load configuration from defaults and an optional user override file.

    Loads the built-in ``defaults.yaml``, deep-merges the user file if provided,
    then applies any CLI-level overrides (``output_dir``, ``num_workers``).

    Parameters:
        path: Optional path to a user YAML override file.
        output_dir: If given, overrides the ``output_dir`` from YAML.
        num_workers: If given, overrides ``num_workers`` from YAML.

    Returns:
        A fully validated :class:`Config` instance.

    Raises:
        FileNotFoundError: If ``path`` is provided but does not exist.
        ValueError: If the configuration is invalid.
        KeyError: If a required YAML key is missing.
    """
    raw = _load_raw(path)
    if output_dir is not None:
        raw['output_dir'] = str(output_dir)
    if num_workers is not None:
        raw['num_workers'] = num_workers
    cfg = _build_config(raw)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Config) -> None:
    """Raise :class:`ValueError` if the config contains invalid values.

    Parameters:
        cfg: A :class:`Config` instance to validate.

    Raises:
        ValueError: If any value is out of range or logically inconsistent.
    """
    if cfg.num_workers < 1:
        raise ValueError(f'num_workers must be >= 1, got {cfg.num_workers}')
    if cfg.noise_samples < 1:
        raise ValueError(f'noise_samples must be >= 1, got {cfg.noise_samples}')

    bvs = cfg.studies.box_vs_sigma
    for bs in bvs.box_sizes:
        if bs < 5 or bs % 2 == 0:
            raise ValueError(
                f'box_vs_sigma.box_sizes: each entry must be an odd integer >= 5, got {bs}'
            )

    mdo = cfg.studies.min_detectable_offset
    for delta in mdo.delta_offsets:
        if delta <= 0:
            raise ValueError(f'min_detectable_offset.delta_offsets: must be positive, got {delta}')

    ns = cfg.studies.noise_sensitivity
    lo, hi = ns.snr_log_range
    if lo >= hi:
        raise ValueError(f'noise_sensitivity.snr_log_range: min ({lo}) must be < max ({hi})')

    sa = cfg.studies.sigma_asymmetry_angle
    for ratio in sa.sigma_ratios:
        if ratio <= 0:
            raise ValueError(f'sigma_asymmetry_angle.sigma_ratios: must be positive, got {ratio}')

    for shape in cfg.studies.constraint_modes.psf_shapes:
        if not (0.0 <= shape.angle <= math.pi):
            raise ValueError(
                f'constraint_modes.psf_shapes: angle must be in [0, pi], got {shape.angle}'
            )

    valid_bkgnd_types = {'none', 'constant', 'linear', 'quadratic', 'noisy_constant'}
    for bt in cfg.studies.background.background_types:
        if bt not in valid_bkgnd_types:
            raise ValueError(
                f'background.background_types: unknown type "{bt}". '
                f'Valid options: {sorted(valid_bkgnd_types)}'
            )


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Convert a :class:`Config` to a plain dict suitable for JSON serialisation.

    Parameters:
        cfg: A :class:`Config` instance.

    Returns:
        A dict representation with all Path objects converted to strings.
    """
    raw = dataclasses.asdict(cfg)
    # Convert Path to string for JSON serialisability.
    raw['output_dir'] = str(cfg.output_dir)
    return raw
