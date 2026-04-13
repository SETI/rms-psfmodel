################################################################################
# characterize_gauss_fit/trial.py
################################################################################

"""Core trial engine: image synthesis, fitting, and result computation.

A *trial* is a single run of the PSF fitter on a synthetically generated image
with known ground-truth parameters. This module provides:

- :class:`TrialSpec` -- plain-data description of one trial (picklable).
- :class:`TrialResult` -- the outcome of one trial (picklable).
- :func:`synthesize_image` -- generate a synthetic image from a :class:`TrialSpec`.
- :func:`run_trial` -- execute a trial and return a :class:`TrialResult`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from psfmodel.gaussian import GaussianPSF

# Sentinel float used for "not applicable" / "fit did not converge".
_NAN = float('nan')

# Background types supported by synthesize_image.
BACKGROUND_TYPE_NONE = 'none'
BACKGROUND_TYPE_CONSTANT = 'constant'
BACKGROUND_TYPE_LINEAR = 'linear'
BACKGROUND_TYPE_QUADRATIC = 'quadratic'
BACKGROUND_TYPE_NOISY_CONSTANT = 'noisy_constant'


# ---------------------------------------------------------------------------
# TrialSpec -- fully describes a single trial (all plain Python types).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrialSpec:
    """Complete specification for a single characterization trial.

    All fields use plain Python types so instances are safely picklable for
    use with :mod:`concurrent.futures.ProcessPoolExecutor`.
    """

    # --- Ground-truth PSF parameters ---
    sigma_y: float
    sigma_x: float
    angle: float
    offset_y: float  # fractional pixel offset from center
    offset_x: float
    scale: float
    base: float
    box_size: int

    # --- Background injection ---
    background_type: str  # one of the BACKGROUND_TYPE_* constants
    background_amplitude: float  # fraction of PSF peak

    # --- Noise ---
    noise_rms: float  # additive Gaussian noise std; 0 = noiseless

    # --- Hot pixels ---
    hot_pixel_count: int
    hot_pixel_amplitude: float  # multiple of PSF peak

    # --- Fitter configuration ---
    # GaussianPSF construction: None means the parameter floats during fitting.
    fit_sigma_y: float | None  # None = float; float = fixed to this value
    fit_sigma_x: float | None
    fit_angle: float | None

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

    # --- Reproducibility ---
    rng_seed: int


# ---------------------------------------------------------------------------
# TrialResult -- the outcome of a single trial.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Outcome of a single characterization trial.

    When ``converged`` is ``False``, all error fields contain ``float('nan')``
    and all ``*_fit`` fields are ``None`` (except ``scale_fit`` which is
    ``float('nan')``). Failures are recorded as data, not skipped.
    """

    # --- Convergence ---
    converged: bool

    # --- True (injected) values ---
    sigma_y_true: float
    sigma_x_true: float
    angle_true: float
    scale_true: float
    offset_y_true: float
    offset_x_true: float

    # --- Position errors (signed and Euclidean) ---
    pos_err_y: float  # fitted_y - true_y  (NaN if not converged)
    pos_err_x: float
    pos_err: float  # sqrt(pos_err_y**2 + pos_err_x**2)

    # --- Fitted parameter values (None if param was not floating) ---
    sigma_y_fit: float | None
    sigma_x_fit: float | None
    angle_fit: float | None
    scale_fit: float  # NaN if not converged

    # --- Relative / absolute errors (None if param was not floating) ---
    # For sigma: (fit - true) / true
    sigma_y_err: float | None
    sigma_x_err: float | None
    # For angle: absolute difference in radians (NaN if degenerate circular PSF)
    angle_err: float | None
    # For scale: (fit - true) / true
    scale_err: float  # NaN if not converged


# ---------------------------------------------------------------------------
# Image synthesis helpers
# ---------------------------------------------------------------------------


def _make_background(
    box_size: int,
    background_type: str,
    amplitude: float,
    *,
    rng: np.random.Generator,
    noise_rms: float,
) -> npt.NDArray[np.float64]:
    """Create a background array to add to the clean PSF image.

    Parameters:
        box_size: Side length of the square patch.
        background_type: One of the ``BACKGROUND_TYPE_*`` constants.
        amplitude: Background amplitude as a fraction of the PSF peak. Used for
            constant, linear, and quadratic types.
        rng: NumPy random number generator (used for noisy_constant type).
        noise_rms: Additive Gaussian noise standard deviation (separate from
            the background amplitude noise in ``noisy_constant``).

    Returns:
        A ``(box_size, box_size)`` float64 array containing the background.

    Raises:
        ValueError: If ``background_type`` is not recognised.
    """
    if background_type == BACKGROUND_TYPE_NONE:
        return np.zeros((box_size, box_size), dtype=np.float64)

    # Normalised coordinate grid in [-1, 1].
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, box_size),
        np.linspace(-1.0, 1.0, box_size),
        indexing='ij',
    )

    if background_type == BACKGROUND_TYPE_CONSTANT:
        return np.full((box_size, box_size), amplitude, dtype=np.float64)

    if background_type == BACKGROUND_TYPE_LINEAR:
        # Tilted plane with slope proportional to amplitude.
        return (amplitude * (0.6 * yy + 0.4 * xx)).astype(np.float64)

    if background_type == BACKGROUND_TYPE_QUADRATIC:
        # Bowl-shaped quadratic surface.
        return (amplitude * (0.5 * yy**2 + 0.3 * xx**2 - 0.2 * yy * xx)).astype(np.float64)

    if background_type == BACKGROUND_TYPE_NOISY_CONSTANT:
        constant = np.full((box_size, box_size), amplitude, dtype=np.float64)
        noise = rng.normal(0.0, noise_rms * 0.5, size=(box_size, box_size))
        return (constant + noise).astype(np.float64)

    raise ValueError(
        f'Unknown background_type "{background_type}". '
        f'Valid options: {BACKGROUND_TYPE_NONE}, {BACKGROUND_TYPE_CONSTANT}, '
        f'{BACKGROUND_TYPE_LINEAR}, {BACKGROUND_TYPE_QUADRATIC}, '
        f'{BACKGROUND_TYPE_NOISY_CONSTANT}'
    )


def synthesize_image(spec: TrialSpec) -> tuple[npt.NDArray[np.float64], float, float]:
    """Generate a synthetic PSF image from a :class:`TrialSpec`.

    Creates a pixel-integrated Gaussian PSF patch of size
    ``(box_size, box_size)``, optionally adds a polynomial background,
    additive Gaussian noise, and hot pixels. The PSF fills the entire image
    (``eval_rect`` size == ``box_size``).

    The true PSF center in the image is ``(box_size//2 + offset_y,
    box_size//2 + offset_x)`` in full-image coordinates (i.e. the integer
    anchor is the image centre, and ``offset_y/x`` is the sub-pixel shift).

    Parameters:
        spec: A fully populated :class:`TrialSpec` describing the trial.

    Returns:
        A tuple ``(image, true_y, true_x)`` where ``image`` is the synthesized
        float64 array and ``true_y``, ``true_x`` are the ground-truth PSF
        centre in full-image pixel coordinates (floating point).
    """
    rng = np.random.default_rng(spec.rng_seed)

    # --- Build clean PSF ---
    psf_gen = GaussianPSF(
        sigma=(spec.sigma_y, spec.sigma_x),
        angle=spec.angle,
    )
    image: npt.NDArray[np.float64] = psf_gen.eval_rect(
        (spec.box_size, spec.box_size),
        offset=(spec.offset_y + 0.5, spec.offset_x + 0.5),
        scale=spec.scale,
        base=spec.base,
        angle=spec.angle,
    ).astype(np.float64)

    psf_peak = float(np.max(image))
    if psf_peak <= 0.0:
        psf_peak = 1.0  # guard against degenerate cases

    # --- Add background ---
    background = _make_background(
        spec.box_size,
        spec.background_type,
        spec.background_amplitude * psf_peak,
        rng=rng,
        noise_rms=spec.noise_rms,
    )
    image = image + background

    # --- Add Gaussian noise ---
    if spec.noise_rms > 0.0:
        image = image + rng.normal(0.0, spec.noise_rms, size=image.shape)

    # --- Inject hot pixels ---
    if spec.hot_pixel_count > 0:
        total_pixels = spec.box_size * spec.box_size
        if spec.hot_pixel_count > total_pixels:
            raise ValueError(
                f'hot_pixel_count ({spec.hot_pixel_count}) exceeds total pixels '
                f'({total_pixels}) for box_size={spec.box_size}'
            )
        hot_amplitude = spec.hot_pixel_amplitude * psf_peak
        # Randomise positions uniformly over the full patch.
        flat_indices = rng.choice(
            spec.box_size * spec.box_size,
            size=spec.hot_pixel_count,
            replace=False,
        )
        rows, cols = np.unravel_index(flat_indices, (spec.box_size, spec.box_size))
        image[rows, cols] += hot_amplitude

    # --- True position in full-image coordinates ---
    # find_position returns positions in pixel-left-edge convention: the centre of
    # pixel i is at coordinate i + 0.5 (pixel 0 spans [0, 1]).  Adding 0.5 to the
    # integer centre index converts to this same convention so that error = 0 for
    # a perfect fit.
    center = spec.box_size // 2
    true_y = float(center) + 0.5 + spec.offset_y
    true_x = float(center) + 0.5 + spec.offset_x

    return image.astype(np.float64), true_y, true_x


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------


def run_trial(spec: TrialSpec) -> TrialResult:
    """Execute a single characterization trial.

    Synthesizes an image from ``spec``, fits it with
    :meth:`~psfmodel.PSF.find_position`, and computes all error metrics.
    If the fitter returns ``None`` (failure or convergence rejection), a
    :class:`TrialResult` with ``converged=False`` and NaN error fields is
    returned. The result is always valid and never raises.

    Parameters:
        spec: A fully populated :class:`TrialSpec`.

    Returns:
        A :class:`TrialResult` describing the fit outcome.
    """
    image, true_y, true_x = synthesize_image(spec)

    # Build the fitter PSF (parameters are fixed at construction or left None to float).
    fitter_psf = GaussianPSF(
        sigma=(spec.fit_sigma_y, spec.fit_sigma_x),
        angle=spec.fit_angle,
    )

    starting_point = (float(spec.box_size // 2), float(spec.box_size // 2))
    box_size = (spec.box_size, spec.box_size)

    find_position_kwargs: dict[str, Any] = {
        'bkgnd_degree': spec.bkgnd_degree,
        'bkgnd_ignore_center': spec.bkgnd_ignore_center,
        'bkgnd_num_sigma': spec.bkgnd_num_sigma,
        'num_sigma': spec.num_sigma,
        'max_bad_frac': spec.max_bad_frac,
        'allow_nonzero_base': spec.allow_nonzero_base,
        'use_angular_params': spec.use_angular_params,
        'tolerance': spec.tolerance,
        'search_limit': spec.search_limit,
        'scale_limit': spec.scale_limit,
    }

    result = fitter_psf.find_position(image, box_size, starting_point, **find_position_kwargs)

    if result is None:
        return TrialResult(
            converged=False,
            sigma_y_true=spec.sigma_y,
            sigma_x_true=spec.sigma_x,
            angle_true=spec.angle,
            scale_true=spec.scale,
            offset_y_true=spec.offset_y,
            offset_x_true=spec.offset_x,
            pos_err_y=_NAN,
            pos_err_x=_NAN,
            pos_err=_NAN,
            sigma_y_fit=None,
            sigma_x_fit=None,
            angle_fit=None,
            scale_fit=_NAN,
            sigma_y_err=None,
            sigma_x_err=None,
            angle_err=None,
            scale_err=_NAN,
        )

    fit_y, fit_x, details = result

    pos_err_y = fit_y - true_y
    pos_err_x = fit_x - true_x
    pos_err = math.hypot(pos_err_y, pos_err_x)

    scale_fit = float(details['scale'])
    scale_err = (scale_fit - spec.scale) / spec.scale if spec.scale != 0.0 else _NAN

    # Retrieve fitted sigma values if they were floating.
    sigma_y_fit: float | None = None
    sigma_x_fit: float | None = None
    sigma_y_err: float | None = None
    sigma_x_err: float | None = None

    if 'sigma_y' in details:
        sigma_y_fit = float(details['sigma_y'])
        sigma_y_err = (
            (sigma_y_fit - spec.sigma_y) / spec.sigma_y if spec.sigma_y != 0.0 else None
        )
    if 'sigma_x' in details:
        sigma_x_fit = float(details['sigma_x'])
        sigma_x_err = (
            (sigma_x_fit - spec.sigma_x) / spec.sigma_x if spec.sigma_x != 0.0 else None
        )

    # Retrieve fitted angle if it was floating.
    angle_fit: float | None = None
    angle_err: float | None = None

    if 'angle' in details:
        angle_fit = float(details['angle'])
        # For circular PSFs (sigma_y == sigma_x), angle is degenerate -- mark as NaN.
        if abs(spec.sigma_y - spec.sigma_x) < 1e-9:
            angle_err = _NAN
        else:
            # Wrap angular difference to [-pi/2, pi/2] because angle has pi symmetry.
            raw_diff = angle_fit - spec.angle
            wrapped = (raw_diff + math.pi / 2) % math.pi - math.pi / 2
            angle_err = abs(wrapped)

    return TrialResult(
        converged=True,
        sigma_y_true=spec.sigma_y,
        sigma_x_true=spec.sigma_x,
        angle_true=spec.angle,
        scale_true=spec.scale,
        offset_y_true=spec.offset_y,
        offset_x_true=spec.offset_x,
        pos_err_y=pos_err_y,
        pos_err_x=pos_err_x,
        pos_err=pos_err,
        sigma_y_fit=sigma_y_fit,
        sigma_x_fit=sigma_x_fit,
        angle_fit=angle_fit,
        scale_fit=scale_fit,
        sigma_y_err=sigma_y_err,
        sigma_x_err=sigma_x_err,
        angle_err=angle_err,
        scale_err=scale_err,
    )
