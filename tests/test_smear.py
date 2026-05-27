################################################################################
# tests/test_smear.py
################################################################################

"""Tests for motion-smear behavior in :meth:`PSF._eval_rect_smeared`.

Covers the base-class loop (validation, sample-count heuristic, base/scale
handling) and the Gaussian-specific separable override.
"""

from __future__ import annotations

import numpy as np
import pytest

from psfmodel.gaussian import GaussianPSF
from psfmodel.psf import PSF

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('granularity', [0.0, -0.1, -1.0])
def test_eval_rect_smeared_nonpositive_granularity_raises(granularity: float) -> None:
    """``movement_granularity`` must be strictly positive; otherwise raises ``ValueError``."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    with pytest.raises(ValueError) as exc_info:
        psf.eval_rect((19, 19), movement=(0.3, 0.3), movement_granularity=granularity)
    assert 'movement_granularity must be positive' in str(exc_info.value)


# ---------------------------------------------------------------------------
# Sum preservation / scale-and-base handling
# ---------------------------------------------------------------------------


def test_eval_rect_smeared_sum_preserved_with_movement() -> None:
    """Smearing preserves total flux: ``sum`` equals ``scale`` (with ``base=0``)."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    rect = psf.eval_rect((21, 21), movement=(0.6, 0.4), scale=2.5)
    assert np.sum(rect) == pytest.approx(2.5, rel=1e-3)


def test_eval_rect_smeared_base_added_once_per_pixel() -> None:
    """A non-zero ``base`` is added once per output pixel after the smear average."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    n_pix = 21 * 21
    base = 0.1
    rect = psf.eval_rect((21, 21), movement=(0.6, 0.4), scale=2.5, base=base)
    assert np.sum(rect) == pytest.approx(2.5 + base * n_pix, rel=1e-3)


def test_eval_rect_smeared_base_only_constant() -> None:
    """With ``scale=0`` and ``base=b``, every pixel equals ``b`` exactly."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    rect = psf.eval_rect((11, 11), movement=(0.6, 0.4), scale=0.0, base=0.25)
    np.testing.assert_allclose(rect, np.full((11, 11), 0.25), rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Sample-count heuristic (ceil-based, midpoint sampling).
# ---------------------------------------------------------------------------


def test_eval_rect_smeared_movement_just_above_granularity_smears() -> None:
    """Movement just larger than the granularity (e.g. 0.15 vs 0.1) actually smears.

    Under the old ``int()`` truncation, ``movement=(0.15, 0)`` with
    ``granularity=0.1`` produced ``num_steps=1`` (step 0.15, wider than the
    requested granularity).  Under the ``math.ceil`` fix, ``S>=2`` so the
    sampling honors the requested granularity.
    """

    psf = GaussianPSF(sigma=(1.0, 1.0))
    still = psf.eval_rect((21, 21), movement=None)
    smeared = psf.eval_rect((21, 21), movement=(0.15, 0.0), movement_granularity=0.1)
    assert not np.allclose(smeared, still)


def test_eval_rect_smeared_movement_below_granularity_passes_through() -> None:
    """Movement strictly below the granularity collapses to a single midpoint sample."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    still = psf.eval_rect((21, 21), movement=None)
    rect = psf.eval_rect((21, 21), movement=(0.05, 0.05), movement_granularity=0.1)
    np.testing.assert_allclose(rect, still, rtol=1e-12, atol=1e-12)


def test_eval_rect_smeared_zero_movement_returns_unsmeared() -> None:
    """A movement tuple of ``(0, 0)`` returns the unsmeared rectangle exactly."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    still = psf.eval_rect((21, 21), movement=None)
    zero_move = psf.eval_rect((21, 21), movement=(0.0, 0.0))
    np.testing.assert_array_equal(still, zero_move)


# ---------------------------------------------------------------------------
# Gaussian-specific separable override matches the base-class loop.
# ---------------------------------------------------------------------------


def test_gaussian_smear_axis_aligned_matches_base_loop() -> None:
    """For ``angle=0`` the separable override matches the loop within float tolerance."""

    psf = GaussianPSF(sigma=(1.5, 1.0))
    rect_size = (21, 21)
    offset = (0.5, 0.5)
    movement = (0.7, 0.3)
    granularity = 0.1

    fast = psf.eval_rect(
        rect_size,
        offset=offset,
        movement=movement,
        movement_granularity=granularity,
        scale=3.0,
        base=0.1,
    )
    slow = PSF._eval_rect_smeared(
        psf,
        rect_size,
        offset=offset,
        movement=movement,
        movement_granularity=granularity,
        scale=3.0,
        base=0.1,
    )
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-12)


def test_gaussian_smear_floating_sigma_override_kwarg() -> None:
    """The separable override honors ``sigma`` passed at call time (float sigma)."""

    psf = GaussianPSF()
    fast = psf.eval_rect((19, 19), movement=(0.5, 0.5), sigma=(1.2, 0.8), scale=1.0)
    slow = PSF._eval_rect_smeared(
        psf, (19, 19), offset=(0.5, 0.5), movement=(0.5, 0.5), sigma=(1.2, 0.8), scale=1.0
    )
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-12)


def test_gaussian_smear_angle_nonzero_falls_back_to_loop() -> None:
    """When ``angle != 0`` the override falls back to the base loop and matches it."""

    psf = GaussianPSF(sigma=(1.5, 1.0), angle=np.pi / 6)
    fast = psf.eval_rect((21, 21), movement=(0.4, 0.2), scale=1.0)
    slow = PSF._eval_rect_smeared(
        psf,
        (21, 21),
        offset=(0.5, 0.5),
        movement=(0.4, 0.2),
        scale=1.0,
    )
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-12)


def test_gaussian_smear_angle_nonzero_smears() -> None:
    """A non-zero angle still produces a smeared result that differs from no smear."""

    psf = GaussianPSF(sigma=(1.5, 1.0), angle=np.pi / 6)
    still = psf.eval_rect((21, 21), movement=None)
    smeared = psf.eval_rect((21, 21), movement=(0.5, 0.3))
    assert not np.allclose(smeared, still)


# ---------------------------------------------------------------------------
# Output dtype and shape
# ---------------------------------------------------------------------------


def test_eval_rect_smeared_returns_float64_array() -> None:
    """The smeared result is a ``float64`` ndarray of the requested shape."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    rect = psf.eval_rect((17, 19), movement=(0.4, 0.2))
    assert rect.dtype == np.float64
    assert rect.shape == (17, 19)


# ---------------------------------------------------------------------------
# Asymmetric movement
# ---------------------------------------------------------------------------


def test_eval_rect_smeared_y_only_movement_extends_y() -> None:
    """Pure Y movement spreads flux only along the Y axis (column-summed equal to still)."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    still = psf.eval_rect((21, 21), movement=None)
    smeared = psf.eval_rect((21, 21), movement=(0.6, 0.0))
    # X-marginal (summing along Y) is unchanged by pure Y smear.
    np.testing.assert_allclose(smeared.sum(axis=0), still.sum(axis=0), rtol=1e-6)


def test_eval_rect_smeared_x_only_movement_extends_x() -> None:
    """Pure X movement spreads flux only along the X axis (row-summed equal to still)."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    still = psf.eval_rect((21, 21), movement=None)
    smeared = psf.eval_rect((21, 21), movement=(0.0, 0.6))
    np.testing.assert_allclose(smeared.sum(axis=1), still.sum(axis=1), rtol=1e-6)


# ---------------------------------------------------------------------------
# Base-class loop paths reached directly (bypassing the GaussianPSF override).
# ---------------------------------------------------------------------------


def test_base_loop_zero_movement_returns_unsmeared() -> None:
    """``PSF._eval_rect_smeared`` returns ``_eval_rect`` unchanged when movement is ``None``."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    direct = psf._eval_rect((19, 19), offset=(0.5, 0.5), scale=2.0)
    via_loop = PSF._eval_rect_smeared(psf, (19, 19), offset=(0.5, 0.5), scale=2.0, movement=None)
    np.testing.assert_array_equal(direct, via_loop)


def test_base_loop_nonpositive_granularity_raises() -> None:
    """``PSF._eval_rect_smeared`` validates ``movement_granularity > 0`` itself."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    with pytest.raises(ValueError) as exc_info:
        PSF._eval_rect_smeared(
            psf,
            (19, 19),
            offset=(0.5, 0.5),
            movement=(0.3, 0.3),
            movement_granularity=0.0,
        )
    assert 'movement_granularity must be positive' in str(exc_info.value)


# ---------------------------------------------------------------------------
# Sigma resolution branches in the Gaussian override.
# ---------------------------------------------------------------------------


def test_gaussian_smear_resolves_separate_sigma_args() -> None:
    """``sigma_y`` and ``sigma_x`` per-call overrides drive the separable path."""

    psf = GaussianPSF()
    rect = psf.eval_rect((19, 19), movement=(0.4, 0.4), sigma_y=1.0, sigma_x=1.5, scale=1.0)
    assert np.sum(rect) == pytest.approx(1.0, rel=1e-3)


def test_gaussian_smear_conflict_instance_and_per_call_sigma_raises() -> None:
    """Specifying both instance sigma and per-call sigma raises ``ValueError``."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    with pytest.raises(ValueError) as exc_info:
        psf.eval_rect((19, 19), movement=(0.4, 0.4), sigma_y=1.5)
    assert 'Cannot specify both sigma' in str(exc_info.value)


def test_gaussian_smear_missing_sigma_raises() -> None:
    """An instance with no fixed sigma and no per-call sigma raises ``ValueError``."""

    psf = GaussianPSF()
    with pytest.raises(ValueError) as exc_info:
        psf.eval_rect((19, 19), movement=(0.4, 0.4))
    assert 'Sigma X and Y must be specified' in str(exc_info.value)


def test_gaussian_smear_per_call_angle_override_falls_back() -> None:
    """Per-call ``angle`` override routes through the loop fallback when non-zero."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    fast = psf.eval_rect((19, 19), movement=(0.4, 0.2), angle=np.pi / 4)
    slow = PSF._eval_rect_smeared(
        psf,
        (19, 19),
        offset=(0.5, 0.5),
        movement=(0.4, 0.2),
        angle=np.pi / 4,
    )
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-12)
