################################################################################
# tests/test_find_position.py
################################################################################

"""Tests for :meth:`PSF.find_position` validation, edge cases, motion smear, and logging."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from psfmodel.gaussian import GaussianPSF


def test_find_position_invalid_box_size_raises() -> None:
    """``find_position`` raises when ``box_size`` is not odd and positive in each axis."""

    psf = GaussianPSF()
    img = np.zeros((21, 21))
    for box in ((4, 5), (5, 4), (-1, 5), (5, -1)):
        with pytest.raises(ValueError) as exc_info:
            psf.find_position(img, box, (10, 10), bkgnd_degree=None, num_sigma=None)
        assert str(exc_info.value) == (
            f'box_size must have odd positive shape in each dimension, got {box}'
        )


def test_find_position_invalid_num_sigma_raises() -> None:
    """``find_position`` raises for non-positive or non-numeric ``num_sigma``."""

    psf = GaussianPSF()
    img = np.zeros((21, 21))

    with pytest.raises(ValueError) as exc_info:
        psf.find_position(img, (5, 5), (10, 10), bkgnd_degree=None, num_sigma=0.0)
    assert 'num_sigma must be > 0' in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        psf.find_position(img, (5, 5), (10, 10), bkgnd_degree=None, num_sigma=-1.0)
    assert 'num_sigma must be > 0' in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info_type:
        psf.find_position(img, (5, 5), (10, 10), bkgnd_degree=None, num_sigma='bad')  # type: ignore[arg-type]
    assert 'num_sigma must be a number or None' in str(exc_info_type.value)


def test_find_position_returns_none_when_starting_point_near_edge() -> None:
    """``find_position`` returns ``None`` when the box does not fit inside the image."""

    psf = GaussianPSF()
    img = np.zeros((11, 11))
    ret = psf.find_position(
        img,
        (7, 7),
        (0, 5),
        bkgnd_degree=None,
        num_sigma=None,
    )
    assert ret is None


def test_find_position_optimizer_failure_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed SciPy minimizer triggers a WARNING containing ``did not succeed``."""

    psf = GaussianPSF()
    gauss2d = psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    fake_result = MagicMock()
    fake_result.success = False
    fake_result.message = 'mocked optimizer failure'
    fake_result.x = np.array([0.0, 0.0, 1.0])
    fake_result.status = 2

    with (
        caplog.at_level(logging.WARNING, logger='psfmodel.psf'),
        patch('psfmodel.psf.sciopt.minimize', return_value=fake_result),
    ):
        ret = psf.find_position(
            gauss2d,
            gauss2d.shape,
            (gauss2d.shape[0] // 2, gauss2d.shape[1] // 2),
            bkgnd_degree=None,
            num_sigma=None,
        )

    assert ret is None
    assert any('did not succeed' in r.message for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_find_position_num_sigma_all_pixels_masked_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When outlier masking removes every pixel, ``find_position`` returns ``None``."""

    psf = GaussianPSF(logger=logging.getLogger('psfmodel.psf'), detailed_logging=True)
    img = np.ones((21, 21))
    with caplog.at_level(logging.INFO, logger='psfmodel.psf'):
        ret = psf.find_position(
            img,
            img.shape,
            (10, 10),
            bkgnd_degree=None,
            num_sigma=0.01,
            max_bad_frac=0.99,
        )
    assert ret is None
    assert any('all pixels masked' in r.message for r in caplog.records)


def test_find_position_num_sigma_too_many_masked_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When too large a fraction of pixels is masked, ``find_position`` returns ``None``."""

    psf = GaussianPSF(logger=logging.getLogger('psfmodel.psf'), detailed_logging=True)
    img = np.ones((21, 21))
    with caplog.at_level(logging.INFO, logger='psfmodel.psf'):
        ret = psf.find_position(
            img,
            img.shape,
            (10, 10),
            bkgnd_degree=None,
            num_sigma=0.05,
            max_bad_frac=0.2,
        )
    assert ret is None
    assert any('too many pixels masked' in r.message for r in caplog.records)


def test_find_position_detailed_logging_emits_info(caplog: pytest.LogCaptureFixture) -> None:
    """With ``detailed_logging=True``, ``find_position`` logs at INFO for key steps."""

    psf = GaussianPSF(logger=logging.getLogger('psfmodel.psf'), detailed_logging=True)
    gauss2d = psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    with caplog.at_level(logging.INFO, logger='psfmodel.psf'):
        ret = psf.find_position(
            gauss2d,
            gauss2d.shape,
            (gauss2d.shape[0] // 2, gauss2d.shape[1] // 2),
            bkgnd_degree=None,
            num_sigma=None,
        )
    assert ret is not None
    messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any('find_position: entering' in m for m in messages)
    assert any('find_position returning' in m for m in messages)


def test_gaussian_eval_rect_with_movement_smears(symmetric_psf: GaussianPSF) -> None:
    """Non-zero ``movement`` exercises motion smearing in :meth:`GaussianPSF.eval_rect`."""

    still = symmetric_psf.eval_rect((19, 19), movement=None)
    smeared = symmetric_psf.eval_rect((19, 19), movement=(0.5, 0.3))
    assert smeared.shape == (19, 19)
    assert np.sum(smeared) == pytest.approx(1.0)
    assert not np.allclose(smeared, still)


def test_gaussian_eval_rect_movement_small_num_steps_branch(symmetric_psf: GaussianPSF) -> None:
    """A small movement relative to ``movement_granularity`` uses the ``num_steps == 0`` path."""

    rect = symmetric_psf.eval_rect(
        (19, 19),
        movement=(0.05, 0.05),
        movement_granularity=0.1,
    )
    assert rect.shape == (19, 19)
    assert np.sum(rect) == pytest.approx(1.0)


def test_find_position_num_sigma_rejects_outlier_pixel(default_psf: GaussianPSF) -> None:
    """``num_sigma`` can mask a bright outlier while still returning a good centroid."""

    gauss2d = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    contaminated = gauss2d.copy()
    contaminated[3, 3] += 200.0
    ret = default_psf.find_position(
        contaminated,
        contaminated.shape,
        (10, 10),
        bkgnd_degree=None,
        num_sigma=5.0,
        max_bad_frac=0.99,
    )
    assert ret is not None
    assert ret[0] == pytest.approx(10.5, abs=0.5)
    assert ret[1] == pytest.approx(10.5, abs=0.5)


# ---------------------------------------------------------------------------
# Quality-metric tests (reduced_chi2, noise_rms, peak_snr, residual_rss)
# ---------------------------------------------------------------------------


def test_find_position_quality_metrics_noise_free(default_psf: GaussianPSF) -> None:
    """Quality metrics are near-zero for a noise-free Gaussian image."""

    img = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret = default_psf.find_position(img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None)
    assert ret is not None
    _, _, details = ret
    assert details['residual_rss'] < 1e-10
    assert details['reduced_chi2'] < 1e-10
    assert details['noise_rms'] < 1e-5
    assert details['peak_snr'] > 1e6


def test_find_position_quality_metrics_noisy() -> None:
    """``reduced_chi2`` approximates per-pixel noise variance for a noisy fit."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    rng = np.random.default_rng(42)
    noise_std = 0.05
    img = psf.eval_rect((21, 21), scale=1.0) + rng.normal(0, noise_std, (21, 21))
    ret = psf.find_position(img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None)
    assert ret is not None
    _, _, details = ret
    assert details['reduced_chi2'] == pytest.approx(noise_std**2, rel=0.3)
    assert details['noise_rms'] == pytest.approx(noise_std, rel=0.2)
    assert details['peak_snr'] == pytest.approx(1.0 / noise_std, rel=0.3)


def test_find_position_quality_metrics_keys_present(default_psf: GaussianPSF) -> None:
    """All four quality-metric keys are present in the returned details dict."""

    img = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret = default_psf.find_position(img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None)
    assert ret is not None
    _, _, details = ret
    for key in ('residual_rss', 'reduced_chi2', 'noise_rms', 'peak_snr'):
        assert key in details
        assert details[key] >= 0.0


# ---------------------------------------------------------------------------
# Parameter-uncertainty tests (x_err, y_err, scale_err, base_err, *_err)
# ---------------------------------------------------------------------------


def test_find_position_position_uncertainties_non_negative(default_psf: GaussianPSF) -> None:
    """Position and scale uncertainties are non-negative for a clean Gaussian fit."""

    img = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret = default_psf.find_position(img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None)
    assert ret is not None
    _, _, details = ret
    assert details['x_err'] >= 0.0
    assert details['y_err'] >= 0.0
    assert details['scale_err'] >= 0.0


def test_find_position_position_uncertainties_small_noise_free(default_psf: GaussianPSF) -> None:
    """Position uncertainties are negligible for a noise-free image."""

    img = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret = default_psf.find_position(img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None)
    assert ret is not None
    _, _, details = ret
    assert details['x_err'] < 1e-3
    assert details['y_err'] < 1e-3


def test_find_position_base_err_zero_when_base_fixed(default_psf: GaussianPSF) -> None:
    """``base_err`` is exactly 0.0 when ``allow_nonzero_base=False``."""

    img = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret = default_psf.find_position(
        img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None, allow_nonzero_base=False
    )
    assert ret is not None
    assert ret[2]['base_err'] == 0.0


def test_find_position_base_err_non_negative_when_base_free(default_psf: GaussianPSF) -> None:
    """``base_err`` is non-negative when ``allow_nonzero_base=True``."""

    img = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret = default_psf.find_position(
        img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None, allow_nonzero_base=True
    )
    assert ret is not None
    assert ret[2]['base_err'] >= 0.0


def test_find_position_additional_param_err_keys_present() -> None:
    """GaussianPSF with floating sigma includes ``sigma_y_err`` and ``sigma_x_err`` in details."""

    psf = GaussianPSF(
        sigma=(None, None),
        sigma_y_range=(0.5, 3.0),
        sigma_x_range=(0.5, 3.0),
    )
    img = psf.eval_rect((21, 21), scale=2.0, sigma=(1.5, 1.5))
    ret = psf.find_position(img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None)
    assert ret is not None
    _, _, details = ret
    assert 'sigma_y_err' in details
    assert 'sigma_x_err' in details
    assert details['sigma_y_err'] >= 0.0
    assert details['sigma_x_err'] >= 0.0


def test_find_position_uncertainty_decreases_with_snr() -> None:
    """Higher-SNR images produce smaller position uncertainties."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    rng = np.random.default_rng(99)
    noise = rng.normal(0, 0.1, (21, 21))
    img_low = psf.eval_rect((21, 21), scale=0.5) + noise
    img_high = psf.eval_rect((21, 21), scale=5.0) + noise

    ret_low = psf.find_position(img_low, (21, 21), (10, 10), bkgnd_degree=None, num_sigma=None)
    ret_high = psf.find_position(img_high, (21, 21), (10, 10), bkgnd_degree=None, num_sigma=None)
    assert ret_low is not None
    assert ret_high is not None
    assert ret_low[2]['x_err'] > ret_high[2]['x_err']
    assert ret_low[2]['y_err'] > ret_high[2]['y_err']


# ---------------------------------------------------------------------------
# compute_uncertainty flag tests
# ---------------------------------------------------------------------------


def test_find_position_compute_uncertainty_false_err_keys_are_nan(
    default_psf: GaussianPSF,
) -> None:
    """With ``compute_uncertainty=False``, ``x_err``, ``y_err``, and ``scale_err`` are NaN."""

    img = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret = default_psf.find_position(
        img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None, compute_uncertainty=False
    )
    assert ret is not None
    _, _, details = ret
    assert np.isnan(details['x_err'])
    assert np.isnan(details['y_err'])
    assert np.isnan(details['scale_err'])


def test_find_position_compute_uncertainty_false_position_and_metrics_unchanged(
    default_psf: GaussianPSF,
) -> None:
    """Skipping uncertainty does not affect the fitted position or quality metrics."""

    img = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret_with = default_psf.find_position(
        img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None, compute_uncertainty=True
    )
    ret_without = default_psf.find_position(
        img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None, compute_uncertainty=False
    )
    assert ret_with is not None
    assert ret_without is not None

    y_with, x_with, d_with = ret_with
    y_without, x_without, d_without = ret_without

    assert y_without == pytest.approx(y_with)
    assert x_without == pytest.approx(x_with)
    for key in ('residual_rss', 'reduced_chi2', 'noise_rms', 'peak_snr'):
        assert d_without[key] == pytest.approx(d_with[key])


def test_find_position_compute_uncertainty_false_base_fixed_base_err_zero(
    default_psf: GaussianPSF,
) -> None:
    """``base_err`` is 0.0 when base is not a free parameter, regardless of the flag."""

    img = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret = default_psf.find_position(
        img,
        img.shape,
        (10, 10),
        bkgnd_degree=None,
        num_sigma=None,
        allow_nonzero_base=False,
        compute_uncertainty=False,
    )
    assert ret is not None
    assert ret[2]['base_err'] == 0.0


def test_find_position_compute_uncertainty_false_free_base_err_nan() -> None:
    """``base_err`` is NaN when ``allow_nonzero_base=True`` and ``compute_uncertainty=False``."""

    psf = GaussianPSF(sigma=(1.0, 1.0))
    img = psf.eval_rect((21, 21), scale=2.0)
    ret = psf.find_position(
        img,
        img.shape,
        (10, 10),
        bkgnd_degree=None,
        num_sigma=None,
        allow_nonzero_base=True,
        compute_uncertainty=False,
    )
    assert ret is not None
    assert np.isnan(ret[2]['base_err'])


def test_find_position_compute_uncertainty_false_additional_param_errs_nan() -> None:
    """``sigma_y_err`` and ``sigma_x_err`` are NaN when ``compute_uncertainty=False``."""

    psf = GaussianPSF(
        sigma=(None, None),
        sigma_y_range=(0.5, 3.0),
        sigma_x_range=(0.5, 3.0),
    )
    img = psf.eval_rect((21, 21), scale=2.0, sigma=(1.5, 1.5))
    ret = psf.find_position(
        img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None, compute_uncertainty=False
    )
    assert ret is not None
    _, _, details = ret
    assert np.isnan(details['sigma_y_err'])
    assert np.isnan(details['sigma_x_err'])


def test_find_position_compute_uncertainty_default_is_true(default_psf: GaussianPSF) -> None:
    """The default (no ``compute_uncertainty`` kwarg) produces finite ``x_err`` and ``y_err``."""

    img = default_psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret = default_psf.find_position(img, img.shape, (10, 10), bkgnd_degree=None, num_sigma=None)
    assert ret is not None
    _, _, details = ret
    assert np.isfinite(details['x_err'])
    assert np.isfinite(details['y_err'])
    assert np.isfinite(details['scale_err'])
