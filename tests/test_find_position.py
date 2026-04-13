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
            psf.find_position(img, box, (10, 10), bkgnd_degree=None, num_sigma=0)
        assert str(exc_info.value) == (
            f'box_size must have odd positive shape in each dimension, got {box}'
        )


def test_find_position_returns_none_when_starting_point_near_edge() -> None:
    """``find_position`` returns ``None`` when the box does not fit inside the image."""

    psf = GaussianPSF()
    img = np.zeros((11, 11))
    ret = psf.find_position(
        img,
        (7, 7),
        (0, 5),
        bkgnd_degree=None,
        num_sigma=0,
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
            num_sigma=0,
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
            num_sigma=0,
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
    contaminated[10, 10] += 200.0
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
