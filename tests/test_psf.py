################################################################################
# tests/test_psf.py
################################################################################

"""Tests for :mod:`psfmodel.psf` helpers (background gradient and validation)."""

from __future__ import annotations

import numpy as np
import numpy.ma as ma
import numpy.testing as npt
import pytest

from psfmodel import PSF


@pytest.mark.parametrize(
    ('shape', 'order', 'expected_msg'),
    [
        ((3, -1), 1, 'Image must have odd positive shape in each dimension, got (3, -1)'),
        ((-1, 3), 1, 'Image must have odd positive shape in each dimension, got (-1, 3)'),
        ((1, 2), 1, 'Image must have odd positive shape in each dimension, got (1, 2)'),
        ((2, 1), 1, 'Image must have odd positive shape in each dimension, got (2, 1)'),
        ((1, 1), -5, 'Order must be non-negative, got -5'),
    ],
)
def test_bkgnd_gradient_coeffs_value_errors(
    shape: tuple[int, int],
    order: int,
    expected_msg: str,
) -> None:
    """``_background_gradient_coeffs`` validates ``shape`` and ``order``."""

    with pytest.raises(ValueError) as exc_info:
        PSF._background_gradient_coeffs(shape, order)
    assert str(exc_info.value) == expected_msg


def test_bkgnd_gradient_coeffs_success() -> None:
    """``_background_gradient_coeffs`` returns expected low-order layouts."""

    ret = PSF._background_gradient_coeffs((1, 1), 1)
    exp = np.array([[[1, 0, 0]]])
    assert np.all(ret == exp)

    ret = PSF._background_gradient_coeffs((1, 1), 2)
    exp = np.array([[[1, 0, 0, 0, 0, 0]]])
    assert np.all(ret == exp)

    ret = PSF._background_gradient_coeffs((1, 1), 3)
    exp = np.array([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    assert np.all(ret == exp)

    ret = PSF._background_gradient_coeffs((3, 3), 1)
    # fmt: off
    exp = np.array([[[1., -1., -1.],
                     [1.,  0., -1.],
                     [1.,  1., -1.]],

                    [[1., -1.,  0.],
                     [1.,  0.,  0.],
                     [1.,  1.,  0.]],

                    [[1., -1.,  1.],
                     [1.,  0.,  1.],
                     [1.,  1.,  1.]]])
    # fmt: on
    assert np.all(ret == exp)

    ret = PSF._background_gradient_coeffs((3, 3), 2)
    # fmt: off
    exp = np.array([[[1., -1., -1.,  1.,  1.,  1.],
                     [1.,  0., -1.,  0., -0.,  1.],
                     [1.,  1., -1.,  1., -1.,  1.]],

                    [[1., -1.,  0.,  1., -0.,  0.],
                     [1.,  0.,  0.,  0.,  0.,  0.],
                     [1.,  1.,  0.,  1.,  0.,  0.]],

                    [[1., -1.,  1.,  1., -1.,  1.],
                     [1.,  0.,  1.,  0.,  0.,  1.],
                     [1.,  1.,  1.,  1.,  1.,  1.]]])
    # fmt: on
    assert np.all(ret == exp)


def test_background_gradient_fit() -> None:
    """``background_gradient_fit`` fits quadratics, honors masks, and validates input."""

    with pytest.raises(ValueError) as exc_info:
        PSF.background_gradient_fit(np.zeros((5,)))
    assert str(exc_info.value) == 'Image must be 2-D, got (5,)'

    with pytest.raises(ValueError) as exc_info:
        PSF.background_gradient_fit(np.zeros((5, 4)))
    assert str(exc_info.value) == 'Image must have odd positive shape in each dimension, got (5, 4)'

    with pytest.raises(ValueError) as exc_info:
        PSF.background_gradient_fit(np.zeros((4, 5)))
    assert str(exc_info.value) == 'Image must have odd positive shape in each dimension, got (4, 5)'

    with pytest.raises(ValueError) as exc_info:
        PSF.background_gradient_fit(np.zeros((5, 5)), order=-10)
    assert str(exc_info.value) == 'Order must be non-negative, got -10'

    # Unmasked
    img = 3 * (np.arange(5.0)[:, np.newaxis] - 2) ** 2 + 2 * (np.arange(5.0)[np.newaxis, :] - 2)
    bkgnd_params, img_mask = PSF.background_gradient_fit(img)
    assert bkgnd_params is not None
    assert img_mask is not None
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 0
    img2 = PSF.background_gradient((5, 5), bkgnd_params)
    npt.assert_array_almost_equal(img, img2)

    bkgnd_params, img_mask = PSF.background_gradient_fit(img, order=3)
    assert bkgnd_params is not None
    assert img_mask is not None
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3, 0, 0, 0, 0]))
    assert np.sum(img_mask) == 0
    img2 = PSF.background_gradient((5, 5), bkgnd_params)
    npt.assert_array_almost_equal(img, img2)

    # All masked
    img = np.zeros((5, 5)).view(ma.MaskedArray)
    img[:, :] = ma.masked
    bkgnd_params, img_mask = PSF.background_gradient_fit(img)
    assert bkgnd_params is None
    assert img_mask is None

    # Ignore center
    img = 3 * (np.arange(5.0)[:, np.newaxis] - 2) ** 2 + 2 * (np.arange(5.0)[np.newaxis, :] - 2)
    img[2, 2] = 1000
    bkgnd_params, img_mask = PSF.background_gradient_fit(img)
    assert bkgnd_params is not None
    assert img_mask is not None
    with np.testing.assert_raises(AssertionError):  # Array not equal
        npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 0
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=0)
    assert bkgnd_params is not None
    assert img_mask is not None
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 1
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=1)
    assert bkgnd_params is not None
    assert img_mask is not None
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 9
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=(1, 1))
    assert bkgnd_params is not None
    assert img_mask is not None
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 9
    img = img.view(ma.MaskedArray)
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=(0, 1))
    assert bkgnd_params is not None
    assert img_mask is not None
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 3
    assert np.sum(img_mask[0]) == 0
    assert np.sum(img_mask[1]) == 0
    assert np.sum(img_mask[2]) == 3
    assert np.sum(img_mask[3]) == 0
    assert np.sum(img_mask[4]) == 0
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=2)
    assert bkgnd_params is None
    assert img_mask is None
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=3)
    assert bkgnd_params is None
    assert img_mask is None

    # Removal of bad pixels
    img[:] = 3 * (np.arange(5.0)[:, np.newaxis] - 2) ** 2 + 2 * (np.arange(5.0)[np.newaxis, :] - 2)
    img = img.view(ma.MaskedArray)
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=5)
    assert bkgnd_params is not None
    assert img_mask is not None
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 0
    img[2, 2] = 10000
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=4)
    assert bkgnd_params is not None
    assert img_mask is not None
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 1
    img[0, 0] = 100000
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=3)
    assert bkgnd_params is not None
    assert img_mask is not None
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 2
    img[0, 4] = 10000
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=3)
    assert bkgnd_params is not None
    assert img_mask is not None
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 3
