################################################################################
# tests/test_psf.py
################################################################################

import numpy as np
import numpy.ma as ma
import numpy.testing as npt
import pytest

from psfmodel import PSF


def test_bkgnd_gradient_coeffs():
    with pytest.raises(ValueError):
        PSF._background_gradient_coeffs((3, -1), 1)
    with pytest.raises(ValueError):
        PSF._background_gradient_coeffs((-1, 3), 1)
    with pytest.raises(ValueError):
        PSF._background_gradient_coeffs((1, 2), 1)
    with pytest.raises(ValueError):
        PSF._background_gradient_coeffs((2, 1), 1)
    with pytest.raises(ValueError):
        PSF._background_gradient_coeffs((1, 1), -5)

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
    exp = np.array([[[1., -1., -1.],
                     [1.,  0., -1.],
                     [1.,  1., -1.]],

                    [[1., -1.,  0.],
                     [1.,  0.,  0.],
                     [1.,  1.,  0.]],

                    [[1., -1.,  1.],
                     [1.,  0.,  1.],
                     [1.,  1.,  1.]]])
    assert np.all(ret == exp)

    ret = PSF._background_gradient_coeffs((3, 3), 2)
    exp = np.array([[[1., -1., -1.,  1.,  1.,  1.],
                     [1.,  0., -1.,  0., -0.,  1.],
                     [1.,  1., -1.,  1., -1.,  1.]],

                    [[1., -1.,  0.,  1., -0.,  0.],
                     [1.,  0.,  0.,  0.,  0.,  0.],
                     [1.,  1.,  0.,  1.,  0.,  0.]],

                    [[1., -1.,  1.,  1., -1.,  1.],
                     [1.,  0.,  1.,  0.,  0.,  1.],
                     [1.,  1.,  1.,  1.,  1.,  1.]]])
    assert np.all(ret == exp)


def test_background_gradient_fit():
    with pytest.raises(ValueError):
        PSF.background_gradient_fit(np.zeros((5,)))
    with pytest.raises(ValueError):
        PSF.background_gradient_fit(np.zeros((5, 4)))
    with pytest.raises(ValueError):
        PSF.background_gradient_fit(np.zeros((4, 5)))
    with pytest.raises(ValueError):
        PSF.background_gradient_fit(np.zeros((5, 5)), order=-10)

    # Unmasked
    img = 3*(np.arange(5.)[:, np.newaxis]-2)**2 + 2*(np.arange(5.)[np.newaxis, :]-2)
    bkgnd_params, img_mask = PSF.background_gradient_fit(img)
    npt.assert_array_almost_equal(np.array(bkgnd_params), np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 0
    img2 = PSF.background_gradient((5, 5), bkgnd_params)
    npt.assert_array_almost_equal(img, img2)

    bkgnd_params, img_mask = PSF.background_gradient_fit(img, order=3)
    npt.assert_array_almost_equal(np.array(bkgnd_params),
                                  np.array([0, 2, 0, 0, 0, 3, 0, 0, 0, 0]))
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
    img = 3*(np.arange(5.)[:, np.newaxis]-2)**2 + 2*(np.arange(5.)[np.newaxis, :]-2)
    img[2, 2] = 1000
    bkgnd_params, img_mask = PSF.background_gradient_fit(img)
    with np.testing.assert_raises(AssertionError):  # Array not equal
        npt.assert_array_almost_equal(np.array(bkgnd_params),
                                      np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 0
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=0)
    npt.assert_array_almost_equal(np.array(bkgnd_params),
                                  np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 1
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=1)
    npt.assert_array_almost_equal(np.array(bkgnd_params),
                                  np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 9
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=(1, 1))
    npt.assert_array_almost_equal(np.array(bkgnd_params),
                                  np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 9
    img = img.view(ma.MaskedArray)
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=(0, 1))
    npt.assert_array_almost_equal(np.array(bkgnd_params),
                                  np.array([0, 2, 0, 0, 0, 3]))
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
    img[:] = 3*(np.arange(5.)[:, np.newaxis]-2)**2 + 2*(np.arange(5.)[np.newaxis, :]-2)
    img = img.view(ma.MaskedArray)
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=5)
    npt.assert_array_almost_equal(np.array(bkgnd_params),
                                  np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 0
    img[2, 2] = 10000
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=4)
    npt.assert_array_almost_equal(np.array(bkgnd_params),
                                  np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 1
    img[0, 0] = 100000
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=3)
    npt.assert_array_almost_equal(np.array(bkgnd_params),
                                  np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 2
    img[0, 4] = 10000
    bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=3)
    npt.assert_array_almost_equal(np.array(bkgnd_params),
                                  np.array([0, 2, 0, 0, 0, 3]))
    assert np.sum(img_mask) == 3
