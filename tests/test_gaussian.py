################################################################################
# tests/test_gaussian.py
################################################################################

import numpy as np
import numpy.testing as npt
import pytest
import scipy.integrate as integrate

from psfmodel.gaussian import GaussianPSF


def test_gaussian_1d():
    assert GaussianPSF.gaussian_1d(0.) == pytest.approx(0.39894228)
    assert GaussianPSF.gaussian_1d(0., scale=2.) == pytest.approx(0.39894228 * 2)
    assert GaussianPSF.gaussian_1d(0., base=-5.) == pytest.approx(0.39894228 - 5.)
    assert GaussianPSF.gaussian_1d(0., scale=2., base=-5.) == \
        pytest.approx(0.39894228 * 2 - 5.)
    assert GaussianPSF.gaussian_1d(1., mean=1.) == pytest.approx(0.39894228)
    assert GaussianPSF.gaussian_1d(1., mean=1., scale=2.) == pytest.approx(0.39894228 * 2)
    assert GaussianPSF.gaussian_1d(1.) == pytest.approx(0.24197072)
    assert GaussianPSF.gaussian_1d(2., sigma=2.) == pytest.approx(0.24197072 / 2)
    npt.assert_array_almost_equal(GaussianPSF.gaussian_1d(np.array([0.])),
                                  np.array([0.39894228]))
    npt.assert_array_almost_equal(GaussianPSF.gaussian_1d(np.array([0., 1.])),
                                  np.array([0.39894228, 0.24197072]))
    npt.assert_array_almost_equal(GaussianPSF.gaussian_1d(np.array([[0., 1.], [1., 0.]])),
                                  np.array([[0.39894228, 0.24197072],
                                            [0.24197072, 0.39894228]]))
    assert integrate.quad(GaussianPSF.gaussian_1d, -10, 10)[0] == pytest.approx(1.)
    assert integrate.quad(lambda x: GaussianPSF.gaussian_1d(x, scale=2.), -10, 10)[0] == \
        pytest.approx(2.)


def test_gaussian_2d():
    assert GaussianPSF.gaussian_2d(0, 0) == pytest.approx(0.15915494309)
    assert GaussianPSF.gaussian_2d(0, 0, scale=2.) == pytest.approx(0.15915494309 * 2)
    assert GaussianPSF.gaussian_2d(0, 0, base=10.) == pytest.approx(0.15915494309 + 10)
    assert GaussianPSF.gaussian_2d(0, 0, scale=2., base=10.) == \
        pytest.approx(0.15915494309 * 2 + 10)
    assert GaussianPSF.gaussian_2d(1., 0., mean_y=1.) == pytest.approx(0.15915494309)
    assert GaussianPSF.gaussian_2d(0., 1., mean_x=1.) == pytest.approx(0.15915494309)
    assert integrate.dblquad(GaussianPSF.gaussian_2d, -10, 10, -10, 10)[0] == \
        pytest.approx(1.)
    assert integrate.dblquad(lambda x, y: GaussianPSF.gaussian_2d(x, y, scale=2.),
                             -10, 10, -10, 10)[0] == pytest.approx(2.)
    assert GaussianPSF.gaussian_2d(1., 0.) == pytest.approx(0.09653235263005391)
    assert GaussianPSF.gaussian_2d(1., 0., sigma_x=2) == pytest.approx(0.048266176315027)
    assert GaussianPSF.gaussian_2d(0., -1., sigma_x=2) != pytest.approx(0.048266176315027)
    assert GaussianPSF.gaussian_2d(0., -1., angle=np.pi/2) == pytest.approx(0.09653235263)
    npt.assert_array_almost_equal(GaussianPSF.gaussian_2d(np.array([0.]), np.array([0.])),
                                  np.array([0.15915494309]))
    npt.assert_array_almost_equal(GaussianPSF.gaussian_2d(np.array([[0.], [1.]]),
                                                          np.array([[0.], [0.]])),
                                  np.array([[0.15915494309], [0.09653235263005391]]))
    npt.assert_array_almost_equal(GaussianPSF.gaussian_2d(np.array([[[0.], [1.]],
                                                                    [[1.], [0.]]]),
                                                          np.array([[[0.], [0.]],
                                                                    [[0.], [0.]]])),
                                  np.array([[[0.15915494309], [0.09653235263005391]],
                                            [[0.09653235263005391], [0.15915494309]]]))

    y_coords = np.tile(np.arange(-10., 11.)/2, 21)
    x_coords = np.repeat(np.arange(-10., 11.)/2, 21)

    gauss2d2 = GaussianPSF.gaussian_2d(y_coords, x_coords, scale=2.,
                                       sigma_x=.25, sigma_y=.5,
                                       base=1.) / 4
    gauss2d3 = GaussianPSF.gaussian_2d(y_coords, x_coords, scale=2.,
                                       sigma_x=.5, sigma_y=.25,
                                       base=1.) / 4
    gauss2d2 = gauss2d2.reshape(21, 21)
    gauss2d3 = gauss2d3.reshape(21, 21)
    with pytest.raises(AssertionError):
        npt.assert_array_almost_equal(gauss2d2, gauss2d3)
    npt.assert_array_almost_equal(np.transpose(gauss2d2), gauss2d3)


def test_gaussian_integral_1d():
    g_0_1 = integrate.quad(GaussianPSF.gaussian_1d, 0., 1.)[0]
    g_n1_1 = integrate.quad(GaussianPSF.gaussian_1d, -1., 1.)[0]
    assert GaussianPSF.gaussian_integral_1d(0., 1.) == \
        pytest.approx(integrate.quad(GaussianPSF.gaussian_1d, 0., 1.)[0])
    assert GaussianPSF.gaussian_integral_1d(-1., 1.) == \
        pytest.approx(g_n1_1)
    assert GaussianPSF.gaussian_integral_1d(-1., 1., mean=2.) == \
        pytest.approx(integrate.quad(GaussianPSF.gaussian_1d, 1., 3.)[0])
    assert GaussianPSF.gaussian_integral_1d(-1., 1., scale=2.) == \
        pytest.approx(g_n1_1 * 2)
    assert GaussianPSF.gaussian_integral_1d(-1., 1., base=5.) == \
        pytest.approx(g_n1_1 + 5)
    assert GaussianPSF.gaussian_integral_1d(-1., 1., scale=2., base=5.) == \
        pytest.approx(g_n1_1 * 2 + 5)
    assert GaussianPSF.gaussian_integral_1d(0., 1.) == \
        pytest.approx(integrate.quad(GaussianPSF.gaussian_1d, 0., 1.)[0])
    assert GaussianPSF.gaussian_integral_1d(np.array([0., -1.]), np.array([1., 1.])) == \
        pytest.approx(np.array([g_0_1, g_n1_1]))
    ret = GaussianPSF.gaussian_integral_1d(np.array([[0., -1.], [-1., 0.]]),
                                           np.array([[1., 1.], [1., 1.]]))
    npt.assert_array_almost_equal(ret, np.array([[g_0_1, g_n1_1], [g_n1_1, g_0_1]]))


def test_gaussian_integral_2d():
    integ1 = integrate.dblquad(lambda y, x: GaussianPSF.gaussian_2d(y, x),
                               0., 3., -2., 1.)[0]
    integ2 = integrate.dblquad(lambda y, x: GaussianPSF.gaussian_2d(y, x),
                               -1., 3., -3., 2.)[0]
    assert GaussianPSF.gaussian_integral_2d(0., 3., -2., 1.) == pytest.approx(integ1)
    assert GaussianPSF.gaussian_integral_2d(0., 3., -2., 1., scale=2., base=5.) == \
        pytest.approx(integ1 * 2 + 5)
    ret = GaussianPSF.gaussian_integral_2d(np.array([0., -1.]), np.array([3., 3.]),
                                           np.array([-2., -3.]), np.array([1., 2.]))
    npt.assert_array_almost_equal(ret, np.array([integ1, integ2]))

    # def rot(y, x):
    #     ang = -np.pi/8
    #     c = np.cos(ang)
    #     s = np.sin(ang)
    #     x2 = c*x + s*y
    #     y2 = -s*x + c*y
    #     return y2, x2

    # y1, x1 = rot(-1, -1)
    # y2, x2 = rot(1, 1)
    # assert GaussianPSF.gaussian_integral_2d(y1+2, y2+2, x1-1, x2-1,
    #                                         mean_y=2., mean_x=-1.,
    #                                         sigma_x=2., angle=np.pi/8) == \
    #     pytest.approx(integrate.dblquad(
    #         lambda y, x: GaussianPSF.gaussian_2d(y, x, sigma_x=2),
    #         -1., 1., -1., 1.)[0])


def test_gaussian_eval_point():
    with pytest.raises(ValueError):
        GaussianPSF(sigma=(1, 1)).eval_point((0, 0), sigma=5)
    with pytest.raises(ValueError):
        GaussianPSF(sigma=(1, 1)).eval_point((0, 0), sigma_x=5)
    with pytest.raises(ValueError):
        GaussianPSF(sigma=(1, 1)).eval_point((0, 0), sigma_y=5)
    with pytest.raises(ValueError):
        GaussianPSF(sigma=None).eval_point((0, 0))
    with pytest.raises(ValueError):
        GaussianPSF(sigma=(None, 1)).eval_point((0, 0))
    with pytest.raises(ValueError):
        GaussianPSF(sigma=(1, None)).eval_point((0, 0))

    psf1 = GaussianPSF()
    psf2 = GaussianPSF(sigma=(1., None))
    psf3 = GaussianPSF(sigma=(None, 2.))
    psf4 = GaussianPSF(sigma=(1., 2.))
    psf5 = GaussianPSF(sigma=(1., 1.))

    assert psf1.eval_point((2, 3), sigma=(1., 2.)) == psf4.eval_point((2, 3))
    assert psf1.eval_point((2, 3), sigma=1.) == psf5.eval_point((2, 3))
    assert psf1.eval_point((2, 3), sigma_y=1., sigma_x=2.) == psf4.eval_point((2, 3))
    assert psf2.eval_point((2, 3), sigma_x=2.) == psf4.eval_point((2, 3))
    assert psf3.eval_point((2, 3), sigma_y=1.) == psf4.eval_point((2, 3))
    assert psf1.eval_point((2, 3), sigma=(1., 2.), base=1, scale=2, angle=np.pi/4) == \
        psf4.eval_point((2, 3), base=1, scale=2, angle=np.pi/4)

    assert psf1.eval_point((2, 3), sigma=(1., 2.), base=1, scale=2, angle=np.pi/4) == \
        pytest.approx(GaussianPSF.gaussian_2d(2, 3, sigma_y=1., sigma_x=2.,
                                              base=1, scale=2, angle=np.pi/4))

    ret = psf1.eval_point((np.array([1, 2]), np.array([2, 3])), sigma=(1., 2.))
    npt.assert_array_almost_equal(ret, np.array([psf4.eval_point((1, 2)),
                                                 psf4.eval_point((2, 3))]))


def test_gaussian_eval_pixel():
    with pytest.raises(ValueError):
        GaussianPSF(sigma=(1, 1)).eval_pixel((0, 0), sigma=5)
    with pytest.raises(ValueError):
        GaussianPSF(sigma=(1, 1)).eval_pixel((0, 0), sigma_x=5)
    with pytest.raises(ValueError):
        GaussianPSF(sigma=(1, 1)).eval_pixel((0, 0), sigma_y=5)
    with pytest.raises(ValueError):
        GaussianPSF().eval_pixel((0, 0))
    with pytest.raises(ValueError):
        GaussianPSF().eval_pixel((0, 0), sigma_x=5)
    with pytest.raises(ValueError):
        GaussianPSF().eval_pixel((0, 0), sigma_y=5)
    with pytest.raises(ValueError):
        GaussianPSF(sigma=None).eval_pixel((0, 0))
    with pytest.raises(ValueError):
        GaussianPSF(sigma=(1, None)).eval_pixel((0, 0))
    with pytest.raises(ValueError):
        GaussianPSF(sigma=(None, 1)).eval_pixel((0, 0))

    integ = GaussianPSF.gaussian_integral_2d(-.5, .5, -.5, .5, sigma_y=2., sigma_x=3.)
    integ2 = GaussianPSF.gaussian_integral_2d(-.5, .5, -.5, .5, sigma_y=2., sigma_x=2.)
    assert GaussianPSF(sigma=(2., 3.)).eval_pixel((0, 0)) == pytest.approx(integ)
    assert GaussianPSF(sigma=2.).eval_pixel((0, 0)) == pytest.approx(integ2)
    assert GaussianPSF().eval_pixel((0, 0), sigma=2.) == pytest.approx(integ2)
    assert GaussianPSF(sigma=(2., 3.)).eval_pixel((0, 0), offset=(0, .25)) == \
        pytest.approx(GaussianPSF.gaussian_integral_2d(0, 1, -.25, .75,
                                                       sigma_y=2., sigma_x=3.))
    assert GaussianPSF(sigma=(2., 3.)).eval_pixel((0, 0), scale=11.) == \
        pytest.approx(integ * 11)
    assert GaussianPSF(sigma=(2., 3.)).eval_pixel((0, 0), base=3.) == \
        pytest.approx(integ + 3)
    assert GaussianPSF().eval_pixel((0, 0), sigma=(2., 3.)) == \
        pytest.approx(integ)
    assert GaussianPSF(sigma=(2., None)).eval_pixel((0, 0), sigma_x=3.) == \
        pytest.approx(integ)
    assert GaussianPSF(sigma=(None, 3.)).eval_pixel((0, 0), sigma_y=2.) == \
        pytest.approx(integ)
    ret = GaussianPSF(sigma=(2., 3.)).eval_pixel((np.array([0, 0]), np.array([0, 0])))
    npt.assert_array_almost_equal(ret, np.array([integ, integ]))

    assert GaussianPSF(sigma=(2., 3.)).eval_pixel((0, 0), angle=np.pi/8) == \
        pytest.approx(GaussianPSF.gaussian_integral_2d(-.5, .5, -.5, .5,
                                                       sigma_y=2., sigma_x=3.,
                                                       angle=np.pi/8))


def test_gaussian_eval_rect():
    assert np.sum(GaussianPSF(sigma=(1, 1)).eval_rect((19, 19))) == pytest.approx(1)
    assert np.sum(GaussianPSF(sigma=(1, 1), angle=np.pi/4).eval_rect((19, 19))) == \
        pytest.approx(1)


@pytest.mark.parametrize('use_angular_params', [True, False])
@pytest.mark.parametrize('bkgnd_degree', [None, 0, 1, 2])
def test_gaussian_find_position(use_angular_params, bkgnd_degree):
    allow_nonzero_base = (bkgnd_degree is not None)

    # centered symmetric PSF, float sigma
    psf = GaussianPSF()
    gauss2d = psf.eval_rect((21, 21), scale=2., sigma=1.)
    # if allow_nonzero_base:
    #     psf._debug_opt = 10
    ret = psf.find_position(gauss2d, gauss2d.shape,
                            starting_point=((gauss2d.shape[0]//2,
                                             gauss2d.shape[1]//2)),
                            bkgnd_degree=bkgnd_degree,
                            allow_nonzero_base=allow_nonzero_base,
                            num_sigma=0,
                            use_angular_params=use_angular_params)
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2)
    assert ret[2]['sigma_y'] == pytest.approx(1., abs=5e-2)
    assert ret[2]['sigma_x'] == pytest.approx(1., abs=5e-2)
    assert ret[2]['scale'] == pytest.approx(2., abs=5e-2)

    # asymmetric PSF, float sigma
    gauss2d = psf.eval_rect((21, 21), scale=2., sigma=(2., 0.5))
    ret = psf.find_position(gauss2d, gauss2d.shape,
                            starting_point=((gauss2d.shape[0]//2,
                                             gauss2d.shape[1]//2)),
                            bkgnd_degree=bkgnd_degree,
                            bkgnd_ignore_center=(4, 4),
                            allow_nonzero_base=allow_nonzero_base,
                            num_sigma=0,
                            use_angular_params=use_angular_params)
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-4)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-4)
    assert ret[2]['sigma_y'] == pytest.approx(2., abs=5e-2)
    assert ret[2]['sigma_x'] == pytest.approx(0.5, abs=5e-2)
    assert ret[2]['scale'] == pytest.approx(2., abs=7e-2)

    # offset PSF created through mean, float sigma
    psf2 = GaussianPSF(mean=(0.5, 0.75))
    gauss2d = psf2.eval_rect((21, 21), scale=0.5, sigma=(0.5, 1.3))

    # find using non-offset PSF
    ret = psf.find_position(gauss2d, gauss2d.shape,
                            starting_point=((gauss2d.shape[0]//2,
                                             gauss2d.shape[1]//2)),
                            bkgnd_degree=bkgnd_degree,
                            bkgnd_ignore_center=(4, 4),
                            allow_nonzero_base=allow_nonzero_base,
                            num_sigma=0,
                            use_angular_params=use_angular_params)
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2 + 0.5, abs=1e-1)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2 + 0.75, abs=1e-1)
    assert ret[2]['sigma_y'] == pytest.approx(0.5, abs=5e-2)
    assert ret[2]['sigma_x'] == pytest.approx(1.3, abs=5e-2)
    assert ret[2]['scale'] == pytest.approx(0.5, abs=5e-2)

    # find using offset PSF
    ret = psf2.find_position(gauss2d, gauss2d.shape,
                             starting_point=((gauss2d.shape[0]//2,
                                              gauss2d.shape[1]//2)),
                             bkgnd_degree=bkgnd_degree,
                             bkgnd_ignore_center=(4, 4),
                             allow_nonzero_base=allow_nonzero_base,
                             num_sigma=0,
                             use_angular_params=use_angular_params)
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-1)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-1)
    assert ret[2]['sigma_y'] == pytest.approx(0.5, abs=5e-2)
    assert ret[2]['sigma_x'] == pytest.approx(1.3, abs=5e-2)
    assert ret[2]['scale'] == pytest.approx(0.5, abs=5e-2)

    # offset PSF created through eval_rect, float sigma
    psf2 = GaussianPSF()
    gauss2d = psf2.eval_rect((21, 21), offset=(0.21, -0.35), scale=1.5, sigma=(0.8, 1.3))
    ret = psf.find_position(gauss2d, gauss2d.shape,
                            starting_point=((gauss2d.shape[0]//2,
                                             gauss2d.shape[1]//2)),
                            bkgnd_degree=bkgnd_degree,
                            bkgnd_ignore_center=(4, 4),
                            allow_nonzero_base=allow_nonzero_base,
                            num_sigma=0,
                            use_angular_params=use_angular_params)
    assert ret[0] == pytest.approx(gauss2d.shape[0] // 2 + 0.21, abs=5e-2)
    assert ret[1] == pytest.approx(gauss2d.shape[1] // 2 - 0.35, abs=5e-2)
    assert ret[2]['sigma_y'] == pytest.approx(0.8, abs=1e-3)
    assert ret[2]['sigma_x'] == pytest.approx(1.3, abs=1e-3)
    assert ret[2]['scale'] == pytest.approx(1.5, abs=1e-2)

    # centered PSF, fixed sigma
    psf2 = GaussianPSF(sigma=(0.9, 1.5))
    gauss2d = psf2.eval_rect((21, 21), scale=1.5)
    ret = psf2.find_position(gauss2d, gauss2d.shape,
                             starting_point=((gauss2d.shape[0]//2,
                                              gauss2d.shape[1]//2)),
                             bkgnd_degree=bkgnd_degree,
                             allow_nonzero_base=allow_nonzero_base,
                             num_sigma=0,
                             use_angular_params=use_angular_params)
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-2)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-2)
    assert 'sigma_y' not in ret[2]
    assert 'sigma_x' not in ret[2]
    # assert ret[2]['scale'] == pytest.approx(1.5, abs=2e-1)  # TODO: Why?

    # centered PSF, fixed sigma_y
    psf2 = GaussianPSF(sigma=(0.9, None))
    gauss2d = psf2.eval_rect((21, 21), scale=1.5, sigma_x=1.1)
    ret = psf2.find_position(gauss2d, gauss2d.shape,
                             starting_point=((gauss2d.shape[0]//2,
                                              gauss2d.shape[1]//2)),
                             bkgnd_degree=bkgnd_degree,
                             allow_nonzero_base=allow_nonzero_base,
                             num_sigma=0,
                             use_angular_params=use_angular_params)
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-2)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-2)
    assert 'sigma_y' not in ret[2]
    # assert ret[2]['sigma_x'] == pytest.approx(1.1, abs=1e-1)  # TODO: Why?
    # assert ret[2]['scale'] == pytest.approx(1.5, abs=1e-1)  # TODO: Why?

    # centered PSF, fixed sigma_x
    psf2 = GaussianPSF(sigma=(None, 0.8))
    gauss2d = psf2.eval_rect((21, 21), scale=1.5, sigma_y=1.1)
    ret = psf2.find_position(gauss2d, gauss2d.shape,
                             starting_point=((gauss2d.shape[0]//2,
                                              gauss2d.shape[1]//2)),
                             bkgnd_degree=bkgnd_degree,
                             allow_nonzero_base=allow_nonzero_base,
                             num_sigma=0,
                             use_angular_params=use_angular_params)
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-2)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-2)
    assert 'sigma_x' not in ret[2]
    # assert ret[2]['sigma_y'] == pytest.approx(1.1, abs=1e-1)  # TODO: Why?
    # assert ret[2]['scale'] == pytest.approx(1.5, abs=1e-1)  # TODO: Why?

    if bkgnd_degree is not None:
        # add background gradient
        gauss2d = psf.eval_rect((21, 21), scale=2., sigma=1.)
        nparams = int((bkgnd_degree+1) * (bkgnd_degree+2) / 2)
        coeffts = np.array([0.5] * nparams)
        gauss2d += GaussianPSF.background_gradient((21, 21), coeffts)

        # if allow_nonzero_base:
        #     psf._debug_opt = 10
        ret = psf.find_position(gauss2d, gauss2d.shape,
                                starting_point=((gauss2d.shape[0]//2,
                                                gauss2d.shape[1]//2)),
                                bkgnd_degree=bkgnd_degree,
                                allow_nonzero_base=allow_nonzero_base,
                                num_sigma=0,
                                use_angular_params=use_angular_params)
        assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-3)
        assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-2)
        assert ret[2]['sigma_y'] == pytest.approx(1., abs=5e-2)
        assert ret[2]['sigma_x'] == pytest.approx(1., abs=5e-2)
        assert ret[2]['scale'] == pytest.approx(2., abs=5e-2)
