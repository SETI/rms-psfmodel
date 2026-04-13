################################################################################
# tests/test_gaussian.py
################################################################################

"""Tests for :mod:`psfmodel.gaussian`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest
import scipy.integrate as integrate

from psfmodel.gaussian import GaussianPSF

_MSG_BOTH_SIGMA = 'Cannot specify both sigma during init and sigma_y/x'
_MSG_EVAL_POINT_SIGMA = (
    'Sigma X and Y must be specified either at object creation or in the call to eval_point'
)
_MSG_EVAL_PIXEL_SIGMA = (
    'Sigma X and Y must be specified either at object creation or in the call to eval_pixel'
)


def _require_position_fit(
    ret: tuple[float, float, dict[str, Any]] | None,
) -> tuple[float, float, dict[str, Any]]:
    """Return ``ret`` after asserting :meth:`PSF.find_position` did not return ``None``."""

    assert ret is not None
    return ret


@pytest.mark.parametrize(
    ('x', 'kwargs', 'expected'),
    [
        (0.0, {}, 0.39894228),
        (0.0, {'scale': 2.0}, 0.39894228 * 2),
        (0.0, {'base': -5.0}, 0.39894228 - 5.0),
        (0.0, {'scale': 2.0, 'base': -5.0}, 0.39894228 * 2 - 5.0),
        (1.0, {'mean': 1.0}, 0.39894228),
        (1.0, {'mean': 1.0, 'scale': 2.0}, 0.39894228 * 2),
        (1.0, {}, 0.24197072),
        (2.0, {'sigma': 2.0}, 0.24197072 / 2),
    ],
)
def test_gaussian_1d_scalar_cases(x: float, kwargs: dict[str, float], expected: float) -> None:
    """``gaussian_1d`` matches reference values for scalar inputs and keyword overrides."""

    assert GaussianPSF.gaussian_1d(x, **kwargs) == pytest.approx(expected)


def test_gaussian_1d_array_broadcast() -> None:
    """``gaussian_1d`` broadcasts correctly over array coordinates."""

    npt.assert_array_almost_equal(GaussianPSF.gaussian_1d(np.array([0.0])), np.array([0.39894228]))
    npt.assert_array_almost_equal(
        GaussianPSF.gaussian_1d(np.array([0.0, 1.0])), np.array([0.39894228, 0.24197072])
    )
    npt.assert_array_almost_equal(
        GaussianPSF.gaussian_1d(np.array([[0.0, 1.0], [1.0, 0.0]])),
        np.array([[0.39894228, 0.24197072], [0.24197072, 0.39894228]]),
    )


def test_gaussian_1d_quadrature_normalization() -> None:
    """The analytic 1-D Gaussian integrates to ``scale`` over the real line."""

    assert integrate.quad(GaussianPSF.gaussian_1d, -10, 10)[0] == pytest.approx(1.0)
    assert integrate.quad(lambda x: GaussianPSF.gaussian_1d(x, scale=2.0), -10, 10)[
        0
    ] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ('y', 'x', 'kwargs', 'expected'),
    [
        (0, 0, {}, 0.15915494309),
        (0, 0, {'scale': 2.0}, 0.15915494309 * 2),
        (0, 0, {'base': 10.0}, 0.15915494309 + 10),
        (0, 0, {'scale': 2.0, 'base': 10.0}, 0.15915494309 * 2 + 10),
        (1.0, 0.0, {'mean_y': 1.0}, 0.15915494309),
        (0.0, 1.0, {'mean_x': 1.0}, 0.15915494309),
        (1.0, 0.0, {}, 0.09653235263005391),
        (1.0, 0.0, {'sigma_x': 2}, 0.048266176315027),
    ],
)
def test_gaussian_2d_scalar_cases(
    y: float,
    x: float,
    kwargs: dict[str, float],
    expected: float,
) -> None:
    """``gaussian_2d`` matches reference values for representative scalar inputs."""

    assert GaussianPSF.gaussian_2d(y, x, **kwargs) == pytest.approx(expected)


def test_gaussian_2d_angle_and_asymmetry() -> None:
    """Rotation and distinct ``sigma_x`` change the 2-D Gaussian as expected."""

    assert GaussianPSF.gaussian_2d(1.0, 0.0, sigma_x=2) == pytest.approx(0.048266176315027)
    assert GaussianPSF.gaussian_2d(0.0, -1.0, sigma_x=2) != pytest.approx(0.048266176315027)
    assert GaussianPSF.gaussian_2d(0.0, -1.0, angle=np.pi / 2) == pytest.approx(0.09653235263)


def test_gaussian_2d_quadrature_normalization() -> None:
    """The 2-D Gaussian integrates to ``scale`` over the plane (numerically)."""

    assert integrate.dblquad(GaussianPSF.gaussian_2d, -10, 10, -10, 10)[0] == pytest.approx(1.0)
    assert integrate.dblquad(
        lambda x, y: GaussianPSF.gaussian_2d(x, y, scale=2.0), -10, 10, -10, 10
    )[0] == pytest.approx(2.0)


def test_gaussian_2d_array_broadcast() -> None:
    """``gaussian_2d`` broadcasts over array coordinates."""

    npt.assert_array_almost_equal(
        GaussianPSF.gaussian_2d(np.array([0.0]), np.array([0.0])), np.array([0.15915494309])
    )
    npt.assert_array_almost_equal(
        GaussianPSF.gaussian_2d(np.array([[0.0], [1.0]]), np.array([[0.0], [0.0]])),
        np.array([[0.15915494309], [0.09653235263005391]]),
    )
    npt.assert_array_almost_equal(
        GaussianPSF.gaussian_2d(
            np.array([[[0.0], [1.0]], [[1.0], [0.0]]]), np.array([[[0.0], [0.0]], [[0.0], [0.0]]])
        ),
        np.array(
            [[[0.15915494309], [0.09653235263005391]], [[0.09653235263005391], [0.15915494309]]]
        ),
    )


def test_gaussian_2d_rotated_ellipses_differ() -> None:
    """Swapping ``sigma_x`` and ``sigma_y`` on a grid yields distinct surfaces."""

    y_coords = np.tile(np.arange(-10.0, 11.0) / 2, 21)
    x_coords = np.repeat(np.arange(-10.0, 11.0) / 2, 21)

    gauss2d2 = np.asarray(
        GaussianPSF.gaussian_2d(y_coords, x_coords, scale=2.0, sigma_x=0.25, sigma_y=0.5, base=1.0)
        / 4
    ).reshape(21, 21)
    gauss2d3 = np.asarray(
        GaussianPSF.gaussian_2d(y_coords, x_coords, scale=2.0, sigma_x=0.5, sigma_y=0.25, base=1.0)
        / 4
    ).reshape(21, 21)
    with pytest.raises(AssertionError):
        npt.assert_array_almost_equal(gauss2d2, gauss2d3)
    npt.assert_array_almost_equal(np.transpose(gauss2d2), gauss2d3)


def test_gaussian_integral_1d() -> None:
    """``gaussian_integral_1d`` matches numerical quadrature for scalars and arrays."""

    g_0_1 = integrate.quad(GaussianPSF.gaussian_1d, 0.0, 1.0)[0]
    g_n1_1 = integrate.quad(GaussianPSF.gaussian_1d, -1.0, 1.0)[0]
    assert GaussianPSF.gaussian_integral_1d(0.0, 1.0) == pytest.approx(
        integrate.quad(GaussianPSF.gaussian_1d, 0.0, 1.0)[0]
    )
    assert GaussianPSF.gaussian_integral_1d(-1.0, 1.0) == pytest.approx(g_n1_1)
    assert GaussianPSF.gaussian_integral_1d(-1.0, 1.0, mean=2.0) == pytest.approx(
        integrate.quad(GaussianPSF.gaussian_1d, 1.0, 3.0)[0]
    )
    assert GaussianPSF.gaussian_integral_1d(-1.0, 1.0, scale=2.0) == pytest.approx(g_n1_1 * 2)
    assert GaussianPSF.gaussian_integral_1d(-1.0, 1.0, base=5.0) == pytest.approx(g_n1_1 + 5)
    assert GaussianPSF.gaussian_integral_1d(-1.0, 1.0, scale=2.0, base=5.0) == pytest.approx(
        g_n1_1 * 2 + 5
    )
    assert GaussianPSF.gaussian_integral_1d(
        np.array([0.0, -1.0]), np.array([1.0, 1.0])
    ) == pytest.approx(np.array([g_0_1, g_n1_1]))
    ret = GaussianPSF.gaussian_integral_1d(
        np.array([[0.0, -1.0], [-1.0, 0.0]]), np.array([[1.0, 1.0], [1.0, 1.0]])
    )
    npt.assert_array_almost_equal(ret, np.array([[g_0_1, g_n1_1], [g_n1_1, g_0_1]]))


def test_gaussian_integral_1d_nonpositive_sigma_raises() -> None:
    """``gaussian_integral_1d`` requires a positive ``sigma``."""

    with pytest.raises(ValueError) as exc_info:
        GaussianPSF.gaussian_integral_1d(0.0, 1.0, sigma=0.0)
    assert str(exc_info.value) == 'sigma must be positive, got 0.0'
    with pytest.raises(ValueError) as exc_info:
        GaussianPSF.gaussian_integral_1d(0.0, 1.0, sigma=-1.0)
    assert str(exc_info.value) == 'sigma must be positive, got -1.0'


def test_gaussian_integral_2d() -> None:
    """``gaussian_integral_2d`` matches ``dblquad`` for sample regions."""

    integ1 = integrate.dblquad(lambda y, x: GaussianPSF.gaussian_2d(y, x), 0.0, 3.0, -2.0, 1.0)[0]
    integ2 = integrate.dblquad(lambda y, x: GaussianPSF.gaussian_2d(y, x), -1.0, 3.0, -3.0, 2.0)[0]
    assert GaussianPSF.gaussian_integral_2d(0.0, 3.0, -2.0, 1.0) == pytest.approx(integ1)
    assert GaussianPSF.gaussian_integral_2d(
        0.0, 3.0, -2.0, 1.0, scale=2.0, base=5.0
    ) == pytest.approx(integ1 * 2 + 5)
    ret = GaussianPSF.gaussian_integral_2d(
        np.array([0.0, -1.0]), np.array([3.0, 3.0]), np.array([-2.0, -3.0]), np.array([1.0, 2.0])
    )
    npt.assert_array_almost_equal(ret, np.array([integ1, integ2]))


@pytest.mark.parametrize(
    ('sigma_init', 'call_kwargs'),
    [
        ((1, 1), {'sigma': 5}),
        ((1, 1), {'sigma_x': 5}),
        ((1, 1), {'sigma_y': 5}),
        (None, {}),
        ((None, 1), {}),
        ((1, None), {}),
    ],
)
def test_gaussian_eval_point_value_errors(
    sigma_init: tuple[int | None, int | None] | None,
    call_kwargs: Mapping[str, Any],
) -> None:
    """``eval_point`` rejects ambiguous or incomplete sigma configuration."""

    with pytest.raises(ValueError) as exc_info:
        GaussianPSF(sigma=sigma_init).eval_point((0, 0), **call_kwargs)
    if sigma_init == (1, 1):
        assert str(exc_info.value) == _MSG_BOTH_SIGMA
    else:
        assert str(exc_info.value) == _MSG_EVAL_POINT_SIGMA


def test_gaussian_eval_point_success_cases() -> None:
    """``eval_point`` agrees with :meth:`GaussianPSF.gaussian_2d` for valid configurations."""

    psf1 = GaussianPSF()
    psf2 = GaussianPSF(sigma=(1.0, None))
    psf3 = GaussianPSF(sigma=(None, 2.0))
    psf4 = GaussianPSF(sigma=(1.0, 2.0))
    psf5 = GaussianPSF(sigma=(1.0, 1.0))

    assert psf1.eval_point((2, 3), sigma=cast(Any, (1.0, 2.0))) == psf4.eval_point((2, 3))
    assert psf1.eval_point((2, 3), sigma=1.0) == psf5.eval_point((2, 3))
    assert psf1.eval_point((2, 3), sigma_y=1.0, sigma_x=2.0) == psf4.eval_point((2, 3))
    assert psf2.eval_point((2, 3), sigma_x=2.0) == psf4.eval_point((2, 3))
    assert psf3.eval_point((2, 3), sigma_y=1.0) == psf4.eval_point((2, 3))
    assert psf1.eval_point(
        (2, 3), sigma=cast(Any, (1.0, 2.0)), base=1, scale=2, angle=np.pi / 4
    ) == psf4.eval_point((2, 3), base=1, scale=2, angle=np.pi / 4)

    assert psf1.eval_point(
        (2, 3), sigma=cast(Any, (1.0, 2.0)), base=1, scale=2, angle=np.pi / 4
    ) == pytest.approx(
        GaussianPSF.gaussian_2d(2, 3, sigma_y=1.0, sigma_x=2.0, base=1, scale=2, angle=np.pi / 4)
    )

    ret = psf1.eval_point(
        (np.array([1, 2]), np.array([2, 3])),
        sigma=cast(Any, (1.0, 2.0)),
    )
    npt.assert_array_almost_equal(ret, np.array([psf4.eval_point((1, 2)), psf4.eval_point((2, 3))]))


@pytest.mark.parametrize(
    ('sigma_init', 'call_kwargs'),
    [
        ((1, 1), {'sigma': 5}),
        ((1, 1), {'sigma_x': 5}),
        ((1, 1), {'sigma_y': 5}),
        (None, {}),
        (None, {'sigma_x': 5}),
        (None, {'sigma_y': 5}),
        ((1, None), {}),
        ((None, 1), {}),
    ],
)
def test_gaussian_eval_pixel_value_errors(
    sigma_init: tuple[int | None, int | None] | None,
    call_kwargs: Mapping[str, Any],
) -> None:
    """``eval_pixel`` rejects ambiguous or incomplete sigma configuration."""

    with pytest.raises(ValueError) as exc_info:
        GaussianPSF(sigma=sigma_init).eval_pixel((0, 0), **call_kwargs)
    if sigma_init == (1, 1):
        assert str(exc_info.value) == _MSG_BOTH_SIGMA
    else:
        assert str(exc_info.value) == _MSG_EVAL_PIXEL_SIGMA


def test_gaussian_eval_pixel_success_cases() -> None:
    """``eval_pixel`` integrates the Gaussian over unit pixels as documented."""

    integ = GaussianPSF.gaussian_integral_2d(-0.5, 0.5, -0.5, 0.5, sigma_y=2.0, sigma_x=3.0)
    integ2 = GaussianPSF.gaussian_integral_2d(-0.5, 0.5, -0.5, 0.5, sigma_y=2.0, sigma_x=2.0)
    assert GaussianPSF(sigma=(2.0, 3.0)).eval_pixel((0, 0)) == pytest.approx(integ)
    assert GaussianPSF(sigma=2.0).eval_pixel((0, 0)) == pytest.approx(integ2)
    assert GaussianPSF().eval_pixel((0, 0), sigma=(2.0, 2.0)) == pytest.approx(integ2)
    assert GaussianPSF(sigma=(2.0, 3.0)).eval_pixel((0, 0), offset=(0, 0.25)) == pytest.approx(
        GaussianPSF.gaussian_integral_2d(0, 1, -0.25, 0.75, sigma_y=2.0, sigma_x=3.0)
    )
    assert GaussianPSF(sigma=(2.0, 3.0)).eval_pixel((0, 0), scale=11.0) == pytest.approx(integ * 11)
    assert GaussianPSF(sigma=(2.0, 3.0)).eval_pixel((0, 0), base=3.0) == pytest.approx(integ + 3)
    assert GaussianPSF().eval_pixel((0, 0), sigma=(2.0, 3.0)) == pytest.approx(integ)
    assert GaussianPSF(sigma=(2.0, None)).eval_pixel((0, 0), sigma_x=3.0) == pytest.approx(integ)
    assert GaussianPSF(sigma=(None, 3.0)).eval_pixel((0, 0), sigma_y=2.0) == pytest.approx(integ)
    ret = GaussianPSF(sigma=(2.0, 3.0)).eval_pixel((np.array([0, 0]), np.array([0, 0])))
    npt.assert_array_almost_equal(ret, np.array([integ, integ]))

    assert GaussianPSF(sigma=(2.0, 3.0)).eval_pixel((0, 0), angle=np.pi / 8) == pytest.approx(
        GaussianPSF.gaussian_integral_2d(
            -0.5, 0.5, -0.5, 0.5, sigma_y=2.0, sigma_x=3.0, angle=np.pi / 8
        )
    )


@pytest.mark.parametrize(
    ('rect_size', 'expected_msg'),
    [
        ((20, 19), 'Rectangle must have odd positive shape in each dimension, got (20, 19)'),
        ((19, 18), 'Rectangle must have odd positive shape in each dimension, got (19, 18)'),
        ((-1, 5), 'Rectangle must have odd positive shape in each dimension, got (-1, 5)'),
        ((5, -3), 'Rectangle must have odd positive shape in each dimension, got (5, -3)'),
    ],
)
def test_gaussian_eval_rect_invalid_shape(rect_size: tuple[int, int], expected_msg: str) -> None:
    """``eval_rect`` requires odd, positive side lengths."""

    with pytest.raises(ValueError) as exc_info:
        GaussianPSF(sigma=(1.0, 1.0)).eval_rect(rect_size)
    assert str(exc_info.value) == expected_msg


@pytest.mark.parametrize(
    ('rect_size', 'sigma', 'scale', 'base', 'offset'),
    [
        ((19, 19), (1.0, 1.0), 1.0, 0.0, (0.5, 0.5)),
        ((15, 15), (0.8, 1.2), 2.5, 0.1, (0.25, 0.75)),
        ((21, 21), (1.5, 1.5), 0.5, 0.02, (0.0, 0.5)),
    ],
)
def test_gaussian_eval_rect_shape_peak_and_nonneg(
    rect_size: tuple[int, int],
    sigma: tuple[float, float],
    scale: float,
    base: float,
    offset: tuple[float, float],
) -> None:
    """``eval_rect`` returns a non-negative patch peaked at the center with expected shape."""

    psf = GaussianPSF(sigma=sigma)
    rect = psf.eval_rect(rect_size, offset=offset, scale=scale, base=base)
    assert rect.shape == rect_size
    assert np.all(rect >= 0)
    cy, cx = rect_size[0] // 2, rect_size[1] // 2
    assert rect[cy, cx] == pytest.approx(np.max(rect))


def test_gaussian_eval_rect_rotated_matches_sum(symmetric_psf: GaussianPSF) -> None:
    """A rotated Gaussian patch remains normalized to unit flux (plus base)."""

    assert np.sum(symmetric_psf.eval_rect((19, 19))) == pytest.approx(1.0)
    assert np.sum(
        GaussianPSF(sigma=(1.0, 1.0), angle=np.pi / 4).eval_rect((19, 19))
    ) == pytest.approx(1.0)


@pytest.mark.parametrize('angle_subsample', [0, 100])
def test_gaussian_init_angle_subsample_invalid_int(angle_subsample: int) -> None:
    """``angle_subsample`` must be an int strictly between 0 and 100."""

    with pytest.raises(ValueError) as exc_info:
        GaussianPSF(angle_subsample=angle_subsample)
    assert str(exc_info.value) == (
        f'angle_subsample must be an int between 1 and 99, got {angle_subsample}'
    )


def test_gaussian_init_angle_subsample_rejects_non_int() -> None:
    """``angle_subsample`` must be an ``int``, not a non-integral type."""

    with pytest.raises(ValueError) as exc_info:
        GaussianPSF(angle_subsample=1.5)  # type: ignore[arg-type]
    assert str(exc_info.value) == 'angle_subsample must be an int between 1 and 99, got 1.5'


@pytest.mark.parametrize('use_angular_params', [True, False])
@pytest.mark.parametrize('bkgnd_degree', [None, 0, 1, 2])
def test_gaussian_find_position(
    use_angular_params: bool,
    bkgnd_degree: int | None,
    default_psf: GaussianPSF,
) -> None:
    """End-to-end centroid and width recovery for synthetic Gaussian patches."""

    allow_nonzero_base = bkgnd_degree is not None

    psf = default_psf
    gauss2d = psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
    ret = _require_position_fit(
        psf.find_position(
            gauss2d,
            gauss2d.shape,
            starting_point=((gauss2d.shape[0] // 2, gauss2d.shape[1] // 2)),
            bkgnd_degree=bkgnd_degree,
            allow_nonzero_base=allow_nonzero_base,
            num_sigma=0,
            use_angular_params=use_angular_params,
        )
    )
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2)
    assert ret[2]['sigma_y'] == pytest.approx(1.0, abs=5e-2)
    assert ret[2]['sigma_x'] == pytest.approx(1.0, abs=5e-2)
    assert ret[2]['scale'] == pytest.approx(2.0, abs=5e-2)

    gauss2d = psf.eval_rect((21, 21), scale=2.0, sigma=(2.0, 0.5))
    ret = _require_position_fit(
        psf.find_position(
            gauss2d,
            gauss2d.shape,
            starting_point=((gauss2d.shape[0] // 2, gauss2d.shape[1] // 2)),
            bkgnd_degree=bkgnd_degree,
            bkgnd_ignore_center=(4, 4),
            allow_nonzero_base=allow_nonzero_base,
            num_sigma=0,
            use_angular_params=use_angular_params,
        )
    )
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-4)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-4)
    assert ret[2]['sigma_y'] == pytest.approx(2.0, abs=5e-2)
    assert ret[2]['sigma_x'] == pytest.approx(0.5, abs=5e-2)
    assert ret[2]['scale'] == pytest.approx(2.0, abs=7e-2)

    psf2 = GaussianPSF(mean=(0.5, 0.75))
    gauss2d = psf2.eval_rect((21, 21), scale=0.5, sigma=(0.5, 1.3))

    ret = _require_position_fit(
        psf.find_position(
            gauss2d,
            gauss2d.shape,
            starting_point=((gauss2d.shape[0] // 2, gauss2d.shape[1] // 2)),
            bkgnd_degree=bkgnd_degree,
            bkgnd_ignore_center=(4, 4),
            allow_nonzero_base=allow_nonzero_base,
            num_sigma=0,
            use_angular_params=use_angular_params,
        )
    )
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2 + 0.5, abs=1e-1)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2 + 0.75, abs=1e-1)
    assert ret[2]['sigma_y'] == pytest.approx(0.5, abs=5e-2)
    assert ret[2]['sigma_x'] == pytest.approx(1.3, abs=5e-2)
    assert ret[2]['scale'] == pytest.approx(0.5, abs=5e-2)

    ret = _require_position_fit(
        psf2.find_position(
            gauss2d,
            gauss2d.shape,
            starting_point=((gauss2d.shape[0] // 2, gauss2d.shape[1] // 2)),
            bkgnd_degree=bkgnd_degree,
            bkgnd_ignore_center=(4, 4),
            allow_nonzero_base=allow_nonzero_base,
            num_sigma=0,
            use_angular_params=use_angular_params,
        )
    )
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-1)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-1)
    assert ret[2]['sigma_y'] == pytest.approx(0.5, abs=5e-2)
    assert ret[2]['sigma_x'] == pytest.approx(1.3, abs=5e-2)
    assert ret[2]['scale'] == pytest.approx(0.5, abs=5e-2)

    psf2 = GaussianPSF()
    gauss2d = psf2.eval_rect((21, 21), offset=(0.21, -0.35), scale=1.5, sigma=(0.8, 1.3))
    ret = _require_position_fit(
        psf2.find_position(
            gauss2d,
            gauss2d.shape,
            starting_point=((gauss2d.shape[0] // 2, gauss2d.shape[1] // 2)),
            bkgnd_degree=bkgnd_degree,
            bkgnd_ignore_center=(4, 4),
            allow_nonzero_base=allow_nonzero_base,
            num_sigma=0,
            use_angular_params=use_angular_params,
        )
    )
    assert ret[0] == pytest.approx(gauss2d.shape[0] // 2 + 0.21, abs=5e-2)
    assert ret[1] == pytest.approx(gauss2d.shape[1] // 2 - 0.35, abs=5e-2)
    assert ret[2]['sigma_y'] == pytest.approx(0.8, abs=1e-3)
    assert ret[2]['sigma_x'] == pytest.approx(1.3, abs=1e-3)
    assert ret[2]['scale'] == pytest.approx(1.5, abs=1e-2)

    psf2 = GaussianPSF(sigma=(0.9, 1.5))
    gauss2d = psf2.eval_rect((21, 21), scale=1.5)
    ret = _require_position_fit(
        psf2.find_position(
            gauss2d,
            gauss2d.shape,
            starting_point=((gauss2d.shape[0] // 2, gauss2d.shape[1] // 2)),
            bkgnd_degree=bkgnd_degree,
            allow_nonzero_base=allow_nonzero_base,
            num_sigma=0,
            use_angular_params=use_angular_params,
        )
    )
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-2)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-2)
    assert 'sigma_y' not in ret[2]
    assert 'sigma_x' not in ret[2]
    # With angular parameters and a polynomial background, optimized ``scale`` and
    # floating ``sigma_*`` values follow angular reparameterization, not literal
    # ``eval_rect`` inputs.
    if bkgnd_degree is None or not use_angular_params:
        assert ret[2]['scale'] == pytest.approx(1.5, abs=2e-1)

    psf2 = GaussianPSF(sigma=(0.9, None))
    gauss2d = psf2.eval_rect((21, 21), scale=1.5, sigma_x=1.1)
    ret = _require_position_fit(
        psf2.find_position(
            gauss2d,
            gauss2d.shape,
            starting_point=((gauss2d.shape[0] // 2, gauss2d.shape[1] // 2)),
            bkgnd_degree=bkgnd_degree,
            allow_nonzero_base=allow_nonzero_base,
            num_sigma=0,
            use_angular_params=use_angular_params,
        )
    )
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-2)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-2)
    assert 'sigma_y' not in ret[2]
    if bkgnd_degree is None or not use_angular_params:
        assert ret[2]['sigma_x'] == pytest.approx(1.1, abs=1e-1)
        assert ret[2]['scale'] == pytest.approx(1.5, abs=1e-1)

    psf2 = GaussianPSF(sigma=(None, 0.8))
    gauss2d = psf2.eval_rect((21, 21), scale=1.5, sigma_y=1.1)
    ret = _require_position_fit(
        psf2.find_position(
            gauss2d,
            gauss2d.shape,
            starting_point=((gauss2d.shape[0] // 2, gauss2d.shape[1] // 2)),
            bkgnd_degree=bkgnd_degree,
            allow_nonzero_base=allow_nonzero_base,
            num_sigma=0,
            use_angular_params=use_angular_params,
        )
    )
    assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-2)
    assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-2)
    assert 'sigma_x' not in ret[2]
    if bkgnd_degree is None or not use_angular_params:
        assert ret[2]['sigma_y'] == pytest.approx(1.1, abs=1e-1)
        assert ret[2]['scale'] == pytest.approx(1.5, abs=1e-1)

    if bkgnd_degree is not None:
        gauss2d = psf.eval_rect((21, 21), scale=2.0, sigma=(1.0, 1.0))
        nparams = int((bkgnd_degree + 1) * (bkgnd_degree + 2) / 2)
        coeffts = np.array([0.5] * nparams)
        gauss2d += GaussianPSF.background_gradient((21, 21), coeffts)

        ret = _require_position_fit(
            psf.find_position(
                gauss2d,
                gauss2d.shape,
                starting_point=((gauss2d.shape[0] // 2, gauss2d.shape[1] // 2)),
                bkgnd_degree=bkgnd_degree,
                allow_nonzero_base=allow_nonzero_base,
                num_sigma=0,
                use_angular_params=use_angular_params,
            )
        )
        assert ret[0] == pytest.approx(gauss2d.shape[0] / 2, abs=1e-3)
        assert ret[1] == pytest.approx(gauss2d.shape[1] / 2, abs=1e-2)
        assert ret[2]['sigma_y'] == pytest.approx(1.0, abs=5e-2)
        assert ret[2]['sigma_x'] == pytest.approx(1.0, abs=5e-2)
        assert ret[2]['scale'] == pytest.approx(2.0, abs=5e-2)
