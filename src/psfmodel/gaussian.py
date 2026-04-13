################################################################################
# psfmodel/gaussian.py
################################################################################

"""Analytic Gaussian PSF models and integrals for :class:`~psfmodel.psf.PSF`.

:class:`GaussianPSF` subclasses :class:`~psfmodel.psf.PSF` for pixel-integrated fitting
and rendering. The module exposes helpers such as :class:`GaussianPSF` and the constant
``INV_SQRT_2`` (used in error-function integrals). Depends on NumPy and
``scipy.special.erf``.
"""

import logging
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.special import erf

from .psf import PSF

INV_SQRT_2 = 2 ** (-0.5)


class GaussianPSF(PSF):
    """A 2-D Gaussian symmetric PSF.

    The PSF can have different standard deviations in the X and Y directions.
    The standard deviations for X and Y can be locked up front when the
    GaussianPSF object is created or left to float so that future calls may
    specify them directly.

    Because these are so fast and easy to compute, we don't cache any results.
    """

    _sigma_y: float | None
    _sigma_x: float | None
    _mean_y: float
    _mean_x: float
    _angle: float | None
    _angle_subsample: int

    def __init__(
        self,
        *,
        sigma: float | tuple[float | None, float | None] | None = None,
        mean: float | tuple[float, float] = 0.0,
        angle: float | None = 0.0,
        sigma_x_range: tuple[float, float] = (0.01, 10.0),
        sigma_y_range: tuple[float, float] = (0.01, 10.0),
        angle_subsample: int = 13,
        logger: logging.Logger | None = None,
        detailed_logging: bool = False,
    ) -> None:
        """Create a GaussianPSF object describing a 2-D Gaussian PSF.

        Parameters:
            sigma: The standard deviation of the Gaussian. May be a scalar, in which case
                the value applies to both X and Y, or a tuple (sigma_y, sigma_x) one of
                which may be None. None for sigma or for one of sigma_x/y means that the
                sigma will be supplied later.
            mean: The mean of the Gaussian. May be a scalar in which case the value
                applies to both X and Y, or a tuple (mean_y, mean_x).
            angle: Rotation angle from 0 to ``pi`` (0 is +X). Default ``0.0`` fixes an
                unrotated Gaussian and keeps pixel integration on the fast axis-aligned
                path. Pass ``None`` to include ``angle`` in ``_additional_params`` so it
                can be fitted in :meth:`~psfmodel.psf.PSF.find_position`; per-call
                overrides are still accepted by :meth:`eval_rect`.
            sigma_x_range: The valid range for sigma_x if it is not specified otherwise.
                This is used during PSF fitting to let sigma_x float to its optimal value.
            sigma_y_range: The valid range for sigma_y if it is not specified otherwise.
                This is used during PSF fitting to let sigma_y float to its optimal value.
            angle_subsample: The amount of subsampling to do in X and Y when computing a
                2-D Gaussian pixel with a non-zero angle.
            logger: Optional logger passed to :class:`PSF`; see :meth:`PSF.__init__`.
            detailed_logging: Passed to :class:`PSF`; see :meth:`PSF.__init__`.
        """

        super().__init__(logger=logger, detailed_logging=detailed_logging)

        if not isinstance(sigma, tuple):
            self._sigma_y = self._sigma_x = float(sigma) if sigma is not None else None
        else:
            self._sigma_y = float(sigma[0]) if sigma[0] is not None else None
            self._sigma_x = float(sigma[1]) if sigma[1] is not None else None
        if not isinstance(mean, (tuple, list)):
            self._mean_y = self._mean_x = float(mean)
        else:
            self._mean_y, self._mean_x = float(mean[0]), float(mean[1])
        self._angle = None if angle is None else float(angle)
        if not isinstance(angle_subsample, int) or not (0 < angle_subsample <= 99):
            raise ValueError(
                f'angle_subsample must be an int between 1 and 99, got {angle_subsample}'
            )
        self._angle_subsample = int(angle_subsample)

        if self._sigma_y is None and sigma_y_range is not None:
            self._additional_params.append(
                (float(sigma_y_range[0]), float(sigma_y_range[1]), 'sigma_y')
            )
        if self._sigma_x is None and sigma_x_range is not None:
            self._additional_params.append(
                (float(sigma_x_range[0]), float(sigma_x_range[1]), 'sigma_x')
            )
        if self._angle is None and angle_subsample > 1:
            self._additional_params.append((0, np.pi, 'angle'))

    @property
    def sigma_y(self) -> float | None:
        """Standard deviation of the Gaussian along the y-axis (pixels), if fixed.

        Returns:
            The locked ``sigma_y`` from construction, or ``None`` when it is left free
            for fitting or per-call overrides.
        """
        return self._sigma_y

    @property
    def sigma_x(self) -> float | None:
        """Standard deviation of the Gaussian along the x-axis (pixels), if fixed.

        Returns:
            The locked ``sigma_x`` from construction, or ``None`` when it is left free
            for fitting or per-call overrides.
        """
        return self._sigma_x

    @property
    def mean_y(self) -> float:
        """Center of the Gaussian along the y-axis in pixel coordinates.

        Returns:
            The mean ``y`` used when evaluating the model (scalar).
        """
        return self._mean_y

    @property
    def mean_x(self) -> float:
        """Center of the Gaussian along the x-axis in pixel coordinates.

        Returns:
            The mean ``x`` used when evaluating the model (scalar).
        """
        return self._mean_x

    @staticmethod
    def gaussian_1d(
        x: float | npt.NDArray[np.floating],
        *,
        sigma: float = 1.0,
        mean: float = 0.0,
        scale: float = 1.0,
        base: float = 0.0,
    ) -> float | npt.NDArray[np.floating]:
        """Return a 1-D Gaussian.

        This simply returns the value of the Gaussian at a point or series of points.
        The Gaussian is normalized so the area under the curve is "scale".

        Parameters:
            x: A scalar or array at which to evaluate the Gaussian function.
            sigma: The standard deviation of the Gaussian.
            mean: The mean of the Gaussian.
            scale: The scale of the Gaussian; the area under the complete curve (excluding
                the base).
            base: The base of the Gaussian; a scalar added to the curve.

        Returns: The value of the Gaussian function at the point(s) defined by x.
        """

        ret = (
            scale / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mean) ** 2) / (2 * sigma**2))
        ) + base
        return cast(float | npt.NDArray[np.floating], ret)

    # @staticmethod
    # def gaussian_2d_rho(y: float | npt.NDArray[np.floating],
    #                     x: float | npt.NDArray[np.floating],
    #                     *,
    #                     sigma_y: float = 1.,
    #                     sigma_x: float = 1.,
    #                     mean_y: float = 0.,
    #                     mean_x: float = 0.,
    #                     scale: float = 1.,
    #                     base: float = 0.,
    #                     rho: float = 0.) -> float | npt.NDArray[np.floating]:
    #     """Return a 2-D Gaussian using rho (X/Y correlation).

    #     This simply returns the value of the 2-D Gaussian at a series of points.
    #     The Gaussian is normalized so the area under the curve is scale.

    #     Input:
    #         y: A scalar or array at which to evaluate the 2-D Gaussian function.
    #         x: A scalar or array at which to evaluate the 2-D Gaussian function.
    #         sigma_y: The standard deviation of the Gaussian in the Y direction.
    #         sigma_x: The standard deviation of the Gaussian in the X direction.
    #         mean_y: The mean of the Gaussian in the Y dimension.
    #         mean_x: The mean of the Gaussian in the X dimension.
    #         scale: The scale of the Gaussian; the area under the complete curve
    #             (excluding the base).
    #         base: The base of the Gaussian; a scalar added to the curve.
    #         rho: The correlation between the X and Y dimensions

    #                    From gaussians-cern.pdf:
    #                        sigma_X^2 = cos^2 sigma_x^2 + sin^2 sigma_y^2
    #                        sigma_Y^2 = cos^2 sigma_y^2 + sin^2 sigma_x^2
    #                     where sigma_X/Y are the size of the ellipse projected
    #                     onto the normal Cartesian axes. Also:
    #                        (1-rho^2) sigma_X^2 sigma_Y^2 = sigma_x^2 sigma_y^2

    #     Returns:
    #         The value of the 2-D Gaussian at the points defined by x and y.
    #     """

    #     # See http://cs229.stanford.edu/section/gaussians.pdf
    #     # See https://indico.cern.ch/category/6015/attachments/192/632/
    #           Statistics_Gaussian_I.pdf
    #     x = x - mean_x
    #     y = y - mean_y

    #     norm_fact = 1. / (2*np.pi * np.sqrt((sigma_x**2 * sigma_y**2 - rho**2)))
    #     expon = (((x*sigma_y)**2 + (y*sigma_x)**2 - 2*x*y*xcorr) /
    #              ((sigma_x*sigma_y)**2 - 2*xcorr))
    #     return scale * norm_fact * np.exp(-0.5 * expon) + base

    @staticmethod
    def gaussian_2d(
        y: float | npt.NDArray[np.floating],
        x: float | npt.NDArray[np.floating],
        *,
        sigma_y: float = 1.0,
        sigma_x: float = 1.0,
        mean_y: float = 0.0,
        mean_x: float = 0.0,
        scale: float = 1.0,
        base: float = 0.0,
        angle: float = 0.0,
    ) -> float | npt.NDArray[np.floating]:
        """Return a 2-D Gaussian using angle (angle 0-pi, 0 at 3 o'clock, CW).

        This simply returns the value of the 2-D Gaussian at a series of points.
        The Gaussian is normalized so the area under the curve is scale.

        Parameters:
            y: A scalar or array at which to evaluate the 2-D Gaussian function.
            x: A scalar or array at which to evaluate the 2-D Gaussian function.
            sigma_y: The standard deviation of the Gaussian in the Y direction.
            sigma_x: The standard deviation of the Gaussian in the X direction.
            mean_y: The mean of the Gaussian in the Y dimension.
            mean_x: The mean of the Gaussian in the X dimension.
            scale: The scale of the Gaussian; the area under the complete curve (excluding
                the base).
            base: The base of the Gaussian; a scalar added to the curve.
            angle: The angle of the ellipse. Angle ranges from 0 to pi, with 0 being
                "3 o'clock" (+X) assuming that (0, 0) is in the top left corner.
                mean_y/x are specified in the rotated coordinate system.

        Returns:
            The value of the 2-D Gaussian at the points defined by x and y.
        """

        # See http://cs229.stanford.edu/section/gaussians.pdf
        # See https://indico.cern.ch/category/6015/attachments/192/632/
        #     Statistics_Gaussian_I.pdf

        # x and y, and mean_x and mean_y are in the rotated coordinate system
        x = x - mean_x
        y = y - mean_y

        # Convert x and y (ellipse coordinates) to xc and yc (Cartesian
        # coordinates)
        c = np.cos(angle)
        s = np.sin(angle)
        xc = c * x + s * y
        yc = -s * x + c * y

        return (
            GaussianPSF.gaussian_1d(xc, sigma=sigma_x)
            * GaussianPSF.gaussian_1d(yc, sigma=sigma_y)
            * scale
            + base
        )

    @staticmethod
    def gaussian_integral_1d(
        x_min: float | npt.NDArray[np.floating],
        x_max: float | npt.NDArray[np.floating],
        *,
        sigma: float = 1.0,
        mean: float = 0.0,
        scale: float = 1.0,
        base: float = 0.0,
    ) -> float | npt.NDArray[np.floating]:
        """Return the integral of a Gaussian.

        The integral is over the limits [xmin, xmax].

        Values are generated via the error function, where the integral from
        ``-inf`` to ``x`` is proportional to

            ``1 + erf((x - mean) / (sqrt(2) * sigma))``

        This function works for both scalar and array values of xmin and xmax.

        Parameters:
            x_min: The lower bound of the integral.
            x_max: The upper bound of the integral.
            sigma: The standard deviation of the Gaussian.
            mean: The mean of the Gaussian.
            scale: The scale of the Gaussian; the area under the complete curve (excluding
                the base).
            base: The base of the Gaussian; a scalar added to the curve.

        Returns:
            The integral of the Gaussian between ``x_min`` and ``x_max``.

        Raises:
            ValueError: If ``sigma`` is not positive.
        """

        if not sigma > 0.0:
            raise ValueError(f'sigma must be positive, got {sigma}')
        xmin_div_sqrt_2 = (x_min - mean) * (INV_SQRT_2 / sigma)
        xmax_div_sqrt_2 = (x_max - mean) * (INV_SQRT_2 / sigma)

        # Handle the scalar case
        if np.shape(x_min) == () and np.shape(x_max) == ():
            ret = (0.5 * (erf(xmax_div_sqrt_2) - erf(xmin_div_sqrt_2)) * scale) + base
            return cast(float | npt.NDArray[np.floating], ret)

        # If either value is an array, broadcast to a common shape
        (xmin_div_sqrt_2, xmax_div_sqrt_2) = np.broadcast_arrays(xmin_div_sqrt_2, xmax_div_sqrt_2)

        result = erf(xmax_div_sqrt_2) - erf(xmin_div_sqrt_2)

        return cast(float | npt.NDArray[np.floating], result * 0.5 * scale + base)

    @staticmethod
    def gaussian_integral_2d(
        y_min: float | npt.NDArray[np.floating],
        y_max: float | npt.NDArray[np.floating],
        x_min: float | npt.NDArray[np.floating],
        x_max: float | npt.NDArray[np.floating],
        *,
        sigma_y: float = 1.0,
        sigma_x: float = 1.0,
        mean_y: float = 0.0,
        mean_x: float = 0.0,
        scale: float = 1.0,
        base: float = 0.0,
        angle: float = 0.0,
        angle_subsample: int = 13,
    ) -> float | npt.NDArray[np.floating]:
        """Return the double integral of a 2-D Gaussian.

        The integral is over the limits [y_min, y_max] and [x_min, x_max].

        For ``angle == 0`` the integral is computed analytically using the
        error function (fast, exact). For a non-zero ``angle``, the integral
        is approximated by uniform subsampling within each pixel using
        ``angle_subsample`` points per axis. When the bounds are arrays (e.g.
        a full image patch), all pixels are evaluated simultaneously in a
        single vectorised NumPy call on an ``(N, S, S)`` grid, where *N* is
        the number of pixels and *S* is ``angle_subsample``; this avoids a
        per-pixel Python loop and is typically 10-20x faster than an
        equivalent element-wise approach.

        Parameters:
            y_min: The lower bound of the integral in the Y dimension.
            y_max: The upper bound of the integral in the Y dimension.
            x_min: The lower bound of the integral in the X dimension.
            x_max: The upper bound of the integral in the X dimension.
            sigma_y: The standard deviation of the Gaussian in the Y dimension.
            sigma_x: The standard deviation of the Gaussian in the X dimension.
            mean_y: The mean of the Gaussian in the Y dimension.
            mean_x: The mean of the Gaussian in the X dimension.
            scale: The scale of the Gaussian; the area under the complete curve (excluding
                the base).
            base: The base of the Gaussian; a scalar added to the curve.
            angle: The angle of the ellipse. Angle ranges from 0 to pi, with 0 being
                "3 o'clock" (+X) assuming that (0, 0) is in the top left corner. Thus
                specifying 0 means that the X and Y axes of the ellipse align with the
                standard X and Y Cartesian axes. mean_y/x are specified in the rotated
                coordinate system, while sigma_y/x are specified in the unrotated
                Cartesian coordinate system.
            angle_subsample: The amount of subsampling to do in X and Y when computing a
                2-D Gaussian pixel with a non-zero angle.

        Returns:
            The integral of the 2-D Gaussian between y_min and y_max, and x_min and x_max.
        """

        if angle == 0.0:
            return (
                GaussianPSF.gaussian_integral_1d(y_min, y_max, sigma=sigma_y, mean=mean_y)
                * GaussianPSF.gaussian_integral_1d(x_min, x_max, sigma=sigma_x, mean=mean_x)
                * scale
                + base
            )

        # Handle the scalar case
        if (
            np.shape(x_min) == ()
            and np.shape(x_max) == ()
            and np.shape(y_min) == ()
            and np.shape(y_max) == ()
        ):
            ys = np.linspace(y_min, y_max, angle_subsample)
            xs = np.linspace(x_min, x_max, angle_subsample)
            xindex, yindex = np.meshgrid(xs, ys)

            ret = GaussianPSF.gaussian_2d(
                yindex,
                xindex,
                sigma_y=sigma_y,
                sigma_x=sigma_x,
                mean_y=mean_y,
                mean_x=mean_x,
                scale=scale,
                base=base,
                angle=angle,
            )
            return cast(float, np.mean(ret))

        x_min, x_max, y_min, y_max = np.broadcast_arrays(x_min, x_max, y_min, y_max)
        orig_shape = x_min.shape
        y_lo = y_min.ravel()
        y_hi = y_max.ravel()
        x_lo = x_min.ravel()
        x_hi = x_max.ravel()

        # Uniform parameter in [0, 1] with angle_subsample steps; shape (S,).
        t = np.linspace(0.0, 1.0, angle_subsample)

        # Sample coordinates for every pixel simultaneously.
        # ys, xs: shape (N, S) where N = number of pixels, S = angle_subsample.
        ys = y_lo[:, None] + (y_hi - y_lo)[:, None] * t
        xs = x_lo[:, None] + (x_hi - x_lo)[:, None] * t

        # Build the full (N, S, S) evaluation grid by broadcasting:
        #   ys[:, :, None]  -- y varies along axis 1, x axis held
        #   xs[:, None, :]  -- x varies along axis 2, y axis held
        # One call to gaussian_2d replaces the entire per-pixel Python loop.
        vals = GaussianPSF.gaussian_2d(
            ys[:, :, None],
            xs[:, None, :],
            sigma_y=sigma_y,
            sigma_x=sigma_x,
            mean_y=mean_y,
            mean_x=mean_x,
            scale=scale,
            base=base,
            angle=angle,
        )
        res = np.asarray(vals, dtype=np.float64).mean(axis=(1, 2)).reshape(orig_shape)

        return cast(float | npt.NDArray[np.floating], res)

    def eval_point(
        self,
        coord: (
            tuple[float | npt.NDArray[np.floating], float | npt.NDArray[np.floating]]
            | npt.NDArray[np.floating]
        ),
        *,
        sigma: float | None = None,
        scale: float = 1.0,
        base: float = 0.0,
        sigma_y: float | None = None,
        sigma_x: float | None = None,
        angle: float | None = None,
    ) -> float | npt.NDArray[np.floating]:
        """Evaluate the 2-D Gaussian PSF at a single, fractional, point.

        (0, 0) is the center of the PSF and x and y may be negative.

        Parameters:
            coord: The coordinate (y, x) at which to evaluate the PSF.
            scale: A scale factor to apply to the resulting PSF.
            base: A scalar added to the resulting PSF.
            sigma: Standard deviations: a scalar (both axes) or a ``(sigma_y, sigma_x)``
                tuple. Must not duplicate values fixed at construction; ``None`` if both
                are already set on the instance.
            sigma_y: An alternative way to specify sigma_y. Used primarily for letting
                sigma_y float during PSF fitting.
            sigma_x: An alternative way to specify sigma_x. Used primarily for letting
                sigma_x float during PSF fitting.
            angle: An alternative way to specify angle. Used primarily for letting
                angle float during PSF fitting.

        Returns:
            The value of the 2-D Gaussian at the point(s) specified by coord.
        """

        sy = self._sigma_y
        sx = self._sigma_x

        if (sx is not None and (sigma is not None or sigma_x is not None)) or (
            sy is not None and (sigma is not None or sigma_y is not None)
        ):
            raise ValueError('Cannot specify both sigma during init and sigma_y/x')

        if sigma is not None:
            if not isinstance(sigma, tuple):
                sy = sx = float(sigma)
            else:
                sy, sx = sigma[0], sigma[1]

        if sigma_y is not None:
            sy = sigma_y
        if sigma_x is not None:
            sx = sigma_x

        if sx is None or sy is None:
            raise ValueError(
                'Sigma X and Y must be specified either at object creation '
                'or in the call to eval_point'
            )

        r: float = 0.0 if self._angle is None else float(self._angle)
        if angle is not None:
            r = float(angle)

        ret = GaussianPSF.gaussian_2d(
            coord[0],
            coord[1],
            sigma_y=sy,
            sigma_x=sx,
            mean_y=self._mean_y,
            mean_x=self._mean_x,
            scale=scale,
            base=base,
            angle=r,
        )
        return cast(float, ret)

    def eval_pixel(
        self,
        coord: (
            tuple[int | npt.NDArray[np.int_], int | npt.NDArray[np.int_]] | npt.NDArray[np.floating]
        ),
        offset: (
            tuple[float | npt.NDArray[np.floating], float | npt.NDArray[np.floating]]
            | npt.NDArray[np.floating]
        ) = (0.5, 0.5),
        *,
        scale: float = 1.0,
        base: float = 0.0,
        sigma: tuple[float, float] | None = None,
        sigma_y: float | None = None,
        sigma_x: float | None = None,
        angle: float | None = None,
    ) -> float | npt.NDArray[np.floating]:
        """Evaluate the Gaussian PSF integrated over an entire integer pixel.

        The returned array has the PSF offset from the center by (offset_y, offset_x). An
        offset of (0, 0) places the PSF in the upper left corner of the center pixel while
        an offset of (0.5, 0.5) places the PSF in the center of the center pixel. The
        angle is applied relative to this new origin, so as angle changes the center of
        the ellipse does not move.

        This essentially performs a 2-D integration of the PSF over the intervals
        [y-offset_y-0.5, y-offset_y+0.5] and [x-offset_x-0.5, x-offset_x+0.5].

        Parameters:
            coord: The integer coordinate(s) (y, x) at which to evaluate the PSF.
            offset: The amount (offset_y, offset_x) to offset the center of the PSF.
            scale: A scale factor to apply to the resulting PSF.
            base: A scalar added to the resulting PSF.
            sigma: Standard deviations: a scalar (both axes) or a ``(sigma_y, sigma_x)``
                tuple. Must not duplicate values fixed at construction; ``None`` if both
                are already set on the instance.
            sigma_y: An alternative way to specify sigma_y. Used primarily for letting
                sigma_y float during PSF fitting.
            sigma_x: An alternative way to specify sigma_x. Used primarily for letting
                sigma_x float during PSF fitting.
            angle: An alternative way to specify angle. Used primarily for letting
                angle float during PSF fitting.

        Returns:
            The integral of the 2-D Gaussian over each full pixel.
        """

        sy = self._sigma_y
        sx = self._sigma_x

        if (sx is not None and (sigma is not None or sigma_x is not None)) or (
            sy is not None and (sigma is not None or sigma_y is not None)
        ):
            raise ValueError('Cannot specify both sigma during init and sigma_y/x')

        if sigma is not None:
            if not isinstance(sigma, tuple):
                sy = sx = float(sigma)
            else:
                sy, sx = sigma[0], sigma[1]

        if sigma_y is not None:
            sy = sigma_y
        if sigma_x is not None:
            sx = sigma_x

        r_angle: float = 0.0 if self._angle is None else float(self._angle)
        if angle is not None:
            r_angle = float(angle)

        if sx is None or sy is None:
            raise ValueError(
                'Sigma X and Y must be specified either at object creation '
                'or in the call to eval_pixel'
            )

        ret = GaussianPSF.gaussian_integral_2d(
            coord[0] - offset[0],
            coord[0] - offset[0] + 1.0,
            coord[1] - offset[1],
            coord[1] - offset[1] + 1.0,
            sigma_y=sy,
            sigma_x=sx,
            mean_y=self._mean_y,
            mean_x=self._mean_x,
            scale=scale,
            base=base,
            angle=r_angle,
            angle_subsample=self._angle_subsample,
        )
        return ret

    # Intentional override of :meth:`PSF._eval_rect`: ``rect_size`` / ``offset`` are
    # narrowed to ``tuple``, Gaussian-specific keyword-only arguments are added, and the
    # return type is ``np.floating``; call sites from the base class remain compatible.
    def _eval_rect(  # type: ignore[override]
        self,
        rect_size: tuple[int, int],
        offset: tuple[float, float] = (0.5, 0.5),
        *,
        scale: float = 1.0,
        base: float = 0.0,
        sigma: tuple[float, float] | None = None,
        sigma_y: float | None = None,
        sigma_x: float | None = None,
        angle: float | None = None,
    ) -> npt.NDArray[np.floating]:
        """Pixel-integrated Gaussian on a rectangle (same grid as :meth:`eval_rect`).

        Parameters:
            rect_size: ``(size_y, size_x)`` patch shape (odd counts).
            offset: Subpixel shift ``(y, x)`` passed to :meth:`eval_pixel`.
            scale: Multiplicative scale for the Gaussian flux.
            base: Additive constant per pixel after scaling.
            sigma: Optional ``(sigma_y, sigma_x)`` pair overriding instance sigmas.
            sigma_y: Override for ``sigma_y`` when fitting.
            sigma_x: Override for ``sigma_x`` when fitting.
            angle: Override for rotation angle (radians); instance default if ``None``.

        Returns:
            A 2-D :class:`numpy.ndarray` of floats with shape ``rect_size``, each entry
            the integral of the Gaussian over that pixel.
        """

        rect_size_y, rect_size_x = rect_size
        y_coords = np.repeat(
            np.arange(-(rect_size_y // 2), rect_size_y // 2 + 1, dtype=np.float64), rect_size_x
        )
        x_coords = np.tile(
            np.arange(-(rect_size_x // 2), rect_size_x // 2 + 1, dtype=np.float64), rect_size_y
        )
        coords = np.empty((2, rect_size_y * rect_size_x))
        coords[0] = y_coords
        coords[1] = x_coords
        rect = self.eval_pixel(
            coords,
            offset,
            scale=scale,
            base=base,
            sigma=sigma,
            sigma_y=sigma_y,
            sigma_x=sigma_x,
            angle=angle,
        )
        rect = cast(npt.NDArray[np.floating], rect)
        rect = rect.reshape(rect_size)

        return rect

    # Same rationale as :meth:`_eval_rect` above: extends :meth:`PSF.eval_rect` with
    # Gaussian kwargs and concrete tuple types while delegating to
    # :meth:`PSF._eval_rect_smeared`.
    def eval_rect(  # type: ignore[override]
        self,
        rect_size: tuple[int, int],
        offset: tuple[float, float] = (0.5, 0.5),
        *,
        movement: tuple[float, float] | None = None,
        movement_granularity: float = 0.1,
        scale: float = 1.0,
        base: float = 0.0,
        sigma: tuple[float, float] | None = None,
        sigma_y: float | None = None,
        sigma_x: float | None = None,
        angle: float | None = None,
    ) -> npt.NDArray[np.floating]:
        """Create a rectangular pixelated Gaussian PSF.

        This is done by evaluating the PSF function from
        [-rect_size_y//2:rect_size_y//2] to [-rect_size_x//2:rect_size_x//2].

        The returned array has the PSF offset from the center by
        (offset_y, offset_x). An offset of (0, 0) places the PSF in the upper
        left corner of the center pixel while an offset of (0.5, 0.5)
        places the PSF in the center of the center pixel. The angle is applied
        relative to this new origin, so as angle changes the center of the
        ellipse does not move.

        Parameters:
            rect_size: The size of the rectangle (rect_size_y, rect_size_x) of the
                returned PSF. Both dimensions must be odd.
            offset: The amount (offset_y, offset_x) to offset the center of the PSF.
            movement: The amount of motion blur in the (Y, X) direction. Must be a tuple
                of scalars. None means no movement.
            movement_granularity: The number of pixels to step for each smear while doing
                motion blur. A smaller granularity means that the resulting PSF will be
                more precise but also take longer to compute.
            scale: A scale factor to apply to the resulting PSF.
            base: A scalar added to the resulting PSF.
            sigma: Standard deviations: a scalar (both axes) or a ``(sigma_y, sigma_x)``
                tuple. Must not duplicate values fixed at construction; ``None`` if both
                are already set on the instance.
            sigma_y: An alternative way to specify sigma_y. Used primarily for letting
                sigma_y float during PSF fitting.
            sigma_x: An alternative way to specify sigma_x. Used primarily for letting
                sigma_x float during PSF fitting.
            angle: An alternative way to specify angle. Used primarily for letting
                angle float during PSF fitting.

        Returns:
            The integral of the 2-D Gaussian over each full pixel in the rectangle.
        """

        rect_size_y, rect_size_x = rect_size

        if rect_size_y < 0 or rect_size_x < 0 or rect_size_y % 2 != 1 or rect_size_x % 2 != 1:
            raise ValueError(
                f'Rectangle must have odd positive shape in each dimension, got {rect_size}'
            )

        return self._eval_rect_smeared(
            rect_size,
            offset=offset,
            movement=movement,
            movement_granularity=movement_granularity,
            scale=scale,
            base=base,
            sigma=sigma,
            sigma_y=sigma_y,
            sigma_x=sigma_x,
            angle=angle,
        )
