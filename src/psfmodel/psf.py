################################################################################
# psfmodel/psf.py
################################################################################

import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import scipy.linalg as linalg
import scipy.optimize as sciopt

# Version
try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = 'Version unspecified'

# Unbounded ends for the additive PSF ``base`` in :meth:`PSF._find_position` when
# ``allow_nonzero_base`` is True and ``use_angular_params`` is False (Powell box
# constraints).
_FIT_PSF_BASE_BOUND_MIN = float('-inf')
_FIT_PSF_BASE_BOUND_MAX = float('inf')


class PSF(ABC):
    """Abstract base for 2-D point-spread models used in fitting and rendering.

    Subclass :class:`PSF` to provide a concrete model. The base class supplies shared
    utilities (for example :meth:`find_position`, background helpers, and motion smear
    via :meth:`_eval_rect_smeared`); evaluation itself is defined by subclasses through
    the abstract API below.

    **Abstract methods (must implement)**

        * :meth:`eval_point` -- Evaluate the continuous PSF at fractional ``(y, x)``
          (scalar or broadcast arrays). Signature includes keyword-only ``scale`` and
          ``base``. Returns a float or a :class:`numpy.ndarray` of floats matching the
          broadcast shape of ``coord``. Origin ``(0, 0)`` is the PSF center; coordinates
          may be negative. Subclasses may add keyword-only parameters.

        * :meth:`eval_rect` -- Build a rectangular, pixel-integrated patch. Signature:
          ``rect_size``, ``offset``, then keyword-only ``movement``,
          ``movement_granularity``, ``scale``, ``base``, and subclass-specific
          ``**kwargs``. Must return :class:`numpy.ndarray` with ``dtype`` ``float64`` and
          shape ``(height, width)`` matching ``rect_size`` as ``(size_y, size_x)``. Should
          validate inputs (for example odd ``rect_size``) and raise :exc:`ValueError` for
          invalid arguments with a clear message.

        * :meth:`_eval_rect` -- Internal hook for the same patch without the checks in
          :meth:`eval_rect`; same core keyword-only ``scale`` and ``base``. Must return a
          ``float64`` array of shape ``(height, width)``. Subclasses often add
          keyword-only model parameters. Callers must pass consistent arguments;
          implementations may omit validation.

    **Protected attributes**

        * ``_logger`` -- :class:`logging.Logger` for this instance (set in
          :meth:`__init__`). Subclasses log warnings and diagnostics through it.

        * ``_additional_params`` -- :class:`list` (initialized empty), each entry a
          ``(lower_bound, upper_bound, name)`` tuple of two floats and a :class:`str`
          keyword name. Used by :meth:`find_position` / :meth:`_find_position` to append
          extra optimized parameters (bounds and :meth:`eval_rect` keyword). Subclasses
          append one tuple per fittable quantity in construction order; leave the list
          empty if there are no extra parameters (for example fixed-width models).

    **Errors and return conventions**

        Public evaluators should reject bad arguments with :exc:`ValueError` (or
        :exc:`TypeError` for wrong types) where feasible. Some higher-level routines
        (notably :meth:`find_position`) signal failure by returning ``None`` instead of
        raising. Numeric outputs are real floating point; ``scale`` multiplies the model
        amplitude and ``base`` adds a constant offset in the same units as the evaluated
        PSF values unless a subclass documents physical units.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        detailed_logging: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a PSF object. Only called by subclasses.

        Parameters:
            logger: Logger for diagnostic messages from this instance. If omitted,
                uses :func:`logging.getLogger` with this module's ``__name__``.
            detailed_logging: If True, emit INFO and DEBUG messages during fitting
                (for example from :meth:`find_position`). Optimizer failure still logs
                at WARNING when this is False.
            **kwargs: Reserved for subclass forward-compatibility; unknown names are
                ignored.
        """

        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self.detailed_logging = detailed_logging
        self._additional_params: list[Any] = []

    @abstractmethod
    def eval_point(
        self,
        coord: (
            tuple[float | npt.NDArray[np.floating], float | npt.NDArray[np.floating]]
            | npt.NDArray[np.floating]
        ),
        *,
        scale: float = 1.0,
        base: float = 0.0,
    ) -> float | npt.NDArray[np.floating]:
        """Evaluate the PSF at a single, fractional, point.

        (0, 0) is the center of the PSF and x and y may be negative.

        Parameters:
            coord: The coordinate (y, x) at which to evaluate the PSF.
            scale: A scale factor to apply to the resulting PSF.
            base: A scalar added to the resulting PSF.

        Other parameters may be available for specific subclasses.

        Returns:
            The PSF value at the given coordinate.
        """
        ...  # pragma: no cover

    # @abstractmethod
    # def eval_pixel(self,
    #                coord: list[int] | tuple[int, int],
    #                offset: list[float] | tuple[float, float] = (0., 0.),
    #                *,
    #                scale: float = 1.,
    #                base: float = 0.,
    #                **kwargs: Any) -> float:
    #     """Evaluate the PSF integrated over an entire integer pixel.

    #     The returned array has the PSF offset from the center by (offset_y,offset_x). An
    #     offset of (0, 0) places the PSF in the upper left corner of the center pixel
    # while
    #     an offset of (0.5, 0.5) places the PSF in the center of the center pixel. The
    #     offset should be limited to the range [0, 1).

    #     Parameters:
    #         coord: The integer coordinate (y, x) at which to evaluate the PSF.
    #         offset: The amount (offset_y, offset_x) to offset the center of the PSF.
    #         scale: A scale factor to apply to the resulting PSF.
    #         base: A scalar added to the resulting PSF.

    #     Other inputs may be available for specific subclasses.
    #     """
    #     ...

    @abstractmethod
    def eval_rect(
        self,
        rect_size: list[int] | tuple[int, int],
        offset: list[float] | tuple[float, float] = (0.5, 0.5),
        *,
        movement: tuple[float, float] | None = None,
        movement_granularity: float = 0.1,
        scale: float = 1.0,
        base: float = 0.0,
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Create a rectangular pixelated PSF.

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

        Other inputs may be available for specific subclasses.

        Returns:
            The integral of the 2-D PSF over each full pixel in the rectangle.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _eval_rect(
        self,
        rect_size: tuple[int, int],
        offset: tuple[float, float] = (0.5, 0.5),
        *,
        scale: float = 1.0,
        base: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Pixel-integrated rectangular PSF; internal counterpart to :meth:`eval_rect`.

        Used by :meth:`_eval_rect_smeared` and subclass implementations. Unlike
        :meth:`eval_rect`, this hook performs no input validation, bounds checking, or
        clipping; callers must supply consistent arguments.

        Parameters:
            rect_size: ``(height, width)`` in pixels, i.e. ``(size_y, size_x)`` (row and
                column counts). This matches the shape of the returned array.
            offset: ``(offset_y, offset_x)`` subpixel shift of the PSF reference in
                fractional pixel coordinates. Default ``(0.5, 0.5)`` centers the model in
                the middle pixel; ``(0.0, 0.0)`` uses the top-left corner of that pixel
                as the reference (same convention as :meth:`eval_rect`).
            scale: Multiplier applied to the PSF amplitude before ``base`` is added.
            base: Additive baseline added to every output pixel after ``scale``.

        Returns:
            A :class:`numpy.ndarray` of dtype ``float64`` with shape ``(height, width)``
            containing the rectangular, pixel-sampled PSF.

        Note:
            Concrete subclasses may add keyword-only parameters for model-specific
            quantities (for example width or angle on a Gaussian).
        """
        ...  # pragma: no cover

    def _eval_rect_smeared(
        self,
        rect_size: tuple[int, int],
        offset: tuple[float, float] = (0.5, 0.5),
        *,
        movement: tuple[float, float] | None = None,
        movement_granularity: float = 0.1,
        scale: float = 1.0,
        base: float = 0.0,
        **kwargs: Any,
    ) -> npt.NDArray[np.floating]:
        """Evaluate and sum a PSF multiple times to simulate motion blur.

        Parameters:
            rect_size: The size of the rectangle (rect_size_y, rect_size_x) of the
                returned PSF. Both dimensions must be odd.
            offset: The amount (offset_y, offset_x) to offset the center of the PSF. A
                positive offset effectively moves the PSF down and to the left. XXX
            movement: The total amount (my, mx) the PSF moves. The movement is assumed to
                be centered on the given offset and exists half on either side.
            movement_granularity: The number of pixels to step for each smear while doing
                motion blur.
            scale: A scale factor to apply to the resulting PSF.
            base: A scalar added to the resulting PSF.

        Other inputs may be available for specific subclasses.
        """

        if movement is None or (movement[0] == 0 and movement[1] == 0):
            return self._eval_rect(rect_size, offset=offset, scale=scale, base=base, **kwargs)

        num_steps = int(
            max(abs(movement[0]) / movement_granularity, abs(movement[1]) / movement_granularity)
        )

        if num_steps == 0:
            step_y = 0.0
            step_x = 0.0
        else:
            step_y = movement[0] / num_steps
            step_x = movement[1] / num_steps

        total_rect = None

        for step in range(num_steps + 1):
            y = offset[0] + step_y * (step - num_steps / 2.0)
            x = offset[1] + step_x * (step - num_steps / 2.0)

            rect = self._eval_rect(rect_size, offset=(y, x), scale=scale, base=base, **kwargs)
            if total_rect is None:
                total_rect = rect
            else:
                total_rect += rect
        if total_rect is None:
            raise RuntimeError('Motion smear loop produced no PSF rectangles')

        total_rect /= float(num_steps + 1)

        return total_rect

    # ==========================================================================
    #
    # Static functions for creating background gradients
    #
    # ==========================================================================

    @staticmethod
    def _background_gradient_coeffs(shape: tuple[int, int], order: int) -> npt.NDArray[np.float64]:
        """Internal routine for creating the coefficient matrix.

        Fundamentally this creates a coefficient matrix indicating the powers of different
        orders of polynomials. For example, an order 1 polynomial (Ax + B) when performed
        in two dimensions becomes (Ax + By + C), which has three free parameters. An order
        2 polynomial (Ax^2 + Bx + C) in two dimensions becomes (Ax^2 + By^2 + Cxy + Dx +
        Ey + F), which has six free parameters. The number of free parameters is (order *
        (order+1)) / 2.

        To make further computation easy, this coefficient matrix is then multiplied with
        a 2-D array that represents the X and Y coordinates, ranging from -N to N such
        that the values are (0, 0) at the center of the image. This is the matrix that is
        returned to the caller.

        The resulting 3-D matrix has the indicies:
            0: Y
            1: X
            2: parameter number
        """

        if shape[0] < 0 or shape[1] < 0 or shape[0] % 2 != 1 or shape[1] % 2 != 1:
            raise ValueError(f'Image must have odd positive shape in each dimension, got {shape}')
        if order < 0:
            raise ValueError(f'Order must be non-negative, got {order}')

        # Create arrays of indexes for line and sample with (0, 0) at the center of the
        # image
        y_values = np.arange(shape[0])[:, np.newaxis] - int(shape[0] / 2)
        x_values = np.arange(shape[1])[np.newaxis, :] - int(shape[1] / 2)

        y_powers: list[float | npt.NDArray[np.floating]] = [1.0]
        x_powers: list[float | npt.NDArray[np.floating]] = [1.0]

        nparams = int((order + 1) * (order + 2) / 2)
        a3d = np.empty((shape[0], shape[1], nparams))
        a3d[:, :, 0] = 1.0  # This is the constant term of the polynomial

        k = 0  # Parameter number
        for p in range(1, order + 1):
            # This creates, sequentially, L, L**2, L**3... and S, S**2, S**3...
            y_powers.append(y_powers[-1] * y_values)
            x_powers.append(x_powers[-1] * x_values)

            # These nested loops walk through all the combinations of L**N * S**M
            # such that N+M == P where P ranges from 1 to <order>. This gives us
            # all combinations like:
            #   1
            #   Y
            #   X
            #   X*Y
            #   Y**2
            #   X**2
            for q in range(p + 1):
                k += 1
                a3d[:, :, k] = y_powers[q] * x_powers[p - q]

        return a3d

    @staticmethod
    def background_gradient_fit(
        image: npt.NDArray[np.floating],
        order: int = 2,
        ignore_center: int | tuple[int, int] | None = None,
        num_sigma: float | None = None,
        debug: bool = False,
        *,
        logger: logging.Logger | None = None,
    ) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None]:
        """Return the polynomial fit to the pixels of an image.

        Parameters:
            image: 2D array to fit; must have odd shape in each dimension.
            order: Order of the polynomial.
            ignore_center: A scalar or tuple (ignore_y, ignore_x) giving the number of
                pixels on either side of the center to ignore while fitting. 0 means
                ignore the center pixel. None means don't ignore anything.
            num_sigma: Outlier rejection uses the fit residual ``image - gradient``:
                unmasked pixels with absolute residual at least ``num_sigma`` times the
                standard deviation of that residual (mask-aware) are masked and the fit
                is repeated. None disables this. Non-positive values disable masking
                after the initial least-squares fit.
            debug: Set to debug bad pixel removal.
            logger: Logger for debug messages; defaults to this module's logger.

        Returns:
            A tuple of the background coefficient array and the mask of ignored pixels.
        """

        fit_logger = logger if logger is not None else logging.getLogger(__name__)

        if len(image.shape) != 2:
            raise ValueError(f'Image must be 2-D, got {image.shape}')
        if (
            image.shape[0] < 0
            or image.shape[1] < 0
            or image.shape[0] % 2 != 1
            or image.shape[1] % 2 != 1
        ):
            raise ValueError(
                f'Image must have odd positive shape in each dimension, got {image.shape}'
            )
        if order < 0:
            raise ValueError(f'Order must be non-negative, got {order}')

        shape = cast(tuple[int, int], image.shape)

        is_masked = False

        if ignore_center is not None or num_sigma is not None:
            if isinstance(image, ma.MaskedArray):
                # We're going to change the mask so make a copy first
                image = image.copy()
            else:
                image = image.view(ma.MaskedArray)

        if isinstance(image, ma.MaskedArray):
            image.mask = cast(npt.NDArray[np.bool_], ma.getmaskarray(image))
            is_masked = True

        if ignore_center is not None:
            if isinstance(ignore_center, int):
                ignore_y = ignore_center
                ignore_x = ignore_center
            else:
                ignore_y, ignore_x = ignore_center
            if ignore_y * 2 + 1 >= shape[0] or ignore_x * 2 + 1 >= shape[1]:
                if debug:  # pragma: no cover
                    fit_logger.debug('Background fit: ignore_center covers entire image')
                return None, None
            ctr_y = shape[0] // 2
            ctr_x = shape[1] // 2
            image[
                ctr_y - ignore_y : ctr_y + ignore_y + 1, ctr_x - ignore_x : ctr_x + ignore_x + 1
            ] = ma.masked

        nparams = int((order + 1) * (order + 2) // 2)

        a3d = PSF._background_gradient_coeffs(shape, order)

        if num_sigma is not None:
            num_bad_pixels = cast(int, ma.count_masked(image))  # type: ignore
            if debug:  # pragma: no cover
                fit_logger.debug(
                    'Background gradient fit: initial masked pixel count %s', num_bad_pixels
                )

        while True:
            # Reshape properly for linalg.lstsq
            a2d = a3d.reshape((image.size, nparams))
            b1d = image.flatten()

            if is_masked:
                # linalg doesn't support masked arrays!
                a2d = a2d[~b1d.mask]  # type: ignore
                b1d = ma.compressed(b1d)

            if a2d.shape[0] < a2d.shape[1]:  # Underconstrained
                if debug:  # pragma: no cover
                    fit_logger.debug(
                        'Background gradient fit: underconstrained system %s', a2d.shape
                    )
                return None, None

            coeffts = linalg.lstsq(a2d, b1d)[0]

            if num_sigma is None:
                break
            num_sigma_f = float(num_sigma)
            if num_sigma_f <= 0:
                break

            gradient = PSF.background_gradient(shape, coeffts)
            delta_img = image - gradient
            sigma = ma.std(delta_img)
            if ma.is_masked(sigma):
                break
            sigma_f = float(sigma)
            if not np.isfinite(sigma_f) or sigma_f <= 0:
                break
            threshold = num_sigma_f * sigma_f
            if debug:  # pragma: no cover
                fit_logger.debug(
                    'Background gradient fit: residual std=%s max_abs=%s threshold=%s',
                    sigma_f,
                    float(ma.max(ma.abs(delta_img))),
                    threshold,
                )
            outlier_mask = ma.filled(ma.abs(delta_img) >= threshold, False)
            image[outlier_mask] = ma.masked

            new_num_bad_pixels = cast(int, ma.count_masked(image))  # type: ignore
            if debug:  # pragma: no cover
                fit_logger.debug(
                    'Background gradient fit: masked pixel count now %s', new_num_bad_pixels
                )
            if new_num_bad_pixels == num_bad_pixels:
                break
            num_bad_pixels = new_num_bad_pixels

        if is_masked:
            return coeffts, ma.getmaskarray(image)  # type: ignore
        else:
            return coeffts, np.zeros(shape, dtype=np.bool_)

    @staticmethod
    def background_gradient(
        rect_size: tuple[int, int], bkgnd_params: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Create a background gradient.

        Parameters:
            rect_size: ``(size_y, size_x)``, the shape of the output grid (height, width)
                in pixels; must match the image shape used when the coefficients were fit.
            bkgnd_params: Coefficients of the background polynomial (1-D array-like). The
                polynomial order is inferred from the number of elements.

        Returns:
            A :class:`numpy.ndarray` of ``dtype`` ``float64`` with shape ``rect_size``
            (i.e. ``(size_y, size_x)``): the evaluated 2-D background polynomial at each
            pixel center of the grid.
        """

        bkgnd_params = np.array(bkgnd_params)

        order = int(np.sqrt(len(bkgnd_params) * 2)) - 1

        a3d = PSF._background_gradient_coeffs(rect_size, order)
        result = np.sum(bkgnd_params * a3d, axis=-1)

        return cast(npt.NDArray[np.float64], result)

    # ==========================================================================
    #
    # Functions for finding astrometric positions
    #
    # ==========================================================================

    def find_position(
        self,
        image: npt.NDArray[np.floating],
        box_size: tuple[int, int],
        starting_point: tuple[float, float],
        *,
        search_limit: float | tuple[float, float] = (1.5, 1.5),
        bkgnd_degree: int | None = 2,
        bkgnd_ignore_center: tuple[int, int] = (2, 2),
        bkgnd_num_sigma: float | None = None,
        tolerance: float = 1e-6,
        num_sigma: float | None = None,
        max_bad_frac: float = 0.2,
        allow_nonzero_base: bool = False,
        scale_limit: float = 1000.0,
        use_angular_params: bool = True,
    ) -> None | tuple[float, float, dict[str, Any]]:
        """Find the (y, x) coordinates that best fit a 2-D PSF to an image.

        Parameters:
            image: The image (2-D).
            box_size: A tuple (box_y, box_x) indicating the size of the PSF to use. This
                governs both the size of the PSF created at each step as well as the size
                of the subimage looked at. Both box_y and box_x must be odd.
            starting_point: A tuple (y, x) indicating the best guess for where the object
                can be found. Searching is limited to a region around this point
                controlled by `search_limit`.
            search_limit: A scalar or tuple (y_limit, x_limit) specifying the maximum
                distance to search from `starting_point`. If a scalar, both x_limit
                and y_limit are the same.
            bkgnd_degree: The degree (order) of the background gradient polynomial. None
                means no background gradient is fit.
            bkgnd_ignore_center: A tuple (ny, nx) giving the number of pixels on each side
                of the center point to ignore when fitting the background. The ignored
                region is thus ny*2+1 by nx*2+1.
            bkgnd_num_sigma: The number of sigma a pixel needs to be beyond the background
                gradient to be ignored. None means don't ignore bad pixels while computing
                the background gradient.
            tolerance: The tolerance (both X and Function) in the Powell optimization
                algorithm.
            num_sigma: The number of sigma for a pixel to be considered bad during PSF
                fitting. None means don't ignore bad pixels while fitting the PSF.
            max_bad_frac: The maximum allowable number of pixels masked during PSF
                fitting. If more pixels than this fraction are masked, the position fit
                fails.
            allow_nonzero_base: If True, allow the base of the PSF (constant bias) to
                vary. Otherwise the base of the PSF is always at zero and can only scale
                in the positive direction.
            scale_limit: The maximum PSF scale allowed.
            use_angular_params: Use angles to optimize parameter values.

        Returns:
            None if no fit found.

            Otherwise returns pos_y, pos_x, metadata. Metadata is a dictionary
            containing::

                'x'                    The offset in X. (Same as pos_x)
                'x_err'                Uncertainty in X.
                'y'                    The offset in Y. (Same as pos_y)
                'y_err'                Uncertainty in Y.
                'scale'                The best fit PSF scale.
                'scale_err'            Uncertainty in PSF scale.
                'base'                 The best fit PSF base.
                'base_err'             Uncertainty in PSF base.
                'subimg'               The box_size area of the original image
                                       surrounding starting_point masked as
                                       necessary using the num_sigma threshold.
                'bkgnd_params'         The tuple of parameters defining the
                                       background gradient.
                'bkgnd_mask'           The mask used during background gradient
                                       fitting.
                'gradient'             The box_size background gradient.
                'subimg-gradient'      The subimg with the background gradient
                                       subtracted.
                'psf'                  The PSF model from eval_rect with the fitted
                                       scale and base (same array as scaled_psf).
                'scaled_psf'           Same as psf; model to compare to
                                       subimg-gradient during outlier rejection.
                'leastsq_cov'          The covariance matrix returned by leastsq
                                       as adjusted by the residual variance.
                'leastsq_infodict'     The infodict returned by leastsq.
                'leastsq_mesg'         The mesg returned by leastsq.
                'leastsq_ier'          The ier returned by leastsq.

            In addition, metadata includes two entries for each "additional
            parameter" used during optimization: one for the value and one for
            the uncertainty ('param' and 'param_err').
        """

        if box_size[0] < 0 or box_size[1] < 0 or box_size[0] % 2 != 1 or box_size[1] % 2 != 1:
            raise ValueError(
                f'box_size must have odd positive shape in each dimension, got {box_size}'
            )

        half_box_size_y = box_size[0] // 2
        half_box_size_x = box_size[1] // 2

        starting_pix = (int(starting_point[0]), int(starting_point[1]))

        if self.detailed_logging:
            self._logger.info('find_position: entering')
            self._logger.info(
                'find_position: image masked=%s num_masked=%s',
                isinstance(image, ma.MaskedArray),
                int(np.sum(ma.getmaskarray(image))),
            )
            self._logger.info(
                'find_position: image min=%s max=%s mean=%s',
                float(np.min(image)),
                float(np.max(image)),
                float(np.mean(image)),
            )
            self._logger.info(
                'find_position: box_size=%s starting_point=%s search_limit=%s',
                box_size,
                starting_point,
                search_limit,
            )
            self._logger.info(
                'find_position: bkgnd_degree=%s bkgnd_ignore_center=%s '
                'bkgnd_num_sigma=%s tolerance=%s num_sigma=%s max_bad_frac=%s '
                'allow_nonzero_base=%s scale_limit=%s use_angular_params=%s',
                bkgnd_degree,
                bkgnd_ignore_center,
                bkgnd_num_sigma,
                tolerance,
                num_sigma,
                max_bad_frac,
                allow_nonzero_base,
                scale_limit,
                use_angular_params,
            )

        # Too close to the edge means we can't search
        if (
            starting_pix[0] - half_box_size_y < 0
            or starting_pix[0] + half_box_size_y >= image.shape[0]
            or starting_pix[1] - half_box_size_x < 0
            or starting_pix[1] + half_box_size_x >= image.shape[1]
        ):
            if self.detailed_logging:
                self._logger.info('find_position: too close to image edge, aborting')
            return None

        sub_img = image[
            starting_pix[0] - half_box_size_y : starting_pix[0] + half_box_size_y + 1,
            starting_pix[1] - half_box_size_x : starting_pix[1] + half_box_size_x + 1,
        ]

        if self.detailed_logging:
            self._logger.info(
                'find_position: subimage min=%s max=%s mean=%s',
                float(np.min(sub_img)),
                float(np.max(sub_img)),
                float(np.mean(sub_img)),
            )

        if not isinstance(search_limit, (list, tuple)):
            search_limit = (float(search_limit), float(search_limit))

        if num_sigma:
            if isinstance(sub_img, ma.MaskedArray):
                # We're going to change the mask so make a copy first
                sub_img = sub_img.copy()
            else:
                sub_img = sub_img.view(ma.MaskedArray)

        num_bad_pixels = 0

        while True:
            if self.detailed_logging:
                self._logger.debug('find_position: outer loop bad_pixel_count=%s', num_bad_pixels)
            ret = self._find_position(
                sub_img,
                search_limit,
                scale_limit,
                bkgnd_degree,
                bkgnd_ignore_center,
                bkgnd_num_sigma,
                tolerance,
                allow_nonzero_base,
                use_angular_params,
            )
            if ret is None:
                if self.detailed_logging:
                    self._logger.info('find_position: inner fit returned None')
                return None

            res_y, res_x, details = ret

            if not num_sigma:
                break

            resid = np.sqrt((details['subimg-gradient'] - details['scaled_psf']) ** 2)
            resid_std = np.std(resid)

            if self.detailed_logging:
                self._logger.debug('find_position: residual per pixel=%s', resid)
                self._logger.debug('find_position: resid_std=%s', resid_std)

            if num_sigma is not None:
                sub_img[np.where(resid > num_sigma * resid_std)] = ma.masked

            new_num_bad_pixels = ma.count_masked(sub_img)  # type: ignore
            if new_num_bad_pixels == num_bad_pixels:
                break
            if new_num_bad_pixels == sub_img.size:
                if self.detailed_logging:
                    self._logger.info('find_position: all pixels masked, returning None')
                return None  # All masked
            if new_num_bad_pixels > max_bad_frac * sub_img.size:
                if self.detailed_logging:
                    self._logger.info('find_position: too many pixels masked, returning None')
                return None  # Too many masked
            num_bad_pixels = new_num_bad_pixels

        if self.detailed_logging:
            msg = f'find_position returning Y {res_y + starting_pix[0]:.4f}'
            # if details['y_err'] is not None:
            #     msg += f' +/- {details["y_err"]:.4f}'
            msg += f' X {res_x + starting_pix[1]:.4f}'
            # if details['x_err'] is not None:
            #     msg += ' +/- {details["x_err"]:.4f}'
            if details['scale'] is not None:
                msg += f' Scale {details["scale"]:.4f} Base {details["base"]:.4f}'
            if 'sigma_y' in details:
                msg += f' SY {details["sigma_y"]:.4f} SX {details["sigma_x"]:.4f}'
            self._logger.info(msg)

        return res_y + starting_pix[0], res_x + starting_pix[1], details

    def _fit_psf_func(
        self,
        params: tuple[float, ...],
        sub_img: npt.NDArray[np.floating],
        search_limit: tuple[float, float],
        scale_limit: float,
        allow_nonzero_base: bool,
        use_angular_params: bool,
        *additional_params: Any,
    ) -> float:
        """Scalar objective for PSF fitting; minimized in :meth:`_find_position`.

        Evaluates :meth:`eval_rect` at the candidate parameters, subtracts the model from
        ``sub_img``, and returns the Euclidean norm of the flattened residual (root sum
        of squared differences).

        Parameters:
            params: Optimizer vector: offset(s), scale, optional ``base`` (if
                ``allow_nonzero_base``), then one value per extra PSF parameter. Meaning
                depends on ``use_angular_params`` (angles vs direct values); see the
                implementation.
            sub_img: 2-D patch (same shape as the PSF grid); background should already be
                subtracted by the caller when applicable.
            search_limit: ``(limit_y, limit_x)`` centroid search half-ranges in pixels,
                used when mapping ``params`` to offsets (see ``use_angular_params``).
            scale_limit: Upper bound on PSF ``scale`` for :meth:`eval_rect`.
            allow_nonzero_base: If ``True``, ``params`` includes a fitted constant
                ``base`` passed to :meth:`eval_rect`; if ``False``, ``base`` is zero.
            use_angular_params: If ``True``, map bounded angles to offsets, scale, extras,
                and optional ``base``; if ``False``, ``params`` are physical values within
                bounds set by the caller.
            additional_params: Zero or more ``(lo, hi, name)`` tuples giving bounds and
                keyword names for subclass-specific :meth:`eval_rect` arguments.

        Returns:
            Non-negative float cost (lower is better).
        """

        # Make an offset of "0" be the center of the pixel (0.5, 0.5)
        if use_angular_params:
            # params are (ang_y, ang_x, ang_scale, ...)
            offset_y = search_limit[0] * np.cos(params[0]) + 0.5
            offset_x = search_limit[1] * np.cos(params[1]) + 0.5
            scale = scale_limit * (np.cos(params[2]) + 1) / 2
        else:
            # params are (y, x, scale, ...)
            offset_y = params[0] + 0.5
            offset_x = params[1] + 0.5
            scale = params[2]
            # This was only needed when using an optimization func that doesn't support
            # bounds.
            # fake_resid = None
            # if not (-search_limit[0] <= params[0] <= search_limit[0]):
            #     fake_resid = abs(params[0]) * 1e10
            # elif not (-search_limit[1] <= params[1] <= search_limit[1]):
            #     fake_resid = abs(params[1]) * 1e10
            # elif not (0.00001 <= scale <= scale_limit):
            #     fake_resid = abs(scale) * 1e10
            # if fake_resid is not None:
            #     fake_return = np.zeros(sub_img.shape).flatten()
            #     fake_return[:] = fake_resid
            #     if self.detailed_logging:
            #         full_resid = np.sqrt(np.sum(fake_return**2))
            #         print('RESID', full_resid)
            #     return fake_return

        base = 0.0
        param_end = 3
        if allow_nonzero_base:
            base = params[3]
            param_end = 4

        addl_vals_dict = {}
        for i, ap in enumerate(additional_params):
            if use_angular_params:
                val = (ap[1] - ap[0]) / 2.0 * (np.cos(params[param_end + i]) + 1.0) + ap[0]
            else:
                val = params[param_end + i]
            addl_vals_dict[ap[2]] = val

        psf = self.eval_rect(
            cast(tuple[int, int], sub_img.shape),
            (offset_y, offset_x),
            scale=scale,
            base=base,
            **addl_vals_dict,
        )

        resid = (sub_img - psf).flatten()

        full_resid = cast(float, np.sqrt(np.sum(resid**2)))

        if self.detailed_logging:
            msg = f'OFFY {offset_y:8.5f} OFFX {offset_x:8.5f} SCALE {scale:9.5f} '
            msg += f'BASE {base:9.5f}'
            for ap in additional_params:
                msg += f' {ap[2].upper()} {addl_vals_dict[ap[2]]:8.5f}'
            msg += f' RESID {full_resid:f}'
            self._logger.debug(msg)

        return full_resid

    def _find_position(
        self,
        sub_img: npt.NDArray[np.floating],
        search_limit: tuple[float, float],
        scale_limit: float,
        bkgnd_degree: int | None,
        bkgnd_ignore_center: tuple[int, int],
        bkgnd_num_sigma: float | None,
        tolerance: float,
        allow_nonzero_base: bool,
        use_angular_params: bool,
    ) -> None | tuple[float, float, dict[str, Any]]:
        """Fit PSF position and shape on a fixed subimage via bounded Powell optimization.

        This is the inner numerical core for :meth:`find_position`: it subtracts an
        optional polynomial background, then runs :func:`scipy.optimize.minimize` (Powell)
        on the scalar objective from :meth:`_fit_psf_func` (root-sum-square residual
        between data and model).

        Parameters:
            sub_img: Cropped 2-D image (float), same shape as the PSF evaluation patch.
                May be a :class:`numpy.ma.MaskedArray`; the background fit omits masked
                pixels, and masked entries do not contribute to the scalar objective in
                :meth:`_fit_psf_func` (masked squared residuals are excluded from the
                sum).
            search_limit: ``(limit_y, limit_x)`` maximum search half-range for subpixel
                offsets, in **pixels**, relative to the subimage. With
                ``use_angular_params`` True, offsets map from bounded angles via cosine
                (see implementation); with False, ``offset_*`` are bounded directly by
                these limits.
            scale_limit: Upper bound on PSF ``scale`` passed to :meth:`eval_rect` (same
                units as that method). The lower bound is a small positive value enforced
                inside :meth:`_fit_psf_func`.
            bkgnd_degree: If ``None``, no background is fit and ``gradient`` is all zeros.
                If an int, polynomial order for :meth:`background_gradient_fit` on
                ``sub_img`` before PSF optimization.
            bkgnd_ignore_center: ``(ny, nx)`` passed to :meth:`background_gradient_fit` as
                ``ignore_center``: a centered block of size ``(2*ny+1, 2*nx+1)`` is masked
                out of the background fit. Ignored when ``bkgnd_degree`` is ``None``.
            bkgnd_num_sigma: Optional outlier rejection for the background fit (sigma
                threshold); ``None`` disables. Only used when ``bkgnd_degree`` is not
                ``None``.
            tolerance: Passed to :func:`scipy.optimize.minimize` as ``tol`` (Powell
                stopping tolerance for both parameter and objective changes, per SciPy).
            allow_nonzero_base: If ``True``, the PSF constant ``base`` in
                :meth:`eval_rect` is a free parameter; if ``False``, ``base`` is fixed at
                zero and only amplitude scaling applies.
            use_angular_params: If ``True``, optimize offsets, scale, optional base, and
                additional parameters via angles in ``[0, pi]`` so box constraints map to
                physical ranges. If ``False``, use direct bounded parameters (offsets
                within ``search_limit``, etc.).

        Returns:
            ``None`` if the background fit fails (:meth:`background_gradient_fit` returns
            ``None``) or if the optimizer reports failure (``success`` is False).

            Otherwise ``(offset_y, offset_x, details)``: subpixel offsets within the
            subimage in pixel units (**y first, then x**), matching ``details['y']`` and
            ``details['x']``. The caller adds integer slice origins to map to full-image
            coordinates.

            ``details`` is a :class:`dict` that always includes at least:

            - ``'y'``, ``'x'``: fitted offsets (float).
            - ``'scale'``, ``'base'``: fitted PSF scale and baseline.
            - ``'subimg'``: reference to the input ``sub_img``.
            - ``'bkgnd_params'``: 1-D coefficient array from the background fit, or
              ``None`` if ``bkgnd_degree`` was ``None``.
            - ``'bkgnd_mask'``: boolean mask from background fitting, or ``None`` if no
              background fit.
            - ``'gradient'``: evaluated background surface (zeros if no background fit).
            - ``'subimg-gradient'``: ``sub_img - gradient`` (used for residuals).
            - ``'psf'``, ``'scaled_psf'``: model patch from :meth:`eval_rect` at the
              solution; identical arrays (``scaled_psf`` supports comparison to
              ``subimg-gradient`` in the outer :meth:`find_position` loop).

            Subclasses append one entry per additional PSF parameter (for example
            ``'sigma_y'`` and ``'sigma_x'`` for :class:`~psfmodel.gaussian.GaussianPSF`),
            using the internal names from ``_additional_params``.

        Note:
            Optimizer ``status``, ``message``, and the final objective value are not
            stored in ``details``; with instance ``detailed_logging`` they may appear in
            DEBUG logs. Uncertainty keys (``'x_err'``, etc.) and least-squares metadata
            described on :meth:`find_position` are not populated by the current
            implementation.
        """

        bkgnd_params = None
        bkgnd_mask = None
        gradient = np.zeros(sub_img.shape)

        if bkgnd_degree is not None:
            bkgnd_params, bkgnd_mask = PSF.background_gradient_fit(
                sub_img,
                order=bkgnd_degree,
                ignore_center=bkgnd_ignore_center,
                num_sigma=bkgnd_num_sigma,
                debug=self.detailed_logging,
                logger=self._logger,
            )
            if bkgnd_params is None:
                return None

            gradient = PSF.background_gradient(cast(tuple[int, int], sub_img.shape), bkgnd_params)

        sub_img_grad = sub_img - gradient

        # Offset Y, Offset X, Scale, AdditionalParams
        if use_angular_params:
            bounds = [(0.0, np.pi), (0.0, np.pi), (0.0, np.pi)]
            starting_guess = [np.pi / 2, np.pi / 2, np.pi / 2]
            if allow_nonzero_base:
                bounds += [(0.0, np.pi)]
                starting_guess += [np.pi / 2]
            for _ in range(len(self._additional_params)):
                bounds += [(0.0, np.pi)]
                starting_guess += [np.pi / 2]
        else:
            bounds = [
                (-search_limit[0], search_limit[0]),
                (-search_limit[1], search_limit[1]),
                (0.0, scale_limit),
            ]
            starting_guess = [0.001, 0.001, scale_limit / 2]
            if allow_nonzero_base:
                bounds += [(_FIT_PSF_BASE_BOUND_MIN, _FIT_PSF_BASE_BOUND_MAX)]
                starting_guess += [0.001]
            for a_min, a_max, _a_name in self._additional_params:
                bounds += [(a_min, a_max)]
                starting_guess.append(np.mean([a_min, a_max]))

        extra_args0 = (
            sub_img_grad,
            search_limit,
            scale_limit,
            allow_nonzero_base,
            use_angular_params,
        )
        if self._additional_params is not None and len(self._additional_params) > 0:
            extra_args = extra_args0 + tuple(self._additional_params)
        else:
            extra_args = extra_args0

        if self.detailed_logging:
            self._logger.debug('-' * 80)
            self._logger.debug('_find_position: starting_guess=%s', starting_guess)
            self._logger.debug('_find_position: bounds=%s', bounds)

        full_result = sciopt.minimize(
            self._fit_psf_func,
            starting_guess,
            args=extra_args,
            bounds=bounds,
            tol=tolerance,
            method='Powell',
            options={'maxiter': len(starting_guess) * 10000},
        )

        result = full_result.x
        success = full_result.success
        status = full_result.status
        message = full_result.message

        if not success:
            self._logger.warning('find_position: optimizer did not succeed: %s', message)
            return None

        # if ier < 1 or ier > 4:
        #     return None

        if use_angular_params:
            offset_y = search_limit[0] * np.cos(result[0]) + 0.5
            offset_x = search_limit[1] * np.cos(result[1]) + 0.5
            scale = scale_limit * (np.cos(result[2]) + 1) / 2
        else:
            offset_y = result[0] + 0.5
            offset_x = result[1] + 0.5
            scale = result[2]

        base = 0.0
        result_end = 3
        if allow_nonzero_base:
            base = result[3]
            result_end = 4

        addl_vals_dict = {}
        for i, ap in enumerate(self._additional_params):
            if use_angular_params:
                val = (ap[1] - ap[0]) / 2.0 * (np.cos(result[result_end + i]) + 1.0) + ap[0]
            else:
                val = result[result_end + i]
            addl_vals_dict[ap[2]] = val

        psf = self.eval_rect(
            cast(tuple[int, int], sub_img.shape),
            (offset_y, offset_x),
            scale=scale,
            base=base,
            **addl_vals_dict,
        )

        details = {}
        details['x'] = offset_x
        details['y'] = offset_y
        details['subimg'] = sub_img
        details['bkgnd_params'] = bkgnd_params
        details['bkgnd_mask'] = bkgnd_mask
        details['gradient'] = gradient
        details['subimg-gradient'] = sub_img_grad
        details['psf'] = psf
        details['scale'] = scale
        details['base'] = base
        details['scaled_psf'] = psf

        # if cov_x is None:
        #     details['leastsq_cov'] = None
        #     details['x_err'] = None
        #     details['y_err'] = None
        #     details['scale_err'] = None
        #     details['base_err'] = None
        #     for i, ap in enumerate(self._additional_params):
        #         details[ap[2]+'_err'] = None
        # else:
        #     # "To obtain the covariance matrix of the parameters x, cov_x must
        #     #  be multiplied by the variance of the residuals"
        #     dof = psf.shape[0]*psf.shape[1]-len(result)
        #     resid_var = np.sum(self._fit_psf_func(result, *extra_args)**2) / dof
        #     cov = cov_x * resid_var  # In angle-parameter space!! (if
        # use_angular_params)
        #     details['leastsq_cov'] = cov
        #     if use_angular_params:
        #         # Deriv of SL0 * sin(R0) is
        #         #   SL0 * cos(R0) * dR0
        #         y_err = np.abs((np.sqrt(cov[0][0]) % (np.pi*2)) *
        #                        search_limit[0] * np.cos(result[0]))
        #         x_err = np.abs((np.sqrt(cov[1][1]) % (np.pi*2)) *
        #                        search_limit[1] * np.cos(result[1]))
        #         # Deriv of SC/2 * (sin(R)+1) = SC/2 * sin(R) + SC/2 is
        #         #   SC/2 * cos(R) * dR
        #         scale_err = np.abs((np.sqrt(cov[2][2]) % (np.pi*2)) *
        #                            scale_limit/2 * np.cos(result[2]))
        #         details['x_err'] = x_err
        #         details['y_err'] = y_err
        #         details['scale_err'] = scale_err
        #         for i, ap in enumerate(self._additional_params):
        #             err = np.abs((np.sqrt(cov[i+3][i+3]) % (np.pi*2)) *
        #                          (ap[1]-ap[0])/2 * np.cos(result[i+3]))
        #             details[ap[2]+'_err'] = err
        #     else:
        #         details['x_err'] = np.sqrt(cov[0][0])
        #         details['y_err'] = np.sqrt(cov[1][1])
        #         details['scale_err'] = np.sqrt(cov[2][2])
        #         for i, ap in enumerate(self._additional_params):
        #             details[ap[2]+'_err'] = np.sqrt(cov[i+result_end][i+result_end])
        #     # Note the base is not computed using angles
        #     details['base_err'] = None
        #     if allow_nonzero_base:
        #         details['base_err'] = np.sqrt(cov[3][3])

        # details['leastsq_infodict'] = infodict
        # details['leastsq_mesg'] = mesg
        # details['leastsq_ier'] = ier

        for key in addl_vals_dict:
            details[key] = addl_vals_dict[key]

        if self.detailed_logging:
            self._logger.debug(
                '_find_position: returning offset_y=%s offset_x=%s', offset_y, offset_x
            )
            self._logger.debug(
                '_find_position: subimage masked pixels=%s', int(np.sum(ma.getmaskarray(sub_img)))
            )
            self._logger.debug('_find_position: bkgnd_params=%s', bkgnd_params)
            if bkgnd_mask is not None:
                self._logger.debug(
                    '_find_position: bkgnd_mask bad pixels=%s',
                    int(np.sum(bkgnd_mask)),
                )
            self._logger.debug('_find_position: PSF scale=%s base=%s', scale, base)
            for key in addl_vals_dict:
                self._logger.debug('_find_position: %s=%s', key, details[key])
            self._logger.debug('_find_position: optimizer message=%s status=%s', message, status)

        return offset_y, offset_x, details
