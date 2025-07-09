################################################################################
# psfmodel/__init__.py
################################################################################

from abc import ABC, abstractmethod
from typing import Any, Optional, cast

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


class PSF(ABC):
    """Abstract superclass for classes that model different types of PSFs."""

    def __init__(self,
                 **kwargs: Any) -> None:
        """Create a PSF object. Only called by subclasses."""

        self._debug_opt = 0
        self._additional_params: list[Any] = []

    @abstractmethod
    def eval_point(self,
                   coord: (tuple[float | npt.NDArray[np.floating],
                                 float | npt.NDArray[np.floating]] |
                           npt.NDArray[np.floating]),
                   *,
                   scale: float = 1.,
                   base: float = 0.) -> float | npt.NDArray[np.floating]:
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
    def eval_rect(self,
                  rect_size: list[int] | tuple[int, int],
                  offset: list[float] | tuple[float, float] = (0.5, 0.5),
                  *,
                  movement: Optional[tuple[float, float]] = None,
                  movement_granularity: float = 0.1,
                  scale: float = 1.,
                  base: float = 0.,
                  **kwargs: Any) -> npt.NDArray[np.float64]:
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
    def _eval_rect(self,
                   rect_size: tuple[int, int],
                   offset: tuple[float, float] = (0.5, 0.5),
                   *,
                   scale: float = 1.,
                   base: float = 0.) -> npt.NDArray[np.float64]:
        """Internal function to create a rectangular pixelated PSF without other checks.
        """
        ...  # pragma: no cover

    def _eval_rect_smeared(self,
                           rect_size: tuple[int, int],
                           offset: tuple[float, float] = (0.5, 0.5),
                           *,
                           movement: Optional[tuple[float, float]] = None,
                           movement_granularity: float = 0.1,
                           scale: float = 1.,
                           base: float = 0.,
                           **kwargs: Any) -> npt.NDArray[np.floating]:
        """Evaluate and sum a PSF multiple times to simulate motion blur.

        Parameters:
            movement: The total amount (my, mx) the PSF moves. The movement is assumed to
                be centered on the given offset and exists half on either side.
            movement_granularity: The number of pixels to step for each smear while doing
                motion blur.
            rect_size: The size of the rectangle (rect_size_y, rect_size_x) of the
                returned PSF. Both dimensions must be odd.
            offset: The amount (offset_y, offset_x) to offset the center of the PSF. A
                positive offset effectively moves the PSF down and to the left. XXX
            scale: A scale factor to apply to the resulting PSF.
            base: A scalar added to the resulting PSF.

        Other inputs may be available for specific subclasses.
        """

        if movement is None or (movement[0] == 0 and movement[1] == 0):
            return self._eval_rect(rect_size, offset=offset,
                                   scale=scale, base=base, **kwargs)

        num_steps = int(max(abs(movement[0]) / movement_granularity,
                            abs(movement[1]) / movement_granularity))

        if num_steps == 0:
            step_y = 0.
            step_x = 0.
        else:
            step_y = movement[0] / num_steps
            step_x = movement[1] / num_steps

        total_rect = None

        for step in range(num_steps+1):
            y = offset[0] + step_y*(step - num_steps/2.)
            x = offset[1] + step_x*(step - num_steps/2.)

            rect = self._eval_rect(rect_size, offset=(y, x),
                                   scale=scale, base=base, **kwargs)
            if total_rect is None:
                total_rect = rect
            else:
                total_rect += rect
        assert total_rect is not None

        total_rect /= float(num_steps+1)

        return total_rect

    #==========================================================================
    #
    # Static functions for creating background gradients
    #
    #==========================================================================

    @staticmethod
    def _background_gradient_coeffs(shape: tuple[int, int],
                                    order: int) -> npt.NDArray[np.float64]:
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
            raise ValueError(
                f'Image must have odd positive shape in each dimension, got {shape}')
        if order < 0:
            raise ValueError(f'Order must be non-negative, got {order}')

        # Create arrays of indexes for line and sample with (0, 0) at the center of the
        # image
        y_values = np.arange(shape[0])[:, np.newaxis] - int(shape[0] / 2)
        x_values = np.arange(shape[1])[np.newaxis, :] - int(shape[1] / 2)

        y_powers: list[float | npt.NDArray[np.floating]] = [1.]
        x_powers: list[float | npt.NDArray[np.floating]] = [1.]

        nparams = int((order+1) * (order+2) / 2)
        a3d = np.empty((shape[0], shape[1], nparams))
        a3d[:, :, 0] = 1.  # This is the constant term of the polynomial

        k = 0  # Parameter number
        for p in range(1, order+1):
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
            for q in range(p+1):
                k += 1
                a3d[:, :, k] = y_powers[q] * x_powers[p-q]

        return a3d

    @staticmethod
    def background_gradient_fit(image: npt.NDArray[np.floating],
                                order: int = 2,
                                ignore_center: Optional[int | tuple[int, int]] = None,
                                num_sigma: Optional[float] = None,
                                debug: bool = False
                                ) -> tuple[npt.NDArray[np.float64] | None,
                                           npt.NDArray[np.float64] | None]:
        """Return the polynomial fit to the pixels of an image.

        Parameters:
            image: 2D array to fit; must have odd shape in each dimension.
            order: Order of the polynomial.
            ignore_center: A scalar or tuple (ignore_y, ignore_x) giving the number of
                pixels on either side of the center to ignore while fitting. 0 means
                ignore the center pixel. None means don't ignore anything.
            num_sigma: The number of sigma a pixel needs to be beyond the background
                gradient to be ignored. None means don't ignore bad pixels.
            debug: Set to debug bad pixel removal.

        Returns:
            A tuple of the background coefficient array and the mask of ignored pixels.
        """

        if len(image.shape) != 2:
            raise ValueError('Image must be 2-D, got {image.shape}')
        if (image.shape[0] < 0 or image.shape[1] < 0 or
            image.shape[0] % 2 != 1 or image.shape[1] % 2 != 1):
            raise ValueError(
                'Image must have odd positive shape in each dimension, got '
                f'{image.shape}')
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
            image.mask = cast(npt.NDArray[np.bool_],
                              ma.getmaskarray(image))  # type: ignore
            is_masked = True

        if ignore_center is not None:
            if isinstance(ignore_center, int):
                ignore_y = ignore_center
                ignore_x = ignore_center
            else:
                ignore_y, ignore_x = ignore_center
            if ignore_y*2+1 >= shape[0] or ignore_x*2+1 >= shape[1]:
                if debug:  # pragma: no cover
                    print('BKGND CENTER IGNORED IS ENTIRE IMAGE')  # XXX
                return None, None
            ctr_y = shape[0] // 2
            ctr_x = shape[1] // 2
            image[ctr_y-ignore_y:ctr_y+ignore_y+1,
                  ctr_x-ignore_x:ctr_x+ignore_x+1] = ma.masked

        nparams = int((order+1) * (order+2) // 2)

        a3d = PSF._background_gradient_coeffs(shape, order)

        if num_sigma is not None:
            num_bad_pixels = cast(int, ma.count_masked(image))  # type: ignore
            if debug:  # pragma: no cover
                print('BKGND GRAD INIT # BAD', num_bad_pixels)

        while True:
            # Reshape properly for linalg.lstsq
            a2d = a3d.reshape((image.size, nparams))
            b1d = image.flatten()

            if is_masked:
                # linalg doesn't support masked arrays!
                a2d = a2d[~b1d.mask]  # type: ignore
                b1d = ma.compressed(b1d)  # type: ignore

            if a2d.shape[0] < a2d.shape[1]:  # Underconstrained
                if debug:  # pragma: no cover
                    print('BKGND UNDERCONSTRAINED', a2d.shape)
                return None, None

            coeffts = linalg.lstsq(a2d, b1d)[0]

            if num_sigma is None:
                break

            # TODO - BITO suggests:
            # worst_sigma = np.max(np.abs(delta_img))
            # if worst_sigma >= sigma*num_sigma:
            #     image[np.abs(delta_img) >= sigma*num_sigma] = ma.masked
            gradient = PSF.background_gradient(shape, coeffts)
            delta_img = image - gradient
            sigma = np.std(delta_img)
            worst_sigma = np.max(np.abs(delta_img))
            if worst_sigma >= sigma*num_sigma:
                image[np.abs(delta_img) >= worst_sigma] = ma.masked

            new_num_bad_pixels = cast(int, ma.count_masked(image))  # type: ignore
            if debug:  # pragma: no cover
                print('BKGD GRAD NEW # BAD', new_num_bad_pixels)
            if new_num_bad_pixels == num_bad_pixels:
                break
            num_bad_pixels = new_num_bad_pixels

        if is_masked:
            return coeffts, ma.getmaskarray(image)  # type: ignore
        else:
            return coeffts, np.zeros(shape, dtype=np.bool_)

    @staticmethod
    def background_gradient(rect_size: tuple[int, int],
                            bkgnd_params: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Create a background gradient.

        Parameters:
            size: A tuple (size_y, size_x) indicating the size of the returned array.
            bkgnd_params: A tuple indicating the coefficients of the background
                polynomial. The order of the polynomial is inferred from the number of
                elements in the tuple.
        """

        bkgnd_params = np.array(bkgnd_params)

        order = int(np.sqrt(len(bkgnd_params)*2))-1

        a3d = PSF._background_gradient_coeffs(rect_size, order)
        result = np.sum(bkgnd_params * a3d, axis=-1)

        return cast(npt.NDArray[np.float64], result)

    #==========================================================================
    #
    # Functions for finding astrometric positions
    #
    #==========================================================================

    def find_position(self,
                      image: npt.NDArray[np.floating],
                      box_size: tuple[int, int],
                      starting_point: tuple[float, float],
                      *,
                      search_limit: float | tuple[float, float] = (1.5, 1.5),
                      bkgnd_degree: int | None = 2,
                      bkgnd_ignore_center: tuple[int, int] = (2, 2),
                      bkgnd_num_sigma: Optional[float] = None,
                      tolerance: float = 1e-6,
                      num_sigma: Optional[float] = None,
                      max_bad_frac: float = 0.2,
                      allow_nonzero_base: bool = False,
                      scale_limit: float = 1000.,
                      use_angular_params: bool = True
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
                'psf'                  The unit-scale and zero-base PSF.
                'scaled_psf'           The fully scaled PSF with the base added.
                'leastsq_cov'          The covariance matrix returned by leastsq
                                       as adjusted by the residual variance.
                'leastsq_infodict'     The infodict returned by leastsq.
                'leastsq_mesg'         The mesg returned by leastsq.
                'leastsq_ier'          The ier returned by leastsq.

            In addition, metadata includes two entries for each "additional
            parameter" used during optimization: one for the value and one for
            the uncertainty ('param' and 'param_err').
        """

        if (box_size[0] < 0 or box_size[1] < 0 or
            box_size[0] % 2 != 1 or box_size[1] % 2 != 1):
            raise ValueError(
                'box_size must have odd positive shape in each dimension, '
                f'got {box_size}')

        half_box_size_y = box_size[0] // 2
        half_box_size_x = box_size[1] // 2

        starting_pix = (int(np.round(starting_point[0])),
                        int(np.round(starting_point[1])))

        if self._debug_opt:
            print('>> Entering psfmodel:find_position')
            print('Image is masked', isinstance(image, ma.MaskedArray))
            print('Image num masked', np.sum(ma.getmaskarray(image)))  # type: ignore
            print('Image min, max, mean', np.min(image), np.max(image), np.mean(image))
            print('Box size', box_size)
            print('Starting point', starting_point)
            print('Search limit', search_limit)
            print('Bkgnd degree', bkgnd_degree)
            print('Bkgnd ignore center', bkgnd_ignore_center)
            print('Bkgnd num sigma', bkgnd_num_sigma)
            print('Tolerance', tolerance)
            print('Num sigma', num_sigma)
            print('Max bad frac', max_bad_frac)
            print('Allow nonzero base', allow_nonzero_base)
            print('Scale limit', scale_limit)
            print('Use angular params', use_angular_params)
            print('-----')

        # Too close to the edge means we can't search
        if (starting_pix[0] - half_box_size_y < 0 or
            starting_pix[0] + half_box_size_y >= image.shape[0] or
            starting_pix[1] - half_box_size_x < 0 or
            starting_pix[1] + half_box_size_x >= image.shape[1]):
            if self._debug_opt:
                print('Too close to the edge - search impossible')
            return None

        sub_img = image[starting_pix[0] - half_box_size_y:
                        starting_pix[0] + half_box_size_y+1,
                        starting_pix[1] - half_box_size_x:
                        starting_pix[1] + half_box_size_x+1]

        if self._debug_opt:
            print('Sub img min, max, mean', np.min(sub_img), np.max(sub_img),
                  np.mean(sub_img))

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
            if self._debug_opt > 1:
                print('MAIN LOOP: FIND POS, # BAD PIXELS', num_bad_pixels)
            ret = self._find_position(sub_img,
                                      search_limit, scale_limit,
                                      bkgnd_degree, bkgnd_ignore_center,
                                      bkgnd_num_sigma, tolerance,
                                      allow_nonzero_base, use_angular_params)
            if ret is None:
                if self._debug_opt:
                    print('find_position returned None')
                return None

            res_y, res_x, details = ret

            if not num_sigma:
                break

            resid = np.sqrt((details['subimg-gradient'] - details['scaled_psf'])**2)
            resid_std = np.std(resid)

            if self._debug_opt > 1:
                print('MAIN LOOP: Resid', resid)
                print('resid_std', resid_std)

            if num_sigma is not None:
                sub_img[np.where(resid > num_sigma*resid_std)] = ma.masked

            new_num_bad_pixels = ma.count_masked(sub_img)  # type: ignore
            if new_num_bad_pixels == num_bad_pixels:
                break
            if new_num_bad_pixels == sub_img.size:
                if self._debug_opt:
                    print('MAIN LOOP: All pixels masked - find_position returning None')
                return None  # All masked
            if new_num_bad_pixels > max_bad_frac*sub_img.size:
                if self._debug_opt:
                    print('MAIN LOOP: Too many pixels masked - '
                          'find_position returning None')
                return None  # Too many masked
            num_bad_pixels = new_num_bad_pixels

        if self._debug_opt:
            msg = f'find_position returning Y {res_y+starting_pix[0]:.4f}'
            # if details['y_err'] is not None:
            #     msg += f' +/- {details["y_err"]:.4f}'
            msg += f' X {res_x+starting_pix[1]:.4f}'
            # if details['x_err'] is not None:
            #     msg += ' +/- {details["x_err"]:.4f}'
            if details['scale'] is not None:
                msg += f' Scale {details["scale"]:.4f} Base {details["base"]:.4f}'
            msg += f' SY {details["sigma_y"]:.4f} SX {details["sigma_x"]:.4f}'
            print(msg)

        return res_y + starting_pix[0], res_x + starting_pix[1], details

    def _fit_psf_func(self,
                      params: tuple[float, ...],
                      sub_img: npt.NDArray[np.floating],
                      search_limit: tuple[float, float],
                      scale_limit: float,
                      allow_nonzero_base: bool,
                      use_angular_params: bool,
                      *additional_params: Any) -> float:

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
            #     if self._debug_opt > 1:
            #         full_resid = np.sqrt(np.sum(fake_return**2))
            #         print('RESID', full_resid)
            #     return fake_return

        base = 0.
        param_end = 3
        if allow_nonzero_base:
            base = params[3]
            param_end = 4

        addl_vals_dict = {}
        for i, ap in enumerate(additional_params):
            if use_angular_params:
                val = ((ap[1] - ap[0]) / 2. *
                       (np.cos(params[param_end+i])+1.) + ap[0])
            else:
                val = params[param_end+i]
            addl_vals_dict[ap[2]] = val

        psf = self.eval_rect(cast(tuple[int, int], sub_img.shape),
                             (offset_y, offset_x),
                             scale=scale, base=base, **addl_vals_dict)

        resid = (sub_img - psf).flatten()

        full_resid = cast(float, np.sqrt(np.sum(resid**2)))

        if self._debug_opt > 1:
            msg = f'OFFY {offset_y:8.5f} OFFX {offset_x:8.5f} SCALE {scale:9.5f} '
            msg += f'BASE {base:9.5f}'
            for ap in additional_params:
                msg += f' {ap[2].upper()} {addl_vals_dict[ap[2]]:8.5f}'
            msg += f' RESID {full_resid:f}'
            print(msg)

        return full_resid

    def _find_position(self,
                       sub_img: npt.NDArray[np.floating],
                       search_limit: tuple[float, float],
                       scale_limit: float,
                       bkgnd_degree: int | None,
                       bkgnd_ignore_center: tuple[int, int],
                       bkgnd_num_sigma: float | None,
                       tolerance: float,
                       allow_nonzero_base: bool,
                       use_angular_params: bool
                       ) -> None | tuple[float, float, dict[str, Any]]:

        bkgnd_params = None
        bkgnd_mask = None
        gradient = np.zeros(sub_img.shape)

        if bkgnd_degree is not None:
            bkgnd_params, bkgnd_mask = PSF.background_gradient_fit(
                                           sub_img,
                                           order=bkgnd_degree,
                                           ignore_center=bkgnd_ignore_center,
                                           num_sigma=bkgnd_num_sigma,
                                           debug=self._debug_opt > 2)
            if bkgnd_params is None:
                return None

            gradient = PSF.background_gradient(cast(tuple[int, int], sub_img.shape),
                                               bkgnd_params)

        sub_img_grad = sub_img - gradient

        # Offset Y, Offset X, Scale, AdditionalParams
        if use_angular_params:
            bounds = [(0., np.pi),
                      (0., np.pi),
                      (0., np.pi)]
            starting_guess = [np.pi/2, np.pi/2, np.pi/2]
            if allow_nonzero_base:
                bounds += [(0., np.pi)]
                starting_guess += [np.pi/2]
            for _ in range(len(self._additional_params)):
                bounds += [(0., np.pi)]
                starting_guess += [np.pi/2]
        else:
            bounds = [(-search_limit[0], search_limit[0]),
                      (-search_limit[1], search_limit[1]),
                      (0., scale_limit)]
            starting_guess = [0.001, 0.001, scale_limit/2]
            if allow_nonzero_base:
                bounds += [(-1e38, 1e38)]
                starting_guess += [0.001]
            for a_min, a_max, a_name in self._additional_params:
                bounds += [(a_min, a_max)]
                starting_guess = starting_guess + [np.mean([a_min, a_max])]

        extra_args0 = (sub_img_grad, search_limit, scale_limit,
                       allow_nonzero_base, use_angular_params)
        if (self._additional_params is not None and
            len(self._additional_params) > 0):
            extra_args = extra_args0 + tuple(self._additional_params)
        else:
            extra_args = extra_args0 + tuple([])

        if self._debug_opt > 3:
            print('-' * 80)
            print(f'STARTING GUESS: {starting_guess}')
            print(f'BOUNDS: {bounds}')

        full_result = sciopt.minimize(self._fit_psf_func,
                                      starting_guess,
                                      args=extra_args,
                                      bounds=bounds,
                                      tol=tolerance,
                                      method='Powell',
                                      options={'maxiter': len(starting_guess) * 10000})

        result = full_result.x
        success = full_result.success
        status = full_result.status
        message = full_result.message

        if not success:
            print('FAIL', message)
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

        base = 0.
        result_end = 3
        if allow_nonzero_base:
            base = result[3]
            result_end = 4

        addl_vals_dict = {}
        for i, ap in enumerate(self._additional_params):
            if use_angular_params:
                val = ((ap[1] - ap[0]) / 2. *
                       (np.cos(result[result_end+i])+1.) + ap[0])
            else:
                val = result[result_end+i]
            addl_vals_dict[ap[2]] = val

        psf = self.eval_rect(cast(tuple[int, int], sub_img.shape), (offset_y, offset_x),
                             scale=scale, base=base, **addl_vals_dict)

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
        details['scaled_psf'] = psf*scale+base

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

        if self._debug_opt > 1:
            print('_find_position RETURNING', offset_y, offset_x)
            print('Subimg num bad pixels',
                  np.sum(ma.getmaskarray(sub_img)))  # type: ignore
            print('Bkgnd params', bkgnd_params)
            print('Bkgnd mask bad pixels',
                  np.sum(ma.getmaskarray(bkgnd_mask)))  # type: ignore
            print('PSF scale', scale)
            print('PSF base', base)
            for key in addl_vals_dict:
                print(key, details[key])
            # print('LEASTSQ COV')
            # cov = details['leastsq_cov']
            # print(cov)
            # if cov is not None:
            #     print('X_ERR', details['x_err'])
            #     print('Y_ERR', details['y_err'])
            #     print('SCALE_ERR', details['scale_err'])
            #     print('BASE_ERR', details['base_err'])
            #     for key in addl_vals_dict:
            #         print(key+'_err', details[key+'_err'])
            print('MESSAGE', message)
            print('STATUS', status)
            print('-----')

        return offset_y, offset_x, details
