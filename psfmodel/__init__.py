################################################################################
# psfmodel/__init__.py
################################################################################

import numpy as np
import numpy.ma as ma
import scipy.linalg as linalg
import scipy.optimize as sciopt

try:
    from _version import __version__
except ImportError as err:
    __version__ = 'Version unspecified'


class PSF(object):
    """Abstract superclass for classes that model different types of PSFs."""

    def __init__(self, movement, movement_granularity):
        """Create a PSF object. Only called by subclasses.

        Input:
            movement          Motion blur in the (Y,X) direction. Must be a
                              tuple of scalars or None.
            movement_granularity
                              The number of pixels to step for each smear doing
                              motion blur.
        """

        self.debug_opt = 0

        if movement is None:
            movement = (0., 0.)
        self.movement = movement
        self.movement_granularity = movement_granularity

        self.additional_params = []

    def eval_point(self, coord, scale=1., base=0., **kwargs):
        """Evaluate the PSF at a single, fractional, point.

        (0,0) is the center of the PSF and x and y may be negative.

        Input:
            coord       The coordinate (y,x) at which to evaluate the PSF.

            scale       A scale factor to apply to the resulting PSF.
            base        A scalar added to the resulting PSF.

        Other inputs may be available for specific subclasses.
        """
        assert False

    def eval_pixel(self, coord, offset=(0.,0.), scale=1., base=0., **kwargs):
        """Evaluate the PSF integrated over an entire integer pixel.

        The returned array has the PSF offset from the center by
        (offset_y,offset_x). An offset of (0,0) places the PSF in the upper
        left corner of the center pixel while an offset of (0.5,0.5)
        places the PSF in the center of the center pixel.

        Input:
            coord       The integer coordinate (y,x) at which to evaluate the
                        PSF.
            offset      The amount (offset_y,offset_x) to offset the center
                        of the PSF.

            scale       A scale factor to apply to the resulting PSF.
            base        A scalar added to the resulting PSF.

        Other inputs may be available for specific subclasses.
        """
        assert False

    def eval_rect(self, rect_size, offset=(0.,0.), scale=1., base=0.,
                  **kwargs):
        """Create a rectangular pixelated PSF.

        This is done by evaluating the PSF function from:
            [-rect_size_y//2:rect_size_y//2] and
            [-rect_size_x//2:rect_size_x//2]

        The returned array has the PSF offset from the center by
        (offset_y,offset_x). An offset of (0,0) places the PSF in the upper
        left corner of the center pixel while an offset of (0.5,0.5)
        places the PSF in the center of the center pixel.

        Input:
            rect_size   The size of the rectangle (rect_size_y,rect_size_x)
                        of the returned PSF. Both dimensions must be odd.
            offset      The amount (offset_y,offset_x) to offset the center
                        of the PSF from the upper left corner of the center
                        pixel of the rectangle.

            scale       A scale factor to apply to the resulting PSF.
            base        A scalar added to the resulting PSF.

        Other inputs may be available for specific subclasses.
        """
        assert False

    def _eval_rect_smeared(self, movement, rect_size, offset=(0.,0.), *args,
                           **kwargs):
        """Evaluate and sum a PSF multiple times to simulate motion blur.

        Input:
            movement    The total amount (my,mx) the PSF moves. The movement
                        is assumed to be centered on the given offset and
                        exists half on either side.
            rect_size   The size of the rectangle (rect_size_y,rect_size_x)
                        of the returned PSF. Both dimensions must be odd.

            offset      The amount (offset_y,offset_x) to offset the center
                        of the PSF. A positive offset effectively moves the
                        PSF down and to the left.
        """

        num_steps = int(max(abs(movement[0])/self.movement_granularity,
                            abs(movement[1])/self.movement_granularity))

        if num_steps == 0:
            step_y = 0.
            step_x = 0.
        else:
            step_y = movement[0] / num_steps
            step_x = movement[1] / num_steps

        total_rect = None

        for step in range(num_steps+1):
            y = offset[0] + step_y*(step-num_steps/2.)
            x = offset[1] + step_x*(step-num_steps/2.)

            rect = self._eval_rect(rect_size, (y, x), *args, **kwargs)
            if total_rect is None:
                total_rect = rect
            else:
                total_rect += rect

        total_rect /= float(num_steps+1)

        return total_rect


    #==========================================================================
    #
    # Static functions for creating background gradients
    #
    #==========================================================================

    @staticmethod
    def _background_gradient_coeffs(shape, order):
        """Internal routine for creating the coefficient matrix."""

        lvalues = np.arange(shape[0])[:,np.newaxis]-int(shape[0]/2)
        svalues = np.arange(shape[1])[np.newaxis,:]-int(shape[1]/2)

        lpowers = [1.]
        spowers = [1.]

        nparams = int((order+1) * (order+2) / 2)
        a3d = np.empty((shape[0], shape[1], nparams))
        a3d[:,:,0] = 1.

        k = 0
        for p in range(1,order+1):
            lpowers.append(lpowers[-1] * lvalues)
            spowers.append(spowers[-1] * svalues)

            for q in range(p+1):
                k += 1
                a3d[:,:,k] = lpowers[q] * spowers[p-q]

        return a3d

    @staticmethod
    def background_gradient_fit(image, order=2, ignore_center=None,
                                num_sigma=None, debug=0):
        """Return the polynomial fit to the pixels of an image.

        Input:
            image           2D array to fit; must have odd shape in each
                            dimension.
            order           Order of the polynomial; default=2 (quadratic).
            ignore_center   A scalar or tuple (ignore_y,ignore_x) giving the
                            number of pixels on either side of the center to
                            ignore while fitting. 0 means ignore the center
                            pixel. None means don't ignore anything.
            num_sigma       The number of sigma a pixel needs to be beyond the
                            background gradient to be ignored. None means don't
                            ignore bad pixels.
            debug           Set to debug bad pixel removal.
        """

        assert image.shape[0] % 2 == 1 # Image shape must be odd
        assert image.shape[1] % 2 == 1

        is_masked = False

        if ignore_center or num_sigma:
            if isinstance(image, ma.MaskedArray):
                # We're going to change the mask so make a copy first
                image = image.copy()
            else:
                image = image.view(ma.MaskedArray)

        if isinstance(image, ma.MaskedArray):
            image.mask = ma.getmaskarray(image)
            is_masked = True

        if ignore_center:
            ctr_y = image.shape[0] // 2
            ctr_x = image.shape[1] // 2
            if np.shape(ignore_center) == ():
                ignore_y = ignore_center
                ignore_x = ignore_center
            else:
                ignore_y, ignore_x = ignore_center
            if (ignore_y*2+1 >= image.shape[0] or
                ignore_x*2+1 >= image.shape[1]):
                if debug:
                    print('BKGND CENTER IGNORED IS ENTIRE IMAGE')
                return None, None
            image[ctr_y-ignore_y:ctr_y+ignore_y+1,
                  ctr_x-ignore_x:ctr_x+ignore_x+1] = ma.masked

        nparams = int((order+1) * (order+2) // 2)

        a3d = PSF._background_gradient_coeffs(image.shape, order)

        if num_sigma:
            num_bad_pixels = ma.count_masked(image)
            if debug:
                print('BKGND GRAD INIT # BAD', num_bad_pixels)

        while True:
            # Reshape properly for linalg.lstsq
            a2d  = a3d.reshape((image.size, nparams))
            b1d  = image.flatten()

            if is_masked:
                a2d = a2d[~b1d.mask] # linalg doesn't support masked arrays!
                b1d = ma.compressed(b1d)

            if a2d.shape[0] < a2d.shape[1]: # Underconstrained
                if debug:
                    print('BKGND UNDERCONSTRAINED', a2d.shape)
                return None, None

            coeffts = linalg.lstsq(a2d, b1d)[0]

            if not num_sigma:
                break

            gradient = PSF.background_gradient(image.shape, coeffts)
            new_img = image-gradient
            sigma = np.std(new_img)
            image[np.where(np.abs(new_img) > sigma*num_sigma)] = ma.masked

            new_num_bad_pixels = ma.count_masked(image)
            if debug:
                print('BKGD GRAD NEW # BAD', new_num_bad_pixels)
            if new_num_bad_pixels == num_bad_pixels:
                break
            num_bad_pixels = new_num_bad_pixels

        if is_masked:
            return coeffts, ma.getmaskarray(image)
        else:
            return coeffts, np.zeros(image.shape)

    @staticmethod
    def background_gradient(size, bkgnd_params):
        """Create a background gradient.

        Input:
            size            A scalar (meaning a square) or a tuple
                            (size_y, size_x) indicating the size of the
                            returned array.
            bkgnd_params    A tuple indicating the coefficients of the
                            background polynomial. The order of the polynomial
                            is inferred from the number of elements in the
                            tuple.
        """

        if np.shape(size) == ():
            size_y = size
            size_x = size
        else:
            size_y, size_x = size

        bkgnd_params = np.array(bkgnd_params)

        order = int(np.sqrt(len(bkgnd_params)*2))-1

        a3d = PSF._background_gradient_coeffs(size, order)
        result = np.sum(bkgnd_params * a3d, axis=-1)

        return result

    #==========================================================================
    #
    # Functions for finding astrometric positions
    #
    #==========================================================================

    def find_position(self, image, box_size,
                      starting_point, search_limit=(1.5, 1.5),
                      bkgnd_degree=2, bkgnd_ignore_center=(2,2),
                      bkgnd_num_sigma=5,
                      tolerance=1e-8, num_sigma=10,
                      max_bad_frac=0.2,
                      allow_nonzero_base=False,
                      scale_limit=1000000.,
                      use_angular_params=True):
        """Find the (y,x) coordinates that best fit a 2-D Gaussian to an image.

        Input:
            image                The image (arbitrary 2-D size, floating
                                 point).

            box_size             A tuple (box_y,box_x) indicating the size of
                                 the PSF to use. This governs both the size of
                                 the PSF created at each step as well as the
                                 size of the subimage looked at. Both box_y
                                 and box_x must be odd.

            starting_point       A tuple (y,x) indicating the best guess for
                                 where the object can be found. Searching is
                                 limited to a region around this point
                                 controlled by "search_limit".

            search_limit         A scalar or tuple (y_limit,x_limit)
                                 specifying the maximum distance to search from
                                 "starting_point". If a scalar, both x_limit
                                 and y_limit are the same.

            bkgnd_degree         The degree (order) of the background gradient
                                 polynomial. None means no background gradient
                                 is fit.

            bkgnd_ignore_center  A tuple (y,x) giving the number of pixels on
                                 each side of the center point to ignore when
                                 fitting the background. The ignored region is
                                 thus n*2+1 on each side.

            bkgnd_num_sigma      The number of sigma a pixel needs to be beyond
                                 the background gradient to be ignored. None
                                 means don't ignore bad pixels while computing
                                 the background gradient.

            tolerance            The tolerance (both X and Function) in the
                                 Powell optimization algorithm.

            num_sigma            The number of sigma for a pixel to be
                                 considered bad during PSF fitting. None means
                                 don't ignore bad pixels while fitting the
                                 PSF.

            max_bad_frac         The maximum allowable number of pixels masked
                                 during PSF fitting. If more pixels than this
                                 fraction are masked, the position fit fails.

            allow_nonzero_base   If True, allow the base of the PSF (constant
                                 bias) to vary. Otherwise the base of the PSF
                                 is always at zero and can only scale in the
                                 positive direction.

            scale_limit          The maximum PSF scale allowed.

            use_angular_params   Use angles to optimize parameter values.

        Returns None if no fit found.

        Otherwise returns pos_y, pos_x, metadata. Metadata is a dictionary
        containing:

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

        assert box_size[0] % 2 == 1 # PSF box size must be odd
        assert box_size[1] % 2 == 1

        half_box_size_y = box_size[0] // 2
        half_box_size_x = box_size[1] // 2

        starting_pix = (int(np.round(starting_point[0])),
                        int(np.round(starting_point[1])))

        if self.debug_opt:
            print('>> Entering psfmodel:find_position')
            print('Image is masked', isinstance(image, ma.MaskedArray))
            print('Image num masked', np.sum(ma.getmaskarray(image)))
            print('Image min, max, mean', np.min(image), np.max(image), end=' ')
            print(np.mean(image))
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
        if (starting_pix[0]-half_box_size_y < 0 or
            starting_pix[0]+half_box_size_y >= image.shape[0] or
            starting_pix[1]-half_box_size_x < 0 or
            starting_pix[1]+half_box_size_x >= image.shape[1]):
            if self.debug_opt:
                print('Too close to the edge - search impossible')
            return None

        sub_img = image[starting_pix[0]-half_box_size_y:
                        starting_pix[0]+half_box_size_y+1,
                        starting_pix[1]-half_box_size_x:
                        starting_pix[1]+half_box_size_x+1]

        if self.debug_opt:
            print('Sub img min, max, mean', np.min(sub_img), end=' ')
            print(np.max(sub_img), np.mean(sub_img))

        if np.shape(search_limit) == ():
            search_limit = (float(search_limit), float(search_limit))

        if num_sigma:
            if isinstance(sub_img, ma.MaskedArray):
                # We're going to change the mask so make a copy first
                sub_img = sub_img.copy()
            else:
                sub_img = sub_img.view(ma.MaskedArray)

        num_bad_pixels = 0

        while True:
            if self.debug_opt > 1:
                print('MAIN LOOP: FIND POS # BAD PIXELS', num_bad_pixels)
            ret = self._find_position(sub_img,
                                      half_box_size_y, half_box_size_x,
                                      search_limit, scale_limit,
                                      bkgnd_degree, bkgnd_ignore_center,
                                      bkgnd_num_sigma, tolerance,
                                      allow_nonzero_base, use_angular_params)
            if ret is None:
                if self.debug_opt:
                    print('find_position returning None')
                return None

            res_y, res_x, details = ret

            if not num_sigma:
                break

            resid = np.sqrt((details['subimg-gradient']-
                             details['scaled_psf'])**2)
            resid_std = np.std(resid)

            if self.debug_opt > 1:
                print('MAIN LOOP: Resid', resid)
                print('resid_std', resid_std)

            if num_sigma is not None:
                sub_img[np.where(resid > num_sigma*resid_std)] = ma.masked

            new_num_bad_pixels = ma.count_masked(sub_img)
            if new_num_bad_pixels == num_bad_pixels:
                break
            if new_num_bad_pixels == sub_img.size:
                if self.debug_opt:
                    print('MAIN LOOP: All pixels masked - find_position returning None')
                return None # All masked
            if new_num_bad_pixels > max_bad_frac*sub_img.size:
                if self.debug_opt:
                    print('MAIN LOOP: Too many pixels masked - find_position returning None')
                return None # Too many masked
            num_bad_pixels = new_num_bad_pixels

        if self.debug_opt:
            print('find_position returning', end=' ')
            print('Y %.4f' % (res_y+starting_pix[0]), end=' ')
            if details['y_err'] is not None:
                print('+/- %.4f' % details['y_err'], end=' ')
            print('X %.4f' % (res_x+starting_pix[1]), end=' ')
            if details['x_err'] is not None:
                print('+/- %.4f' % details['x_err'], end=' ')
            print()
            if details['scale'] is not None:
                print('                        Scale %.4f Base %.4f' % (
                        details['scale'], details['base']))

        return res_y+starting_pix[0], res_x+starting_pix[1], details

    def _fit_psf_func(self, params, sub_img, search_limit, scale_limit,
                      allow_nonzero_base, use_angular_params,
                      *additional_params):
        if self.debug_opt > 1:
            print(params)
        # Make an offset of "0" be the center of the pixel (0.5,0.5)
        if use_angular_params:
            offset_y = search_limit[0] * np.sin(params[0]) + 0.5
            offset_x = search_limit[1] * np.sin(params[1]) + 0.5
            scale = scale_limit * (np.sin(params[2]) + 1) / 2
        else:
            offset_y = params[0] + 0.5
            offset_x = params[1] + 0.5
            scale = params[2]
            fake_resid = None
            if not -search_limit[0] <= params[0] <= search_limit[0]:
                fake_resid = abs(params[0]) * 1e10
            elif not -search_limit[1] <= params[1] <= search_limit[1]:
                fake_resid = abs(params[1]) * 1e10
            elif not 0.00001 <= scale <= scale_limit:
                fake_resid = abs(scale) * 1e10
            if fake_resid is not None:
                fake_return = np.zeros(sub_img.shape).flatten()
                fake_return[:] = fake_resid
                if self.debug_opt > 1:
                    full_resid = np.sqrt(np.sum(fake_return**2))
                    print('RESID', full_resid)
                return fake_return

        base = 0.
        param_end = 3
        if allow_nonzero_base:
            base = params[3]
            param_end = 4

        addl_vals_dict = {}
        for i, ap in enumerate(additional_params):
            if use_angular_params:
                val = ((ap[1] - ap[0]) / 2. *
                       (np.sin(params[param_end+i])+1.) + ap[0])
            else:
                val = params[param_end+i]
            addl_vals_dict[ap[2]] = val

        if self.debug_opt > 1:
            print(('OFFY %8.5f OFFX %8.5f SCALE %9.5f BASE %9.5f' %
                   (offset_y, offset_x, scale, base)), end=' ')
            for ap in additional_params:
                print('%s %8.5f' % (ap[2].upper(), addl_vals_dict[ap[2]]), end=' ')
            print()

        psf = self.eval_rect(sub_img.shape, (offset_y, offset_x),
                             scale=scale, base=base, **addl_vals_dict)

        resid = (sub_img-psf).flatten()

        if self.debug_opt > 1:
            full_resid = np.sqrt(np.sum(resid**2))
            print('RESID', full_resid)

        return resid

    def _find_position(self, sub_img, half_box_size_y, half_box_size_x,
                       search_limit, scale_limit, bkgnd_degree,
                       bkgnd_ignore_center, bkgnd_num_sigma, tolerance,
                       allow_nonzero_base, use_angular_params):

        bkgnd_params = None
        bkgnd_mask = None
        gradient = np.zeros(sub_img.shape)

        if bkgnd_degree is not None:
            bkgnd_params, bkgnd_mask = PSF.background_gradient_fit(
                                           sub_img,
                                           order=bkgnd_degree,
                                           ignore_center=bkgnd_ignore_center,
                                           num_sigma=bkgnd_num_sigma,
                                           debug=self.debug_opt > 1)
            if bkgnd_params is None:
                return None

            gradient = PSF.background_gradient(sub_img.shape, bkgnd_params)

        sub_img_grad = sub_img - gradient

        # Offset Y, Offset X, Scale, AdditionalParams
        if use_angular_params:
            starting_guess = [0.001,0.001,np.pi/8]
            if allow_nonzero_base:
                starting_guess = starting_guess + [0.001]
            starting_guess = starting_guess + [0.001] * len(self.additional_params)
        else:
            starting_guess = [0.001,0.001,1.]
            if allow_nonzero_base:
                starting_guess = starting_guess + [0.001]
            for a_min, a_max, a_name in self.additional_params:
                starting_guess = starting_guess + [np.mean([a_min,a_max])]

        extra_args = (sub_img_grad, search_limit, scale_limit,
                      allow_nonzero_base, use_angular_params)
        if (self.additional_params is not None and
            len(self.additional_params) > 0):
            extra_args = extra_args + tuple(self.additional_params)
        else:
            extra_args = extra_args + tuple([])
        full_result = sciopt.leastsq(self._fit_psf_func, starting_guess,
                                     args=extra_args,
                                     full_output=True,
                                     xtol=tolerance)

        result, cov_x, infodict, mesg, ier = full_result

        if ier < 1 or ier > 4:
            return None

        if use_angular_params:
            offset_y = search_limit[0] * np.sin(result[0]) + 0.5
            offset_x = search_limit[1] * np.sin(result[1]) + 0.5
            scale = scale_limit * (np.sin(result[2]) + 1) / 2
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
        for i, ap in enumerate(self.additional_params):
            if use_angular_params:
                val = ((ap[1] - ap[0]) / 2. *
                       (np.sin(result[result_end+i])+1.) + ap[0])
            else:
                val = result[result_end+i]
            addl_vals_dict[ap[2]] = val

        psf = self.eval_rect(sub_img.shape, (offset_y, offset_x),
                             **addl_vals_dict)

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

        if cov_x is None:
            details['leastsq_cov'] = None
            details['x_err'] = None
            details['y_err'] = None
            details['scale_err'] = None
            details['base_err'] = None
            for i, ap in enumerate(self.additional_params):
                details[ap[2]+'_err'] = None
        else:
            # "To obtain the covariance matrix of the parameters x, cov_x must
            #  be multiplied by the variance of the residuals"
            dof = psf.shape[0]*psf.shape[1]-len(result)
            resid_var = np.sum(self._fit_psf_func(result, *extra_args)**2) / dof
            cov = cov_x * resid_var # In angle-parameter space!! (if use_angular_params)
            details['leastsq_cov'] = cov
            if use_angular_params:
                # Deriv of SL0 * sin(R0) is
                #   SL0 * cos(R0) * dR0
                y_err = np.abs((np.sqrt(cov[0][0]) % (np.pi*2)) * search_limit[0] * np.cos(result[0]))
                x_err = np.abs((np.sqrt(cov[1][1]) % (np.pi*2)) * search_limit[1] * np.cos(result[1]))
                # Deriv of SC/2 * (sin(R)+1) = SC/2 * sin(R) + SC/2 is
                #   SC/2 * cos(R) * dR
                scale_err = np.abs((np.sqrt(cov[2][2]) % (np.pi*2)) * scale_limit/2 * np.cos(result[2]))
                details['x_err'] = x_err
                details['y_err'] = y_err
                details['scale_err'] = scale_err
                for i, ap in enumerate(self.additional_params):
                    err = np.abs((np.sqrt(cov[i+3][i+3]) % (np.pi*2)) *
                                 (ap[1]-ap[0])/2 * np.cos(result[i+3]))
                    details[ap[2]+'_err'] = err
            else:
                details['x_err'] = np.sqrt(cov[0][0])
                details['y_err'] = np.sqrt(cov[1][1])
                details['scale_err'] = np.sqrt(cov[2][2])
                for i, ap in enumerate(self.additional_params):
                    details[ap[2]+'_err'] = np.sqrt(cov[i+result_end][i+result_end])
            # Note the base is not computed using angles
            details['base_err'] = None
            if allow_nonzero_base:
                details['base_err'] = np.sqrt(cov[3][3])

        details['leastsq_infodict'] = infodict
        details['leastsq_mesg'] = mesg
        details['leastsq_ier'] = ier

        for key in addl_vals_dict:
            details[key] = addl_vals_dict[key]

        if self.debug_opt > 1:
            print('_find_position RETURNING', offset_y, offset_x)
            print('Subimg num bad pixels', np.sum(ma.getmaskarray(sub_img)))
            print('Bkgnd params', bkgnd_params)
            print('Bkgnd mask bad pixels', np.sum(ma.getmaskarray(bkgnd_mask)))
            print('PSF scale', scale)
            print('PSF base', base)
            for key in addl_vals_dict:
                print(key, details[key])
            print('LEASTSQ COV')
            cov = details['leastsq_cov']
            print(cov)
            if cov is not None:
                print('X_ERR', details['x_err'])
                print('Y_ERR', details['y_err'])
                print('SCALE_ERR', details['scale_err'])
                print('BASE_ERR', details['base_err'])
                for key in addl_vals_dict:
                    print(key+'_err', details[key+'_err'])
            print('LEASTSQ MESG', mesg)
            print('LEASTSQ IER', ier)
            print('-----')

        return (offset_y, offset_x, details)
