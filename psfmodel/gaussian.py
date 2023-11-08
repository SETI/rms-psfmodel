################################################################################
# psfmodel/gaussian.py
################################################################################

import numpy as np
from scipy.special import erf

from psfmodel import PSF


INV_SQRT_2 = 2**(-0.5)

class GaussianPSF(PSF):
    """A 2-D Gaussian symmetric PSF.

    The PSF can have different standard deviations in the X and Y directions.
    The standard deviations for X and Y can be locked up front when the
    GaussianPSF object is created or left to float so that future calls may
    specify them directly.

    Because these are so fast and easy to compute, we don't cache any results.
    """

    def __init__(self, sigma=None, mean=0., sigma_angle=0., movement=None,
                 movement_granularity=0.1,
                 sigma_x_range=(0.01,10.), sigma_y_range=(0.01,10.),
                 angle_subsample=7):
        """Create a GaussianPSF object describing a 2-D Gaussian PSF.

        Input:
            sigma     The standard deviation of the Gaussian. May be a scalar
                      in which case the value applies to both X and Y, or a
                      tuple (sigma_y,sigma_x) one of which may be None.
                      None for sigma or for sigma_x/y means that the sigma
                      will be supplied later.

            mean      The mean of the Gaussian. May be a scalar in which case
                      the value applies to both X and Y, or a tuple
                      (mean_y,mean_x).

            sigma_angle
                      The angle of the Gaussian. If None, angle is allowed to
                      float during optimization. angle ranges from 0 to pi, with
                      0 being "noon" (-Y) and pi/2 being "3" (+X) assuming that
                      (0,0) is in the top left corner.

            movement  The amount of motion blur in the (Y,X) direction. Must
                      be a tuple of scalars.

            movement_granularity
                      The number of pixels to step for each smear while doing
                      motion blur.

            sigma_x_range
                      The valid range for sigma_x if it is not specified
                      otherwise. This is used during PSF fitting to let
                      sigma_x float to its optimal value.

            sigma_y_range
                      The valid range for sigma_y if it is not specified
                      otherwise. This is used during PSF fitting to let
                      sigma_y float to its optimal value.

            angle_subsample
                      The amount of subsampling done during computation of the
                      2-D Gaussian integral when angle != 0.
        """

        PSF.__init__(self, movement, movement_granularity)

        if np.shape(sigma) == ():
            self.sigma_y = sigma
            self.sigma_x = sigma
        else:
            self.sigma_y, self.sigma_x = sigma
        if np.shape(mean) == ():
            self.mean_y = mean
            self.mean_x = mean
        else:
            self.mean_y, self.mean_x = mean
        self.sigma_angle = sigma_angle
        self.sigma_angle_subsample = angle_subsample

        if self.sigma_y is None:
            if sigma_y_range is not None:
                self.additional_params.append(tuple(sigma_y_range)+('sigma_y',))
        if self.sigma_x is None:
            if sigma_x_range is not None:
                self.additional_params.append(tuple(sigma_x_range)+('sigma_x',))
        if self.sigma_angle is None:
                self.additional_params.append((0, np.pi, 'sigma_angle'))

    @staticmethod
    def gaussian_1d(x, sigma=1., mean=0., scale=1.0, base=0.):
        """Return a 1-D Gaussian.

        This simply returns the value of the Gaussian at a series of points.
        The Gaussian is normalized so the area under the curve is "scale".

        Input:
            x          A scalar or array
            sigma      The standard deviation of the Gaussian
            mean       The mean of the Gaussian
            scale      The scale of the Gaussian; the area under the complete
                       curve (excluding the base)
            base       The base of the Gaussian; a scalar added to the curve
        """

        return (scale/np.sqrt(2*np.pi*sigma**2) *
                np.exp(-(x-mean)**2/(2*sigma**2))) + base

    @staticmethod
    def gaussian_2d_rho(y, x, sigma_y=1., sigma_x=1., mean_y=0., mean_x=0.,
                        scale=1., base=0., rho=0.):
        """Return a 2-D Gaussian using rho (X/Y correlation).

        This simply returns the value of the 2-D Gaussian at a series of points.
        The Gaussian is normalized so the area under the curve is scale.

        Input:
            y          A scalar or array
            x          A scalar or array

            sigma_y    The standard deviation of the Gaussian in the Y
                       direction
            sigma_x    The standard deviation of the Gaussian in the X
                       direction

            mean_y     The mean of the Gaussian in the Y dimension
            mean_x     The mean of the Gaussian in the X dimension

            scale      The scale of the Gaussian; the area under the complete
                       curve (excluding the base)
            base       The base of the Gaussian; a scalar added to the curve

            rho        The correlation between the X and Y dimensions
                       From gaussians-cern.pdf:
                           sigma_X^2 = cos^2 sigma_x^2 + sin^2 sigma_y^2
                           sigma_Y^2 = cos^2 sigma_y^2 + sin^2 sigma_x^2
                        where sigma_X/Y are the size of the ellipse projected
                        onto the normal Cartesian axes. Also:
                           (1-rho^2) sigma_X^2 sigma_Y^2 = sigma_x^2 sigma_y^2
        """

        # See http://cs229.stanford.edu/section/gaussians.pdf
        # See https://indico.cern.ch/category/6015/attachments/192/632/Statistics_Gaussian_I.pdf
        x = x-mean_x
        y = y-mean_y

        norm_fact = 1./(2*np.pi*np.sqrt((sigma_x**2 * sigma_y**2 - rho**2)))
        expon = (((x*sigma_y)**2 + (y*sigma_x)**2 - 2*x*y*xcorr) /
                 ((sigma_x*sigma_y)**2-2*xcorr))
        return scale * norm_fact * np.exp(-0.5 * expon) + base

    @staticmethod
    def gaussian_2d(y, x, sigma_y=1., sigma_x=1., mean_y=0., mean_x=0.,
                    scale=1., base=0., sigma_angle=0.):
        """Return a 2-D Gaussian using angle (angle 0-pi, 0 at noon, CW).

        This simply returns the value of the 2-D Gaussian at a series of points.
        The Gaussian is normalized so the area under the curve is scale.

        Input:
            y          A scalar or array
            x          A scalar or array

            sigma_y    The standard deviation of the Gaussian in the Y
                       direction
            sigma_x    The standard deviation of the Gaussian in the X
                       direction

            mean_y     The mean of the Gaussian in the Y dimension
            mean_x     The mean of the Gaussian in the X dimension

            scale      The scale of the Gaussian; the area under the complete
                       curve (excluding the base)
            base       The base of the Gaussian; a scalar added to the curve

            sigma_angle
                       The angle of the ellipse in the range (0,pi). angle
                       ranges from 0 to pi, with 0 being "noon" (-Y) and pi/2
                       being "3" (+X) assuming that (0,0) is in the top left corner.
        """

        # See http://cs229.stanford.edu/section/gaussians.pdf
        # See https://indico.cern.ch/category/6015/attachments/192/632/Statistics_Gaussian_I.pdf

        # x and y, and sigma_x and sigma_y are in the rotated coordinate system
        x = x-mean_x
        y = y-mean_y

        # Convert angle (normally 0 is to the right, CW) to clock angle
        # (0 is noon, pi/2 is 3, pi is 6)
        sigma_angle = sigma_angle - np.pi/2

        # Convert x and y (ellipse coordinates) to X and Y (Cartesian
        # coordinates)
        c = np.cos(sigma_angle)
        s = np.sin(sigma_angle)
        X = c*x + s*y
        Y = -s*x + c*y

        norm_fact = 1./(2*np.pi*np.sqrt((sigma_x**2 * sigma_y**2)))
        expon = (((X*sigma_y)**2 + (Y*sigma_x)**2) /
                 ((sigma_x*sigma_y)**2))
        return scale * norm_fact * np.exp(-0.5 * expon) + base

    @staticmethod
    def gaussian_integral_1d(xmin, xmax, sigma=1., mean=0.,
                             scale=1., base=0.):
        """Return the integral of a Gaussian.

        The integral is over the limits [xmin,xmax].

        Values are generated via the error function, where the integral from
        -inf to x is equal to
                    (1 + erf((x - mean_x) / (sqrt(2)*sigma_x)) / 2

        This function works for both scalar and array values of xmin and xmax.

        Input:
            xmin       The lower bound of the integral; may be a scalar or 1-D
                       array
            xmax       The upper bound of the integral; may be a scalar or 1-D
                       array
            sigma_x    The standard deviation of the Gaussian
            mean_x     The mean of the Gaussian
            scale      The scale of the Gaussian; the area under the complete
                       curve (excluding the base)
            base       The base of the Gaussian; a scalar added to the curve
        """

        # Normalize xmin and xmax
        assert sigma > 0.
        xmin_div_sqrt_2 = (xmin - mean) * (INV_SQRT_2/sigma)
        xmax_div_sqrt_2 = (xmax - mean) * (INV_SQRT_2/sigma)

        # Handle the scalar case
        if np.shape(xmin) == () and np.shape(xmax) == ():
            return (0.5 * (erf(xmax_div_sqrt_2) -
                           erf(xmin_div_sqrt_2)) * scale) + base

        # If either value is an array, broadcast to a common shape
        (xmin_div_sqrt_2,
         xmax_div_sqrt_2) = np.broadcast_arrays(xmin_div_sqrt_2,
                                                xmax_div_sqrt_2)

        result = erf(xmax_div_sqrt_2) - erf(xmin_div_sqrt_2)

        return result * 0.5 * scale + base

    @staticmethod
    def gaussian_integral_2d(ymin, ymax, xmin, xmax, sigma_y=1., sigma_x=1.,
                             mean_y=0., mean_x=0., scale=1., base=0.,
                             sigma_angle=0., sigma_angle_subsample=7):
        """Return the double integral of a 2-D Gaussian.

        The integral is over the limits [ymin,ymax] and [xmin,xmax].

        This function works for both scalar and array values of
        xmin/xmax/ymin/ymax.

        Inputs:
            ymin       The lower bound of the integral in the Y dimension;
                       may be a scalar or 1-D array
            ymax       The upper bound of the integral in the Y dimension;
                       may be a scalar or 1-D array
            xmin       The lower bound of the integral in the X dimension;
                       may be a scalar or 1-D array
            xmax       The upper bound of the integral in the X dimension;
                       may be a scalar or 1-D array

            sigma_y    The standard deviation of the Gaussian in the Y
                       dimension
            sigma_x    The standard deviation of the Gaussian in the X
                       dimension

            mean_y     The mean of the Gaussian in the Y dimension
            mean_x     The mean of the Gaussian in the X dimension

            scale      The scale of the Gaussian; the area under the complete
                       curve (excluding the base)
            base       The base of the Gaussian; a scalar added to the curve

            sigma_angle
                       The angle of the ellipse in the range (0,pi)

            sigma_angle_subsample
                      The amount of subsampling done during computation of the
                      2-D Gaussian integral when angle != 0.
        """

        if sigma_angle == 0.:
            return (GaussianPSF.gaussian_integral_1d(ymin, ymax,
                                                     sigma_y, mean_y) *
                    GaussianPSF.gaussian_integral_1d(xmin, xmax,
                                                     sigma_x, mean_x) *
                    scale + base)

        # Handle the scalar case
        if (np.shape(xmin) == () and np.shape(xmax) == () and
            np.shape(ymin) == () and np.shape(ymax) == ()):
            ys = np.linspace(ymin, ymax, sigma_angle_subsample)
            xs = np.linspace(xmin, xmax, sigma_angle_subsample)
            xindex, yindex = np.meshgrid(xs, ys)

            ret = GaussianPSF.gaussian_2d(ys, xs, sigma_y, sigma_x,
                                          mean_y, mean_x, scale, base,
                                          sigma_angle)
            return np.mean(ret)

        res = np.empty(xmin.shape)
        for x in range(xmin.shape[0]):
            ys = np.linspace(ymin[x], ymax[x], sigma_angle_subsample)
            xs = np.linspace(xmin[x], xmax[x], sigma_angle_subsample)
            xindex, yindex = np.meshgrid(xs, ys)

            ret = GaussianPSF.gaussian_2d(ys, xs, sigma_y, sigma_x,
                                          mean_y, mean_x, scale, base,
                                          sigma_angle)
            res[x] = np.mean(ret)

        return res
        # If any value is an array, broadcast to a common shape
#         (xmin, xmax, ymin, ymax) = np.broadcast_arrays(xmin, xmax,
#                                                        ymin, ymax)


    def eval_point(self, coord, scale=1., base=0., sigma=None,
                   sigma_y=None, sigma_x=None, sigma_angle=None):
        """Evaluate the 2-D Gaussian PSF at a single, fractional, point.

        (0,0) is the center of the PSF and x and y may be negative.

        Input:
            coord       The coordinate (y,x) at which to evaluate the PSF.

            scale       A scale factor to apply to the resulting PSF.
            base        A scalar added to the resulting PSF.

            sigma       The standard deviation of the Gaussian. It may be
                        specified here or during the creation of the
                        GaussianPSF object but not both. May be a scalar
                        or a tuple (sigma_y,sigma_x), or None if specified
                        at creation time.

            sigma_y     An alternative way to specify sigma_y. Used primarily
                        for letting sigma_y float during PSF fitting.
            sigma_x     An alternative way to specify sigma_x. Used primarily
                        for letting sigma_x float during PSF fitting.
            sigma_angle An alternative way to specify angle. Used primarily
                        for letting angle float during PSF fitting.
        """

        sy = self.sigma_y
        sx = self.sigma_x

        if sigma is not None:
            if np.shape(sigma) == ():
                sy = sigma
                sx = sigma
            else:
                sy, sx = sigma

        if sigma_y is not None:
            sy = sigma_y
        if sigma_x is not None:
            sx = sigma_x

        r = self.sigma_angle
        if sigma_angle is not None:
            r = sigma_angle

        return PSFGaussian.gaussian_2d(coord[0], coord[1], sy, sx,
                                       self.mean_y, self.mean_x,
                                       scale, base, r)

    def eval_pixel(self, coord, offset=(0.,0.), scale=1., base=0., sigma=None,
                   sigma_y=None, sigma_x=None, sigma_angle=None):
        """Evaluate the Gaussian PSF integrated over an entire integer pixel.

        The returned array has the PSF offset from the center by
        (offset_y,offset_x). An offset of (0,0) places the PSF in the upper
        left corner of the center pixel while an offset of (0.5,0.5)
        places the PSF in the center of the center pixel. The angle is applied
        relative to this new origin, so as angle changes the center of the
        ellipse does not move.

        This essentially performs a 2-D integration of the PSF over the intervals
            [y-offset_y-0.5,y-offset_y+0.5] and
            [x-offset_x-0.5,x-offset_x+0.5].

        Input:
            coord       The integer coordinate (y,x) at which to evaluate the
                        PSF.
            offset      The amount (offset_y,offset_x) to offset the center
                        of the PSF.

            scale       A scale factor to apply to the resulting PSF.
            base        A scalar added to the resulting PSF.

            sigma       The standard deviation of the Gaussian. It may be
                        specified here or during the creation of the
                        GaussianPSF object but not both. May be a scalar
                        or a tuple (sigma_y,sigma_x), or None if specified
                        at creation time.

            sigma_y     An alternative way to specify sigma_y. Used primary
                        for letting sigma_y float during astrometry.
            sigma_x     an alternative way to specify sigma_x. Used primary
                        for letting sigma_x float during astrometry.
            sigma_angle An alternative way to specify angle. Used primarily
                        for letting angle float during PSF fitting.
        """

        sy = self.sigma_y
        sx = self.sigma_x

        if sigma is not None:
            if np.shape(sigma) == ():
                sy = sigma
                sx = sigma
            else:
                sy, sx = sigma

        if sigma_y is not None:
            sy = sigma_y
        if sigma_x is not None:
            sx = sigma_x

        r = self.sigma_angle
        if sigma_angle is not None:
            r = sigma_angle

        return (GaussianPSF.gaussian_integral_2d(coord[0]-offset[0],
                                                 coord[0]-offset[0]+1.,
                                                 coord[1]-offset[1],
                                                 coord[1]-offset[1]+1,
                                                 sy, sx,
                                                 self.mean_y, self.mean_x,
                                                 scale, base, r,
                                                 self.sigma_angle_subsample))

    def _eval_rect(self, rect_size, offset=(0.,0.), scale=1., base=0.,
                   sigma=None, sigma_y=None, sigma_x=None, sigma_angle=None):
        rect_size_y, rect_size_x = rect_size
        y_coords = np.repeat(np.arange(-(rect_size_y//2), rect_size_y//2+1),
                             rect_size_x).astype('float')
        x_coords = np.tile(np.arange(-(rect_size_x//2), rect_size_x//2+1),
                           rect_size_y).astype('float')
        coords = np.empty((2,rect_size_y*rect_size_x))
        coords[0] = y_coords
        coords[1] = x_coords
        rect = self.eval_pixel(coords, offset, scale, base, sigma,
                               sigma_y, sigma_x, sigma_angle)
        rect = rect.reshape(rect_size)

        return rect

    def eval_rect(self, rect_size, offset=(0.,0.), scale=1., base=0.,
                  sigma=None, sigma_y=None, sigma_x=None, sigma_angle=None):
        """Create a rectangular pixelated Gaussian PSF.

        This is done by evaluating the PSF function from:
            [-rect_size_y//2:rect_size_y//2] and
            [-rect_size_x//2:rect_size_x//2]

        The returned array has the PSF offset from the center by
        (offset_y,offset_x). An offset of (0,0) places the PSF in the upper
        left corner of the center pixel while an offset of (0.5,0.5)
        places the PSF in the center of the center pixel. The angle is applied
        relative to this new origin, so as angle changes the center of the
        ellipse does not move.

        Input:
            rect_size   The size of the rectangle (rect_size_y,rect_size_x)
                        of the returned PSF. Both dimensions must be odd.
            offset      The amount (offset_y,offset_x) to offset the center
                        of the PSF from the upper left corner of the center
                        pixel of the rectangle.

            scale       A scale factor to apply to the resulting PSF.
            base        A scalar added to the resulting PSF.

            sigma       The standard deviation of the Gaussian. It may be
                        specified here or during the creation of the
                        GaussianPSF object but not both. May be a scalar
                        or a tuple (sigma_y,sigma_x), or None if specified
                        at creation time.

            sigma_y     An alternative way to specify sigma_y. Used primary
                        for letting sigma_y float during astrometry.
            sigma_x     An alternative way to specify sigma_x. Used primary
                        for letting sigma_x float during astrometry.
            sigma_angle An alternative way to specify angle. Used primarily
                        for letting angle float during PSF fitting.
        """

        rect_size_y, rect_size_x = rect_size
        offset_y, offset_x = offset

        assert rect_size_y % 2 == 1 # Odd
        assert rect_size_x % 2 == 1 # Odd

        if self.movement[0] == 0. and self.movement[1] == 0.:
            return self._eval_rect(rect_size, offset, scale, base, sigma,
                                   sigma_y, sigma_x, sigma_angle)

        return self._eval_rect_smeared(self.movement, rect_size, offset,
                                       scale, base, sigma, sigma_y, sigma_x,
                                       sigma_angle)



################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_GaussianPSF(unittest.TestCase):

    def runTest(self):

        # gaussian_1d
        x_coords = np.arange(-20., 21.)/4
        gauss1d = GaussianPSF.gaussian_1d(x_coords)/4
        self.assertEqual(np.argmax(gauss1d), 20)
        self.assertAlmostEqual(np.sum(gauss1d), 1., places=5)

        gauss1d_scalar = GaussianPSF.gaussian_1d(0.)/4
        self.assertEqual(gauss1d_scalar, gauss1d[20])

        gauss1d2 = GaussianPSF.gaussian_1d(x_coords, sigma=0.2)/4
        self.assertAlmostEqual(np.sum(gauss1d2), 1., places=4)
        self.assertTrue(np.sum(np.abs(gauss1d-gauss1d2)) > 1e-5)

        gauss1d3 = GaussianPSF.gaussian_1d(x_coords, mean=0.25)/4
        self.assertEqual(np.argmax(gauss1d3), 21)
        self.assertAlmostEqual(np.sum(gauss1d3), 1., places=5)
        self.assertTrue(np.sum(np.abs(gauss1d-gauss1d3)) > 1e-5)

        gauss1d4 = GaussianPSF.gaussian_1d(x_coords, scale=2.)/4
        self.assertAlmostEqual(np.sum(gauss1d4), 2., places=5)

        gauss1d5 = GaussianPSF.gaussian_1d(x_coords, base=1.)/4
        self.assertAlmostEqual(np.sum(gauss1d5-.25), 1., places=5)

        # gaussian_2d
        y_coords = np.tile(np.arange(-10., 11.)/2, 21)
        x_coords = np.repeat(np.arange(-10., 11.)/2, 21)
        gauss2d = GaussianPSF.gaussian_2d(y_coords, x_coords)/4
        self.assertAlmostEqual(np.sum(gauss2d), 1., places=5)

        gauss2d_scalar = GaussianPSF.gaussian_2d(0., 0.)/4
        self.assertEqual(gauss2d_scalar, gauss2d[10*21+10])

        gauss2d = GaussianPSF.gaussian_2d(y_coords, x_coords, scale=2.)/4
        self.assertAlmostEqual(np.sum(gauss2d), 2., places=5)

        gauss2d5 = GaussianPSF.gaussian_2d(y_coords, x_coords, scale=2.,
                                           sigma_x=.5, sigma_y=.5)/4
        self.assertAlmostEqual(np.sum(gauss2d5), 2., places=5)
        self.assertTrue(np.sum(np.abs(gauss2d-gauss2d5)) > 1e-5)

        gauss2d51 = GaussianPSF.gaussian_2d(y_coords, x_coords, scale=2.,
                                           sigma_x=.5, sigma_y=.5,
                                           base=1.)/4
        self.assertAlmostEqual(np.sum(gauss2d51-0.25), 2., places=5)
        self.assertTrue(np.sum(np.abs(gauss2d51-gauss2d5)) > 1e-5)

        gauss2d2 = GaussianPSF.gaussian_2d(y_coords, x_coords, scale=2.,
                                           sigma_x=.25, sigma_y=.5,
                                           base=1.)/4
        gauss2d3 = GaussianPSF.gaussian_2d(y_coords, x_coords, scale=2.,
                                           sigma_x=.5, sigma_y=.25,
                                           base=1.)/4
        gauss2d2 = gauss2d2.reshape(21,21)
        gauss2d3 = gauss2d3.reshape(21,21)
        self.assertTrue(np.sum(np.abs(gauss2d2-gauss2d3)) > 1e-5)
        self.assertTrue(np.sum(np.abs(np.transpose(gauss2d2)-gauss2d3)) < 1e-5)

        # gaussian_integral_1d
        i1 = GaussianPSF.gaussian_integral_1d(-10., 10.)
        self.assertAlmostEqual(i1, 1.)
        i2 = GaussianPSF.gaussian_integral_1d(-10., 0.)
        i3 = GaussianPSF.gaussian_integral_1d(0., 10.)
        self.assertAlmostEqual(i2, i3)
        i4 = GaussianPSF.gaussian_integral_1d(0., 10., sigma=5.)
        self.assertNotAlmostEqual(i2, i4)
        i5 = GaussianPSF.gaussian_integral_1d(0., 10., mean=1.)
        self.assertNotAlmostEqual(i2, i5)
        i6 = GaussianPSF.gaussian_integral_1d(-10., 1., mean=1., sigma=0.25)
        i7 = GaussianPSF.gaussian_integral_1d(1., 10., mean=1., sigma=0.25)
        self.assertAlmostEqual(i6, i7)
        i8 = GaussianPSF.gaussian_integral_1d(-10., 10., scale=2., base=1.)
        self.assertAlmostEqual(i8, 3.)

        # NEED TO TEST ARRAY VERSION

        # gaussian_integral_2d
        i1 = GaussianPSF.gaussian_integral_2d(-10., 10., -10., 10.)
        self.assertAlmostEqual(i1, 1.)
        i2 = GaussianPSF.gaussian_integral_2d(-10., 0., -10., 10.)
        i3 = GaussianPSF.gaussian_integral_2d(0., 10., -10., 10.)
        self.assertAlmostEqual(i2, i3)
        i2 = GaussianPSF.gaussian_integral_2d(-10., 10., -10., 0.)
        i3 = GaussianPSF.gaussian_integral_2d(-10., 10., 0., 10.)
        self.assertAlmostEqual(i2, i3)
        i4 = GaussianPSF.gaussian_integral_2d(-10., 10., -10., 0., sigma_x=5.)
        self.assertNotAlmostEqual(i2, i4)
        i4 = GaussianPSF.gaussian_integral_2d(-10., 10., -10., 0., sigma_y=5.)
        self.assertNotAlmostEqual(i2, i4)
        i5 = GaussianPSF.gaussian_integral_2d(-10., 10., -10., 0., mean_x=1.)
        self.assertNotAlmostEqual(i2, i5)
        i5 = GaussianPSF.gaussian_integral_2d(-10., 10., -10., 0., mean_y=10.)
        self.assertNotAlmostEqual(i2, i5)
        i6 = GaussianPSF.gaussian_integral_2d(-10., 0., -10., 10.,
                                              sigma_x=0.5, sigma_y=2.0)
        i7 = GaussianPSF.gaussian_integral_2d(0., 10., -10., 10.,
                                              sigma_x=0.5, sigma_y=2.0)
        self.assertAlmostEqual(i6, i7)
        i8 = GaussianPSF.gaussian_integral_2d(-10., 10., -10., 10.,
                                              scale=2., base=1.)
        self.assertAlmostEqual(i8, 3.)

        # eval_point

        # eval_rect

        # eval_pixel and optimize_from_data
        psf = GaussianPSF()
        y_coords = np.repeat(np.arange(-10., 11.), 21)
        x_coords = np.tile(np.arange(-10., 11.), 21)
        coords = np.empty((2,21*21))
        coords[0] = y_coords
        coords[1] = x_coords
        gauss2d = psf.eval_pixel(coords, scale=2., sigma=1.)
        gauss2d = gauss2d.reshape(21,21)
        ret = psf.find_position(gauss2d, gauss2d.shape,
                                starting_point=((gauss2d.shape[0]//2,
                                                 gauss2d.shape[1]//2)),
                                bkgnd_degree=0, num_sigma=0)
        self.assertAlmostEqual(ret[0], gauss2d.shape[0]//2)
        self.assertAlmostEqual(ret[1], gauss2d.shape[1]//2)
        self.assertAlmostEqual(ret[2]['sigma_y'], 1.)
        self.assertAlmostEqual(ret[2]['sigma_x'], 1.)
        self.assertAlmostEqual(ret[2]['scale'], 2., places=6)

        y_coords = np.repeat(np.arange(-10., 11.), 21)
        x_coords = np.tile(np.arange(-10., 11.), 21)
        coords = np.empty((2,21*21))
        coords[0] = y_coords
        coords[1] = x_coords
        gauss2d = psf.eval_pixel(coords, scale=2., sigma=(2., 0.5))
        gauss2d = gauss2d.reshape(21,21)
        ret = psf.find_position(gauss2d, gauss2d.shape,
                                starting_point=((gauss2d.shape[0]//2,
                                                 gauss2d.shape[1]//2)),
                                bkgnd_degree=0, num_sigma=0)
        self.assertAlmostEqual(ret[0], gauss2d.shape[0]//2)
        self.assertAlmostEqual(ret[1], gauss2d.shape[1]//2)
        self.assertAlmostEqual(ret[2]['sigma_y'], 2.)
        self.assertAlmostEqual(ret[2]['sigma_x'], 0.5)
        self.assertAlmostEqual(ret[2]['scale'], 2., places=6)

        psf2 = GaussianPSF(mean=(0.5,0.75))
        y_coords = np.repeat(np.arange(-10., 11.), 21)
        x_coords = np.tile(np.arange(-10., 11.), 21)
        coords = np.empty((2,21*21))
        coords[0] = y_coords
        coords[1] = x_coords
        gauss2d = psf2.eval_pixel(coords, scale=0.5, sigma=(0.3, 1.3))
        gauss2d = gauss2d.reshape(21,21)

        ret = psf.find_position(gauss2d, gauss2d.shape,
                                starting_point=((gauss2d.shape[0]//2,
                                                 gauss2d.shape[1]//2)),
                                bkgnd_degree=0, num_sigma=0)
        self.assertAlmostEqual(ret[0], gauss2d.shape[0]//2+0.5)
        self.assertAlmostEqual(ret[1], gauss2d.shape[1]//2+0.75)
        self.assertAlmostEqual(ret[2]['sigma_y'], 0.3)
        self.assertAlmostEqual(ret[2]['sigma_x'], 1.3)
        self.assertAlmostEqual(ret[2]['scale'], 0.5, places=6)

        ret = psf2.find_position(gauss2d, gauss2d.shape,
                                 starting_point=((gauss2d.shape[0]//2,
                                                  gauss2d.shape[1]//2)),
                                 bkgnd_degree=0, num_sigma=0)
        self.assertAlmostEqual(ret[0], gauss2d.shape[0]//2)
        self.assertAlmostEqual(ret[1], gauss2d.shape[1]//2)
        self.assertAlmostEqual(ret[2]['sigma_y'], 0.3)
        self.assertAlmostEqual(ret[2]['sigma_x'], 1.3)
        self.assertAlmostEqual(ret[2]['scale'], 0.5, places=6)

        psf2 = GaussianPSF()
        y_coords = np.repeat(np.arange(-10., 11.), 21)
        x_coords = np.tile(np.arange(-10., 11.), 21)
        coords = np.empty((2,21*21))
        coords[0] = y_coords
        coords[1] = x_coords
        gauss2d = psf2.eval_pixel(coords, offset=(0.21, -0.35),
                                  scale=0.5, sigma=(0.3, 1.3))
        gauss2d = gauss2d.reshape(21,21)
        ret = psf.find_position(gauss2d, gauss2d.shape,
                                starting_point=((gauss2d.shape[0]//2,
                                                 gauss2d.shape[1]//2)),
                                bkgnd_degree=0, num_sigma=0)
        self.assertAlmostEqual(ret[0], gauss2d.shape[0]//2+0.21)
        self.assertAlmostEqual(ret[1], gauss2d.shape[1]//2-0.35)
        self.assertAlmostEqual(ret[2]['sigma_y'], 0.3)
        self.assertAlmostEqual(ret[2]['sigma_x'], 1.3)
        self.assertAlmostEqual(ret[2]['scale'], 0.5, places=6)


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
