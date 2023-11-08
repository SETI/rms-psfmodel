################################################################################
# psfmodel/hst.py
################################################################################

import os

import astropy.io.fits as pyfits
import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.signal as scisig

from psfmodel import PSF


#===============================================================================
#
# HST CAMERA DETAILS
#
#===============================================================================

#=====================
# ACS HRC is 1024x1024
#=====================

# TinyTim returns the following size PSFs given the FOV in arcsec.
# Subsample > 1 may make the returned PSF slightly larger than predicted
ACS_HRC_RETURNED_SIZES = [
    (0.5,  27),
    (0.75, 39),
    (1.0,  50),
    (1.25, 62),
    (1.5,  75),
    (1.75, 87),
    (2.0,  98),
    (2.25, 111),
    (2.5,  123),
    (2.75, 136),
    (3.0,  146),
    (4.0,  194),
    (5.0,  243),
    (6.0,  291)
]

ACS_HRC_PIXEL_SIZE = 0.025 # arc sec
ACS_HRC_DEFAULT_RECTSIZEGUESS = 11 # Safely inside 0.75 arcsec
ACS_HRC_DEFAULT_FOV = 0.75 # arc sec
ACS_HRC_DEFAULT_JITTER = 10
ACS_HRC_DEFAULT_SUBSAMPLE = 5
ACS_HRC_DEFAULT_LINE = 512
ACS_HRC_DEFAULT_SAMPLE = 512

#=====================
# WFPC2 PC1 is 800x800
#=====================

# TinyTim returns the following size PSFs given the FOV in arcsec.
# Subsample > 1 may make the returned PSF slightly larger than predicted
WFPC2_PC1_RETURNED_SIZES = [
    (0.5,  10),
    (0.75, 16),
    (1.0,  20),
    (1.25, 26),
    (1.5,  32),
    (1.75, 38),
    (2.0,  42),
    (2.25, 48),
    (2.5,  54),
    (2.75, 60),
    (3.0,  64),
    (4.0,  86),
    (5.0,  108),
    (6.0,  130)
]

WFPC2_PC1_PIXEL_SIZE = 0.046 # arc sec
WFPC2_PC1_DEFAULT_RECTSIZEGUESS = 11 # Safely inside 0.75 arcsec
WFPC2_PC1_DEFAULT_FOV = 0.75 # arc sec
WFPC2_PC1_DEFAULT_JITTER = 20
WFPC2_PC1_DEFAULT_SUBSAMPLE = 5
WFPC2_PC1_DEFAULT_LINE = 400
WFPC2_PC1_DEFAULT_SAMPLE = 400

#=======================
# WFC3 IR is 1014x1014
# WFC3 UVIS is 4096x2051
#=======================

# TinyTim returns the following size PSFs given the FOV in arcsec.
# Subsample > 1 may make the returned PSF slightly larger than predicted
WFC3_UVIS_RETURNED_SIZES = [
    (0.5,  22),
    (0.75, 32),
    (1.0,  28),
    (1.25, 36),
    (1.5,  44),
    (1.75, 51),
    (2.0,  59),
    (2.25, 66),
    (2.5,  74),
    (2.75, 81),
    (3.0,  89),
    (4.0,  119),
    (5.0,  149),
    (6.0,  179)
]

WFC3_UVIS_PIXEL_SIZE = 0.0396 # arc sec
WFC3_UVIS_DEFAULT_RECTSIZEGUESS = 11 # Safely inside 0.75 arcsec
WFC3_UVIS_DEFAULT_FOV = 0.75 # arc sec
WFC3_UVIS_DEFAULT_JITTER = 10
WFC3_UVIS_DEFAULT_SUBSAMPLE = 5
WFC3_UVIS_DEFAULT_LINE = None
WFC3_UVIS_DEFAULT_SAMPLE = None

# TinyTim returns the following size PSFs given the FOV in arcsec.
# Subsample > 1 may make the returned PSF slightly larger than predicted
WFC3_IR_RETURNED_SIZES = [ # XXX
    (0.5,  4),
    (0.75, 6),
    (1.0,  9),
    (1.25, 11),
    (1.5,  13),
    (1.75, 15),
    (2.0,  18),
    (2.25, 20),
    (2.5,  22),
    (2.75, 24),
    (3.0,  27),
    (4.0,  36),
    (5.0,  45),
    (6.0,  54)
]

WFC3_IR_PIXEL_SIZE = 0.128 # arc sec
WFC3_IR_DEFAULT_RECTSIZEGUESS = 11 # Safely inside 2.0 arcsec
WFC3_IR_DEFAULT_FOV = 2.0 # arc sec
WFC3_IR_DEFAULT_JITTER = 40
WFC3_IR_DEFAULT_SUBSAMPLE = 5
WFC3_IR_DEFAULT_LINE = None
WFC3_IR_DEFAULT_SAMPLE = None


RETURNED_SIZES = {
    ('ACS',   'HRC'):  ACS_HRC_RETURNED_SIZES,
    ('WFPC2', 'PC1'):  WFPC2_PC1_RETURNED_SIZES,
    ('WFC3',  'UVIS'): WFC3_UVIS_RETURNED_SIZES,
    ('WFC3',  'IR'):   WFC3_IR_RETURNED_SIZES,
}

DEFAULT_RECTSIZEGUESS = {
    ('ACS',   'HRC'):  ACS_HRC_DEFAULT_RECTSIZEGUESS,
    ('WFPC2', 'PC1'):  WFPC2_PC1_DEFAULT_RECTSIZEGUESS,
    ('WFC3',  'UVIS'): WFC3_UVIS_DEFAULT_RECTSIZEGUESS,
    ('WFC3',  'IR'):   WFC3_IR_DEFAULT_RECTSIZEGUESS,
}

DEFAULT_FOV = {
    ('ACS',   'HRC'):  ACS_HRC_DEFAULT_FOV,
    ('WFPC2', 'PC1'):  WFPC2_PC1_DEFAULT_FOV,
    ('WFC3',  'UVIS'): WFC3_UVIS_DEFAULT_FOV,
    ('WFC3',  'IR'):   WFC3_IR_DEFAULT_FOV,
}

DEFAULT_JITTER = {
    ('ACS',   'HRC'):  ACS_HRC_DEFAULT_JITTER,
    ('WFPC2', 'PC1'):  WFPC2_PC1_DEFAULT_JITTER,
    ('WFC3',  'UVIS'): WFC3_UVIS_DEFAULT_JITTER,
    ('WFC3',  'IR'):   WFC3_IR_DEFAULT_JITTER,
}

DEFAULT_SUBSAMPLE = {
    ('ACS',   'HRC'):  ACS_HRC_DEFAULT_SUBSAMPLE,
    ('WFPC2', 'PC1'):  WFPC2_PC1_DEFAULT_SUBSAMPLE,
    ('WFC3',  'UVIS'): WFC3_UVIS_DEFAULT_SUBSAMPLE,
    ('WFC3',  'IR'):   WFC3_IR_DEFAULT_SUBSAMPLE,
}

DEFAULT_LINE = {
    ('ACS',   'HRC'):  ACS_HRC_DEFAULT_LINE,
    ('WFPC2', 'PC1'):  WFPC2_PC1_DEFAULT_LINE,
    ('WFC3',  'UVIS'): WFC3_UVIS_DEFAULT_LINE,
    ('WFC3',  'IR'):   WFC3_IR_DEFAULT_LINE,
}

DEFAULT_SAMPLE = {
    ('ACS',   'HRC'):  ACS_HRC_DEFAULT_SAMPLE,
    ('WFPC2', 'PC1'):  WFPC2_PC1_DEFAULT_SAMPLE,
    ('WFC3',  'UVIS'): WFC3_UVIS_DEFAULT_SAMPLE,
    ('WFC3',  'IR'):   WFC3_IR_DEFAULT_SAMPLE,
}


#===============================================================================
#
# ENVIRONMENT VARIABLES AND UTILITY ROUTINES
#
#===============================================================================

TINY_TIM_DIR = os.environ['TINYTIM']
PSF_CACHE_DIR = os.environ['PSF_CACHE_DIR']

if os.getcwd()[1] == ':':
    # Windows
    DEV_NULL = 'NUL'
else:
    # Linux
    DEV_NULL = '/dev/null'

def path_join(*args):
    """A prettier version of os.path.join that works better under Windows."""
    return os.path.join(*args).replace('\\', '/')


#===============================================================================
#
# THE HSTPSF CLASS
#
#===============================================================================

class HSTPSF(PSF):
    """A PSF that represents one of the HST cameras."""

    def __init__(self, instrument, detector, filter, line=None, sample=None,
                 rect_size_guess=None, fov=None, subsample=None, movement=None,
                 movement_granularity=0.3,
                 jitter_x=None, jitter_y=None, jitter_z=None, **kwargs):
        """Create an HSTPSF object.

        Input:
            instrument        The HST instrument (ACS, WFPC2, WFC3)

            detector          The HST detector
                                  For ACS: HRC
                                  For WFPC2: PC1
                                  For WFC3: UVIS, IR

            filter            The HST filter

            line              The line (Y) coordinate of the center of the PSF;
                              used for off-axis instruments because of
                              location-specific distortions

            sample            The sample (X) coordinate of the center of the
                              PSF; used for off-axis instruments because of
                              location-specific distortions

            rect_size_guess   A guess for the largest PSF size that's going
                              to be asked for later; used to cache the initial
                              PSF so that it doesn't need to be recomputed
                              multiple times

            fov               The field of view of the camera used to generate
                              the PSF from TinyTim in arcsec;
                              recommended max 3.0

            subsample         Override the default subsample amount; only valid
                              for ACS; Must be odd

            movement          Motion blur in the (Y,X) direction. Must be a
                              tuple of scalars.

            movement_granularity
                              The number of pixels to step for each smear doing
                              motion blur. The default of 0.1 is empirically a
                              good choice for speed vs. accuracy (<< 0.01
                              pixels).

            jitter_x/y/z      The amount of jitter (in mas) to apply to the
                              PSF; z is the angle; None means use the default
                              for the instrument/detector
        """

        PSF.__init__(self, movement, movement_granularity)

        self.instrument = instrument
        self.detector = detector
        self.filter = filter
        self.line = line
        self.sample = sample
        self.rect_size_guess = rect_size_guess
        self.fov = fov
        self.subsample = subsample
        self.jitter_x = jitter_x
        self.jitter_y = jitter_y
        self.jitter_z = jitter_z

        self.do_subsample_diffusion = False # Only for certain instruments

        if self.line is None:
            self.line = DEFAULT_LINE[(self.instrument, self.detector)]
        if self.sample is None:
            self.sample = DEFAULT_SAMPLE[(self.instrument, self.detector)]
        if self.rect_size_guess is None:
            self.rect_size_guess = DEFAULT_RECTSIZEGUESS[(self.instrument,
                                                          self.detector)]
        if self.fov is None:
            self.fov = DEFAULT_FOV[(self.instrument, self.detector)]
        if self.subsample is None:
            self.subsample = DEFAULT_SUBSAMPLE[(self.instrument,
                                                self.detector)]
        assert self.subsample % 2 == 1 # Must be odd
        if self.jitter_x is None:
            self.jitter_x = DEFAULT_JITTER[(self.instrument, self.detector)]
        if self.jitter_y is None:
            self.jitter_y = DEFAULT_JITTER[(self.instrument, self.detector)]
        if self.jitter_z is None:
            self.jitter_z = 0.

        self.cached_psf_size = None
        self.cached_psf = None
        self.cached_offset = None
        self.cached_spline_order = None
        self.cached_pixelated_psf = None

        if instrument == 'ACS':
            self._init_ACS(**kwargs)
        elif instrument == 'WFPC2':
            self._init_WFPC2(**kwargs)
        elif instrument == 'WFC3':
            self._init_WFC3(**kwargs)
        else:
            print('UNKNOWN INSTRUMENT', instrument)
            assert False

    def _init_ACS(self, **kwargs):
        assert self.detector == 'HRC'
        self.do_subsample_diffusion = True

    def _init_WFPC2(self, **kwargs):
        assert self.detector == 'PC1'
        self.do_subsample_diffusion = False

    def _init_WFC3(self, **kwargs):
        assert self.detector == 'UVIS' or self.detector == 'IR'
        self.do_subsample_diffusion = True

        # See http://www.stsci.edu/hst/observatory/apertures/wfc3.html
        self.aperture = kwargs['aperture']

        if self.detector == 'UVIS':
            if self.aperture == 'UVIS2-C512C-SUB':
                x_offset = 0  # Lower left corner of chip 2 is at (0,0)
                y_offset = 0 +2048 # Will be taken off in run_tinytim after
                                   # chip number is detected
                self.wfc3_uvis_chip = 2
                if self.line is None:
                    self.line = 256
                if self.sample is None:
                    self.sample = 256
            elif self.aperture == 'UVIS1-C512B-SUB':
                x_offset = 4096-512
                y_offset = 2048-512  # Upper right corner of chip 1 is at
                                     # (4095, 2047)
                self.wfc3_uvis_chip = 1
                if self.line is None:
                    self.line = 256
                if self.sample is None:
                    self.sample = 256
            elif self.aperture == 'UVIS2-M512C-SUB':
                x_offset = 2048-512  # Lower left corner of chip 2 is at (0,0)
                y_offset = 2048-512 +2048 # Will be taken off in run_tinytim
                                          # after chip number is detected
                self.wfc3_uvis_chip = 2
                if self.line is None:
                    self.line = 256
                if self.sample is None:
                    self.sample = 256
            else:
                print('UNKNOWN WFC3 UVIS APERTURE', self.aperture)
                assert False
        else:
            if self.aperture != 'IRSUB256':
                print('UNKNOWN WFC3 IR APERTURE', self.aperture)
                assert False
            x_offset = 1014//2-256//2
            y_offset = 1014//2-256//2
            if self.line is None:
                self.line = 128
            if self.sample is None:
                self.sample = 128

        self.line += y_offset
        self.sample += x_offset

    @staticmethod
    def run_tinytim(instrument, detector, y_ctr, x_ctr, filter,
                    psf_size_pixels, fov, subsample_amt,
                    jitter_x=0., jitter_y=0., jitter_z=0.,
                    force_run_tinytim=False, output_dev_null=True,
                    **kwargs):
        """Run TinyTim to produce a PSF.

        Note that the TinyTim PSF is centered on the center pixel, not on the
        corner.

        Input:

        instrument         The HST instrument (ACS, WFC3)

        detector           The HST detector
                               ACS: HRC
                               WFPC2: PC1
                               WFC3: UVIS, IR

        y_ctr              The center of the PSF in the Y dimension on the
                           image plane

        x_ctr              The center of the PSF in the X dimension on the
                           image plane

        filter             The filter name (case irrelevant)

        psf_size_pixels    The desired size of the returned PSF in pixels
                           (the PSF is square)

        fov                The camera FOV in arcsec used to create the PSF

        subsample_amt      The amount of subsampling desired (1-10)
                              [ACS and WFC3 ONLY!]
                           NOTE: Subsampling > 1 means the charge diffusion
                                 has not been done!

        jitter_x/y/z       The amount of jitter (in mas) to apply to the PSF;
                           z is the angle

        force_run_tinytim  If True, rerun TinyTim even if there is a cached
                           version.

        Return:

        The PSF as an array of size (psf_size_pixels, psf_size_pixels)
        The convolution matrix required for charge diffusion if
        subsample_amt>1, or None
        """

        if instrument == 'ACS':
            assert detector == 'HRC'
        elif instrument == 'WFPC2':
            assert detector == 'PC1'
        elif instrument == 'WFC3':
            assert detector == 'UVIS' or detector == 'IR'
        else:
            assert False

        min_fov = None
        for fov_size, pixel_size in RETURNED_SIZES[(instrument, detector)]:
            if pixel_size > psf_size_pixels:
                min_fov = fov_size
                break
        assert min_fov is not None

        if min_fov*.7 > fov: # Give a little slack for the distortion
            print('WARNING: OVERRIDING SPECIFIED FOV', fov, end=' ')
            print('BECAUSE REQUESTED PSF IS TOO BIG')
            print('NEW FOV', min_fov, 'PSF SIZE', psf_size_pixels)
            fov = min_fov

        x_ctr = int(x_ctr)
        y_ctr = int(y_ctr)
        filter = filter.upper()

        if jitter_x != 0. and jitter_y != 0.:
            fits_base = ('%s_%s_%s_%04d_%04d_%04.2f_%02d_%06.2f_%06.2f_'
                         '%06.2f_00.fits' % (instrument, detector, filter,
                                             y_ctr, x_ctr,
                                             fov, subsample_amt,
                                             jitter_x, jitter_y, jitter_z))
        else:
            fits_base = ('%s_%s_%s_%04d_%04d_%4.2f_%02d_00.fits' %
                             (instrument, detector, filter, y_ctr, x_ctr,
                              fov, subsample_amt))
        fits_filename = path_join(PSF_CACHE_DIR, fits_base)
        if not os.path.exists(fits_filename) or force_run_tinytim:
            assert psf_size_pixels % 2 == 1 # Must be odd
            orig_cwd = os.getcwd()
            os.chdir(TINY_TIM_DIR)
            psf_filename = 'temp_psf' + str(os.getpid()) + '_'
            full_psf_filename = psf_filename + '00_psf.fits'
            temp_fits_filename = psf_filename + '00.fits'
            params_filename = 'params_input' + str(os.getpid()) + '.txt'
            temp_filename = 'temp' + str(os.getpid()) + '.in'

            params_fp = open(params_filename, 'w')

            if instrument == 'ACS' and detector == 'HRC':
                print(16, file=params_fp)               # ACS HRC
            elif instrument == 'WFPC2' and detector == 'PC1':
                print(6, file=params_fp)                # WFPC2 PC1
            elif instrument == 'WFC3' and detector == 'UVIS':
                print(22, file=params_fp)               # WFC3 UVIS
                if y_ctr > 2047:
                    print(2, file=params_fp)            # UVIS chip (1 or 2)
                    y_ctr -= 2048
                else:
                    print(1, file=params_fp)            # UVIS chip (1 or 2)
            elif instrument == 'WFC3' and detector == 'IR':
                print(23, file=params_fp)               # WFC3 IR

            print(x_ctr, y_ctr, file=params_fp)     # PSF location on detector
            if filter.lower() == 'fqch4n15' and instrument == 'WFPC2':
                # This filter is not in the tinytim database
                print('mono', file=params_fp)       # Monochromatic
                print(620, file=params_fp)          # nm
            elif filter.lower() == 'fqch4p15' and instrument == 'WFPC2':
                # This filter is not in the tinytim database
                print('mono', file=params_fp)       # Monochromatic
                print(893, file=params_fp)          # nm
            else:
                print(filter, file=params_fp)           # Filter
                if filter[-1].lower() != 'n':
                    # Only wide-band filters require an input spectrum
                    print(1, file=params_fp)            # Spectrum # (stars)
                    print(11, file=params_fp)           # G2V - our Sun
            print(fov, file=params_fp)  # PSF size in arcsec
            if instrument == 'WFPC2':
                if subsample_amt == 1:
                    print('N', file=params_fp)          # Subsample NO
                else:
                    print('Y', file=params_fp)          # Subsample YES
                    print(subsample_amt, file=params_fp)
            print(0, file=params_fp)                # Focus offset
            print(psf_filename, file=params_fp)     # PSF filename
            params_fp.close()

            if output_dev_null:
                redir = ' > ' + DEV_NULL
            else:
                print('*** PARAMS FOR TINY1 ***')
                os.system('cat '+params_filename)
                redir = ''
            if jitter_x != 0. and jitter_y != 0.:
                os.system('./tiny1 ' + temp_filename + ' major='+str(jitter_x) +
                          ' minor='+str(jitter_y) + ' angle='+str(jitter_z) +
                          ' < ' + params_filename + redir)
            else:
                os.system('./tiny1 ' + temp_filename + ' < ' + params_filename +
                          ' > ' + DEV_NULL)

            os.system('./tiny2 ' + temp_filename + redir)

            if instrument == 'ACS' or instrument == 'WFC3':
                if not output_dev_null:
                    print('*** TINY3 SUB='+str(subsample_amt)+' ***')
                os.system('./tiny3 ' + temp_filename + ' SUB=' +
                          str(subsample_amt) + redir)
                os.unlink(psf_filename + '.tt3')
                os.unlink(full_psf_filename)

            if not os.path.exists(temp_fits_filename):
                print('RUN_TINYTIM: ERROR CREATING FITS FILE', psf_filename)
                assert False

            psf_file = pyfits.open(temp_fits_filename)
            psf_file.writeto(fits_filename, overwrite=True)
            psf_file.close()

            os.unlink(temp_fits_filename)
            os.unlink(params_filename)
            os.unlink(temp_filename)

            os.chdir(orig_cwd)

        psf_file = pyfits.open(fits_filename)
        psf_data = psf_file[0].data
        psf_file.close()

        diffusion_matrix = None
        if subsample_amt > 1 and (instrument == 'ACS' or instrument == 'WFC3'):
            # Subsample > 1 requires later convolution for charge diffusion
            # We need to extract the convolution matrix from the header
            # comments
            comment_list = psf_file[0].header['COMMENT']
            diffusion_matrix = np.empty((3,3))
            for diffusion_y in range(3):
                comment_str = comment_list[diffusion_y+3].strip()
                comment_vals = comment_str.split(' ')
                for diffusion_x in range(3):
                    diffusion_matrix[diffusion_y, diffusion_x] = \
                                float(comment_vals[diffusion_x])

        psf_size = psf_data.shape
        psf_size_pixels *= subsample_amt
        psf_halfsize = psf_size_pixels//2
        if (psf_halfsize > psf_size[0]//2 or
            psf_halfsize > psf_size[1]//2):
            print('FATAL ERROR: TINYTIM returned a PSF smaller', psf_size, 'than we wanted', psf_halfsize)
            assert False
        psf_data = psf_data[psf_size[1]//2-psf_halfsize:
                            psf_size[1]//2+psf_halfsize+1,
                            psf_size[0]//2-psf_halfsize:
                            psf_size[0]//2+psf_halfsize+1]

        return psf_data, diffusion_matrix

    #===============================================================================
    # Instrument-independent private routines
    #===============================================================================

    def _cache_psf(self, min_size, **kwargs):
        max_xy = max(min_size, self.rect_size_guess)
        max_xy = (max_xy//2)*2+1 # Make it odd
        if self.cached_psf_size is None or max_xy > self.cached_psf_size:
            psf, diffusion_matrix = HSTPSF.run_tinytim(self.instrument,
                                                       self.detector,
                                                       self.line, self.sample,
                                                       self.filter,
                                                       max_xy, self.fov,
                                                       self.subsample,
                                                       self.jitter_x,
                                                       self.jitter_y,
                                                       self.jitter_z,
                                                       **kwargs)
            self.cached_psf_size = psf.shape[0]
            self.cached_psf = psf
            self.cached_diffusion_matrix = diffusion_matrix
            assert psf.shape[0] % 2 == 1 # Odd
            assert psf.shape[1] % 2 == 1 # Odd
            self.psf_zero_offset = psf.shape[0]//2
            self.cached_pixelated_psf = None

    def _cache_pixelation(self, offset, **kwargs):
        spline_order = 3
        if 'spline_order' in kwargs:
            spline_order = kwargs['spline_order']

        # We need (0,0) to be the upper left corner of the pixel
        offset = (offset[0]-0.5, offset[1]-0.5)

        if (self.cached_offset == offset and
            self.cached_spline_order == spline_order and
            self.cached_pixelated_psf is not None):
            return

        self.cached_offset = offset
        self.cached_spline_order = spline_order

        subsample_psf_size = self.cached_psf.shape[0]
        psf_size = subsample_psf_size // self.subsample
        # This is NOT the same as subsample_psf_size!
        psf_size_resampled = psf_size * self.subsample

        # First: Make a new subsampled PSF from the existing subsampled PSF
        # but offset by offset_y,offset_x using linear interpolation. This
        # includes doing motion blur.

        # The X and Y indices in the subsampled PSF
        native_indices = (np.arange(float(subsample_psf_size))-
                          self.psf_zero_offset) / self.subsample

        # Create a linear interpolation spline
        spline = RectBivariateSpline(native_indices, native_indices,
                                     self.cached_psf,
                                     kx=spline_order, ky=spline_order)

        num_steps = int(max(abs(self.movement[0])/self.movement_granularity,
                            abs(self.movement[1])/self.movement_granularity))

        if num_steps == 0:
            step_y = 0.
            step_x = 0.
        else:
            step_y = self.movement[0] / num_steps
            step_x = self.movement[1] / num_steps

        total_rect = None

        for step in range(num_steps+1):
            y = offset[0] + step_y*(step-num_steps/2.)
            x = offset[1] + step_x*(step-num_steps/2.)

            # The X and Y indices in our new offset coordinate system in the
            # subsampled PSF
            desired_y_indices = native_indices - y
            desired_x_indices = native_indices - x
            desired_y_indices = np.repeat(desired_y_indices,
                                          psf_size_resampled)
            desired_x_indices = np.tile(desired_x_indices,
                                        psf_size_resampled)

            interp_psf = spline.ev(desired_y_indices, desired_x_indices)
            interp_psf = interp_psf.reshape((psf_size_resampled,
                                             psf_size_resampled))

            if total_rect is None:
                total_rect = interp_psf
            else:
                total_rect += interp_psf

        total_rect /= float(num_steps+1)

        # Second: Downsample the offset subsampled PSF by taking the sum over
        # adjacent pixels
        psf = total_rect.reshape([psf_size, self.subsample, psf_size,
                                  self.subsample])
        psf = np.sum(psf, axis=3)
        psf = np.sum(psf, axis=1)

        if self.subsample > 1 and self.do_subsample_diffusion:
            # We have to do the charge diffusion ourselves
            psf = scisig.convolve2d(psf, self.cached_diffusion_matrix)
        self.cached_pixelated_psf = psf
        self.pixelated_psf_zero_offset = psf.shape[0] // 2


    #===============================================================================
    # Public routines
    #===============================================================================

    def eval_point(self, point, scale=1., base=0.):
        """Evaluate the PSF at a single, fractional, point.

        (0,0) is the center of the PSF and x and y may be negative.

        Input:
            coord       the coordinate (y,x) at which to evaluate the PSF.
            scale       a scale factor to apply to the resulting PSF.
            base        a scalar added to the resulting PSF.
        """

        assert False

    def eval_pixel(self, coord, offset=(0.,0.), scale=1., base=0., **kwargs):
        """Evaluate the PSF integrated over an entire integer pixel.

        The returned array has the PSF offset from the center by
        (offset_y,offset_x). An offset of (0,0) places the PSF in the upper
        left corner of the center pixel while an offset of (0.5,0.5)
        places the PSF in the center of the center pixel.

        NOTE: The pixel value includes the effect of motion blur.

        Input:
            coord       the integer coordinate (y,x) at which to evaluate the
                        PSF.
            offset      the amount (offset_y,offset_x) to offset the center
                        of the PSF.
            scale       a scale factor to apply to the resulting PSF.
            base        a scalar added to the resulting PSF.
        """

        self._cache_psf(max(abs(y)*2+1, abs(x)*2+1), **kwargs)
        self._cache_pixelation(offset, **kwargs)

        index_y = coord[0]+self.pixelated_psf_zero_offset
        index_x = coord[1]+self.pixelated_psf_zero_offset

        return self.cached_pixelated_psf[index_y, index_x] * scale + base

    def eval_rect(self, rect_size, offset=(0.,0.), scale=1., base=0., **kwargs):
        """Create a rectangular pixelated PSF.

        This is done by evaluating the PSF function from:
            [-rect_size_y//2:rect_size_y//2] and
            [-rect_size_x//2:rect_size_x//2]

        The returned array has the PSF offset from the center by
        (offset_y,offset_x). An offset of (0,0) places the PSF in the upper
        left corner of the center pixel while an offset of (0.5,0.5)
        places the PSF in the center of the center pixel.

        Input:
            rect_size   the size of the rectangle (rect_size_y,rect_size_x)
                        of the returned PSF. Both dimensions must be odd.
            offset      the amount (offset_y,offset_x) to offset the center
                        of the PSF from the upper left corner of the center
                        pixel of the rectangle.
            scale       a scale factor to apply to the resulting PSF.
            base        a scalar added to the resulting PSF.
        """

        rect_size_y, rect_size_x = rect_size

        assert rect_size_y % 2 == 1 # Odd
        assert rect_size_x % 2 == 1 # Odd

        half_rect_size_y = rect_size_y // 2
        half_rect_size_x = rect_size_x // 2

        self._cache_psf(max(rect_size_x, rect_size_y), **kwargs)
        self._cache_pixelation(offset, **kwargs)

        rect = self.cached_pixelated_psf[self.pixelated_psf_zero_offset-half_rect_size_y:
                                         self.pixelated_psf_zero_offset+half_rect_size_y+1,
                                         self.pixelated_psf_zero_offset-half_rect_size_x:
                                         self.pixelated_psf_zero_offset+half_rect_size_x+1]

        rect = np.maximum(rect, 0.)

        return rect * scale + base
