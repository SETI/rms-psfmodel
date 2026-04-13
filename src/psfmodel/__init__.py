"""Point-spread function (PSF) models and fitting for image data.

This package defines an abstract :class:`~psfmodel.psf.PSF` API and concrete
implementations such as :class:`~psfmodel.gaussian.GaussianPSF` for analytic
Gaussians, pixel integration, background handling, and astrometric fitting.

The public surface re-exports ``PSF`` and ``GaussianPSF`` (see ``__all__``).
Additional modules (for example instrument-specific PSFs) are imported from
their submodules when needed. A :class:`logging.NullHandler` is attached to the
package logger so library logging is opt-in for applications.
"""

import logging

from .gaussian import GaussianPSF
from .psf import PSF

__all__ = ['PSF', 'GaussianPSF']

logging.getLogger(__name__).addHandler(logging.NullHandler())
