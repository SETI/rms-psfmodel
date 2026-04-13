################################################################################
# characterize_gauss_fit/__init__.py
################################################################################

"""Characterization tool for Gaussian PSF fitting accuracy.

This package systematically explores the input parameter space of the Gaussian
PSF fitter and produces plots and tabular data showing how fitting accuracy
depends on subimage size, subpixel offset, PSF shape, parameter constraints,
background conditions, noise, and bad pixel rejection.

Entry point: ``characterize_gauss_fit`` command or ``python -m characterize_gauss_fit``.
"""

__all__: list[str] = []
