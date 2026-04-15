################################################################################
# tests/conftest.py
################################################################################

"""Shared pytest fixtures for psfmodel tests."""

import pytest

from psfmodel.gaussian import GaussianPSF


@pytest.fixture
def default_psf() -> GaussianPSF:
    """Return a :class:`GaussianPSF` built with default constructor arguments."""

    return GaussianPSF()


@pytest.fixture
def symmetric_psf() -> GaussianPSF:
    """Return a :class:`GaussianPSF` with equal ``sigma_y`` and ``sigma_x``."""

    return GaussianPSF(sigma=(1.0, 1.0))


@pytest.fixture
def asymmetric_psf() -> GaussianPSF:
    """Return a :class:`GaussianPSF` with distinct ``sigma_y`` and ``sigma_x``."""

    return GaussianPSF(sigma=(2.0, 3.0))
