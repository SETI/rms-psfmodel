"""Sphinx configuration for the standalone PSF fitter performance report."""

project = "rms-psfmodel Performance Report"
copyright = "2026, SETI Institute"
author = "SETI Institute"
release = ""

extensions = [
    "sphinx.ext.intersphinx",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
