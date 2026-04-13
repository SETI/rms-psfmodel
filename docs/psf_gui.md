# `psf_gui` -- Interactive PSF Explorer

## Overview

`psf_gui` is an optional interactive desktop application for visually exploring
Gaussian PSF models and fitting results. It is built with Python's standard
Tkinter library and ships as part of the `rms-psfmodel` package.

The GUI lets you interactively adjust PSF parameters (sigma, angle, offset, scale,
background) and immediately see the resulting pixel-integrated image alongside
the fitting residuals. It is intended as a diagnostic and educational tool rather
than a batch analysis tool; for systematic characterization use
`characterize_gauss_fit` instead.

## Prerequisites

`psf_gui` requires a working Tcl/Tk installation. On most systems this is
provided by the `python3-tk` package or equivalent:

| Platform | Command |
|----------|---------|
| Debian / Ubuntu | `sudo apt install python3-tk` |
| Fedora / RHEL | `sudo dnf install python3-tkinter` |
| macOS (Homebrew Python) | Tk is bundled; no extra install needed |
| macOS (python.org installer) | Tk is bundled; no extra install needed |
| Windows | Tk is bundled with the official Python installer |

Verify your installation by running:

```sh
python -c "import tkinter; tkinter._test()"
```

A small test window should appear.

## Running

After installing `rms-psfmodel`, start the GUI with:

```sh
psf_gui
```

Or, if the `src` directory is on `PYTHONPATH` (development mode):

```sh
python -m psf_gui
```

## Features

- **Real-time PSF rendering** -- adjusting any slider immediately re-renders
  the pixel-integrated Gaussian patch.
- **Background subtraction** -- polynomial background fitting with configurable
  degree and ignore-center region.
- **Noise simulation** -- add Gaussian noise to the synthetic PSF image and
  observe the effect on fitting accuracy.
- **Residual display** -- view the difference between the fitted model and the
  data after position optimisation.
- **Parameter readout** -- fitted position, sigma, angle, and scale are
  displayed numerically alongside their true injected values.
