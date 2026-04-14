# `characterize_gauss_fit` -- PSF Fitting Characterization Tool

## Overview

`characterize_gauss_fit` is a standalone command-line program that
systematically measures the accuracy of the Gaussian PSF fitter across a
large, configurable parameter space. It generates synthetic PSF images with
known ground-truth parameters, fits them, and reports how closely the fitter
recovers position, sigma, angle, and scale.

Eight focused *studies* each vary a small set of parameters while holding
others fixed. Every study produces:

- **PNG plots** for visual inspection (heatmaps, line plots with error bands,
  grouped bar charts).
- **`trials.csv`** -- one row per trial with all input parameters and all
  result metrics, loadable by pandas or any data-analysis tool.
- **`summary.json`** -- aggregate statistics per parameter group plus the
  exact configuration used, formatted for AI-assisted analysis.

## Installation

Install the extra dependencies required by this tool:

```sh
pip install rms-psfmodel[characterize]
```

This adds `matplotlib` and `pyyaml` to your environment.

## Quick Start

Run all studies with default settings:

```sh
characterize_gauss_fit
```

Run a quick smoke test across all studies using the bundled reduced-grid
configuration (completes in roughly 30--120 seconds):

```sh
characterize_gauss_fit --config src/characterize_gauss_fit/test_config.yaml
```

Run a single study:

```sh
characterize_gauss_fit --study box_vs_sigma
```

Run with a custom override file and parallel workers:

```sh
characterize_gauss_fit --config my_config.yaml --num-workers 8
```

List all available study names:

```sh
characterize_gauss_fit --list-studies
```

Copy the built-in default configuration to a local file for editing:

```sh
characterize_gauss_fit --copy-default-config-to my_config.yaml
```

Copy the built-in reduced-grid test configuration to a local file:

```sh
characterize_gauss_fit --copy-test-config-to test_config.yaml
```

Copy the built-in high-resolution configuration to a local file:

```sh
characterize_gauss_fit --copy-hires-config-to hires_config.yaml
```

## CLI Reference

```text
usage: characterize_gauss_fit [--config FILE] [--study NAME] [--output-dir DIR]
                               [--num-workers N] [--list-studies]
                               [--copy-default-config-to FILE]
                               [--copy-test-config-to FILE]
                               [--copy-hires-config-to FILE] [--verbose]

Options:
  --config FILE               Path to a YAML override file merged onto
                              built-in defaults.
  --study NAME                Run only this study (repeatable). Default: all
                              enabled studies. Use --list-studies to see names.
  --output-dir DIR            Override the output directory from the config
                              file.
  --num-workers N             Number of parallel worker processes. 1 (default)
                              runs sequentially in the main process. >1 uses
                              concurrent.futures.ProcessPoolExecutor.
  --list-studies              Print available study names and exit.
  --copy-default-config-to FILE
                              Write the built-in default configuration to FILE
                              and exit. No studies are run.
  --copy-test-config-to FILE  Write the built-in reduced-grid test
                              configuration to FILE and exit. No studies are
                              run.
  --copy-hires-config-to FILE Write the built-in high-resolution configuration
                              to FILE and exit. No studies are run.
  --verbose, -v               Enable DEBUG-level logging.
```

Exit code is 0 on success. Exit code 1 if any study raises an unhandled
exception; individual trial failures (fitter non-convergence) are not fatal
and are recorded as data.

## Configuration Reference

All parameters have built-in defaults. You can override any subset by
providing a YAML file with `--config`. The file is deep-merged onto the
defaults: scalar values and lists replace the default; nested dicts are
merged recursively.

### Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_dir` | string (path) | `./gauss_fit_results` | Root directory for all output files. |
| `num_workers` | int | `1` | Worker processes for parallel execution. |
| `noise_samples` | int | `50` | Default noise realisations for stochastic studies. |

### `fitting` section

Global defaults passed to `PSF.find_position`. Any study section may include
a `fitting` subsection to override these for that study only.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `bkgnd_degree` | int or null | `2` | Polynomial degree for background fitting. `null` = no background subtraction. |
| `bkgnd_ignore_center` | [int, int] | `[2, 2]` | Half-size of central region excluded from background fit (rows, cols). The excluded region is `(2*ny+1) x (2*nx+1)`. |
| `bkgnd_num_sigma` | float or null | `null` | Sigma-clipping threshold for background residuals. `null` = disabled. |
| `num_sigma` | float or null | `null` | Sigma-clipping threshold for PSF residuals (bad-pixel rejection). `null` = disabled. |
| `max_bad_frac` | float | `0.2` | Maximum fraction of pixels that can be masked before the fit is abandoned. |
| `allow_nonzero_base` | bool | `false` | Fit a constant base level in addition to the polynomial background. |
| `use_angular_params` | bool | `true` | Reparametrise fit variables as angles for bounded optimization. |
| `tolerance` | float | `1e-6` | Powell optimizer convergence tolerance. |
| `search_limit` | [float, float] | `[1.5, 1.5]` | Maximum allowed position offset from the starting point [y, x] in pixels. |
| `scale_limit` | float | `1000.0` | Maximum allowed PSF amplitude scale factor. |

### `generation` section

Controls the synthetic PSF image generator.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `scale` | float | `1.5` | PSF amplitude scale factor applied to the normalised Gaussian. |
| `base` | float | `0.0` | Additive base level on the clean PSF before background injection. |

### `studies` section

Each study has an `enabled` flag and its own parameter set. Per-study
`fitting` subsections override the global defaults for that study only.

---

#### `box_vs_sigma`

Study 1: How box size relative to PSF sigma affects fitting accuracy.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable this study. |
| `box_sizes` | list[int] | `[5,7,9,11,13,17,21,25,31]` | Odd box sizes to test. Each must be >= 5. |
| `sigmas` | list[float] | `[0.3,0.5,0.8,1.0,1.5,2.0,3.0,5.0]` | Symmetric PSF sigma values (pixels). |
| `offset` | [float, float] | `[0.25, 0.25]` | Sub-pixel offset [y, x] applied to the PSF centre. |
| `angle` | float | `0.0` | PSF rotation angle (radians). |
| `scale` | float | `1.0` | PSF amplitude scale factor (overrides `generation.scale`). |
| `fitting` | dict | `{bkgnd_degree: null}` | Per-study fitting overrides. |

---

#### `subpixel_offset`

Study 2: How fractional pixel position introduces systematic bias.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable this study. |
| `offset_steps` | int | `11` | Number of evenly-spaced steps in each axis. |
| `offset_range` | [float, float] | `[0.0, 0.5]` | Range of offset values. Values outside [0, 0.5] are redundant by symmetry. |
| `sigmas` | list[float] | `[0.5, 1.0, 2.0]` | Sigma values used as a panel variable. |
| `box_size` | int | `21` | Fixed box size. |
| `angle` | float | `0.0` | Fixed PSF angle. |
| `fitting` | dict | `{bkgnd_degree: null}` | Per-study fitting overrides. |

---

#### `min_detectable_offset`

Study 3: Minimum offset delta reliably recoverable vs. PSF sigma and noise.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable this study. |
| `delta_offsets` | list[float] | `[0.001,...,0.5]` | Offset deltas to test (pixels). |
| `sigmas` | list[float] | `[0.3,...,3.0]` | Symmetric sigma values. |
| `box_size` | int | `21` | Fixed box size. |
| `noise_samples` | int | `50` | Noise realisations per stochastic condition. |
| `snr_values` | list[float] | `[50.0, 100.0, 500.0]` | SNR values (peak / noise RMS) to test. |
| `include_noiseless` | bool | `true` | Also run a noiseless trial (numerical precision floor). |
| `fitting` | dict | `{bkgnd_degree: null}` | Per-study fitting overrides. |

---

#### `sigma_asymmetry_angle`

Study 4: Sigma asymmetry and angle recovery for elongated, rotated PSFs.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable this study. |
| `sigma_ratios` | list[float] | `[0.25,...,4.0]` | Ratios sigma_y / sigma_x. |
| `angle_steps` | int | `13` | Evenly-spaced angles from 0 to pi (inclusive). Must be >= 2. |
| `sigma_x_values` | list[float] | `[0.5, 1.0, 2.0]` | sigma_x values (panel variable). |
| `box_size` | int | `25` | Fixed box size. |
| `offset` | [float, float] | `[0.25, 0.25]` | Fixed sub-pixel offset. |
| `fitting` | dict | `{bkgnd_degree: null}` | Per-study fitting overrides. |

---

#### `constraint_modes`

Study 5: Effect of fixing vs. floating sigma/angle on all output metrics.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable this study. |
| `sigma_error_fractions` | list[float] | `[0.0, 0.2, 0.5]` | Fractional errors applied to sigma when fixed. 0.0 = correct value; 0.2 = fixed at 1.2x true. |
| `angle_error_rad` | float | `0.3` | Absolute angle error (radians) when angle is fixed incorrectly. |
| `psf_shapes` | list[{sigma: [y,x], angle: rad}] | 3 shapes | PSF shapes to test. Each entry must have `sigma` (list of two floats) and `angle` (float in [0, pi]). |
| `box_size` | int | `21` | Fixed box size. |
| `offset` | [float, float] | `[0.25, 0.25]` | Fixed sub-pixel offset. |
| `scale` | float | `1.5` | PSF amplitude scale factor. |

---

#### `background`

Study 6: How injected background and fitting model choice interact.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable this study. |
| `background_amplitudes` | list[float] | `[0.01, 0.1, 0.5]` | Background amplitude as fraction of PSF peak. |
| `bkgnd_degrees` | list[int] | `[0, 1, 2]` | Polynomial degrees to use when fitting the background. |
| `bkgnd_degrees_with_null` | bool | `true` | Also test `bkgnd_degree=null` (no fitting). |
| `bkgnd_ignore_centers` | list[[int,int]] | `[[1,1],[2,2],[4,4]]` | `bkgnd_ignore_center` values to test. |
| `background_types` | list[str] | `[none,constant,linear,quadratic,noisy_constant]` | Background types to inject. |
| `box_size` | int | `21` | Fixed box size. |
| `sigma` | [float, float] | `[1.0, 1.0]` | Fixed PSF sigma [y, x]. |
| `offset` | [float, float] | `[0.25, 0.25]` | Fixed sub-pixel offset. |

Valid `background_types` values:

| Value | Description |
|-------|-------------|
| `none` | No background injected. |
| `constant` | Flat pedestal at `amplitude * PSF_peak`. |
| `linear` | Tilted plane (linear gradient). |
| `quadratic` | Bowl-shaped quadratic surface. |
| `noisy_constant` | Flat pedestal plus Gaussian noise at 0.5x the main noise level. |

---

#### `noise_sensitivity`

Study 7: Position, sigma, and scale accuracy as a function of SNR.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable this study. |
| `snr_log_range` | [float, float] | `[0.5, 3.5]` | Log10 range for SNR (min, max). |
| `snr_steps` | int | `15` | Number of log-spaced SNR points. |
| `sigmas` | list[float] | `[0.5, 1.0, 2.0]` | Sigma values (panel variable). |
| `noise_samples` | int | `50` | Noise realisations per (SNR, sigma) point. |
| `box_size` | int | `21` | Fixed box size. |

---

#### `hot_pixel_rejection`

Study 8: Effectiveness of `num_sigma` bad-pixel rejection.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable this study. |
| `num_hot_pixels` | list[int] | `[0,1,3,5,10]` | Number of hot pixels to inject. |
| `num_sigma_values` | list[float] | `[2.0, 3.0, 5.0]` | `num_sigma` rejection thresholds to test. |
| `num_sigma_with_null` | bool | `true` | Also test `num_sigma=null` (rejection disabled). |
| `hot_amplitudes` | list[float] | `[5.0, 20.0, 100.0]` | Hot pixel amplitude as multiple of PSF peak. |
| `noise_samples` | int | `20` | Noise realisations per combination (hot pixel positions randomised). |
| `snr` | float | `100.0` | Background Gaussian noise SNR (peak / noise RMS). |
| `box_size` | int | `21` | Fixed box size. |
| `sigma` | [float, float] | `[1.0, 1.0]` | Fixed PSF sigma [y, x]. |
| `offset` | [float, float] | `[0.25, 0.25]` | Fixed sub-pixel offset. |

---

## Output Format Reference

Each study writes its results to `{output_dir}/{study_name}/`.

### `trials.csv`

One row per trial. Columns:

| Column | Description |
|--------|-------------|
| `study` | Study name string. |
| `box_size` | Subimage side length (pixels). |
| `sigma_y_true`, `sigma_x_true` | True PSF sigma values. |
| `angle_true` | True rotation angle (radians). |
| `offset_y_true`, `offset_x_true` | True sub-pixel offset. |
| `scale_true` | True amplitude scale factor. |
| `fit_sigma_y`, `fit_sigma_x` | Value sigma was fixed to (empty = floated). |
| `fit_angle` | Value angle was fixed to (empty = floated). |
| `background_type` | Injected background type string. |
| `background_amplitude` | Injected background amplitude (fraction of peak). |
| `noise_rms` | Additive Gaussian noise standard deviation. |
| `num_hot_pixels` | Number of hot pixels injected. |
| `hot_pixel_amplitude` | Hot pixel amplitude (multiple of PSF peak). |
| `bkgnd_degree` | Background fitting polynomial degree (empty = null). |
| `num_sigma` | Bad-pixel rejection threshold (empty = null). |
| `bkgnd_ignore_center_y`, `bkgnd_ignore_center_x` | Ignore-center half-sizes. |
| `converged` | `true` or `false`. |
| `pos_err_y`, `pos_err_x` | Signed position errors (fitted - true). |
| `pos_err` | Euclidean position error. |
| `sigma_y_fit`, `sigma_x_fit` | Fitted sigma values (empty if fixed). |
| `angle_fit` | Fitted angle (empty if fixed). |
| `scale_fit` | Fitted scale factor. |
| `sigma_y_err`, `sigma_x_err` | Relative sigma errors (fit - true) / true. |
| `angle_err` | Absolute angle error in radians. |
| `scale_err` | Relative scale error (fit - true) / true. |

**Conventions:**
- Non-applicable fields are empty strings (e.g. `sigma_y_fit` when sigma was fixed).
- Failed / non-converged trials have `NaN` for all error fields.
- Load with `pandas.read_csv(..., na_values=['NaN', ''])`.

> **Naming asymmetry warning.** Two pairs of columns have similar names but
> opposite roles:
>
> - `fit_sigma_y` / `fit_sigma_x` are **inputs** (the value sigma was
>   *constrained to* before fitting; empty string when sigma was left to float).
> - `sigma_y_fit` / `sigma_x_fit` are **outputs** (the sigma value *returned
>   by the fitter*; empty string when sigma was fixed and not fitted).
>
> In short: `fit_*` columns describe what you told the fitter; `*_fit` columns
> describe what the fitter found.
>
> Example (Python / pandas):
>
> ```python
> # Trials where sigma_y was constrained (input column non-empty):
> constrained = df[df['fit_sigma_y'].notna()]
> # Trials where sigma_y was floated and a fitted value was returned:
> fitted = df[df['sigma_y_fit'].notna()]
> ```

### `summary.json`

```text
{
  "study": "box_vs_sigma",
  "total_trials": 72,
  "converged_trials": 68,
  "convergence_rate": 0.944,
  "overall": { ... aggregate stats ... },
  "groups": [
    {
      "box_size": 5,
      "sigma": 0.3,
      "n_trials": 1,
      "n_converged": 1,
      "convergence_rate": 1.0,
      "pos_err_mean": 0.00123,
      "pos_err_std": null,
      "sigma_y_err_mean": 0.005,
      "scale_err_mean": 0.002,
      "angle_err_mean": null
    }
  ],
  "config_used": { ... full config dict ... }
}
```

`null` in JSON corresponds to `None` in Python (not enough data to compute,
or metric not applicable). The `config_used` field contains the exact
configuration that produced these results for full reproducibility.

### AI / automated analysis

To load all studies for analysis:

```python
import json
import pathlib
import pandas as pd

results_dir = pathlib.Path('./gauss_fit_results')
dfs = []
for csv_file in results_dir.glob('*/trials.csv'):
    dfs.append(pd.read_csv(csv_file, na_values=['NaN', '']))
all_results = pd.concat(dfs, ignore_index=True)
```

---

## Study Descriptions

### Study 1: Box Size vs. Sigma (`box_vs_sigma`)

**Question:** How large must the subimage be relative to the PSF width for
accurate fitting?

Sweeps box size and PSF sigma on a 2-D grid. The PSF always fills the entire
image (`eval_rect` size == `box_size`). Background is disabled so only the
intrinsic truncation effect is measured. Sigma is left to float.

**Key plots:** Heatmaps of log10(position error), log10(sigma error), and
log10(scale error) as functions of box size (rows) and sigma (columns). Cells
where the fitter did not converge are shown in grey.

**Interpretation:** Expect a sharp accuracy cliff when
`box_size < 4 * sigma + 1`. Small sigmas are well-fitted even in tiny boxes;
large sigmas in small boxes truncate most of the PSF flux.

---

### Study 2: Subpixel Offset (`subpixel_offset`)

**Question:** Does the fractional pixel position of the PSF centre introduce
systematic bias?

Sweeps offset_y and offset_x from 0 to 0.5 pixels in a grid. The range
[0, 0.5] is sufficient by symmetry. Three sigma panel values are tested.

**Key plots:** 2-D heatmap of position error vs. (offset_y, offset_x) for
each sigma. Line plot of error vs. offset_x at fixed offset_y.

**Interpretation:** The fitter may exhibit a small systematic oscillation at
the pixel-period scale due to aliasing. Very small sigmas show larger absolute
error because the PSF peak is narrower than a pixel.

---

### Study 3: Minimum Detectable Offset (`min_detectable_offset`)

**Question:** How small an offset delta can the fitter reliably recover as a
function of PSF size and noise?

Tests a log-spaced grid of offset deltas, applied purely in the X direction,
for each (sigma, SNR) combination. Recovery fraction is defined as the fraction
of trials where `|pos_err| < delta/2` (within 50% of the true offset).

**Key plots:** Log-log line plot of mean position error vs. delta for each
sigma (one panel per SNR). Recovery fraction heatmap (sigma vs. delta, one
plot per SNR level).

**Interpretation:** The precision floor in the noiseless case reveals
numerical resolution limits. Noise raises the floor to approximately
`sigma / SNR`. Recovery fraction drops below 0.5 near the precision floor.

---

### Study 4: Sigma Asymmetry and Angle (`sigma_asymmetry_angle`)

**Question:** How well are elongated, rotated PSFs recovered?

Sweeps sigma_ratio (sigma_y / sigma_x) and rotation angle for several sigma_x
values. All parameters float. For circular PSFs (ratio ~= 1.0), angle is
degenerate and angle error is not meaningful -- those cells show NaN.

**Key plots:** Heatmaps of position error, angle error, and sigma_y error as
functions of (ratio, angle), one panel per sigma_x.

**Interpretation:** Near-circular PSFs (ratio near 1) have degenerate angle;
the fitter can converge to any angle without affecting position accuracy.
Highly elongated PSFs at edge-case angles (0, pi/2, pi) may have increased
error due to the optimizer landscape.

---

### Study 5: Constraint Modes (`constraint_modes`)

**Question:** How does fixing vs. floating sigma and angle affect position,
scale, sigma, and angle accuracy?

Tests eight constraint configurations on three PSF shapes. Reports all four
accuracy metrics.

**Key plots:** 4-panel grouped bar chart: position error, relative scale
error, relative sigma_y error, and absolute angle error, grouped by PSF shape.

**Interpretation:** Correctly fixing sigma reduces fitting degrees of freedom
and generally improves position accuracy at the cost of sigma recovery
information. Incorrectly fixed sigma can bias all metrics. Floating angle on
circular PSFs wastes degrees of freedom but rarely hurts position accuracy.

---

### Study 6: Background Conditions (`background`)

**Question:** How do injected background and fitting model choice interact?

Combines five background types with four fitting-degree options and three
ignore-center sizes.

**Key plots:** Heatmap matrix -- rows = injected background type,
columns = fitting degree. One heatmap per (amplitude, ignore-center)
combination.

**Interpretation:** Fitting a constant background (degree 0) is generally
sufficient for constant injected backgrounds. Higher-degree backgrounds require
matching or higher fitting degrees. Using `bkgnd_degree=null` with any
non-zero background will show degraded accuracy.

---

### Study 7: Noise Sensitivity (`noise_sensitivity`)

**Question:** At what SNR does fitting accuracy degrade?

Sweeps a log-spaced SNR range with multiple noise realisations per point.
Random offsets are used so position error reflects typical (not best-case)
accuracy.

**Key plots:** Line plots with shaded ±1 std bands: position error, sigma_y
error, sigma_x error, and scale error vs. SNR (log scale). One line per sigma
panel value.

**Interpretation:** All metrics improve roughly as 1/SNR in the noise-limited
regime. The saturation at high SNR reveals the floor set by optimizer
precision and pixel-integration discretisation.

---

### Study 8: Hot Pixel Rejection (`hot_pixel_rejection`)

**Question:** How effectively does `num_sigma` rejection handle hot pixels?

Varies the number of hot pixels (0 to 10), their amplitude (5x to 100x peak),
and the `num_sigma` rejection threshold. Multiple noise realisations randomise
hot pixel positions.

**Key plots:** Line plot per amplitude panel: X = number of hot pixels,
Y = mean position error, lines = `num_sigma` settings.

**Interpretation:** `num_sigma=null` shows the baseline degradation from
unrejected hot pixels. Aggressive rejection (low `num_sigma`) can mask real
PSF pixels near the core. `num_sigma=3` is typically a good balance.

---

## Bundled Configuration Files

Three YAML configurations ship alongside the package source. Each can be
copied to a local file for editing with the corresponding `--copy-*-to`
option.

### Default configuration (`defaults.yaml`)

The primary reference configuration. All parameters are set to sensible
defaults that provide a thorough survey of each study's parameter space.
Output is written to `./gauss_fit_results/`.

```sh
characterize_gauss_fit --copy-default-config-to my_config.yaml
```

### Reduced-grid test configuration (`test_config.yaml`)

Runs all eight studies with the smallest viable parameter grids so the
entire suite completes in roughly 30--120 seconds on a single core. Use it
to verify that all code paths execute after code changes.

```sh
characterize_gauss_fit --copy-test-config-to test_config.yaml
characterize_gauss_fit --config test_config.yaml
```

Output is written to `./gauss_fit_test/` by default.

| Study | Grid size | Approx. trials |
|-------|-----------|----------------|
| `box_vs_sigma` | 2 box sizes x 3 sigmas | 6 |
| `subpixel_offset` | 3 x 3 offset grid, 1 sigma | 9 |
| `min_detectable_offset` | 3 deltas x 2 sigmas x 2 noise conditions x 3 samples | ~18 |
| `sigma_asymmetry_angle` | 3 ratios x 3 angles x 1 sigma_x | 9 |
| `constraint_modes` | 5 modes x 2 PSF shapes | ~18 |
| `background` | 2 background types x 2 fitting degrees | 4 |
| `noise_sensitivity` | 4 SNR points x 1 sigma x 3 samples | 12 |
| `hot_pixel_rejection` | 2 num_hot x 1 threshold x 1 amplitude x 3 samples | 12 |

### High-resolution configuration (`hires_config.yaml`)

Runs all eight studies with denser parameter grids and larger
`noise_samples` counts compared with the defaults, while staying within the
same parameter ranges. The goal is smoother heatmaps, less noisy line plots,
and more nuanced detail at intermediate parameter values. Estimated runtime
is 10--30x longer than the default configuration; use `--num-workers` to
parallelise across CPU cores.

```sh
characterize_gauss_fit --copy-hires-config-to hires_config.yaml
characterize_gauss_fit --config hires_config.yaml --num-workers 8
```

Output is written to `./gauss_fit_hires/` by default.

| Study | Denser axes | Key changes vs. defaults |
|-------|-------------|--------------------------|
| `box_vs_sigma` | 12 box sizes, 13 sigmas | adds 15, 19, 41 px boxes; intermediate sigma values |
| `subpixel_offset` | 21 x 21 offset grid, 4 sigmas | 0.025 px step; adds sigma=1.5 |
| `min_detectable_offset` | 11 deltas, 4 SNR conditions, 200 samples | adds SNR=20 condition |
| `sigma_asymmetry_angle` | 10 ratios, 25 angle steps | 7.5 deg angular resolution |
| `constraint_modes` | 6 sigma-error fractions, 4 PSF shapes | adds 0.1, 0.3, 0.75 fractions |
| `background` | 7 amplitudes, degree=3, 4 ignore_center sizes | finer amplitude sweep |
| `noise_sensitivity` | 25 SNR points, 4 sigmas, 200 samples | adds sigma=1.5; 4x more samples |
| `hot_pixel_rejection` | 8 hot-pixel counts, 6 amplitudes, 4 thresholds, 50 samples | fills gaps in all axes |

### Obtaining any bundled file

All three copy commands write the file and exit immediately — no studies are
run. The destination path must not already exist.

```sh
characterize_gauss_fit --copy-default-config-to my_config.yaml
characterize_gauss_fit --copy-test-config-to test_config.yaml
characterize_gauss_fit --copy-hires-config-to hires_config.yaml
```

All parameters in any bundled config can be further overridden by combining
it with a second user config file or with CLI flags. For example, to run
only study 1 with the test grid:

```sh
characterize_gauss_fit --config test_config.yaml --study box_vs_sigma
```

---

## Example User Override File

To override a subset of parameters on top of the defaults, create a YAML
file containing only the keys you want to change. The following example
runs two studies with custom grids:

```yaml
output_dir: ./my_results

studies:
  box_vs_sigma:
    box_sizes: [7, 11, 21]
    sigmas: [0.5, 1.0, 2.0, 3.0]

  noise_sensitivity:
    snr_steps: 10
    noise_samples: 20
    sigmas: [0.5, 1.0, 2.0]

  subpixel_offset:
    enabled: false
  min_detectable_offset:
    enabled: false
  sigma_asymmetry_angle:
    enabled: false
  constraint_modes:
    enabled: false
  background:
    enabled: false
  hot_pixel_rejection:
    enabled: false
```

Run with:

```sh
characterize_gauss_fit --config my_overrides.yaml
```
