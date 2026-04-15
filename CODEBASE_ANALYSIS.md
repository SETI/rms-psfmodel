# Codebase Analysis: rms-psfmodel

## Summary

`rms-psfmodel` is a Python library for PSF (Point Spread Function) model fitting,
supporting analytic Gaussian PSFs and HST/TinyTim-based PSFs. The Gaussian path
(`gaussian.py`, `psf.py`) is moderately well-structured, partially typed, and has
reasonable test coverage for its core math. However, the codebase has several critical
numerical bugs, a complete absence of the `logging` module (all diagnostic output uses
bare `print()`), significant dead and commented-out code, and an `hst.py` module that is
entirely untested, uses `os.system()` for shell execution, and reads environment variables
at import time — crashing any user who does not have `TINYTIM` and `PSF_CACHE_DIR` set.
Test coverage is only ~45% overall (68% for `psf.py`, 0% for `hst.py`).

**Top priorities:**

1. Fix the numerical correctness bugs in `gaussian_integral_1d` and
   `background_gradient_fit` (data-integrity risk).
2. Replace all `print()` debugging with `logging` throughout the library.
3. Address the `hst.py` import-time crash, `os.system()` shell injection, and `assert`
   misuse.

---

## 1. Algorithms and Numerical Accuracy

### 1.1 CRITICAL — `gaussian_integral_1d` uses `np.abs()` on signed integral (array path)

- **Evidence:** `gaussian.py` lines 267–281
- **Issue:** The scalar path correctly computes
  `erf(xmax) - erf(xmin)`, which is signed (positive when `xmax > xmin`, negative
  otherwise). The array path wraps the difference in `np.abs()`:

  ```python
  result = np.abs(erf(xmax_div_sqrt_2) - erf(xmin_div_sqrt_2))
  ```

  This means the array path always returns a non-negative value regardless of the
  ordering of `x_min` and `x_max`, while the scalar path preserves the sign. The two code
  paths are therefore inconsistent. If a caller ever passes `x_min > x_max` (e.g. to
  express a reversed integration direction), the scalar and array paths disagree. More
  subtly, when `base != 0`, the `base` is added *after* the absolute value, so the result
  for `scale * integral + base` differs from the scalar result even for `x_min < x_max`
  when `scale` is negative.
- **Impact:** Silent numerical errors in any pipeline that relies on the array path with
  non-standard argument ordering or negative scale. This is the most serious correctness
  issue in the codebase.
- **Suggestion:** Remove `np.abs()` to match the scalar path.

### 1.2 CRITICAL — `background_gradient_fit` missing f-string prefix

- **Evidence:** `psf.py` line 284
- **Issue:** The error message reads:

  ```python
  raise ValueError('Image must be 2-D, got {image.shape}')
  ```

  This is a plain string, not an f-string. The user sees the literal text
  `{image.shape}` instead of the actual shape, making the error message useless for
  diagnosis.
- **Suggestion:** Change to `f'Image must be 2-D, got {image.shape}'`.

### 1.3 HIGH — `gaussian_integral_1d` uses `assert` for input validation

- **Evidence:** `gaussian.py` line 263
- **Issue:** `assert sigma > 0.` is used to validate a public-API parameter. Asserts are
  removed under `python -O`, silently allowing `sigma <= 0` which produces `nan`/`inf`
  results.
- **Suggestion:** Replace with `if sigma <= 0: raise ValueError(...)`.

### 1.4 HIGH — `gaussian_integral_2d` angle path uses mean-of-samples (not true integral)

- **Evidence:** `gaussian.py` lines 330–365
- **Issue:** When `angle != 0`, the 2-D Gaussian integral falls back to sampling the
  function on a `linspace` grid of size `angle_subsample` (default 13) and returning
  `np.mean(ret)`. This is a crude midpoint-rule quadrature on a 13×13 grid. For narrow
  Gaussians (`sigma < 0.5`) or large integration regions, 169 samples may not capture the
  peak adequately. The `np.mean()` approximation also does not correctly scale by the area
  of the integration domain — it returns the average value, not the integral. For the
  integral to be correct it should be multiplied by the area
  `(y_max - y_min) * (x_max - x_min)`, which is not done here.

  **However**, this same function is called in the angle=0 path from `eval_pixel` where it
  evaluates over a unit pixel (area = 1), so the mean equals the integral only in that
  specific case. For non-unit-pixel regions or direct calls with `angle != 0`, the result
  is mathematically incorrect.
- **Impact:** Quantitative errors for any rotated Gaussian integration over non-unit
  regions. Even for unit pixels, accuracy is limited to ~1% for `sigma < 1`.
- **Suggestion:** Either multiply by the area `(y_max - y_min) * (x_max - x_min)`, or
  switch to a proper 2-D quadrature (e.g. `scipy.integrate.dblquad` or rotate the
  coordinate system analytically to separate the integral). Document the accuracy
  limitation if the grid approach is intentional.

### 1.5 MEDIUM — `background_gradient` infers order from parameter count via `sqrt`

- **Evidence:** `psf.py` line 390
- **Issue:** `order = int(np.sqrt(len(bkgnd_params)*2))-1` uses a floating-point square
  root to recover the polynomial order from the parameter count. For the standard counts
  (3, 6, 10) this works, but for unusual counts the `int()` truncation could silently pick
  the wrong order. No validation is performed on the result.
- **Suggestion:** Validate that the inferred order is consistent with the actual parameter
  count, or pass the order explicitly.

### 1.6 MEDIUM — `_find_position` applies `scaled_psf = psf * scale + base` redundantly

- **Evidence:** `psf.py` line 804
- **Issue:** `details['scaled_psf'] = psf * scale + base`, but `psf` was already created
  by `self.eval_rect(..., scale=scale, base=base, ...)` (line 790–791), meaning `psf`
  already includes the scale and base. The `scaled_psf` entry therefore double-applies the
  scaling: `(psf_raw * scale + base) * scale + base`. The residual computation
  `sub_img_grad - details['scaled_psf']` at line 569 uses this double-scaled PSF, which is
  incorrect.
- **Impact:** The bad-pixel rejection loop in `find_position` (lines 569–591) computes
  residuals against a double-scaled PSF, so the `num_sigma` threshold is compared to an
  incorrect residual. The final returned position is still from the optimizer, so the
  position result itself is not affected, but the `scaled_psf` metadata entry and the
  bad-pixel masking logic are wrong.
- **Suggestion:** Either compute `psf` with `scale=1., base=0.` and store `psf * scale +
  base` in `scaled_psf`, or call `eval_rect` with the full parameters and set
  `details['scaled_psf'] = psf`.

### 1.7 LOW — Numerical stability of Gaussian evaluation for extreme sigma

- **Evidence:** `gaussian.py` lines 123–124
- **Issue:** `np.exp(-(x-mean)**2 / (2 * sigma**2))` underflows to 0.0 for `|x-mean|`
  much larger than `sigma`, and `1 / sigma**2` overflows for extremely small sigma. The
  `sigma_x_range` default of `(0.01, 10.)` keeps sigma in a safe range during fitting,
  but the static methods accept arbitrary sigma.
- **Suggestion:** Document the valid range or add a guard for sigma near zero.

---

## 2. Performance and Resource Use

### 2.1 HIGH — `_eval_rect` creates flat coordinate arrays instead of using meshgrid

- **Evidence:** `gaussian.py` lines 527–541
- **Issue:** `_eval_rect` constructs `y_coords` via `np.repeat` and `x_coords` via
  `np.tile`, then creates a `(2, N)` coords array. This allocates three large arrays where
  a simple `np.meshgrid` + reshape would suffice and be clearer. More importantly, these
  flat coordinate arrays are passed to `eval_pixel`, which calls
  `gaussian_integral_2d` element-by-element for the `angle != 0` path (the array branch
  of `gaussian_integral_2d` has an explicit Python `for` loop at lines 353–363). For a
  21×21 PSF with a non-zero angle, this means 441 separate `np.linspace` + `meshgrid` +
  `gaussian_2d` calls.
- **Suggestion:** For the rotated case, vectorize the computation. Consider using a
  rotated-coordinate analytic integral or at least batch the meshgrid operation.

### 2.2 HIGH — Powell optimizer called with `maxiter = len(starting_guess) * 10000`

- **Evidence:** `psf.py` line 752
- **Issue:** For a typical Gaussian with 5–7 parameters, this sets `maxiter` to
  50,000–70,000. Powell's method evaluates the objective function many times per iteration.
  Each evaluation calls `eval_rect`, which for a 21×21 Gaussian is fast but for an HST PSF
  involves spline interpolation, reshaping, and optional convolution. The large iteration
  cap can cause `find_position` to run for minutes if convergence is slow.
- **Suggestion:** Consider a more modern optimizer (e.g. `scipy.optimize.minimize` with
  `method='L-BFGS-B'` for bounded problems) and add a callback or timeout mechanism. At
  minimum, log the number of iterations used.

### 2.3 MEDIUM — `_eval_rect_smeared` allocates a new array per step in the loop

- **Evidence:** `psf.py` lines 173–189
- **Issue:** Each step in the motion-blur loop calls `_eval_rect` which returns a new
  array, then adds it to `total_rect`. For `num_steps` up to several hundred (large
  motion, small granularity), this allocates many temporary arrays.
- **Suggestion:** Pre-allocate `total_rect = np.zeros(rect_size)` and accumulate in place.

### 2.4 MEDIUM — `hst.py` `_cache_pixelation` evaluates spline per-point

- **Evidence:** `hst.py` lines 672–681
- **Issue:** `desired_y_indices` and `desired_x_indices` are flattened to 1-D, then
  `spline.ev()` is called on them. For a 39×5 = 195 pixel subsampled PSF, this evaluates
  ~38,000 points individually. `RectBivariateSpline.__call__` on a 2-D grid is much faster
  because it exploits the tensor-product structure.
- **Suggestion:** Use `spline(desired_y_unique, desired_x_unique)` on the unique
  coordinate arrays to get a 2-D grid evaluation.

### 2.5 LOW — No caching of `_background_gradient_coeffs`

- **Evidence:** `psf.py` line 326, called inside a `while True` loop at line 333
- **Issue:** `_background_gradient_coeffs` is called once before the loop, but
  `background_gradient` (line 357) calls it again inside the loop on every iteration. For
  a fixed image shape and order, the result is always the same.
- **Suggestion:** Cache the coefficients or pass them into `background_gradient`.

---

## 3. Debugging and Logging

### 3.1 CRITICAL — No `logging` module usage anywhere in the library

- **Evidence:** `grep -r "logging" src/psfmodel/` returns zero matches.
- **Issue:** The entire library uses bare `print()` for diagnostic output, controlled by
  the integer `self._debug_opt` attribute. There are **50+** `print()` calls in `psf.py`
  and **30+** in `hst.py`. This violates the project's own rule: "ALWAYS include
  meaningful, structured logging (use the `logging` module)... NEVER use bare `print()`
  for diagnostic output in library code."
- **Impact:** Users cannot selectively enable/disable debug output, redirect it to a log
  file, or integrate it with their own logging configuration. The `print()` calls write
  directly to stdout, polluting output in Jupyter notebooks, pipelines, and web services.
- **Suggestion:** Replace all `print()` calls with `logging.getLogger(__name__)` calls at
  appropriate levels:
  - `logger.debug()` for `_debug_opt > 1` messages (per-iteration detail)
  - `logger.info()` for `_debug_opt == 1` messages (entry/exit, final results)
  - `logger.warning()` for the unconditional `'FAIL'` message at line 760
  - `logger.error()` for fatal errors in `hst.py`

  Add `logging.getLogger(__name__).addHandler(logging.NullHandler())` in
  `__init__.py` as per library best practice.

### 3.2 HIGH — `_debug_opt` is a public mutable attribute with no API

- **Evidence:** `psf.py` line 28
- **Issue:** `self._debug_opt = 0` is set in `__init__`, but tests set it directly
  (e.g. `psf._debug_opt = 10`). There is no method, property, or documentation for this.
  The integer levels (0, 1, 2, 3) are undocumented and scattered across the code.
- **Suggestion:** Remove once `logging` is adopted. If levels are still needed, use
  standard `logging` levels (`DEBUG`, `INFO`, etc.).

### 3.3 HIGH — Debug `print()` on optimizer failure is unconditional

- **Evidence:** `psf.py` line 760
- **Issue:** `print('FAIL', message)` fires regardless of `_debug_opt`, meaning any
  user calling `find_position` will see "FAIL ..." on stdout if the optimizer does not
  converge. This should be a `logger.warning()` at minimum, or the failure should be
  communicated via the return value (which it already is — `None`).

### 3.4 MEDIUM — Unreachable `print('hi')` at end of `_find_position`

- **Evidence:** `psf.py` line 885
- **Issue:** `print('hi')` appears after `return offset_y, offset_x, details` at line 883.
  This is dead code that will never execute but suggests leftover debugging.
- **Suggestion:** Delete.

---

## 4. Structure and Layout

### 4.1 HIGH — Dead code: `_dead_code()` function and unreachable `print('hi')`

- **Evidence:** `psf.py` lines 885–889
- **Issue:** A standalone function `_dead_code()` containing only `pass` exists at module
  level, preceded by unreachable `print('hi')`. This is clearly leftover scaffolding.
- **Suggestion:** Delete both.

### 4.2 HIGH — Massive amounts of commented-out code

- **Evidence:** `psf.py` lines 55–79 (commented abstract `eval_pixel`), lines 618–644
  (commented bounds logic), lines 806–850 (commented covariance/error computation),
  lines 851–854 (commented leastsq metadata). `gaussian.py` lines 127–175 (commented
  `gaussian_2d_rho`). Test files also have commented-out assertions with `# TODO: Why?`.
- **Impact:** Makes the code harder to read and maintain. Version control preserves
  history; commented-out code adds noise.
- **Suggestion:** Remove all commented-out code blocks. File issues for features that were
  partially implemented (e.g. the covariance/error estimation).

### 4.3 HIGH — `hst.py` excluded from Ruff and mypy

- **Evidence:** `pyproject.toml` line 121: `exclude = ["src/psfmodel/hst.py"]`,
  `.mypy.ini` line 3: `exclude = hst.py`
- **Issue:** The largest module (789 lines) is exempt from all linting and type checking.
  It uses bare `assert` for control flow (20 instances), `print()` for error reporting,
  `os.system()` for shell execution, and has no type annotations.
- **Suggestion:** Incrementally bring `hst.py` under lint and type checking. Start by
  adding type annotations to public methods and replacing `assert False` with proper
  exceptions.

### 4.4 MEDIUM — `psf.py` header says `# psfmodel/__init__.py`

- **Evidence:** `psf.py` line 2
- **Issue:** Stale header comment; the module is `psf.py`, not `__init__.py`.
- **Suggestion:** Fix to `# psfmodel/psf.py`.

### 4.5 MEDIUM — `__init__.py` does not export `HSTPSF`

- **Evidence:** `src/psfmodel/__init__.py`
- **Issue:** `__all__` exports `PSF` and `GaussianPSF` but not `HSTPSF`. Users must do
  `from psfmodel.hst import HSTPSF`. If `HSTPSF` is part of the public API, it should be
  in `__all__`. If it is internal/experimental, it should be prefixed with `_` or clearly
  documented as such.
- **Suggestion:** Decide on the public status of `HSTPSF` and update `__all__` and docs
  accordingly. Note that importing `HSTPSF` in `__init__.py` would trigger the
  import-time crash for users without `TINYTIM` set (see Security section).

### 4.6 LOW — Duplicate mypy configuration

- **Evidence:** Both `.mypy.ini` and `[tool.mypy]` in `pyproject.toml` exist. Per pytest
  configuration discovery rules, `.mypy.ini` takes precedence over `pyproject.toml`, so
  the `pyproject.toml` mypy settings (e.g. `disallow_subclassing_any = false`) may not
  be applied.
- **Suggestion:** Consolidate into `pyproject.toml` and delete `.mypy.ini`.

### 4.7 LOW — Stale `.flake8` and `setup.cfg` files

- **Evidence:** `.flake8` configures flake8 (superseded by Ruff), `setup.cfg` contains
  only `[metadata] name = rms-psfmodel` (superseded by `pyproject.toml`).
- **Suggestion:** Delete both files.

---

## 5. Best Practices Alignment

### 5.1 CRITICAL — `hst.py` reads environment variables at module import time

- **Evidence:** `hst.py` lines 198–199

  ```python
  TINY_TIM_DIR = os.environ['TINYTIM']
  PSF_CACHE_DIR = os.environ['PSF_CACHE_DIR']
  ```

- **Issue:** Any `import psfmodel.hst` (or even indirect import) crashes with `KeyError`
  if these environment variables are not set. This prevents the entire module from being
  imported for testing, documentation generation, or by users who only need `GaussianPSF`.
- **Suggestion:** Defer the lookup to when TinyTim is actually called. Use
  `os.environ.get()` with a `None` default and validate at the point of use.

### 5.2 CRITICAL — `os.system()` used for shell execution with string concatenation

- **Evidence:** `hst.py` lines 528–544
- **Issue:** `os.system('./tiny1 ' + temp_filename + ' < ' + params_filename + ...)` is
  vulnerable to shell injection if any of the file paths or parameters contain shell
  metacharacters. `os.system()` also provides no error handling — the return code is
  ignored.
- **Suggestion:** Use `subprocess.run()` with a list of arguments (no `shell=True`).
  Check the return code and raise on failure.

### 5.3 HIGH — `assert` used for control flow in `hst.py` (20 instances)

- **Evidence:** `hst.py` lines 300, 322, 325, 329, 333, 369, 373, 436, 438, 440, 442,
  449, 473, 550, 586, 615, 616, 719, 772, 773
- **Issue:** `assert False` is used to signal errors (e.g. unknown instrument, PSF too
  small). Asserts are stripped under `python -O`, so these checks vanish in optimized
  mode, leading to silent misbehavior or later crashes with confusing tracebacks.
- **Suggestion:** Replace every `assert` used for validation with `raise ValueError(...)`.

### 5.4 HIGH — `open()` without context manager or `encoding=`

- **Evidence:** `hst.py` line 482: `params_fp = open(params_filename, 'w')`
- **Issue:** The file is opened without a `with` statement and without specifying
  `encoding='utf-8'`. If an exception occurs between `open()` and `close()` (line 522),
  the file handle leaks. The default encoding depends on the platform locale.
- **Suggestion:** Use `with open(params_filename, 'w', encoding='utf-8') as params_fp:`.

### 5.5 HIGH — `hst.py` shadows the built-in `filter`

- **Evidence:** `hst.py` line 222: `def __init__(self, ..., filter, ...)`
- **Issue:** The parameter name `filter` shadows the Python built-in. Per project rules
  (python_best_practices.mdc): "Do NOT use variable or function names that shadow Python
  built-ins... append a single underscore (e.g. `filter_`)."
- **Suggestion:** Rename to `filter_` throughout `hst.py`.

### 5.6 HIGH — `hst.py` `os.getcwd()` used to detect OS

- **Evidence:** `hst.py` lines 201–206

  ```python
  if os.getcwd()[1] == ':':
      DEV_NULL = 'NUL'
  else:
      DEV_NULL = '/dev/null'
  ```

- **Issue:** Detecting Windows by checking if the second character of `cwd` is `:` is
  fragile. It fails for UNC paths (`\\server\share`), and the check runs at import time
  making it dependent on the cwd at that moment.
- **Suggestion:** Use `sys.platform == 'win32'` or `os.devnull` (which is the
  platform-correct value).

### 5.7 MEDIUM — `hst.py` `eval_pixel` references undefined variables `y` and `x`

- **Evidence:** `hst.py` line 740

  ```python
  self._cache_psf(max(abs(y)*2+1, abs(x)*2+1), **kwargs)
  ```

  The parameters are `coord`, `offset`, `scale`, `base`, `**kwargs`. There are no local
  variables `y` or `x` before this line. This is a `NameError` at runtime.
- **Impact:** `eval_pixel` is completely broken and would crash on any call.
- **Suggestion:** Replace with `coord[0]` and `coord[1]`.

### 5.8 MEDIUM — `HSTPSF.__init__` passes extra positional args to `PSF.__init__`

- **Evidence:** `hst.py` line 272: `PSF.__init__(self, movement, movement_granularity)`
- **Issue:** `PSF.__init__` accepts only `**kwargs`, not positional arguments. This would
  raise a `TypeError` at runtime. The `movement` and `movement_granularity` parameters are
  handled by `_eval_rect_smeared` via `eval_rect`, not by the base class.
- **Impact:** `HSTPSF` cannot be instantiated. This module is entirely non-functional in
  its current state.
- **Suggestion:** Remove the extra arguments from the `PSF.__init__` call.

### 5.9 MEDIUM — `hst.py` `_cache_pixelation` references `self.movement` (never set)

- **Evidence:** `hst.py` lines 654–655
- **Issue:** `self.movement` is never assigned in `HSTPSF.__init__`. The base class
  `PSF.__init__` does not set it either. This would raise `AttributeError` at runtime.
- **Suggestion:** If motion blur is needed, store `self._movement` and
  `self._movement_granularity` as instance attributes in `HSTPSF.__init__`.

### 5.10 LOW — `num_sigma` checked as truthy instead of `is not None`

- **Evidence:** `psf.py` lines 542, 566
- **Issue:** `if num_sigma:` treats `0.0` as falsy. If a caller passes `num_sigma=0.0`
  (meaning "reject no pixels"), the code skips the sigma-rejection loop, which is correct
  by coincidence but not by intent.
- **Suggestion:** Use `if num_sigma is not None:` for clarity.

---

## 6. Types and Static Checks

### 6.1 HIGH — mypy is commented out of dev dependencies

- **Evidence:** `pyproject.toml` line 69: `# "mypy>=1.0",`
- **Issue:** mypy is not installed by `pip install -e ".[dev]"`, so developers are not
  running type checks locally. The CI workflow *does* run mypy, but the local dev
  experience is inconsistent.
- **Suggestion:** Uncomment mypy in dev dependencies.

### 6.2 HIGH — `_eval_rect` has `# type: ignore` on method signature

- **Evidence:** `gaussian.py` lines 515, 545
- **Issue:** Both `_eval_rect` and `eval_rect` have `# type: ignore` on the `def` line
  because the override signatures do not match the base class. The base class defines
  `_eval_rect(self, ...) -> npt.NDArray[np.float64]` while the override adds `sigma`,
  `sigma_y`, `sigma_x`, and `angle` keyword arguments.
- **Suggestion:** Align the base-class signature (use `**kwargs: Any` in the base) or use
  `@overload` to express the extended signatures. The `# type: ignore` suppresses
  real type errors.

### 6.3 MEDIUM — `hst.py` has zero type annotations

- **Evidence:** All 789 lines of `hst.py`
- **Issue:** No function signatures, no return types, no variable annotations. The module
  is excluded from mypy so this is not flagged.
- **Suggestion:** Add annotations incrementally, starting with the public API
  (`HSTPSF.__init__`, `eval_point`, `eval_pixel`, `eval_rect`, `run_tinytim`).

### 6.4 LOW — Inconsistent return type annotations

- **Evidence:** `gaussian.py` `eval_point` returns `cast(float, ret)` but the actual
  return could be an array; `gaussian_1d` returns `cast(float | npt.NDArray, ret)`.
- **Suggestion:** Ensure cast types match actual possible returns.

---

## 7. Testing

### 7.1 HIGH — Overall coverage is only ~45%

- **Evidence:** pytest-cov output: `TOTAL 791 435 302 19 45%`
- **Issue:** `hst.py` is 0% covered (331 statements). `psf.py` is 68% covered — the
  entire `find_position` debug-output and bad-pixel rejection paths are untested. The
  coverage target in `pyproject.toml` is set to `fail_under = 40` (marked `# TODO`).
- **Suggestion:** Raise the coverage target incrementally. Add tests for
  `find_position` with `num_sigma`, edge-of-image, and all-masked scenarios. Add basic
  `hst.py` unit tests (mocking TinyTim).

### 7.2 HIGH — No tests for `HSTPSF` at all

- **Evidence:** `tests/` contains only `test_gaussian.py` and `test_psf.py`.
- **Issue:** The entire HST PSF path is untested. Given the numerous bugs identified above
  (`NameError`, broken `__init__`, missing attributes), the module is likely non-functional.
- **Suggestion:** Add a test module `test_hst.py` with mocked TinyTim calls.

### 7.3 MEDIUM — Tests do not assert on exception messages

- **Evidence:** All `pytest.raises(ValueError)` calls in both test files check only the
  exception type, not the message content (e.g. `test_gaussian.py` lines 127–138,
  `test_psf.py` lines 14–23).
- **Suggestion:** Use `pytest.raises(ValueError, match="...")` or assert on
  `str(exc_info.value)`.

### 7.4 MEDIUM — Commented-out assertions with `# TODO: Why?`

- **Evidence:** `test_gaussian.py` lines 292, 302, 310, 318
- **Issue:** Several assertions for `scale` and `sigma` in `test_gaussian_find_position`
  are commented out with `# TODO: Why?`. This suggests known fitting accuracy issues that
  are not understood or tracked.
- **Suggestion:** Investigate and either fix the fitting or document the known limitation
  with a linked issue.

### 7.5 MEDIUM — No `conftest.py` or shared fixtures

- **Evidence:** No `tests/conftest.py` exists.
- **Issue:** Test setup is duplicated across test functions (e.g. creating `GaussianPSF()`
  instances, generating test images).
- **Suggestion:** Create shared fixtures for common PSF instances and test images.

### 7.6 LOW — Tests access private `_background_gradient_coeffs` directly

- **Evidence:** `test_psf.py` line 15: `PSF._background_gradient_coeffs((3, -1), 1)`
- **Issue:** Tests import and call `_`-prefixed methods directly, coupling them to
  implementation details.
- **Suggestion:** Test through the public API (`background_gradient_fit`,
  `background_gradient`) where possible.

---

## 8. Security and Robustness

### 8.1 CRITICAL — Shell injection via `os.system()` in `hst.py`

- **Evidence:** `hst.py` lines 528–544
- **Issue:** `os.system('./tiny1 ' + temp_filename + ' < ' + params_filename + redir)`
  concatenates user-influenced strings into a shell command. If `PSF_CACHE_DIR` or a filter
  name contained shell metacharacters (e.g. `; rm -rf /`), arbitrary commands could execute.
- **Suggestion:** Use `subprocess.run([...], check=True)` without `shell=True`.

### 8.2 HIGH — No path traversal protection in `hst.py`

- **Evidence:** `hst.py` lines 471, 474–476
- **Issue:** `fits_filename = path_join(PSF_CACHE_DIR, fits_base)` and
  `os.chdir(TINY_TIM_DIR)` use environment-variable paths with no validation. A malicious
  `PSF_CACHE_DIR` could write FITS files to arbitrary locations.
- **Suggestion:** Validate that paths resolve within expected directories.

### 8.3 MEDIUM — `hst.py` reads FITS header comments to extract diffusion matrix

- **Evidence:** `hst.py` lines 571–578
- **Issue:** The diffusion matrix is parsed from FITS header COMMENT lines by splitting on
  spaces and calling `float()`. No validation is done on the number of values, their range,
  or the comment format. A corrupted or modified FITS file could cause `IndexError`,
  `ValueError`, or inject extreme values into the convolution.
- **Suggestion:** Validate the parsed matrix (shape, value ranges) before use.

---

## 9. Dependencies and Tooling

### 9.1 HIGH — Runtime dependencies have no minimum versions

- **Evidence:** `pyproject.toml` lines 11–15

  ```toml
  dependencies = [
    "astropy",
    "numpy",
    "scipy"
  ]
  ```

- **Issue:** Per project rules, library dependencies should specify minimum compatible
  versions (e.g. `numpy>=1.24`). Without minimums, users may install ancient versions
  that lack required features.
- **Suggestion:** Add minimum versions based on the features used (e.g. `numpy>=1.24`,
  `scipy>=1.10`, `astropy>=5.3`).

### 9.2 MEDIUM — `pyproject.toml` has stale TODO markers

- **Evidence:** `pyproject.toml` lines 23 (`keywords = ["TODO"]`), 59
  (`"TODO" = ["py.typed"]`), 88 (`#TODO = "main.psfmodel:main"`), 103
  (`fail_under = 40  # TODO`)
- **Suggestion:** Replace `keywords` with actual keywords (e.g. `"PSF"`, `"astronomy"`,
  `"Gaussian"`). Fix the `py.typed` package-data key to `"psfmodel"`. Remove or fill in
  the script entry point. Set `fail_under` to a meaningful target (e.g. 80).

### 9.3 MEDIUM — Duplicate/conflicting linter configurations

- **Evidence:** `.flake8` (max-line-length 90), `pyproject.toml` Ruff (line-length 100),
  `.mypy.ini` vs `[tool.mypy]` in `pyproject.toml`
- **Issue:** Multiple config files for the same tools create confusion about which is
  active. Ruff has replaced flake8 in this project.
- **Suggestion:** Delete `.flake8` and `.mypy.ini`. Consolidate everything into
  `pyproject.toml`.

### 9.4 LOW — `setup.cfg` still exists

- **Evidence:** `setup.cfg` contains only `[metadata] name = rms-psfmodel`
- **Issue:** Redundant with `pyproject.toml`. May confuse build tools.
- **Suggestion:** Delete.

---

## 10. Maintainability and Extensibility

### 10.1 HIGH — `find_position` returns `tuple[float, float, dict[str, Any]]`

- **Evidence:** `psf.py` line 418
- **Issue:** The return type is `None | tuple[float, float, dict[str, Any]]`. The
  `details` dict has 10+ keys with mixed types, no schema, and no documentation beyond
  the docstring. Callers must remember string keys like `'subimg-gradient'` and
  `'scaled_psf'`.
- **Suggestion:** Define a `@dataclass` (e.g. `FitResult`) with typed fields. This
  enables IDE autocompletion, type checking, and clearer documentation.

### 10.2 HIGH — `_additional_params` is a list of tuples with magic indices

- **Evidence:** `gaussian.py` lines 74–82, `psf.py` lines 651–658
- **Issue:** Each additional parameter is stored as `(min, max, name)` in a list. Code
  accesses `ap[0]`, `ap[1]`, `ap[2]` with no named fields. This makes the code fragile
  and hard to extend.
- **Suggestion:** Use a `NamedTuple` or `@dataclass` (e.g.
  `ParamSpec(min: float, max: float, name: str)`).

### 10.3 MEDIUM — Docs only automodule `psfmodel`, missing `gaussian` and `hst`

- **Evidence:** `docs/module.rst` lines 1–10
- **Issue:** The Sphinx docs only have `.. automodule:: psfmodel`. The `gaussian` and
  `hst` submodules are not documented. Since `GaussianPSF` is re-exported from
  `__init__.py` it may appear, but `HSTPSF`, static methods, and utility functions are
  invisible in the docs.
- **Suggestion:** Add `.. automodule:: psfmodel.gaussian` and
  `.. automodule:: psfmodel.psf` sections (and `psfmodel.hst` when ready).

### 10.4 MEDIUM — README describes wrong project

- **Evidence:** `README.md` line 33: "psfmodel is a set of classes for reading and
  searching star catalogs. Currently NAIF SPICE star catalogs, the Yale Bright Star
  Catalog (YBSC), and UCAC4 are supported."
- **Issue:** This is a copy-paste from a different project. The actual project is a PSF
  model fitting library.
- **Suggestion:** Rewrite the introduction to describe PSF modeling and fitting.

### 10.5 LOW — GUI program in `programs/psf_gui.py` uses wildcard import

- **Evidence:** `psf_gui.py` line 5: `from tkinter import *`
- **Issue:** Wildcard imports pollute the namespace and make it unclear which names come
  from tkinter.
- **Suggestion:** Use `import tkinter as tk` and prefix all names with `tk.`.

---

## 11. Packaging and Distribution

### 11.1 MEDIUM — `py.typed` package-data key is `"TODO"`

- **Evidence:** `pyproject.toml` line 59: `"TODO" = ["py.typed"]`
- **Issue:** The `py.typed` marker file exists in `src/psfmodel/`, but the `pyproject.toml`
  key should be `"psfmodel"`, not `"TODO"`. The marker file is not included in the built
  wheel.
- **Suggestion:** Change to `"psfmodel" = ["py.typed"]`.

### 11.2 LOW — `requirements.txt` is not present or useful

- **Evidence:** A `requirements.txt` exists but was not read; per the dependency management
  rule it should contain only `-e .` if kept.
- **Suggestion:** Verify contents or delete.

---

## 12. Technical Debt and Risk

### 12.1 HIGH — `hst.py` is effectively non-functional

- **Evidence:** Multiple runtime errors (NameError, TypeError, AttributeError) identified
  in code review, zero test coverage, excluded from all linting.
- **Issue:** This module cannot be imported without specific environment variables and
  cannot be instantiated due to broken `__init__`. It appears to be legacy code that has
  not been maintained alongside the refactoring of `psf.py`.
- **Suggestion:** Either invest in bringing `hst.py` up to standard (fix bugs, add types,
  add tests, remove `os.system`) or mark it explicitly as experimental/unsupported and
  gate the import behind a try/except with a clear message.

### 12.2 MEDIUM — Commented-out covariance/error estimation in `_find_position`

- **Evidence:** `psf.py` lines 806–850
- **Issue:** The switch from `scipy.optimize.leastsq` to `scipy.optimize.minimize` (with
  Powell) means covariance information is no longer available. The old code is commented
  out, and the docstring still documents `leastsq_cov`, `x_err`, `y_err`, etc., that are
  never populated. This is misleading to users who expect uncertainty estimates.
- **Suggestion:** Either implement uncertainty estimation (e.g. via numerical Hessian or
  bootstrap) or remove the references from the docstring and file an issue to track the
  feature gap.

### 12.3 MEDIUM — `gaussian_2d_rho` commented out but referenced in documentation context

- **Evidence:** `gaussian.py` lines 127–175
- **Issue:** An alternative parameterization of the 2-D Gaussian using correlation `rho`
  is commented out. The mathematical notes in the comments contain an undefined variable
  `xcorr` (should be `rho * sigma_x * sigma_y`), suggesting the implementation was never
  completed.
- **Suggestion:** Delete the commented-out code. File an issue if the `rho`
  parameterization is needed in the future.

---

## Recommended Priorities

1. **Fix `gaussian_integral_1d` `np.abs()` bug** — silent numerical errors in array
   integrals. One-line fix with high data-integrity impact.

2. **Fix `background_gradient_fit` missing f-string** — users get unhelpful error
   messages. One-character fix.

3. **Fix `_find_position` double-scaling of `scaled_psf`** — incorrect bad-pixel
   rejection. Straightforward logic fix.

4. **Replace all `print()` with `logging`** — the single largest quality-of-life
   improvement for library users. Systematic but not complex.

5. **Triage `hst.py`** — decide whether to fix or deprecate. If fixing: address
   import-time crash, `os.system()`, `assert` misuse, undefined variables, and missing
   `__init__` arguments. If deprecating: gate the import and document.

6. **Raise test coverage** — add tests for `find_position` edge cases, exception
   messages, and `HSTPSF` (mocked). Increase `fail_under` from 40% to 80%+.

7. **Clean up commented-out code and TODOs** — remove dead code, fix stale comments, and
   resolve `pyproject.toml` TODO markers.

8. **Consolidate configuration** — delete `.flake8`, `.mypy.ini`, `setup.cfg`. Uncomment
   mypy in dev dependencies. Fix `py.typed` package-data key.

9. **Fix README** — rewrite the introduction to describe PSF modeling instead of star
   catalogs.

10. **Add minimum dependency versions** — specify `numpy>=X`, `scipy>=Y`, `astropy>=Z`
    in `pyproject.toml`.
