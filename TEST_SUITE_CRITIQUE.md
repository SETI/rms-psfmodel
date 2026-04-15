# Test Suite Critique Report

**Generated:** 2026-04-12
**Scope:** tests/ (no conftest.py present)

## Executive summary

The test suite covers the core mathematical functions of the `psfmodel` library
(Gaussian 1-D/2-D evaluation, integration, pixel evaluation, and PSF fitting)
with reasonable numeric precision assertions. However, there are significant
gaps:

**Strengths:**

- Numeric assertions use `pytest.approx` and `npt.assert_array_almost_equal`
  consistently, which is appropriate for floating-point comparisons.
- `test_gaussian_find_position` uses `@pytest.mark.parametrize` to cover
  multiple `bkgnd_degree` and `use_angular_params` combinations (8 variants).
- Tests validate both scalar and array inputs for mathematical functions.
- Integration tests (via `scipy.integrate`) confirm normalization properties.

**Main gaps (high priority):**

1. **Coverage is 45%** (target: 90%). The `hst.py` module (335 statements) has
   0% coverage. Even excluding `hst.py`, `psf.py` is only at 70% with large
   uncovered regions (`find_position`, `_find_position`, `_fit_psf_func`,
   `_eval_rect_smeared`).
2. **No exception-message assertions.** All 25 `pytest.raises(ValueError)` calls
   lack `as exc_info` and message-content checks.
3. **No `conftest.py` or shared fixtures** -- setup logic is duplicated across
   tests.
4. **No tests for `PSF.__init__`**, `GaussianPSF.__init__` validation (e.g.
   `angle_subsample` range), `eval_rect` validation, `find_position` edge
   cases, `_eval_rect_smeared` motion blur, or `background_gradient`.
5. **No logging assertions** despite extensive `self._logger` usage in `psf.py`.
6. **`hst.py` is entirely untested** (excluded from mypy/ruff but still shipped).

**Nice-to-have improvements:**

- Parameterize repetitive test cases in `test_gaussian_1d`, `test_gaussian_2d`.
- Add type annotations to test functions.
- Configure `filterwarnings = ["error"]` and `--strict-markers` in pytest.

---

## 1. Return values and assertions

**Strengths:**

- Numeric return values are checked with `pytest.approx` (explicit tolerances
  where needed) and `npt.assert_array_almost_equal`.
- `test_gaussian_find_position` checks specific dictionary keys (`sigma_y`,
  `sigma_x`, `scale`) with explicit values.

**Issues:**

- **`test_gaussian_eval_rect` (line 226-230):** Only checks
  `np.sum(...) == pytest.approx(1)`. Does not assert shape, individual pixel
  values, or that the center pixel is the maximum. This is a weak existence
  assertion.
- **`test_gaussian_find_position`:** Several assertions are commented out with
  `# TODO: Why?` (lines 349, 366-367, 384-385), meaning those return values
  are not verified at all.
- **`test_background_gradient_fit` (line 84):** Checks `np.sum(img_mask) == 0`
  but not the shape or dtype of `img_mask`.
- **`test_bkgnd_gradient_coeffs`:** Uses `assert np.all(ret == exp)` which
  gives poor diagnostic messages on failure compared to
  `npt.assert_array_equal`.

---

## 2. Success and failure conditions

### GaussianPSF

| Method | Success tested | Failure tested | Missing |
|---|---|---|---|
| `gaussian_1d` | Yes (scalars, arrays) | No | Negative sigma, sigma=0 |
| `gaussian_2d` | Yes (scalars, arrays, rotation) | No | Invalid sigma, angle edge cases |
| `gaussian_integral_1d` | Yes (scalars, arrays, params) | No | sigma<=0 (has `assert sigma > 0`), xmin > xmax |
| `gaussian_integral_2d` | Yes (scalars, arrays) | No | Invalid inputs |
| `eval_point` | Yes | Partial (6 ValueError) | No message assertions; missing edge cases for `angle` param |
| `eval_pixel` | Yes | Partial (9 ValueError) | No message assertions |
| `eval_rect` | Minimal (sum only) | No | Odd-size validation, negative size, shape check |
| `__init__` | Implicit only | No | `angle_subsample` out of range, invalid sigma types |

### PSF

| Method | Success tested | Failure tested | Missing |
|---|---|---|---|
| `_background_gradient_coeffs` | Yes (orders 1-3) | Partial (5 ValueError) | No message assertions |
| `background_gradient_fit` | Yes (masked, unmasked, sigma) | Partial (4 ValueError) | No message assertions |
| `background_gradient` | Yes (indirect) | No | Invalid params |
| `find_position` | Yes (via GaussianPSF) | No | box_size validation, edge-of-image None return, optimizer failure, all-pixels-masked |
| `_eval_rect_smeared` | No | No | Motion blur (movement != None) |
| `__init__` | Implicit only | No | Logger, detailed_logging |

---

## 3. Consistency

- **Naming:** Test names follow `test_<class>_<method>` or
  `test_<function_name>` consistently. However, they do not encode conditions
  (e.g., `test_gaussian_1d_with_scale`), making it hard to tell which scenario
  failed.
- **Structure:** `test_gaussian_1d` and `test_gaussian_2d` mix many scenarios
  into a single test function (scalar, array, integration, parameter
  variations) rather than separating them into focused tests.
- **Fixtures:** No fixtures are used anywhere. Common PSF objects (e.g.,
  `GaussianPSF()`, `GaussianPSF(sigma=(1,1))`) are recreated in each test.
- **Assertion style:** Mixed use of `assert ... == pytest.approx(...)`,
  `npt.assert_array_almost_equal`, and `assert np.all(ret == exp)`. The last
  form gives poor failure diagnostics.

---

## 4. Completeness

### Coverage map

| Module | Public methods | Tested | Untested |
|---|---|---|---|
| `gaussian.py` | `__init__`, `gaussian_1d`, `gaussian_2d`, `gaussian_integral_1d`, `gaussian_integral_2d`, `eval_point`, `eval_pixel`, `eval_rect` | All partially | `__init__` validation, `eval_rect` shape/edge, `_eval_rect` directly |
| `psf.py` | `__init__`, `eval_point` (abstract), `eval_rect` (abstract), `_eval_rect_smeared`, `_background_gradient_coeffs`, `background_gradient_fit`, `background_gradient`, `find_position` | `_background_gradient_coeffs`, `background_gradient_fit`, `find_position` (partial), `background_gradient` (indirect) | `__init__`, `_eval_rect_smeared`, `find_position` edge cases, `_fit_psf_func`, `_find_position` directly |
| `hst.py` | `HSTPSF.__init__`, `run_tinytim`, `eval_point`, `eval_pixel`, `eval_rect` | None | Everything (0% coverage) |
| `__init__.py` | `__all__` exports | Yes (implicit) | N/A |

### Docstring gaps

- `GaussianPSF.eval_pixel` documents the `sigma` parameter as
  `tuple[float, float] | None` but also accepts a scalar `float` in
  `eval_point`. Tests exercise both, but `eval_pixel` does not accept scalar
  sigma, which is inconsistent with `eval_point`.
- `find_position` documents many metadata keys (e.g., `x_err`, `y_err`,
  `scale_err`) that are currently commented out in the code. Tests do not
  verify which keys are present/absent.

---

## 5. Redundancy

- **`test_gaussian_integral_1d` lines 90-92 and 102-104:** Both assert
  `gaussian_integral_1d(0.0, 1.0) == approx(integrate.quad(...))` with
  identical inputs. The second is a duplicate.
- **`test_gaussian_eval_pixel` and `test_gaussian_eval_point`:** Both test
  sigma-conflict `ValueError` raises in nearly identical patterns (lines
  145-156 vs 184-201). The validation logic is the same and could share a
  helper or parametrize.
- **`test_gaussian_find_position`:** Tests multiple PSF configurations in a
  single test function with repeated setup patterns. Breaking into separate
  tests per scenario would improve isolation and diagnostics.

---

## 6. Parallel execution

- **No global mutable state:** Tests do not modify module-level variables or
  singletons. Each test creates its own `GaussianPSF` instances.
- **No shared files or external resources:** All data is generated in-memory.
- **Parallel safe:** The tests should run correctly with `pytest -n auto`.
  The current config uses `-n 4`.

**No issues detected.**

---

## 7. Mocking and dependency isolation

- **No external calls:** The tested modules (`gaussian.py`, `psf.py`) do not
  make HTTP requests or access the file system (other than `hst.py` which is
  untested).
- **No time-sensitive logic** in tested code paths.
- **No mocks used:** The tests are integration-style, calling real scipy
  integration functions. This is appropriate for numerical code.
- **`hst.py`** calls `os.system`, `os.environ`, `os.getcwd`, `pyfits.open`,
  and file I/O. If/when tests are added, these will need to be mocked.

**No issues in current tests, but `hst.py` will need heavy mocking when
tested.**

---

## 8. Security and input validation

- **Input validation:** `GaussianPSF.__init__` validates `angle_subsample` but
  there is no test for it. `eval_rect` validates odd positive shape but has no
  test for it. `find_position` validates `box_size` but has no test for it.
- **No sensitive data:** Tests use only numeric constants; no credential risk.
- **No path traversal risk** in tested code (`hst.py` handles paths but is
  untested).

**Missing validation tests:**

- `GaussianPSF(angle_subsample=0)` -- should raise `ValueError`
- `GaussianPSF(angle_subsample=100)` -- should raise `ValueError`
- `GaussianPSF(angle_subsample=3.5)` -- should raise `ValueError` (not int)
- `GaussianPSF().eval_rect((4, 4))` -- even dimensions
- `GaussianPSF().eval_rect((-1, 5))` -- negative dimensions
- `PSF.find_position(...)` with bad `box_size`

---

## 9. Parameterization

**Good:**

- `test_gaussian_find_position` uses `@pytest.mark.parametrize` for
  `use_angular_params` and `bkgnd_degree`.

**Should be parameterized:**

- **`test_gaussian_1d` (lines 14-33):** 14 separate assertions with different
  parameter combinations. These should be parametrized over
  `(x, kwargs, expected)` tuples.
- **`test_gaussian_2d` (lines 37-67):** 16+ separate assertions. Same
  recommendation.
- **`test_gaussian_eval_point` (lines 145-156):** Six `pytest.raises` calls
  with different sigma configurations. Should be parametrized.
- **`test_gaussian_eval_pixel` (lines 184-201):** Nine `pytest.raises` calls.
  Same recommendation.
- **`test_bkgnd_gradient_coeffs` (lines 14-23):** Five `pytest.raises` calls.

**Missing boundary tests:**

- sigma at 0 (should assert `sigma > 0` raises)
- Very large sigma values
- Very large/small `scale` and `base` values
- `angle_subsample` at boundaries (1, 99)

---

## 10. Async (if applicable)

Not applicable -- no async code in the project.

---

## 11. Output and contract

- **`find_position` return shape:** The docstring specifies a detailed
  `dict[str, Any]` with many keys (`x`, `y`, `scale`, `base`, `subimg`,
  `bkgnd_params`, etc.). Tests only check `sigma_y`, `sigma_x`, and `scale`
  from the metadata dict. Many documented keys are never verified:
  - `x`, `y`, `base`, `subimg`, `bkgnd_params`, `bkgnd_mask`, `gradient`,
    `subimg-gradient`, `psf`, `scaled_psf`.
- **`eval_rect` return shape:** Not asserted (only `np.sum` is checked).
- **Exception types:** Tested (all use `pytest.raises(ValueError)`), but
  exception messages are never asserted.

---

## 12. Error handling

- **25 `pytest.raises(ValueError)` calls across both files**, but **zero**
  assert on message content. Every `raise ValueError(...)` in the source
  includes a descriptive message string; tests should verify these.
- **Specific issues:**
  - `test_gaussian_eval_point` line 145: `GaussianPSF(sigma=(1, 1)).eval_point((0, 0), sigma=5)`
    should check message contains "Cannot specify both sigma".
  - `test_gaussian_eval_point` line 152: `GaussianPSF(sigma=None).eval_point((0, 0))`
    should check message contains "must be specified".
  - `test_bkgnd_gradient_coeffs` lines 14-23: should check message contains
    "odd positive shape" or "non-negative".
- **`gaussian_integral_1d` line 280:** Uses `assert sigma > 0.0` instead of
  `raise ValueError`. This means invalid sigma produces an `AssertionError`
  that disappears with `-O`. No test covers this.

---

## 13. State and workflow

- **No state machines or lifecycle transitions** in the tested code.
- **Idempotency:** `eval_point`, `eval_pixel`, `eval_rect` are pure functions
  given the same PSF object -- no idempotency concerns.
- **Side effects:** `find_position` logs messages but tests do not verify
  logging side effects (see Section 21).
- **`hst.py` caching:** `HSTPSF._cache_psf` and `_cache_pixelation` are
  stateful with caching. Idempotency and cache invalidation are completely
  untested.

---

## 14. Test data and fixtures

- **No `conftest.py`** exists. Common objects like `GaussianPSF()`,
  `GaussianPSF(sigma=(1, 1))`, `GaussianPSF(sigma=(2.0, 3.0))` are recreated
  in multiple tests. These should be fixtures.
- **Realistic data:** Test data is mathematically generated, which is
  appropriate for a numerical library. Edge cases with `np.nan`, `np.inf`, or
  very large arrays are not tested.
- **Cleanup:** No external resources are created; no cleanup needed.
- **Fixture scope:** N/A (no fixtures used).
- **Fixture depth:** N/A.

**Recommended fixtures:**

```python
@pytest.fixture
def default_psf():
    return GaussianPSF()

@pytest.fixture
def symmetric_psf():
    return GaussianPSF(sigma=(1.0, 1.0))

@pytest.fixture
def asymmetric_psf():
    return GaussianPSF(sigma=(2.0, 3.0))
```

---

## 15. Flakiness indicators

- **No time-based assertions.**
- **No order dependence** detected -- tests do not share mutable state.
- **No external dependencies** in tested code paths.
- **No random data.**
- **`test_gaussian_find_position`** uses numerical optimization
  (`scipy.optimize.minimize`) which could theoretically produce slightly
  different results on different platforms. The tolerances (`abs=5e-2`,
  `abs=1e-1`) are generous enough to avoid flakiness, but this is worth noting.

**Low flakiness risk overall.**

---

## 16. Regression and documentation

- **No bug references** in test comments or docstrings.
- **Commented-out code:** `test_gaussian_find_position` has multiple
  commented-out assertions (lines 349, 366-367, 384-385) with `# TODO: Why?`.
  These may represent known regressions or unfinished investigations.
- **Commented-out test code:** `test_gaussian_integral_2d` has a large
  commented-out block (lines 126-141) that appears to be an unfinished rotation
  test.
- **`filterwarnings`:** Not configured in `pyproject.toml`. Unexpected warnings
  are silently swallowed.
- **Deprecation warnings:** No `pytest.warns` usage. If numpy or scipy emit
  deprecation warnings, they go unnoticed.

---

## 17. Other

- **Type annotations:** Test functions lack return type annotations and
  parameter annotations (mypy is configured with `strict = false` for test
  files).
- **Clarity:** Test function names describe the method under test but not the
  scenario. `test_gaussian_1d` tests 14 different scenarios in one function.
- **AAA pattern:** Most tests follow Arrange-Act-Assert, but large tests like
  `test_gaussian_find_position` interleave multiple arrange-act-assert cycles.
- **Single responsibility:** `test_gaussian_1d` tests scalars, arrays,
  multidimensional arrays, and integration in a single function. These are
  conceptually different behaviors.
- **Speed:** `test_gaussian_find_position` runs 8 parametrized variants, each
  performing multiple `scipy.optimize.minimize` calls. This is the slowest
  test. Consider marking it `@pytest.mark.slow`.
- **Logic in tests:** `test_gaussian_find_position` lines 387-409 contain an
  `if bkgnd_degree is not None:` branch. This conditional logic in a test can
  mask failures if the condition is wrong.

---

## 18. Code coverage

- **Overall coverage: 45.18%** (target: 90%). Measured by running the entire
  test suite with `pytest tests/ --cov=src`.
- **`hst.py`: 0%** (335 statements). This module is excluded from mypy and
  ruff but still ships as part of the package. It requires TinyTim environment
  variables (`TINYTIM`, `PSF_CACHE_DIR`) at import time, making it hard to
  test without mocking.
- **`gaussian.py`: 95%** Missing lines: 75, 89, 93, 97, 101, 105, 664.
  - Lines 75, 89, 93, 97, 101, 105: Property accessors (`sigma_y`, `sigma_x`,
    `mean_y`, `mean_x`) and `angle_subsample` validation dead branch.
  - Line 664: `eval_rect` validation (odd positive shape).
- **`psf.py`: 70%** Missing 82 statements.
  - `_eval_rect_smeared` (lines 187-213): Motion blur path untested.
  - `find_position` edge cases (lines 532-669): box_size validation, edge of
    image, optimizer failure, bad pixel masking.
  - `_fit_psf_func` detailed logging (lines 739-744).
  - `_find_position` detailed logging and optimizer failure (lines 818-820,
    838-839, 941-956).
- **`__init__.py`: 100%**

**To reach 90%:** `hst.py` must be either tested or excluded from coverage
measurement. `psf.py` needs tests for `_eval_rect_smeared`,
`find_position` edge cases, and error paths.

---

## 19. Pytest markers

- **No custom markers are registered** in `pyproject.toml` (`markers` key is
  absent).
- **`--strict-markers` is not enabled.** Any marker typo would be silently
  ignored.
- **Markers used:** Only `@pytest.mark.parametrize` (built-in).
- **No `xfail` or `skip` markers.**
- **No `@pytest.mark.slow`** despite `test_gaussian_find_position` being
  notably slower than others.

**Recommendations:**

- Add `markers = []` to `[tool.pytest.ini_options]`.
- Add `"--strict-markers"` and `"--strict-config"` to `addopts`.
- Consider `@pytest.mark.slow` for `test_gaussian_find_position`.

---

## 20. Test boundary

- **Private imports:** `test_psf.py` line 15 calls
  `PSF._background_gradient_coeffs(...)` directly. This is a private method
  (`_`-prefixed). The test is tightly coupled to the internal implementation.
- **Public API coverage:** `background_gradient_fit` and `find_position` are
  tested through the public API, which is good. However,
  `background_gradient` is tested only indirectly (as a helper in
  `background_gradient_fit` tests).
- **Over-mocking:** No mocking is used, so this is not an issue.
- **`hst.py` public API:** `HSTPSF.eval_point`, `HSTPSF.eval_pixel`,
  `HSTPSF.eval_rect` are completely untested.

---

## 21. Logging assertions

- **`psf.py` contains 43 logger calls** (mix of `.info`, `.debug`, `.warning`)
  used extensively in `find_position` and `_find_position`.
- **Zero `caplog` usage** in the test suite.
- **Key untested logging:**
  - `find_position`: Logs "optimizer did not succeed" at WARNING level when
    optimization fails (line 838). This is the only warning-level log and
    should have a test.
  - `find_position`: Logs entry/exit at INFO level when `detailed_logging=True`.
  - `_fit_psf_func`: Logs per-iteration diagnostics at DEBUG level.
- **Recommendation:** Add at least one test with `caplog` that verifies the
  optimizer-failure warning message.

---

## 22. Pytest configuration

- **Active config file:** `pyproject.toml` (`[tool.pytest.ini_options]`). No
  higher-precedence files (`pytest.toml`, `.pytest.toml`, `pytest.ini`,
  `.pytest.ini`) exist.
- **`testpaths`:** Not set. Pytest collects from the entire repo. Should be set
  to `["tests"]`.
- **`addopts`:** `["-n", "4", "--cov=src"]`. Missing `--strict-markers`,
  `--strict-config`, and `-W error::DeprecationWarning`.
- **`filterwarnings`:** Not configured. Unexpected warnings are silently
  ignored.
- **Plugins installed:** `pytest-xdist` (for `-n`), `pytest-cov`. Both are
  used.
- **Missing plugins:** `pytest-randomly` (for order-independence testing) is
  not listed but would be beneficial.
- **No duplicate config files** detected.

**Recommendations:**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-n", "4", "--cov=src", "--strict-markers", "--strict-config"]
filterwarnings = ["error"]
markers = []
```

---

## 23. Snapshot and golden-file testing

- **No snapshot or golden-file tests** are used.
- **Candidates:** `find_position` returns a complex dict with many keys.
  The full return shape could benefit from a snapshot test rather than
  cherry-picking individual keys.
- **`eval_rect` output:** The full 2-D array could be snapshot-tested for
  regression, though inline `npt.assert_array_almost_equal` is adequate for
  small arrays.

**Not critical for this project, but `find_position` metadata dict is a good
candidate.**

---

## Prompt for an AI agent to fix tests

You are an AI agent tasked with improving the test suite for the `psfmodel`
Python library. The library lives in `src/psfmodel/` and tests are in `tests/`.

**Do not modify any production code.** Only add, modify, or reorganize test
files and `conftest.py`. Preserve all existing passing behavior -- do not remove
or weaken any existing assertion.

### Context

The test suite currently has 45% line coverage (target: 90%). There are two test
files: `tests/test_gaussian.py` (8 test functions) and `tests/test_psf.py`
(2 test functions). There is no `conftest.py`.

### Tasks (ordered by priority)

1. **Exception message assertions:** All 25 `pytest.raises(ValueError)` calls
   lack message assertions. Add `as exc_info` and assert on
   `str(exc_info.value)` for each. Match the exact message from the source
   code.

2. **Add missing failure/validation tests:**
   - `GaussianPSF.__init__`: `angle_subsample` out of range (0, 100, non-int).
   - `GaussianPSF.eval_rect`: even dimensions, negative dimensions.
   - `PSF.find_position`: invalid `box_size` (even, negative).
   - `PSF.find_position`: starting point too close to image edge (returns
     None).
   - `PSF.find_position`: optimizer failure (returns None with warning).
   - `GaussianPSF.gaussian_integral_1d`: sigma <= 0 (currently an `assert`).

3. **Increase coverage for `psf.py` (currently 70%):**
   - Test `_eval_rect_smeared` with non-zero movement via
     `GaussianPSF.eval_rect(..., movement=(0.5, 0.3))`.
   - Test `find_position` edge cases: all pixels masked, too many pixels
     masked, `num_sigma` pixel rejection.
   - Test `find_position` with `detailed_logging=True` and verify log output
     via `caplog`.

4. **Add logging tests:**
   - Test that optimizer failure emits a WARNING containing "did not succeed".
   - Test that `detailed_logging=True` emits INFO-level messages.

5. **Create `tests/conftest.py`** with shared fixtures:
   - `default_psf` -> `GaussianPSF()`
   - `symmetric_psf` -> `GaussianPSF(sigma=(1.0, 1.0))`
   - `asymmetric_psf` -> `GaussianPSF(sigma=(2.0, 3.0))`

6. **Parametrize repetitive tests:**
   - `test_gaussian_1d`: parametrize over `(x, kwargs, expected)` tuples.
   - `test_gaussian_2d`: parametrize over `(y, x, kwargs, expected)` tuples.
   - `test_gaussian_eval_point` ValueError cases: parametrize over
     `(sigma_init, call_kwargs)`.
   - `test_gaussian_eval_pixel` ValueError cases: same approach.
   - `test_bkgnd_gradient_coeffs` ValueError cases: parametrize over
     `(shape, order)`.

7. **Improve `test_gaussian_eval_rect`:**
   - Assert output shape equals `rect_size`.
   - Assert center pixel is the maximum.
   - Assert all values are non-negative.
   - Test with different sigma, scale, base, and offset values.

8. **Remove duplicate assertion:** `test_gaussian_integral_1d` lines 102-104
   duplicate lines 90-92.

9. **Resolve or document TODO comments:** Lines 349, 366-367, 384-385 in
   `test_gaussian_find_position` have commented-out assertions with
   `# TODO: Why?`. Investigate and either fix or add `@pytest.mark.xfail` with
   an issue reference.

10. **Update pytest configuration in `pyproject.toml`:**
    - Add `testpaths = ["tests"]`.
    - Add `"--strict-markers"` and `"--strict-config"` to `addopts`.
    - Add `filterwarnings = ["error"]`.
    - Add `markers = []`.

11. **Coverage target:** Run the full test suite with
    `pytest tests/ --cov=src --cov-report=term-missing` and ensure at least
    90% line coverage for `gaussian.py` and `psf.py`. `hst.py` may be excluded
    from coverage if testing it requires TinyTim; add it to `[tool.coverage.run]
    omit`.

### Constraints

- Do not modify files in `src/`.
- Do not remove or weaken existing assertions.
- All tests must pass with `pytest -n auto`.
- Add type annotations to all new test functions.
- Follow Google-style docstrings for new test functions.
- Use `pytest.approx` for floating-point comparisons.
- Use `npt.assert_array_almost_equal` for array comparisons.
