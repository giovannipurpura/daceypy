# Changelog

All notable changes to DACEyPy are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases up to and including 1.3.1 predate this file and were reconstructed
from the git history; they are summarised more briefly than later entries.

## [Unreleased]

## [1.4.0] - 2026-08-22

### Added

- Adaptive Domain Splitting utilities (`ADS_utils`) and an optimised ADS
  integrator (`ADSintegrator_optimized`), together with `assign_taylor_to_da`
  and a new ADS example, `docs/ADS/4OptimizedADS-Ex.py` (#13).
- A test suite under `tests/`, run with pytest. It covers the DA scalar type,
  the `daceypy.array` type contract, the ADS primitives and the integrators,
  and runs every documented example as a smoke test.
- Continuous integration on GitHub Actions: lint (ruff), type checking
  (mypy), tests across Python 3.9-3.13 on Linux plus macOS and Windows, both
  ends of the supported numpy range, and a packaging check that installs the
  built wheel into a clean environment.
- A `test` extra (`pip install daceypy[test]`) pulling in pytest, matplotlib
  and scipy, and a `lint` extra pinning the ruff and mypy versions CI uses.
- This changelog, and `CONTRIBUTING.md`.

### Fixed

- **Import failure on numpy 2.5 and later.** `import daceypy` raised
  `TypeError: metaclass conflict`, because `numpy.typing.NDArray` became a
  PEP 695 type alias and could no longer be used as a base class. Since a
  fresh `pip install` resolves the newest numpy, the previously published
  releases are effectively uninstallable for new users.
- **`ZeroDivisionError` when propagating to time zero.** The integrator's
  final-time test was purely relative in the final time, so any propagation
  ending at `t = 0` crashed, including the backward leg of a forward-backward
  round trip.
- `ADS.eval` could not be silenced the way its docstring described:
  `log_fun` is called with several arguments, so the suggested
  `lambda s: None` raised `TypeError`. The docstring and the annotation now
  describe the real signature.
- `ADS.canSplit` returns a `bool` instead of a `numpy.bool_`.
- `docs/ADS/1Basics-Ex.py` assigned a one-element sequence to a scalar array
  slot, an error since numpy 2.0.
- **Examples crashed on Windows.** `Example11DARK78.py` and
  `4OptimizedADS-Ex.py` printed non-ASCII characters, which raise
  `UnicodeEncodeError` on a console using the legacy code page. Their output
  is now plain ASCII, and a test checks that every example's string literals
  stay that way.

### Changed

- **Minimum Python is now 3.9** (was 3.7). Python 3.7 and 3.8 are long past
  end of life.
- Packaging metadata moved from `setup.cfg` to `pyproject.toml` (PEP 621),
  with the license declared as a PEP 639 expression. `setup.cfg` and
  `.flake8` are gone.
- Assorted mechanical lint cleanups: import ordering, unused imports,
  trailing whitespace.

### Notes

- `ADS.canSplit` deliberately allows a domain to reach `N_max + 1` splits, for
  parity with the legacy C++ ADS implementation (issue #6, changed in 1.2.0).
  It reads like an off-by-one and is not one.
- The bundled DACE core is still built from commit `2bda904` (2022-10-09).
  Norm estimation on polynomials with very few monomials raises there; it is
  fixed upstream but not yet in the bundled binaries.

## [1.3.1] - 2026-06-13

### Fixed

- Typo in the `RK78_DP` A-matrix entry `A[10, 6]`, which was missing a digit
  in the denominator (#11).

## [1.3.0] - 2025-12-24

### Added

- New propagation modes and utilities for the integrator class (#9).
- License file and updated Python version classifiers.

### Fixed

- Bug in the auxiliary function `_eliminate` causing DA matrix inversion
  errors (#10).

## [1.2.1] - 2024-03-06

### Fixed

- Issue #7 (#8).

### Added

- Python 3.12 metadata.

## [1.2.0] - 2024-03-05

### Added

- Online ADS and further ADS features; improved type hinting of the
  integrator.

### Fixed

- `ADS.canSplit` used a strict comparison, which disagreed with the legacy
  C++ ADS implementation (#6).

### Changed

- Reversed the order of the split check, for performance.

## [1.1.0] - 2023-10-09

### Added

- Adaptive Domain Splitting library and a modular integrator (#4).

## [1.0.5] - 2023-08-02

### Fixed

- `assign` method when keyword arguments are used.

## [1.0.4] - 2023-01-28

### Added

- ARM64 support.

## [1.0.3] - 2023-01-08

### Changed

- Bumped the bundled DACE core to commit `2bda904`.

## [1.0.2] - 2022-10-11

### Fixed

- `compiledDA.eval` method.

## [1.0.1] - 2022-07-26

### Changed

- Rebuilt the Windows and Linux DA core binaries.

[Unreleased]: https://github.com/giovannipurpura/daceypy/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/giovannipurpura/daceypy/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/giovannipurpura/daceypy/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/giovannipurpura/daceypy/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/giovannipurpura/daceypy/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/giovannipurpura/daceypy/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/giovannipurpura/daceypy/compare/v1.0.5...v1.1.0
[1.0.5]: https://github.com/giovannipurpura/daceypy/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/giovannipurpura/daceypy/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/giovannipurpura/daceypy/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/giovannipurpura/daceypy/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/giovannipurpura/daceypy/releases/tag/v1.0.1
