# DACEyPy

DACEyPy is a Python wrapper for
[DACE, the Differential Algebra Computational Toolbox](https://github.com/dacelib/dace).
It exposes differential algebra scalars, NumPy-like DA arrays, mathematical
operators, Taylor-map utilities, Runge-Kutta integrators, and Adaptive Domain
Splitting (ADS) tools from Python.

The package ships with precompiled DACE native libraries for the supported
platforms, so the core differential algebra engine can be used directly after
installation on common Windows, Linux, and macOS systems.

## Features

- `DA`: differential algebra scalar objects with arithmetic, elementary
  functions, derivatives, integrals, truncation, evaluation, norms, bounds,
  coefficient access, text parsing, serialization, and compilation.
- `array`: a NumPy `ndarray` subclass for vectors, matrices, and higher
  dimensional containers of DA objects.
- `op`: vectorized operator functions such as `sin`, `cos`, `sqrt`, `exp`,
  `log`, `atan2`, `erf`, `GammaFunction`, `PsiFunction`, `cons`, and `vnorm`.
- `compiledDA`: compiled Taylor maps for faster repeated evaluation.
- `Monomial`: coefficient/order metadata for individual DA monomials.
- `integrator` and `integrator_optimized`: adaptive Runge-Kutta propagation for
  numeric and DA states.
- `RK`: built-in Runge-Kutta coefficient sets including `RK78`, `RK78_DP`,
  `RK54`, and `RK45`.
- `ADS`: Adaptive Domain Splitting primitives for splitting DA domains when
  truncation errors become too large.
- `ADSintegrator` and `ADSintegrator_optimized`: DA propagation with automatic
  domain splitting.
- `DA_utils`: helpers for extracting Taylor-map data and working with symmetric
  tensor-style coefficient layouts.
- `ADS_utils`: helpers for extracting ADS boxes, assigning sample points to
  domains, preparing visualization data, and reporting assignment statistics.

## Installation

DACEyPy requires Python 3.9 or newer and NumPy.

```bash
pip install daceypy
```

For local development from this repository:

```bash
python -m pip install -e ".[test]"
```

The `test` extra adds pytest, plus the SciPy and Matplotlib that the
documentation examples and the ADS visualization helpers use. The core
package needs only NumPy.

## Supported Platforms

The repository includes precompiled dynamic-link libraries in `daceypy/lib` for:

- Windows x86, x64, and ARM64
- Linux i686, x86_64, and aarch64
- macOS x86_64 and ARM64

On other architectures, the DACE core must be recompiled as a dynamic library.
See [`daceypy/lib/README.md`](https://github.com/giovannipurpura/daceypy/blob/master/daceypy/lib/README.md)
for details and the DACE reference revision used for the bundled binaries.

## Quick Start

```python
from daceypy import DA, array
from daceypy.op import sin

# Initialize DACE: order = 3, number of variables = 6
DA.init(3, 6)

# Create the first DA variable
x = DA(1)

# Use either op functions or DA methods
sin_x = sin(x)
same_result = x.sin()

# Create a DA vector [x1, x2, ..., x6]
state = array.identity(6)
state += [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# Operators work on DA arrays too
sin_state = sin(state)

# Expressions can also be parsed from text
parsed = DA.fromText("sin(x)")

print(sin_x)
print(sin_state)
print(parsed - sin_x)
```

More examples are available in
[`docs/basic_example.py`](https://github.com/giovannipurpura/daceypy/blob/master/docs/basic_example.py)
and [`docs/basic_example.ipynb`](https://github.com/giovannipurpura/daceypy/blob/master/docs/basic_example.ipynb).

## Core API Overview

The main public imports are exposed from `daceypy`:

```python
from daceypy import (
    DA,
    array,
    compiledDA,
    Monomial,
    ADS,
    integrator,
    integrator_optimized,
    ADSintegrator,
    ADSintegrator_optimized,
    RK,
    op,
    DA_utils,
    ADS_utils,
)
```

Convenience aliases are also available:

- `daceypy.init` -> `DA.init`
- `daceypy.isInitialized` -> `DA.isInitialized`
- `daceypy.zeros` -> `array.zeros`
- `daceypy.identity` -> `array.identity`

## Differential Algebra Scalars

`DA` objects represent truncated Taylor polynomials. They support regular Python
arithmetic and many DA-specific operations:

- construction from variables, constants, serialized bytes, strings, or text
  expressions
- coefficient and monomial inspection with `getCoefficient`, `setCoefficient`,
  `getMonomial`, and `getMonomials`
- differential operations with `deriv`, `integ`, `trim`, `trunc`, and `round`
- elementary and special functions such as `sqrt`, `exp`, `log`, `sin`, `cos`,
  `erf`, `BesselJFunction`, and `GammaFunction`
- evaluation through `eval`, `evalScalar`, or direct call syntax
- bounds and convergence helpers such as `bound`, `convRadius`, `norm`,
  `orderNorm`, and `estimNorm`
- performance helpers such as `compile`, `cache_enable`, `cache_disable`, and
  `cache_manager`

## DA Arrays

`daceypy.array` inherits from NumPy's `ndarray`, so it can represent vectors,
matrices, and higher-dimensional arrays while preserving DA-aware operations.

Useful methods include:

- constructors: `array.identity`, `array.zeros`, `array.fromText`
- algebra: `inv`, `det`, `cross`, `concat`, `normalize`, `vnorm`
- elementwise math: `sin`, `cos`, `sqrt`, `exp`, `log`, `erf`,
  `GammaFunction`, and other operators mirrored from `DA`
- DA operations: `cons`, `linear`, `deriv`, `integ`, `trim`, `plug`,
  `invert`, `getTruncationErrors`
- evaluation through `eval`, `evalScalar`, or direct call syntax

## Operator Functions

The `daceypy.op` submodule provides functions that work on DA scalars, DA
arrays, Python numeric values, lists, and NumPy arrays where applicable.

Examples:

```python
from daceypy import DA, array
from daceypy import op

DA.init(4, 2)
x = DA(1)
y = DA(2)

f = op.sqrt(1 + x * x) + op.atan2(y, x)
vec = op.exp(array([x, y]))
```

## Integrators

The repository includes adaptive Runge-Kutta integrators for ordinary numeric
states and DA states:

- `integrator`: base adaptive propagation interface
- `integrator_optimized`: optimized propagation interface with support for
  evaluated time grids
- `RK.RK78`, `RK.RK78_DP`, `RK.RK54`, `RK.RK45`: available coefficient sets

The ADS-aware integrators extend this propagation workflow with automatic domain
splitting:

- `ADSintegrator`
- `ADSintegrator_optimized`

## Adaptive Domain Splitting

ADS enables DA-based uncertainty propagation in nonlinear dynamics: it
automatically splits a DA domain into sub-domains whenever the truncation
error of its Taylor expansion grows past a configurable threshold.

See [the ADS overview, references, and examples](https://github.com/giovannipurpura/daceypy/blob/master/docs/ADS/README.md) for the
full API (`ADS`, `ADSintegrator`, `ADSintegrator_optimized`, and the
`ADS_utils` extraction/visualization helpers).

## Repository Layout

```text
daceypy/
  daceypy/
    __init__.py              Public package exports
    core.py                  ctypes bindings to the native DACE library
    _DA.py                   DA scalar implementation
    _array.py                NumPy-like DA array implementation
    _compiledDA.py           Compiled DA/Taylor map support
    _Monomial.py             DA monomial coefficient/order metadata
    _ADS.py                  Adaptive Domain Splitting domain object
    _integrator.py           Adaptive RK integrators
    _ADSintegrator.py        ADS-aware integrators
    RK.py                    Runge-Kutta states and coefficients
    op.py / op.pyi           Operator functions and type stubs
    DA_utils.py              Taylor-map extraction utilities
    ADS_utils.py             ADS extraction/assignment/visualization utilities
    _DACEException.py        Python exception wrapping native DACE error codes
    _PrettyType.py           Metaclass giving DACEyPy classes a clean repr/module name
    _version.py              Package version
    get_platform.py          Platform detection for the bundled native libraries
    lib/                     Bundled native DACE libraries
  docs/
    index.md                 Documentation entry point
    basic_example.py         Minimal Python example
    basic_example.ipynb      Minimal notebook example
    differences.md           Differences from DACE C++
    ADS/                     ADS examples
    Tutorials/               Python translations of DACE C++ tutorials
  tests/                     pytest suite
  .github/workflows/ci.yml   Lint, type check, tests, packaging check
  README.md
  CHANGELOG.md
  CONTRIBUTING.md
  LICENSE
  NOTICE
  pyproject.toml             Packaging, pytest, ruff and mypy configuration
```


## Documentation and Examples

See [`docs/index.md`](https://github.com/giovannipurpura/daceypy/blob/master/docs/index.md)
for the full documentation entry point. The tutorials under `docs/Tutorials`
include Python translations of the original DACE C++ tutorial material.

## Contributing

[`CONTRIBUTING.md`](https://github.com/giovannipurpura/daceypy/blob/master/CONTRIBUTING.md)
covers the development setup, the checks that run in CI, and how pull requests
are handled.
[`CHANGELOG.md`](https://github.com/giovannipurpura/daceypy/blob/master/CHANGELOG.md)
has the release history.

## Notes on DACE Compatibility

DACEyPy replicates most DACE C++ features, with Python-oriented differences:

- arrays are represented by one NumPy-based `daceypy.array` class instead of
  separate algebraic vector/matrix classes
- mathematical operators are available through `daceypy.op` and as methods on
  `DA` or `array`
- DA objects and DA arrays can be evaluated using call syntax
- DA objects can be used as both bases and exponents in power expressions
- DA caching and output arguments are available for performance-sensitive code
- `DA.fromText` and `array.fromText` can parse expressions, but should only be
  used with trusted input

See [`docs/differences.md`](https://github.com/giovannipurpura/daceypy/blob/master/docs/differences.md)
for the full notes.

## License

DACEyPy is licensed under the Apache License, Version 2.0. See
[`LICENSE`](https://github.com/giovannipurpura/daceypy/blob/master/LICENSE) and
[`NOTICE`](https://github.com/giovannipurpura/daceypy/blob/master/NOTICE).
