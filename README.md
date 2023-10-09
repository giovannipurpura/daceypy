# DACEyPy

Python wrapper of DACE, the Differential Algebra Computational Toolbox.

## Introduction

DACEyPy is a Python wrapper of DACE, the Differential Algebra Computational Toolbox
(https://github.com/dacelib/dace).

Additionally, since DACEyPy 1.1.0, the Adaptive Domain Splitting (ADS) library is included.
This is documented in the [ADS](https://github.com/giovannipurpura/daceypy/tree/master/docs/ADS) doc folder.

## Setup

DACEyPy requires Python >= 3.7, running under:
- Windows (x86, x64, ARM64),
- Linux (x86, x64, ARM64),
- MacOs (x64, ARM64).

To use this library in other architectures, it is necessary to recompile the
DACE core (as a dynamic-link library) and set the path in the code, see the lib folder.

DACEyPy can be installed using pip.

## Documentation

The documentation is available [here](https://github.com/giovannipurpura/daceypy/blob/master/docs/index.md).

## Tutorials

The original DACE C++ tutorials have been translated to Python and are available in the
[Tutorials](https://github.com/giovannipurpura/daceypy/tree/master/docs/Tutorials) folder.
