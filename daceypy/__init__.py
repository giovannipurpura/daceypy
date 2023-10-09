"""
# Differential Algebra Core Engine in Python - DACEyPy

DACEyPy is a Python wrapper of DACE,
the Differential Algebra Computational Toolbox
(https://github.com/dacelib/dace).

------------------------------------------------------------------------------

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------
"""

__all__ = (
    "__version__",
    "DACEException",
    "DA",
    "compiledDA",
    "array",
    "Monomial",
    "init",
    "isInitialized",
    "zeros",
    "identity",
    "op",
    "ADS",
    "RK",
    "integrator",
)

from ._version import __version__
from ._DACEException import DACEException
from ._DA import DA
from ._compiledDA import compiledDA
from ._array import array
from ._Monomial import Monomial
from . import op
from ._ADS import ADS
from . import RK
from ._integrator import integrator

init = DA.init
isInitialized = DA.isInitialized
zeros = array.zeros
identity = array.identity
