"""
# Differential Algebra Core Engine in Python - DACEyPy

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Union, overload

import numpy
from numpy.typing import NDArray

import daceypy


def DA(x: Union[daceypy.DA, float, int]) -> daceypy.DA:
    """
    Converts a value to a DA object.

    Args:
        x: value to be converted.

    Returns:
        A constant DA if x is a float, the x-th DA var if x is an int,
        a copy of x if x is a DA object.
    """
    ...


def array(
    x: Union[list, tuple, daceypy.array, NDArray[numpy.double]],
) -> daceypy.array:
    """
    Converts a value to a DACEyPy array.

    Args:
        x: value to be converted.

    Returns:
        A constant DA array if x is composed of floats or ints,
        a copy of x if x is a DACEyPy array.
    """
    ...


@overload
def round(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `round`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def round(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `round`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def round(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `round`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def round(x: float) -> float:
    """
    Compute `round`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def round(x: daceypy.array) -> daceypy.array:
    """
    Compute `round`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def round(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `round`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def root(x: List[float], p: int = 2) -> NDArray[numpy.double]:
    """
    Compute `root`.

    Args:
        x: input value (List[float]).
        p: root to be computed (int, default 2).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def root(x: List[daceypy.DA], p: int = 2) -> daceypy.array:
    """
    Compute `root`.

    Args:
        x: input value (List[daceypy.DA]).
        p: root to be computed (int, default 2).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def root(x: daceypy.DA, p: int = 2) -> daceypy.DA:
    """
    Compute `root`.

    Args:
        x: input value (daceypy.DA).
        p: root to be computed (int, default 2).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def root(x: float, p: int = 2) -> float:
    """
    Compute `root`.

    Args:
        x: input value (float).
        p: root to be computed (int, default 2).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def root(x: daceypy.array, p: int = 2) -> daceypy.array:
    """
    Compute `root`.

    Args:
        x: input value (DACEyPy array).
        p: root to be computed (int, default 2).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def root(x: NDArray[numpy.double], p: int = 2) -> NDArray[numpy.double]:
    """
    Compute `root`.

    Args:
        x: input value (NumPy array).
        p: root to be computed (int, default 2).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def minv(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `minv`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def minv(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `minv`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def minv(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `minv`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def minv(x: float) -> float:
    """
    Compute `minv`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def minv(x: daceypy.array) -> daceypy.array:
    """
    Compute `minv`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def minv(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `minv`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def sqr(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `sqr`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def sqr(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `sqr`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def sqr(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `sqr`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def sqr(x: float) -> float:
    """
    Compute `sqr`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def sqr(x: daceypy.array) -> daceypy.array:
    """
    Compute `sqr`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def sqr(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `sqr`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def sqrt(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `sqrt`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def sqrt(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `sqrt`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def sqrt(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `sqrt`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def sqrt(x: float) -> float:
    """
    Compute `sqrt`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def sqrt(x: daceypy.array) -> daceypy.array:
    """
    Compute `sqrt`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def sqrt(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `sqrt`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def isrt(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `isrt`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def isrt(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `isrt`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def isrt(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `isrt`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def isrt(x: float) -> float:
    """
    Compute `isrt`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def isrt(x: daceypy.array) -> daceypy.array:
    """
    Compute `isrt`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def isrt(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `isrt`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def cbrt(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `cbrt`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def cbrt(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `cbrt`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def cbrt(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `cbrt`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def cbrt(x: float) -> float:
    """
    Compute `cbrt`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def cbrt(x: daceypy.array) -> daceypy.array:
    """
    Compute `cbrt`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def cbrt(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `cbrt`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def icrt(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `icrt`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def icrt(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `icrt`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def icrt(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `icrt`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def icrt(x: float) -> float:
    """
    Compute `icrt`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def icrt(x: daceypy.array) -> daceypy.array:
    """
    Compute `icrt`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def icrt(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `icrt`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def hypot(x: daceypy.DA, other: daceypy.DA) -> daceypy.DA:
    """
    Compute `hypot`.

    Args:
        x: input value (daceypy.DA).
        other: daceypy.DA

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def hypot(x: float, other: float) -> float:
    """
    Compute `hypot`.

    Args:
        x: input value (float).
        other: float

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def exp(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `exp`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def exp(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `exp`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def exp(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `exp`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def exp(x: float) -> float:
    """
    Compute `exp`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def exp(x: daceypy.array) -> daceypy.array:
    """
    Compute `exp`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def exp(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `exp`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def log(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `log`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def log(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `log`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def log(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `log`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def log(x: float) -> float:
    """
    Compute `log`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def log(x: daceypy.array) -> daceypy.array:
    """
    Compute `log`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def log(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `log`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def logb(x: List[float], b: float = 10.0) -> NDArray[numpy.double]:
    """
    Compute `logb`.

    Args:
        x: input value (List[float]).
        b: logarithm base, default 10.0.

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def logb(x: List[daceypy.DA], b: float = 10.0) -> daceypy.array:
    """
    Compute `logb`.

    Args:
        x: input value (List[daceypy.DA]).
        b: logarithm base, default 10.0.

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def logb(x: daceypy.DA, b: float = 10.0) -> daceypy.DA:
    """
    Compute `logb`.

    Args:
        x: input value (daceypy.DA).
        b: logarithm base, default 10.0.

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def logb(x: float, b: float = 10.0) -> float:
    """
    Compute `logb`.

    Args:
        x: input value (float).
        b: logarithm base, default 10.0.

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def logb(x: daceypy.array, b: float = 10.0) -> daceypy.array:
    """
    Compute `logb`.

    Args:
        x: input value (DACEyPy array).
        b: logarithm base, default 10.0.

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def logb(x: NDArray[numpy.double], b: float = 10.0) -> NDArray[numpy.double]:
    """
    Compute `logb`.

    Args:
        x: input value (NumPy array).
        b: logarithm base, default 10.0.

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def log10(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `log10`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def log10(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `log10`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def log10(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `log10`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def log10(x: float) -> float:
    """
    Compute `log10`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def log10(x: daceypy.array) -> daceypy.array:
    """
    Compute `log10`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def log10(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `log10`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def log2(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `log2`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def log2(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `log2`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def log2(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `log2`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def log2(x: float) -> float:
    """
    Compute `log2`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def log2(x: daceypy.array) -> daceypy.array:
    """
    Compute `log2`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def log2(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `log2`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def sin(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `sin`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def sin(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `sin`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def sin(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `sin`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def sin(x: float) -> float:
    """
    Compute `sin`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def sin(x: daceypy.array) -> daceypy.array:
    """
    Compute `sin`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def sin(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `sin`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def cos(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `cos`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def cos(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `cos`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def cos(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `cos`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def cos(x: float) -> float:
    """
    Compute `cos`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def cos(x: daceypy.array) -> daceypy.array:
    """
    Compute `cos`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def cos(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `cos`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def tan(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `tan`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def tan(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `tan`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def tan(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `tan`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def tan(x: float) -> float:
    """
    Compute `tan`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def tan(x: daceypy.array) -> daceypy.array:
    """
    Compute `tan`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def tan(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `tan`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def asin(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `asin`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def asin(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `asin`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def asin(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `asin`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def asin(x: float) -> float:
    """
    Compute `asin`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def asin(x: daceypy.array) -> daceypy.array:
    """
    Compute `asin`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def asin(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `asin`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def acos(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `acos`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def acos(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `acos`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def acos(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `acos`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def acos(x: float) -> float:
    """
    Compute `acos`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def acos(x: daceypy.array) -> daceypy.array:
    """
    Compute `acos`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def acos(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `acos`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def atan(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `atan`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def atan(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `atan`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def atan(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `atan`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def atan(x: float) -> float:
    """
    Compute `atan`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def atan(x: daceypy.array) -> daceypy.array:
    """
    Compute `atan`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def atan(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `atan`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def atan2(x: List[float], other: List[float]) -> NDArray[numpy.double]:
    """
    Compute `atan2`.

    Args:
        x: input value (List[float]).
        other: List[float]

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def atan2(x: List[daceypy.DA], other: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `atan2`.

    Args:
        x: input value (List[daceypy.DA]).
        other: List[daceypy.DA]

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def atan2(x: daceypy.DA, other: daceypy.DA) -> daceypy.DA:
    """
    Compute `atan2`.

    Args:
        x: input value (daceypy.DA).
        other: daceypy.DA

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def atan2(x: float, other: float) -> float:
    """
    Compute `atan2`.

    Args:
        x: input value (float).
        other: float

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def atan2(x: daceypy.array, other: daceypy.array) -> daceypy.array:
    """
    Compute `atan2`.

    Args:
        x: input value (DACEyPy array).
        other: daceypy.array

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def atan2(
    x: NDArray[numpy.double], other: numpy.ndarray
) -> NDArray[numpy.double]:
    """
    Compute `atan2`.

    Args:
        x: input value (NumPy array).
        other: numpy.ndarray

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def sinh(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `sinh`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def sinh(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `sinh`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def sinh(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `sinh`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def sinh(x: float) -> float:
    """
    Compute `sinh`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def sinh(x: daceypy.array) -> daceypy.array:
    """
    Compute `sinh`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def sinh(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `sinh`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def cosh(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `cosh`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def cosh(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `cosh`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def cosh(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `cosh`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def cosh(x: float) -> float:
    """
    Compute `cosh`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def cosh(x: daceypy.array) -> daceypy.array:
    """
    Compute `cosh`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def cosh(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `cosh`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def tanh(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `tanh`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def tanh(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `tanh`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def tanh(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `tanh`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def tanh(x: float) -> float:
    """
    Compute `tanh`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def tanh(x: daceypy.array) -> daceypy.array:
    """
    Compute `tanh`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def tanh(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `tanh`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def asinh(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `asinh`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def asinh(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `asinh`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def asinh(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `asinh`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def asinh(x: float) -> float:
    """
    Compute `asinh`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def asinh(x: daceypy.array) -> daceypy.array:
    """
    Compute `asinh`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def asinh(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `asinh`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def acosh(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `acosh`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def acosh(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `acosh`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def acosh(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `acosh`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def acosh(x: float) -> float:
    """
    Compute `acosh`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def acosh(x: daceypy.array) -> daceypy.array:
    """
    Compute `acosh`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def acosh(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `acosh`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def atanh(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `atanh`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def atanh(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `atanh`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def atanh(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `atanh`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def atanh(x: float) -> float:
    """
    Compute `atanh`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def atanh(x: daceypy.array) -> daceypy.array:
    """
    Compute `atanh`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def atanh(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `atanh`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def erf(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `erf`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def erf(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `erf`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def erf(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `erf`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def erf(x: float) -> float:
    """
    Compute `erf`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def erf(x: daceypy.array) -> daceypy.array:
    """
    Compute `erf`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def erf(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `erf`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def erfc(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `erfc`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def erfc(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `erfc`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def erfc(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `erfc`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def erfc(x: float) -> float:
    """
    Compute `erfc`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def erfc(x: daceypy.array) -> daceypy.array:
    """
    Compute `erfc`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def erfc(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `erfc`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def GammaFunction(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `GammaFunction`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def GammaFunction(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `GammaFunction`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def GammaFunction(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `GammaFunction`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def GammaFunction(x: float) -> float:
    """
    Compute `GammaFunction`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def GammaFunction(x: daceypy.array) -> daceypy.array:
    """
    Compute `GammaFunction`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def GammaFunction(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `GammaFunction`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def LogGammaFunction(x: List[float]) -> NDArray[numpy.double]:
    """
    Compute `LogGammaFunction`.

    Args:
        x: input value (List[float]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def LogGammaFunction(x: List[daceypy.DA]) -> daceypy.array:
    """
    Compute `LogGammaFunction`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def LogGammaFunction(x: daceypy.DA) -> daceypy.DA:
    """
    Compute `LogGammaFunction`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def LogGammaFunction(x: float) -> float:
    """
    Compute `LogGammaFunction`.

    Args:
        x: input value (float).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def LogGammaFunction(x: daceypy.array) -> daceypy.array:
    """
    Compute `LogGammaFunction`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def LogGammaFunction(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `LogGammaFunction`.

    Args:
        x: input value (NumPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def PsiFunction(x: List[float], n: int) -> NDArray[numpy.double]:
    """
    Compute `PsiFunction`.

    Args:
        x: input value (List[float]).
        n: order of the Psi function (n >= 0).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def PsiFunction(x: List[daceypy.DA], n: int) -> daceypy.array:
    """
    Compute `PsiFunction`.

    Args:
        x: input value (List[daceypy.DA]).
        n: order of the Psi function (n >= 0).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def PsiFunction(x: daceypy.DA, n: int) -> daceypy.DA:
    """
    Compute `PsiFunction`.

    Args:
        x: input value (daceypy.DA).
        n: order of the Psi function (n >= 0).

    Returns:
        A daceypy.DA with the result of the operation.
    """
    ...


@overload
def PsiFunction(x: float, n: int) -> float:
    """
    Compute `PsiFunction`.

    Args:
        x: input value (float).
        n: order of the Psi function (n >= 0).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def PsiFunction(x: daceypy.array, n: int) -> daceypy.array:
    """
    Compute `PsiFunction`.

    Args:
        x: input value (DACEyPy array).
        n: order of the Psi function (n >= 0).

    Returns:
        A DACEyPy array with the result of the operation.
    """
    ...


@overload
def PsiFunction(x: NDArray[numpy.double]) -> NDArray[numpy.double]:
    """
    Compute `PsiFunction`.

    Args:
        x: input value (NumPy array).
        n: order of the Psi function (n >= 0).
    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def cons(x: List[daceypy.DA]) -> NDArray[numpy.double]:
    """
    Compute `cons`.

    Args:
        x: input value (List[daceypy.DA]).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def cons(x: daceypy.DA) -> float:
    """
    Compute `cons`.

    Args:
        x: input value (daceypy.DA).

    Returns:
        A float with the result of the operation.
    """
    ...


@overload
def cons(x: daceypy.array) -> NDArray[numpy.double]:
    """
    Compute `cons`.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A NumPy array with the result of the operation.
    """
    ...


@overload
def vnorm(x: daceypy.array) -> daceypy.DA:
    """
    Compute vector norm.

    Args:
        x: input value (DACEyPy array).

    Returns:
        A DA object with the result of the operation.
    """
    ...


@overload
def vnorm(x: NDArray[numpy.double]) -> float:
    """
    Compute vector norm.

    Args:
        x: input value (NumPy array).
    Returns:
        A float with the result of the operation.
    """
    ...
