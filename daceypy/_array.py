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

from __future__ import annotations

from copy import copy
from typing import (Any, List, Mapping, Optional, Sequence, Tuple, Union,
                    overload)

import numpy as np
from numpy.typing import ArrayLike, NDArray

import daceypy

from ._PrettyType import PrettyType


class array(NDArray[np.object_], metaclass=PrettyType):
    """
    N-dimensional array of DA objects.
    """

    def __new__(cls, input_array) -> array:
        arr = np.asarray(input_array, dtype=np.object_)
        return number_to_DA_ufunc(arr).view(cls)

    def copy(self, order="K") -> array:
        """
        Create a copy of the DACEyPy array,
        creating a copy also of all the referenced DA objects.
        """
        return DA_copy_ufunc(super().copy(order))

    __copy__ = copy

    def __bytes__(self):
        raise NotImplementedError

    @staticmethod
    def fromText(
        text: str,
        alphabet: Union[
            Mapping[str, Union[float, daceypy.DA]],
            Sequence[str],
        ] = ("x", "y", "z", "xx", "yy", "zz"),
    ) -> array:
        """
        Compute a symbolic expression using DA.

        Args:
            text: string to evaluate.
            alphabet:
              sequence of strings that will be converted in order to DA vars,
              or dictionary with keys = names and vals = DA objects.

        Returns:
            Result of the evaluation as DA object.

        Raises:
            DACEException
            ValueError
        """

        out = daceypy.DA.fromText_general(text, alphabet)
        if isinstance(out, daceypy.DA):
            raise ValueError(
                "The expression returned a scalar instead of an array. "
                "Use daceypy.DA.fromText() instead of this method.")
        return out

    def vnorm(self) -> daceypy.DA:
        """
        Compute the Euclidean vector norm (length) of a DACEyPy array.

        Returns:
            A DA object containing the result of the operation.

        Derived from C++:
            `DA vnorm(const AlgebraicVector<DA> &obj)`
        """
        return self.sqr().sum().sqrt().item()

    def normalize(self) -> array:
        """
        Normalize the vector.

        Returns:
            A DACEyPy array of unit length.

        Raises:
            DACEException

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::normalize()`

        """
        return self / self.vnorm()

    def inv(self) -> array:
        """
        Compute the matrix inverse.
        Algorithm based on the Gauss elimination with full pivot
        (from the Numerical Cookbook).
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array that is the inverse of the original.

        Raises:
            ValueError

        Derived from C++:
            `AlgebraicMatrix<DA> AlgebraicMatrix<DA>::inv()`
        """
        # Check if matrix is 2D and square
        if self.ndim != 2 or self.shape[0] != self.shape[1]:
            raise ValueError("Matrix must be 2D square to compute the inverse")

        n = self.shape[0]
        k = 0
        det = daceypy.DA(1.0)
        R = list(range(n))
        P = [0] * n
        C1 = [0] * n
        C2 = [0] * n
        AA = self.copy()

        for i in range(n):
            k, err = _pivot(k, i, AA, P, R, C1, C2, det)
            if err:
                raise ValueError("Singolar matrix")
            _eliminate(k, AA, R)

        P = list(range(n))
        for i in range(n - 1, -1, -1):
            P[C1[i]], P[C2[i]] = P[C2[i]], P[C1[i]]

        AI = [[AA[R[i], P[j]] for j in range(n)] for i in range(n)]
        return array(AI)

    def det(self) -> daceypy.DA:
        """
        Compute the determinant of a square DACEyPy array.

        Returns:
            A DA object that is the determinant of the matrix.

        Raises:
            ValueError

        Derived from C++:
            `DA AlgebraicMatrix<DA>::det()`
        """
        # Check if matrix is 2D and square
        if self.ndim != 2 or self.shape[0] != self.shape[1]:
            raise ValueError("Matrix must be 2D square to compute the inverse")

        n = self.shape[0]
        k = 0
        det = daceypy.DA(1.0)
        R = list(range(n))
        P = [0] * n
        C1 = [0] * n
        C2 = [0] * n
        AA = self.copy()

        for i in range(n):
            k, err = _pivot(k, i, AA, P, R, C1, C2, det)
            if err:
                return daceypy.DA(0.0)
            _eliminate(k, AA, R)

        return det

    # *************************************************************************
    # *     Coefficient access routines
    # *************************************************************************

    def cons(self) -> NDArray[np.double]:
        """
        Extract the constant part of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array that contains the constant part of the original.

        Raises:
            DACEException

        Derived from C++:
            `AlgebraicMatrix<double> AlgebraicVector<DA>::cons()`
        """
        return DA_to_number_ufunc(self).view(np.ndarray)

    def linear(self) -> NDArray[np.double]:
        """
        Get the linear part of a DACEyPy array.

        Returns:
            A NumPy array with an additional dimension, that represents the DA
            variable of which the linear part is taken.

        Raises:
            DACEException

        Derived from C++:
            `AlgebraicMatrix<double> AlgebraicVector<DA>::linear()`
        """
        nvar = daceypy.DA.getMaxVariables()
        out = np.empty(self.shape + (nvar, ))
        for i in range(self.shape[0]):
            out[i] = self[i].linear()
        return out

    def concat(self, other: array) -> array:
        """
        Append a DACEyPy array to the end of the current one and return the new vector.

        Args:
            obj: the DACEyPy array to be appended.

        Returns:
            A new DACEyPy array containing the elements of both arrays.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::concat(const std::vector<DA> &obj)`
        """

        return np.concatenate((self, other)).view(array).copy()

    def trim(self, min: int, max: Optional[int] = None) -> array:
        """
        Returns a DACEyPy array with all monomials of order
        less than min and greater than max removed (trimmed).
        The result is copied in a new DACEyPy array.

        Args:
            min: minimum order to be preserved.
            max: maximum order to be preserved.

        Returns:
            A new DACEyPy array containing the result of the trimming.

        Raises:
            DACEException

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::trim(const unsigned int min, const unsigned int max = DA::getMaxOrder())`
        """
        if max is None:
            max = daceypy.DA.getMaxOrder()
        return np.frompyfunc(lambda da: daceypy.DA.trim(da, min, max), 1, 1)(self)

    # *************************************************************************
    # *     Math routines
    # *************************************************************************

    def cross(self, other) -> array:
        """
        Compute the cross product with another DACEyPy array.
        """
        return np.cross(self, other).view(array)

    def __pow__(self, p: Union[float, daceypy.DA]) -> array:  # type: ignore
        """
        Elevate a DACEyPy array to a given power.
        The result is copied in a new DACEyPy array.

        Args:
            p: power at which the DACEyPy array is elevated.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::pow(const int p)`
        """
        return np.power(self, p)  # type: ignore

    pow = __pow__

    def sqrt(self) -> array:
        """
        Compute the square root of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::sqrt()`
        """
        return np.sqrt(self)  # type: ignore[return-value]

    def exp(self) -> array:
        """
        Compute the exponent of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::exp()`
        """
        return np.exp(self)  # type: ignore[return-value]

    def log(self) -> array:
        """
        Compute the natural logarithm of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::log()`
        """
        return np.log(self)  # type: ignore[return-value]

    def sin(self) -> array:
        """
        Compute the sine of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::sin()`
        """
        return np.sin(self)  # type: ignore[return-value]

    def cos(self) -> array:
        """
        Compute the cosine of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::cos()`
        """
        return np.cos(self)  # type: ignore[return-value]

    def tan(self) -> array:
        """
        Compute the tangent of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::tan()`
        """
        return np.tan(self)  # type: ignore[return-value]

    def asin(self) -> array:
        """
        Compute the arcsine of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::asin()`
        """
        return np.arcsin(self)  # type: ignore[return-value]

    arcsin = asin

    def acos(self) -> array:
        """
        Compute the arccosine of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::acos()`
        """
        return np.arccos(self)  # type: ignore[return-value]

    arccos = acos

    def atan(self) -> array:
        """
        Compute the arctangent of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::atan()`
        """
        return np.arctan(self)  # type: ignore[return-value]

    arctan = atan

    def atan2(self, obj: array) -> array:
        """
        Compute the four-quadrant arctangent of Y/X. Y is the current vector,
        whereas X is the AlgebraicVector<DA> in input.
        The result is copied in a new DACEyPy array.

        Args:
            obj: AlgebraicVector<DA>

        Returns:
            A new DACEyPy array containing the result of the operation Y/X in [-pi, pi].

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::atan2(const AlgebraicVector<DA> &obj)`
        """
        return np.arctan2(self, obj)  # type: ignore[return-value]

    arctan2 = atan2

    def sinh(self) -> array:
        """
        Compute the hyperbolic sine of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::sinh()`
        """
        return np.sinh(self)  # type: ignore[return-value]

    def cosh(self) -> array:
        """
        Compute the hyperbolic cosine of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::cosh()`
        """
        return np.cosh(self)  # type: ignore[return-value]

    def tanh(self) -> array:
        """
        Compute the hyperbolic tangent of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::tanh()`
        """
        return np.tanh(self)  # type: ignore[return-value]

    def logb(self, b: float = 10.0) -> array:
        """
        Compute the logarithm of a DACEyPy array with respect to a given base.
        The result is copied in a new DACEyPy array.

        Args:
            b: base with respect to which the logarithm is computed (default = 10).

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::logb(const double b)`
        """
        return np.frompyfunc(lambda da: daceypy.DA.logb(da, b), 1, 1)(self)

    def log10(self) -> array:
        """
        Compute the base 10 logarithm of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.
        """
        return np.frompyfunc(lambda da: daceypy.DA.log10(da), 1, 1)(self)

    def log2(self) -> array:
        """
        Compute the base 2 logarithm of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.
        """
        return np.frompyfunc(lambda da: daceypy.DA.log2(da), 1, 1)(self)

    def isrt(self) -> array:
        """
        Compute the inverse square root of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::isrt()`
        """
        return np.frompyfunc(lambda da: daceypy.DA.isrt(da), 1, 1)(self)

    def cbrt(self) -> array:
        """
        Compute the cubic root of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.
        """
        return np.frompyfunc(lambda da: daceypy.DA.cbrt(da), 1, 1)(self)

    def icrt(self) -> array:
        """
        Compute the inverse cubic root of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.
        """
        return np.frompyfunc(lambda da: daceypy.DA.icrt(da), 1, 1)(self)

    def sqr(self) -> array:
        """
        Compute the square of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::sqr()`
        """
        return np.frompyfunc(lambda da: daceypy.DA.sqr(da), 1, 1)(self)

    def minv(self) -> array:
        """
        Compute the multiplicative inverse of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::minv()`
        """
        return np.frompyfunc(lambda da: daceypy.DA.minv(da), 1, 1)(self)

    def root(self, p: int) -> array:
        """
        Compute the p-th root of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Args:
            p: root to be computed (default = 2).

        Returns:
            A new DACEyPy array.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::root(const int p)`
        """
        return np.frompyfunc(lambda da: daceypy.DA.root(da, p), 1, 1)(self)

    def asinh(self) -> array:
        """
        Compute the hyperbolic arcsine of a DACEyPy array.
        The result is copied in a new DACEyPy array.


        Returns:
            A new DACEyPy array containing the result of the operation.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::asinh()`
        """
        return np.arcsinh(self)  # type: ignore[return-value]

    arcsinh = asinh

    def acosh(self) -> array:
        """
        Compute the hyperbolic arccosine of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array containing the result of the operation.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::acosh()`
        """
        return np.arccosh(self)  # type: ignore[return-value]

    arccosh = acosh

    def atanh(self) -> array:
        """
        Compute the hyperbolic arctangent of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array containing the result of the operation.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::atanh()`
        """
        return np.arctanh(self)  # type: ignore[return-value]

    arctanh = atanh

    def erf(self) -> array:
        """
        Compute the error function of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array containing the result of the operation.
        """
        return np.frompyfunc(lambda da: daceypy.DA.erf(da), 1, 1)(self)

    def erfc(self) -> array:
        """
        Compute the cumulative error function of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array containing the result of the operation.
        """
        return np.frompyfunc(lambda da: daceypy.DA.erfc(da), 1, 1)(self)

    def GammaFunction(self) -> array:
        """
        Compute the GammaFunction function of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array containing the result of the operation.
        """
        return np.frompyfunc(lambda da: daceypy.DA.GammaFunction(da), 1, 1)(self)

    def LogGammaFunction(self) -> array:
        """
        Compute the LogGammaFunction function of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Returns:
            A new DACEyPy array containing the result of the operation.
        """
        return np.frompyfunc(lambda da: daceypy.DA.LogGammaFunction(da), 1, 1)(self)

    def PsiFunction(self, n: int) -> array:
        """
        Compute the PsiFunction function of a DACEyPy array.
        The result is copied in a new DACEyPy array.

        Args:
            n: order of the Psi function (n >= 0).

        Returns:
            A new DACEyPy array containing the result of the operation.
        """
        return np.frompyfunc(lambda da: daceypy.DA.PsiFunction(da, n), 1, 1)(self)

    def deriv(self, p: int) -> array:
        """
        Compute the derivative of a DACEyPy array with respect to variable p.
        The result is copied in a new DACEyPy array.

        Args:
            p: variable with respect to which the derivative is calculated.

        Returns:
            A new DACEyPy array containing the result of the derivation.

        Raises:
            DACEException

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::deriv(const unsigned int p)`
        """
        return np.frompyfunc(lambda da: daceypy.DA.deriv(da, p), 1, 1)(self)

    def integ(self, p: int) -> array:
        """
        Compute the integral of a DACEyPy array with respect to variable p.
        The result is copied in a new DACEyPy array.

        Args:
            p: variable with respect to which the integral is calculated.

        Returns:
            A new DACEyPy array containing the result of the integration.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::integ(const unsigned int p)`
        """
        return np.frompyfunc(lambda da: daceypy.DA.integ(da, p), 1, 1)(self)

    def __setitem__(self, key, value) -> Any:
        """
        This is overridden to be sure to always use the already present
        DA objects in the array as assignment destination.
        """
        for dst_, val_ in np.nditer([self[key], value], ["refs_ok"]):
            # dst_ and val_ are np arrays of shape (1, )
            dst: daceypy.DA = dst_.item()  # get the DA object
            val = val_.item()  # get the object to be copied
            if isinstance(val, int):
                # if val is an int, we want to consider it a constant, not
                # the DA variable #val, so we cast it to a float
                val = float(val)
            # assign val to the content of the DA object dst
            dst.assign(val)

    # The following definitions are only to avoid type checking errors

    def __getitem__(self, key) -> Any:
        return super().__getitem__(key)

    def __add__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__add__(other).view(array)  # type: ignore

    def __sub__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__sub__(other).view(array)  # type: ignore

    def __mul__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__mul__(other).view(array)  # type: ignore

    def __matmul__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__matmul__(other).view(array)  # type: ignore

    def __truediv__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__truediv__(other).view(array)  # type: ignore

    def __radd__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__radd__(other).view(array)  # type: ignore

    def __rsub__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__rsub__(other).view(array)  # type: ignore

    def __rmul__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__rmul__(other).view(array)  # type: ignore

    def __rmatmul__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__rmatmul__(other).view(array)  # type: ignore

    def __rtruediv__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__rtruediv__(other).view(array)  # type: ignore

    def __iadd__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__iadd__(other).view(array)  # type: ignore

    def __isub__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__isub__(other).view(array)  # type: ignore

    def __imul__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__imul__(other).view(array)  # type: ignore

    def __itruediv__(self, other: Union[ArrayLike, daceypy.DA]) -> array:  # type: ignore
        return super().__itruediv__(other).view(array)  # type: ignore

    # *************************************************************************
    # *     Polynomial evaluation routines
    # *************************************************************************

    def compile(self) -> daceypy.compiledDA:
        """
        Compile vector of polynomials and create a compiledDA object.

        Returns:
            The compiled DA object.

        Derived from C++:
            `compiledDA AlgebraicVector<DA>::compile()`
        """
        return daceypy.compiledDA(self)

    @overload
    def eval(self, args: List[float]) -> List[float]:
        """
        Evaluate a DACEyPy array using a list of floats.

        Args:
            args: list of float to use for the evaluation.

        Returns:
            Result of the evaluation as list of floats.

        Raises:
            DACEException

        Derived from C++:
            `DA AlgebraicVector<DA>::eval(const std::vector<double> &args, std::vector<double> &res)`
        """
        ...

    @overload
    def eval(self, args: array) -> array:
        """
        Evaluate a DACEyPy array using a DACEyPy array.

        Args:
            args: DACEyPy array to use for the evaluation.

        Returns:
            Result of the evaluation as DACEyPy array.

        Raises:
            DACEException

        Derived from C++:
            `DA AlgebraicVector<DA>::eval(const std::vector<DA> &args, std::vector<DA> &res)`
        """
        ...

    @overload
    def eval(self, args: NDArray[np.double]) -> NDArray[np.double]:
        """
        Evaluate a DACEyPy array using a NumPy array of doubles.

        Args:
            args: NumPy array of doubles to use for the evaluation.

        Returns:
            Result of the evaluation as NumPy array of doubles.

        Raises:
            DACEException

        Derived from C++:
            `DA AlgebraicVector<DA>::eval(const std::vector<double> &args, std::vector<double> &res)`
        """
        ...

    @overload
    def eval(self, args: List[daceypy.DA]) -> List[daceypy.DA]:
        """
        Evaluate a DACEyPy array using a list of DA objects.

        Args:
            args: list of DA objects to use for the evaluation.

        Returns:
            Result of the evaluation as list of DA objects.

        Raises:
            DACEException

        Derived from C++:
            `DA AlgebraicVector<DA>::eval(const std::vector<DA> &args, std::vector<DA> &res)`
        """
        ...

    def eval(
        self,
        args: Union[
            List[float], List[daceypy.DA], NDArray[np.double], array],
    ) -> Union[List[float], List[daceypy.DA], NDArray[np.double], array]:
        return self.compile().eval(args)

    @overload
    def evalScalar(self, arg: float) -> NDArray[np.double]:
        """
        Evaluate a DACEyPy array with a single argument of type
        float and return vector of results.

        Args:
            arg:
              The value of the first independent DA variable to evaluate with.
              All remaining independent DA variables are assumed to be zero.

        Returns:
            NumPy array with the result of the evaluation.
        """
        ...

    @overload
    def evalScalar(self, arg: daceypy.DA) -> array:
        """
        Evaluate a DACEyPy array with a single argument of type
        DA and return vector of results.

        Args:
            arg: The value of the first independent DA variable to evaluate with.
              All remaining independent DA variables are assumed to be zero.

        Returns:
            DACEyPy array with the result of the evaluation.
        """
        ...

    def evalScalar(self, arg: Union[float, daceypy.DA]) \
            -> Union[NDArray[np.double], array]:
        args = np.array([arg]) if isinstance(arg, (float, int)) else array([arg])
        return self.eval(args)

    @overload
    def __call__(self, arg: float) -> NDArray[np.double]: ...

    @overload
    def __call__(self, arg: daceypy.DA) -> array: ...

    @overload
    def __call__(self, arg: NDArray[np.double]) -> NDArray[np.double]: ...

    @overload
    def __call__(self, arg: List[float]) -> List[float]: ...

    @overload
    def __call__(self, arg: array) -> array: ...

    @overload
    def __call__(self, arg: List[daceypy.DA]) -> List[daceypy.DA]: ...

    def __call__(
        self, arg: Union[
            float, daceypy.DA, NDArray[np.double],
            List[float], array, List[daceypy.DA],
        ]
    ) -> Union[NDArray[np.double], List[float], array, List[daceypy.DA]]:

        if isinstance(arg, (list, array, np.ndarray)):
            return self.eval(arg)
        return self.evalScalar(arg)

    def plug(self, var: int, val: float) -> array:
        """
        Partial evaluation of vector of polynomials. In each element of the vector,
        variable var is replaced by the value val. The resulting vector of DAs
        is returned.

        Args:
            var: variable number to be replaced.
            val: value by which to replace the variable.

        Returns:
            A new DACEyPy array containing the result of the operation.

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::plug(const unsigned int var, const double val)`
        """
        return np.frompyfunc(lambda da: daceypy.DA.plug(da, var, val), 1, 1)(self)

    def invert(self) -> array:
        """
        Invert the polynomials map given by the DACEyPy array.

        Returns:
            The inverted polynomials.

        Raises:
            TypeError
            Exception

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::invert()`
        """
        if self.ndim != 1:
            raise TypeError("This function works only on 1D objects")

        ord = daceypy.DA.getTO()
        nvar = len(self)

        if nvar > daceypy.DA.getMaxVariables():
            raise Exception(
                "DACE::AlgebraicVector<DA>::inverse: "
                "dimension of vector exceeds maximum number of DA variables.")

        # Create DA identity
        DDA = array.identity(nvar)

        # Split map into constant part AC,
        # non-constant part M, and non-linear part AN
        AC = self.cons()
        M = self.trim(1)
        AN = M.trim(2)

        # Extract the linear coefficients matrix
        AL = M.linear()

        # Compute the inverse of linear coefficients matrix
        AI = np.linalg.inv(AL)

        # Compute DA representation of the inverse of the linear part
        # of the map and its composition with non-linear part AN
        AIoAN_: array = AI @ AN  # type: ignore
        AIoAN = daceypy.compiledDA(AIoAN_)
        Linv = array(AI @ DDA)

        # Iterate to obtain the inverse map
        MI: array = Linv
        for i in range(ord):
            daceypy.DA.setTO(i + 1)
            MI = Linv - AIoAN.eval(MI)

        DDA_m_AC: array = DDA - AC
        return MI.eval(DDA_m_AC)

    # *************************************************************************
    # *     Input/Output routines
    # *************************************************************************

    def __str__(self) -> str:
        """
        Get a string representation of the DACEyPy array.

        Returns:
            Reference to output stream out.

        Derived from C++:
            `std::ostream& operator<<(std::ostream &out, const AlgebraicVector<DA> &obj)
            `std::ostream& operator<<(std::ostream &out, const AlgebraicMatrix<DA> &obj)
        """

        out: List[str] = []

        if self.ndim == 1:
            out.append(f"[[[ {len(self)} vector:\n")
            out.extend(f"{i}\n\n" for i in self)
            out.append("]]]")
        elif self.ndim == 2:
            nrows, ncols = self.shape
            out.append(f"[[[ {nrows}x{ncols} matrix:\n")
            for j in range(ncols):
                out.append(f"    Column {j + 1}\n")
                out.extend(f"{self[i, j]}\n" for i in range(nrows))
            out.append("]]]")
        else:
            return f"[[[ {'x'.join(map(str, self.shape))} DACEyPy array ]]]"

        return "".join(out)

    # *****************************************************************************
    # *     Auxiliary functions for ADS
    # *****************************************************************************

    def getTruncationErrors(self, type_: int = 0) -> NDArray[np.double]:
        """
        Return a NumPy array with truncation errors computed
        according to the given norm for each of the elements of the DA vector.

        Args:
            type_: type of the norm to be used, see documentation for DA.estimNorm.

        Raises:
            DACEException
        """
        errors = np.empty(self.shape[0])
        ord = daceypy.DA.getTO()
        el: daceypy.DA
        for i, el in enumerate(self):
            err, _ = el.estimNorm(0, type_, ord + 1)
            errors[i] = err[-1]
        return errors

    # *************************************************************************
    # *     Static factory routines
    # *************************************************************************

    @classmethod
    def identity(cls, n: Optional[int] = None) -> array:
        """
        Return the DA identity of dimension n.

        Args:
            n: dimension of the identity (default: n. of DA vars.).

        Raises:
            DACEException

        Derived from C++:
            `AlgebraicVector<DA> AlgebraicVector<DA>::identity(const size_t n)`
        """
        if n is None:
            n = daceypy.DA.getMaxVariables()
        return cls([daceypy.DA(i + 1) for i in range(n)])

    @classmethod
    def zeros(cls, shape: Union[int, Tuple[int, ...]]) -> array:
        """
        Return a DACEyPy array with null DA values.

        Args:
            shape: shape of the array.

        Raises:
            DACEException
        """
        return cls(np.zeros(shape))


# *****************************************************************************
# *     Auxiliary functions for inverse and determinant computation
# *****************************************************************************

def _pivot(
    k: int, ii: int, A: array, P: List[int], R: List[int],
    C1: List[int], C2: List[int], det: daceypy.DA,
) -> Tuple[int, bool]:
    """Auxiliary function for inverse and determinant computation"""

    im = t = m = 0
    n = A.shape[0]
    for i in range(n):
        if P[i] == 0:
            for j in range(n):
                if P[j] == 0:
                    t = abs(A[R[i], j])
                    if t >= m:
                        im = i
                        k = j
                        m = t
    if m < 1.e-12:
        return k, True
    else:
        det *= A[R[im], k]  # multiply pivot into determinant
        if im != k:
            det *= -1.0  # adjust sign of determinant
        R[im], R[k] = R[k], R[im]
        P[k] = 1  # mark column as done
        # record original row/column of pivot
        C1[ii] = im
        C2[ii] = k
    return k, False


def _eliminate(k: int, A: array, R: List[int]):
    """Auxiliary function for inverse and determinant computation"""

    n = A.shape[0]

    # # Better speed, less accuracy
    # for j in range(n):
    #     if j != k:
    #         A[R[k], j] = A[R[k], j] * A[R[k], k]

    # Better accuracy, less speed
    for j in range(n):
        if j != k:
            A[R[k], j] /= A[R[k], k]

    A[R[k], k] **= -1

    for i in range(n):
        if i != k:
            for j in range(n):
                if j != k:
                    A[R[i], j] -= A[R[i], k] * A[R[k], j]
            A[R[i], k] *= -A[R[k], k]


# *****************************************************************************
# *     Auxiliary functions for "vectorized" functions
# *****************************************************************************

def number_to_DA(n: float) -> daceypy.DA:
    return daceypy.DA.fromNumber(n)


def DA_to_number(da: daceypy.DA) -> float:
    return da.cons()


number_to_DA_ufunc = np.frompyfunc(number_to_DA, 1, 1)
DA_to_number_ufunc = np.vectorize(DA_to_number, [np.double])
DA_copy_ufunc = np.frompyfunc(copy, 1, 1)
