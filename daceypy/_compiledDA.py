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

from ctypes import POINTER, c_double, c_uint, pointer
from typing import List, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray

from . import _DA, _array, core
from ._DACEException import DACEException
from ._PrettyType import PrettyType


class compiledDA(metaclass=PrettyType):

    __slots__ = "dim", "ac", "terms", "vars", "ord"

    # *************************************************************************
    # *     Constructors & Destructors
    # *************************************************************************

    def __init__(self, da: Union[_DA.DA, List[_DA.DA], _array.array]):
        """
        Create a compiledDA object from one or more DA objects.

        Args:
            da: DA object(s) to use as source.

        Raises:
            DACEException

        Derived from C++:
            `compiledDA::compiledDA(const std::vector<DA> &da)`
            `compiledDA::compiledDA(const DA &da)`
        """

        if isinstance(da, _DA.DA):
            da = [da]
        elif isinstance(da, _array.array):
            if da.ndim != 1:
                raise TypeError("This function works only on 1D objects")

        dim = len(da)

        if not dim:
            raise DACEException(16, 4)

        c_ac = (c_double * (_DA.DA.getMaxMonomials() * (dim + 2)))()
        c_terms = c_uint()
        c_vars = c_uint()
        c_ord = c_uint()

        da_array_t = POINTER(core.DACEDA) * dim
        da_array = da_array_t(*(pointer(da_elem.m_index) for da_elem in da))
        core.EvalTree(da_array, dim, c_ac, c_terms, c_vars, c_ord)

        self.dim: int = dim
        self.ac: Tuple[float] = tuple(c_ac)  # type: ignore
        self.terms: int = c_terms.value
        self.vars: int = c_vars.value
        self.ord: int = c_ord.value

    def __hash__(self) -> int:
        return hash((self.dim, self.ac, self.terms, self.vars, self.ord))

    def __eq__(self, other) -> bool:
        if not isinstance(other, compiledDA):
            return False
        if self.dim != other.dim:
            return False
        if self.ac != other.ac:
            return False
        if self.terms != other.terms:
            return False
        if self.vars != other.vars:
            return False
        if self.ord != other.ord:
            return False
        return True

    # Notes on methods not ported from C++:
    # - `compiledDA::compiledDA(const compiledDA &cda)`
    #   Create a copy of a compiledDA object.
    #   -> can be achieved using deepcopy(cda)
    # - `compiledDA::~compiledDA()` (destructor)
    #   -> not necessary since Python garbage collector frees memory
    # - `compiledDA& compiledDA::operator=(const compiledDA &cda)` (assignment)
    #   -> in order to make compiledDA immutable and therefore hashable

    # *************************************************************************
    # *     Evaluation overloads and template specialization
    # *************************************************************************

    @overload
    def eval(self, args: List[float]) -> List[float]:
        """
        Evaluate the compiledDA object using a list of floats.

        Args:
            args: list of float to use for the evaluation.

        Returns:
            Result of the evaluation as list of floats.

        Raises:
            DACEException

        Derived from C++:
            `void compiledDA::eval(const std::vector<double> &args, std::vector<double> &res)`
        """
        ...

    @overload
    def eval(self, args: _array.array) -> _array.array:
        """
        Evaluate the compiledDA object using a DACEyPy array.

        Args:
            args: DACEyPy array to use for the evaluation.

        Returns:
            Result of the evaluation as DACEyPy array.

        Raises:
            DACEException

        Derived from C++:
            `void compiledDA::eval(const std::vector<DA> &args, std::vector<DA> &res)`
        """
        ...

    @overload
    def eval(self, args: NDArray[np.double]) -> NDArray[np.double]:
        """
        Evaluate the compiledDA object using a NumPy array of doubles.

        Args:
            args: NumPy array of doubles to use for the evaluation.

        Returns:
            Result of the evaluation as NumPy array of doubles.

        Raises:
            DACEException

        Derived from C++:
            `void compiledDA::eval(const std::vector<double> &args, std::vector<double> &res)`
        """
        ...

    @overload
    def eval(self, args: List[_DA.DA]) -> List[_DA.DA]:
        """
        Evaluate the compiledDA object using a list of DA objects.

        Args:
            args: list of DA objects to use for the evaluation.

        Returns:
            Result of the evaluation as list of DA objects.

        Raises:
            DACEException

        Derived from C++:
            `void compiledDA::eval(const std::vector<DA> &args, std::vector<DA> &res)`
        """
        ...

    def eval(
        self,
        args: Union[
            List[float], List[_DA.DA], NDArray[np.double], _array.array],
    ) -> Union[List[float], List[_DA.DA], NDArray[np.double], _array.array]:

        if isinstance(args, np.ndarray) and args.ndim != 1:
            raise TypeError("This function works only on 1D objects")

        narg = len(args)

        xm: Union[_array.array, NDArray[np.double]]
        res: Union[_array.array, NDArray[np.double]]

        if narg == 0 or isinstance(args[0], (float, int, np.number)):
            p = 2
            xm = np.zeros(self.ord + 1)
            res = np.zeros(self.dim)

            # prepare temporary powers
            xm[0] = 1.0

            # constant part
            for i in range(self.dim):
                res[i] = self.ac[p]
                p += 1

            # higher order terms
            for i in range(1, self.terms):
                jl = int(self.ac[p])
                p += 1
                jv = int(self.ac[p]) - 1
                p += 1
                if jv < narg:
                    xm[jl] = xm[jl - 1] * args[jv]
                else:
                    xm[jl] = 0.0
                for j in range(self.dim):
                    res[j] += xm[jl] * self.ac[p]
                    p += 1
        else:
            jlskip = self.ord + 1
            p = 2

            xm = _array.array.zeros(self.ord + 1)
            tmp = _DA.DA()
            res = _array.array.zeros(self.dim)

            # prepare temporary powers
            xm[0] = 1.0

            # constant part
            for i in range(self.dim):
                res[i] = self.ac[p]
                p += 1

            # higher order terms
            for i in range(1, self.terms):
                jl = int(self.ac[p])
                p += 1
                jv = int(self.ac[p]) - 1
                p += 1
                if jl > jlskip:
                    p += self.dim
                    continue
                if jv >= narg:
                    jlskip = jl
                    p += self.dim
                    continue

                jlskip = self.ord + 1
                core.Multiply(xm[jl - 1], args[jv], xm[jl])
                for j in range(self.dim):
                    if(self.ac[p] != 0.0):
                        core.MultiplyDouble(xm[jl], self.ac[p], tmp)
                        core.Add(res[j].m_index, tmp, res[j].m_index)
                    p += 1

        if isinstance(args, list):
            return res.tolist()

        return res

    @overload
    def evalScalar(self, arg: float) -> NDArray[np.double]:
        """
        Evaluate the compiled polynomial with a single argument of type
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
    def evalScalar(self, arg: _DA.DA) -> _array.array:
        """
        Evaluate the compiled polynomial with a single argument of type
        DA and return vector of results.

        Args:
            arg: The value of the first independent DA variable to evaluate with.
              All remaining independent DA variables are assumed to be zero.

        Returns:
            DACEyPy array with the result of the evaluation.
        """
        ...

    def evalScalar(self, arg: Union[float, _DA.DA]) \
            -> Union[NDArray[np.double], _array.array]:
        args = np.array([arg]) if isinstance(arg, (float, int)) else _array.array([arg])
        return self.eval(args)

    @overload
    def __call__(self, arg: float) -> NDArray[np.double]: ...

    @overload
    def __call__(self, arg: _DA.DA) -> _array.array: ...

    @overload
    def __call__(self, arg: NDArray[np.double]) -> NDArray[np.double]: ...

    @overload
    def __call__(self, arg: List[float]) -> List[float]: ...

    @overload
    def __call__(self, arg: _array.array) -> _array.array: ...

    @overload
    def __call__(self, arg: List[_DA.DA]) -> List[_DA.DA]: ...

    def __call__(
        self, arg: Union[
            float, _DA.DA, NDArray[np.double],
            List[float], _array.array, List[_DA.DA],
        ]
    ) -> Union[NDArray[np.double], List[float], _array.array, List[_DA.DA]]:

        if isinstance(arg, (list, _array.array, np.ndarray)):
            return self.eval(arg)
        return self.evalScalar(arg)

    # *************************************************************************
    # *     Member access routines
    # *************************************************************************

    def getAc(self) -> Tuple[float]:
        return self.ac

    def getDim(self) -> int:
        return self.dim

    def getOrd(self) -> int:
        return self.ord

    def getVars(self) -> int:
        return self.vars

    def getTerms(self) -> int:
        return self.terms

    def __getattr__(self, k):
        raise TypeError

    def __delattr__(self, k):
        raise TypeError
