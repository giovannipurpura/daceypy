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

import collections.abc
from ctypes import (byref, c_double, c_int, c_uint, c_void_p, cast,
                    create_string_buffer)
from typing import (Any, List, Mapping, Optional, Sequence, Tuple, Union,
                    overload)

import numpy as np
from numpy.typing import NDArray

import daceypy

from . import _compiledDA, core
from ._DACEException import DACEException
from ._Monomial import Monomial
from ._PrettyType import PrettyType

c_double_p = core.c_double_p

DACE_MAJOR_VERSION = 2
DACE_MINOR_VERSION = 0
DACE_PATCH_VERSION = 1

DACE_STRLEN = 140


class DA(metaclass=PrettyType):

    __slots__ = "m_index", "_as_parameter_"

    initialized = False
    TOstack: List[int] = []
    _DACEDA_cache_enabled: bool = False
    _DACEDA_cache: List[Tuple[core.DACEDA, Any]] = []

    @classmethod
    def init(cls, ord: int, nvar: int) -> None:
        """
        Initialize the DACE control arrays and set the maximum order and the
        maximum number of variables.

        Notes:
        MUST BE CALLED BEFORE ANY OTHER DA ROUTINE CAN BE USED.
        This routine performs a mandatory version check to compare the version
        of the Python interface to the version of the DACE library that is
        linked dynamically at runtime.

        Args:
            ord: order of the Taylor polynomials.
            nvar: number of variables considered.

        Raises:
            DACEException

        See also:
            DA.checkVersion

        Derived from C++:
            `void DA::init(const unsigned int ord, const unsigned int nvar)`
        """
        cls.checkVersion()
        core.Initialize(ord, nvar)
        cls.initialized = True

    @classmethod
    def isInitialized(cls) -> bool:
        """
        Get the inizialization status of the DACE.

        Returns:
            True if the DACE has previously been initialized by a call
            to DA.init, False otherwise.

        See also:
            DA.init

        Derived from C++:
            `bool DA::isInitialized()`
        """
        return cls.initialized

    @staticmethod
    def version() -> Tuple[int, int, int]:
        """
        Get the major, minor and patch version of DACE core.

        Returns:
            Tuple of integers repesenting the DACE core version as:
            (major, minor, patch).

        Raises:
            DACEException

        Derived from C++:
            `void DA::version(int &maj, int &min, int &patch)`
        """
        maj = c_int()
        min = c_int()
        patch = c_int()
        core.GetVersion(maj, min, patch)
        return maj.value, min.value, patch.value

    @classmethod
    def checkVersion(cls) -> None:
        """
        Check the DACE core library version linked to this C++ interface
        against the interface version and throw an exception if the versions
        don't match.

        This routine is called automatically by DA.init()
        to ensure compatibility with the current runtime environment.

        Raises:
            DACEException

        Derived from C++:
            `void DA::checkVersion()`
        """
        maj, min, _ = cls.version()
        if maj != DACE_MAJOR_VERSION or min != DACE_MINOR_VERSION:
            raise DACEException(20, 99)

    @staticmethod
    def getMaxOrder() -> int:
        """
        Get the maximum order currently set for the computations.

        Returns:
            Maximum order for the computations.

        Raises:
            DACEException

        Derived from C++:
            `unsigned int DA::getMaxOrder()`
        """
        return core.GetMaxOrder()

    @staticmethod
    def setEps(eps: float) -> float:
        """
        Set the cutoff value eps to a new value and return the previous value.

        Args:
            eps: new cutoff value.

        Returns:
            The previous cutoff value of eps, or zero if undefined.

        Raises:
            DACEException

        Derived from C++:
            `double DA::setEps(const double eps)`
        """
        return core.SetEpsilon(eps)

    @staticmethod
    def getEps() -> float:
        """
        Return the cutoff value eps currently set for the computations.

        Returns:
            The cutoff value of eps, or zero if undefined.

        Raises:
            DACEException

        Derived from C++:
            `double DA::getEps()`
        """
        return core.GetEpsilon()

    @staticmethod
    def getEpsMac() -> float:
        """
        Return the machine epsilon (pessimistic estimate).

        Returns:
            The machine epsilon, or zero if undefined.

        Raises:
            DACEException

        Derived from C++:
            `double DA::getEpsMac()`
        """
        return core.GetMachineEpsilon()

    @staticmethod
    def getMaxVariables() -> int:
        """
        Return the maximum number of variables set for the computations.

        Returns:
            The maximum number of variables, or zero if undefined.

        Raises:
            DACEException

        Derived from C++:
            `unsigned int DA::getMaxVariables()`
        """
        return core.GetMaxVariables()

    @staticmethod
    def getMaxMonomials() -> int:
        """
        Return the maximum number of monomials available with the
        order and number of variables specified.

        Returns:
            The maximum number of monomials, or zero if undefined.

        Raises:
            DACEException

        Derived from C++:
            `unsigned int DA::getMaxMonomials()`
        """
        return core.GetMaxMonomials()

    @staticmethod
    def setTO(ot: Optional[int] = None) -> int:
        """
        Set the truncation order to a new value and return the previous one.
        All terms larger than the truncation order are discarded
        in subsequent operations.

        Args:
            ot: new truncation order, default DA.getMaxOrder().

        Returns:
            previous truncation order, or zero if undefined.

        Raises:
            DACEException

        See also:
            DA.getTO
            DA.pushTO
            DA.popTO

        Derived from C++:
            `unsigned int DA::setTO(const unsigned int ot = DA::getMaxOrder())`
        """
        if ot is None:
            ot = DA.getMaxOrder()
        return core.SetTruncationOrder(ot)

    @staticmethod
    def getTO() -> int:
        """
        Get the truncation order currently set for the computations.
        All terms larger than the truncation order are discarded
        in subsequent operations.

        Returns:
            current truncation order, or zero if undefined.

        Raises:
            DACEException

        See also:
            DA.setTO
            DA.pushTO
            DA.popTO

        Derived from C++:
            `unsigned int DA::getTO()`
        """
        return core.GetTruncationOrder()

    @classmethod
    def pushTO(cls, ot: Optional[int] = None) -> None:
        """
        Set a new truncation order, saving the previous one on the truncation
        order stack. All terms larger than the truncation order are discarded
        in subsequent operations.

        Args:
            ot: new truncation order, default DA.getMaxOrder().

        Raises:
            DACEException

        See also:
            DA.getTO
            DA.setTO
            DA.popTO

        Derived from C++:
            `void DA::pushTO(const unsigned int ot)`
        """
        if ot is None:
            ot = DA.getMaxOrder()
        cls.TOstack.append(core.SetTruncationOrder(ot))

    @classmethod
    def popTO(cls) -> Optional[int]:
        """
        Restore the previous truncation order from the truncation order stack.
        All terms larger than the truncation order are discarded
        in subsequent operations.

        Returns:
            Previous truncation order or None.

        Raises:
            DACEException

        See also:
            DA.getTO
            DA.setTO
            DA.pushTO

        Derived from C++:
            `void DA::popTO()`
        """
        if cls.TOstack:
            return core.SetTruncationOrder(cls.TOstack.pop())
        return None

    # *************************************************************************
    # *     Constructors & Destructors
    # *************************************************************************

    @overload
    def __init__(self) -> None:
        """
        Create an empty DA object representing the constant zero function.

        Raises:
            DACEException

        Derived from C++:
            `DA::DA()`
        """
        ...

    @overload
    def __init__(self, da: DA) -> None:
        """
        Create a copy of a DA object.

        Args:
            da: DA object to copy

        Raises:
            DACEException

        Derived from C++:
            `DA::DA(const DA &da)`
        """
        ...

    @overload
    def __init__(self, da: bytes) -> None:
        """
        Create a DA object from a DA blob.

        Args:
            da: DA blob to be decoded

        Raises:
            DACEException
        """
        ...

    @overload
    def __init__(self, da: str) -> None:
        """
        Create a DA object from a DA string.

        Args:
            da: DA string to be decoded

        Raises:
            DACEException

        Derived from C++:
            `DA DA::fromString(const std::vector<std::string> &str)`
        """
        ...

    @overload
    def __init__(self, i: int, c: float = 1.0) -> None:
        """
        Create a DA object as c times the independent variable number i.

        Args:
            i: independent variable number.
            c: variable coefficient.

        Raises:
            DACEException

        Derived from C++:
            `DA::DA(const unsigned int i, const double c)`
        """
        ...

    @overload
    def __init__(self, c: float) -> None:
        """
        Create a DA object with the constant part equal to c.

        Args:
            c: constant value to be assigned to the DA object.

        Raises:
            DACEException

        Derived from C++:
            `DA::DA(const double c)`
        """
        ...

    def __init__(self, *args, **kwargs) -> None:

        try:
            self.m_index, self._as_parameter_ = self._DACEDA_cache.pop()
        except IndexError:
            # Allocate a DACEDA C struct
            self.m_index = core.DACEDA()
            # Assign the reference to m_index to _as_parameter_,
            # to be able to use the DA object as a parameter to C functions
            self._as_parameter_ = byref(self.m_index)

        # Assign the value
        if args or kwargs:
            self.assign(*args, **kwargs)

    def copy(self) -> DA:
        """
        Create a copy of the DA object.

        Raises:
            DACEException
        """
        return DA(self)

    __copy__ = copy

    def __del__(self) -> None:
        """
        Destroy a DA object and free the associated object in the DACE core.

        Derived from C++:
            `DA::~DA() throw()`
        """
        if self._DACEDA_cache_enabled:
            self._DACEDA_cache.append((self.m_index, self._as_parameter_))

    @classmethod
    def fromText(
        cls,
        text: str,
        alphabet: Union[
            Mapping[str, Union[float, DA]],
            Sequence[str],
        ] = ("x", "y", "z", "xx", "yy", "zz"),
    ) -> DA:
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

        out = cls.fromText_general(text, alphabet)
        if isinstance(out, daceypy.array):
            raise ValueError(
                "The expression returned an array instead of a scalar. "
                "Use daceypy.array.fromText() instead of this method.")
        return out

    @classmethod
    def fromText_general(
        cls,
        text: str,
        alphabet: Union[
            Mapping[str, Union[float, DA]],
            Sequence[str],
        ] = ("x", "y", "z", "xx", "yy", "zz"),
    ) -> Union[DA, daceypy.array]:
        """
        Compute a symbolic expression using DA.

        Args:
            text: string to evaluate.
            alphabet:
              sequence of strings that will be converted in order to DA vars,
              or dictionary with keys = names and vals = DA objects.

        Returns:
            Result of the evaluation as DA object or DACEyPy array,
            depending on the expression.

        Raises:
            DACEException
            ValueError
        """

        class translation_table(collections.abc.Mapping):

            def __iter__(self): ...

            def __len__(self): ...

            def __getitem__(self, k):
                try:
                    if isinstance(alphabet, collections.abc.Sequence):
                        return DA(alphabet.index(k) + 1)
                    return alphabet[k]
                except (ValueError, KeyError):
                    try:
                        return getattr(daceypy.op, k)
                    except AttributeError:
                        raise ValueError(
                            f"Unknown variable or function: '{k}'") from None

        result = eval(text, {}, translation_table())
        if isinstance(result, DA):
            return result
        if isinstance(result, (float, int)):
            return DA.fromNumber(result)
        if isinstance(result, (collections.abc.Sequence, np.ndarray)):
            return daceypy.array(result)
        if isinstance(result, daceypy.array):
            return result
        raise ValueError(
            f"Unexpected type of return value of expression: {type(result)}")

    # *************************************************************************
    # *     Coefficient access routines
    # *************************************************************************

    def cons(self) -> float:
        """
        Return the constant part of a DA object.

        Returns:
            A double corresponding to the constant part of the DA object.

        Raises:
            DACEException

        Derived from C++:
            `double DA::cons()`
        """
        return core.GetConstant(self)

    def linear(self) -> NDArray[np.double]:
        """
        Get linear part of a DA object.

        Returns:
            NumPy array of doubles containing the linear coefficients of
            each independent DA variable in the DA object.

        Raises:
            DACEException

        Derived from C++:
            `AlgebraicVector<double> DA::linear()`
        """
        v = np.empty(core.GetMaxVariables())
        address = cast(
            c_void_p(v.__array_interface__["data"][0]), c_double_p)
        core.GetLinear(self, address)
        return v

    def gradient(self) -> daceypy.array:
        """
        Compute the gradient of the DA object.

        Returns:
            A DACEArray containing the derivatives of the DA object
            with respect to all independent DA variables.

        Raises:
            DACEException

        Derived from C++:
            `AlgebraicVector<DA> DA::gradient()`
        """
        nvar = core.GetMaxVariables()
        return daceypy.array([self.deriv(i) for i in range(1, nvar + 1)])

    def getCoefficient(self, jj: List[int]) -> float:
        """
        Get a specific coefficient of a DA object.

        Args:
            jj: vector of the exponents of the coefficient to retrieve.

        Returns:
            The coefficient of the DA object corresponding
            to the given vector of exponents.

        Raises:
            DACEException

        Derived from C++:
            `double DA::getCoefficient(const std::vector<unsigned int> &jj)`
        """
        nvar: int = core.GetMaxVariables()
        temp = (c_uint * nvar)(*jj[:nvar])
        coeff: float = core.GetCoefficient(self, temp)
        return coeff

    def setCoefficient(self, jj: List[int], coeff: float) -> None:
        """
        Set a specific coefficient into a DA object.

        Args:
            jj: vector of the exponents of the coefficient to be set.
            coeff: value to be set as coefficient.

        Raises:
            DACEException

        Derived from C++:
            `void DA::setCoefficient(const std::vector<unsigned int> &jj, const double coeff)`
        """
        nvar: int = core.GetMaxVariables()
        temp = (c_uint * nvar)(*jj[:nvar])
        core.SetCoefficient(self, temp, coeff)

    def getMonomial(self, npos: int, out: Optional[Monomial] = None) -> Monomial:
        """
        Return the Monomial corresponding to the non-zero coefficient at the
        given position in the DA object (monomials use one based indexing!).

        Args:
            npos: position within the DA object.
              The ordering of the Monomials within a DA object is
              implementation dependent and does not correspond to the order
              in which Monomials are listed in the ASCII output routines.
            m: destination Monomial. If None, a new Monomial is created.

        Returns:
            A Monomial object containing both the coefficient and
            the vector of exponents corresponding to the given position.
            If the requested monomial is not present in the DA object,
            a Monomial with coefficient set to 0.0 is returned.

        Raises:
            DACEException

        See also:
            Monomial
            DA.getMonomial

        Derived from C++:
            `void DA::getMonomial(const unsigned int npos, Monomial &m)`
        """
        if out is None:
            out = Monomial()
        core.GetCoefficientAt(self, npos, out.m_jj, out.m_coeff)
        return out

    def getMonomials(self) -> List[Monomial]:
        """
        Get a list of all Monomials in the DA object
        (differently from getMonomial() where only a single Monomial,
        corresponding to a specified position in the DA object, is returned).

        Returns:
            A list of Monomial objects containing both the coefficient and
            the exponents corresponding to each monomial in the DA object.
            The monomials are returned in the same order as in the DACE ASCII
            output (that is, they are sorted by order).

        Raises:
            DACEException

        Derived from C++:
            `std::vector<Monomial> DA::getMonomials()`
        """
        s: int = self.size()
        out: List[Monomial] = []

        for i in range(s):
            m = Monomial()
            core.GetCoefficientAt(self, i + 1, m.m_jj, m.m_coeff)
            out.append(m)

        # sort monomials by order
        out.sort(key=lambda m: m.order())

        return out

    # *************************************************************************
    # *     Assignments
    # *************************************************************************

    @overload
    def assign(self, da: DA) -> None:
        """
        Assign to the current object a copy of the DA object da.

        Args:
            da: DA object to copy

        Raises:
            DACEException
        """
        ...

    @overload
    def assign(self, da: bytes) -> None:
        """
        Assign to the current object data from a DA blob.

        Args:
            da: DA blob to be decoded

        Raises:
            DACEException
        """
        ...

    @overload
    def assign(self, da: str) -> None:
        """
        Assign to the current object data from a DA string.

        Args:
            da: DA string to be decoded

        Raises:
            DACEException
        """
        ...

    @overload
    def assign(self, i: int, c: float) -> None:
        """
        Assign to the current object c times the independent variable number i.

        Args:
            i: independent variable number.
            c: variable coefficient.

        Raises:
            DACEException
        """
        ...

    @overload
    def assign(self, c: float) -> None:
        """
        Assign to the current object a constant equal to c.

        Args:
            c: constant value to be assigned to the DA object.

        Raises:
            DACEException
        """
        ...

    def assign(self, *args, **kwargs) -> None:
        i = c = da = None

        # Process keyword arguments
        for kw, val in kwargs.items():
            if kw == "i":
                i = val
                if not isinstance(i, int):
                    raise TypeError(
                        "Keyword argument i must be of type int, "
                        f"\"{type(i).__name__}\" was given")
            elif kw == "c":
                c = val
                if not isinstance(c, float):
                    raise TypeError(
                        "Keyword argument c must be of type float, "
                        f"\"{type(c).__name__}\" was given")
            elif kw == "da":
                da = val
                if not isinstance(da, (DA, bytes, str)):
                    raise TypeError(
                        "Keyword argument da must be of type "
                        "DA or bytes or str, "
                        f"\"{type(da).__name__}\" was given")
                if args or len(kwargs) > 1:
                    raise TypeError(
                        "No other argument can be passed when da is used")
            else:
                raise TypeError(f"Unexpected keyword argument '{kw}'")

        # Get positional arguments
        if not args:
            v1 = v2 = None
        elif len(args) == 1:
            v1 = args[0]
            v2 = None
        elif len(args) == 2:
            v1, v2 = args
        else:
            raise TypeError(
                "Maximum 2 positional arguments are accepted, "
                f"but {len(args)} were given")

        # Process positional arguments
        if isinstance(v1, int):
            if i is not None:
                raise TypeError("Got multiple values for argument 'i'")
            i = v1
            if v2 is None:
                pass
            elif isinstance(v2, float):
                if c is not None:
                    raise TypeError("Got multiple values for argument 'c'")
                c = v2
            else:
                raise TypeError(
                    "The second positional argument must be a float")
        elif isinstance(v1, float):
            if v2 is not None:
                raise TypeError(
                    "When the first positional argument is a float, "
                    "no other argument can be passed")
            if c is not None:
                raise TypeError("Got multiple values for argument 'c'")
            c = v1
            core.CreateConstant(self, c)
        elif isinstance(v1, (DA, bytes, str)):
            if v2 is not None:
                raise TypeError(
                    "When the first positional argument is a DA object or "
                    "bytes or str, no other argument can be passed")
            da = v1

        # Execute the corresponding core function
        if da is not None:
            if isinstance(da, DA):
                core.Copy(da, self)
            elif isinstance(da, str):
                # split in lines
                lines = da.splitlines()
                # create corresponding bytes,
                # padding each line up to DACE_STRLEN
                ss = create_string_buffer(
                    b"".join(
                        line.encode().ljust(DACE_STRLEN) for line in lines
                    )
                )
                core.Read(self, ss, len(lines))
            else:
                core.ImportBlob(da, self)
        elif c is not None:
            if i is None:
                core.CreateConstant(self, c)
            else:
                core.CreateVariable(self, i, c)
        elif i is not None:
            core.CreateVariable(self, i, 1.0)
        else:
            raise TypeError("At least an argument must be passed")

    def __iadd__(self, other: Union[DA, float]) -> DA:
        """
        Compute the sum with another DA object or a float.
        The result is directly copied into the current DA object.

        Args:
            other: DA object or float to add.

        Returns:
            Current DA object with modified contents.

        Raises:
            DACEException

        Derived from C++:
            `DA& DA::operator+=(const DA &da | const double c)`
        """
        if isinstance(other, DA):
            core.Add(self, other, self)
        elif isinstance(other, (float, int)):
            core.AddDouble(self, other, self)
        else:
            return NotImplemented
        return self

    def __isub__(self, other: Union[DA, float]) -> DA:
        """
        Compute the difference with another DA object or a float.
        The result is directly copied into the current DA object.
        """
        if isinstance(other, DA):
            core.Subtract(self, other, self)
        elif isinstance(other, (float, int)):
            core.SubtractDouble(self, other, self)
        else:
            return NotImplemented
        return self

    @overload
    def __imul__(self, other: Union[DA, float]) -> DA: ...

    @overload
    def __imul__(self, other: np.ndarray) -> daceypy.array: ...

    def __imul__(  # type: ignore
        self, other: Union[DA, float, np.ndarray]
    ) -> Union[DA, daceypy.array]:
        """
        Compute the product with another DA object, a float or a NumPy array.
        The result is directly copied into the current DA object if the product
        is with a DA object or a float, whereas if the product is with a NumPy
        array a new DACEyPy array is created.
        """
        if isinstance(other, DA):
            core.Multiply(self, other, self)
        elif isinstance(other, (float, int)):
            core.MultiplyDouble(self, other, self)
        elif isinstance(other, np.ndarray):
            return other.view(daceypy.array).__rmul__(self)
        else:
            return NotImplemented
        return self

    def __itruediv__(self, other: Union[DA, float]) -> DA:
        """
        Compute the quotient with another DA object or a float.
        The result is directly copied into the current DA object.
        """
        if isinstance(other, DA):
            core.Divide(self, other, self)
        elif isinstance(other, (float, int)):
            core.DivideDouble(self, other, self)
        else:
            return NotImplemented
        return self

    # *************************************************************************
    # *     Algebraic operations
    # *************************************************************************

    def __eq__(self, other) -> bool:
        """
        Check value equality with another object.
        """
        if isinstance(other, (DA, float, int)):
            return (self - other).size() == 0
        return False

    def __neg__(self) -> DA:
        """
        Compute the additive inverse of the given DA object.
        The result is copied in a new DA vector.
        """
        out = DA()
        core.MultiplyDouble(self, -1.0, out)
        return out

    def __add__(self, other: Union[DA, float]) -> DA:
        """
        Compute the sum with another DA object or a float.
        The result is copied in a new DA object.
        """
        out = DA()
        if isinstance(other, DA):
            core.Add(self, other, out)
        elif isinstance(other, (float, int)):
            core.AddDouble(self, other, out)
        else:
            return NotImplemented
        return out

    def __radd__(self, other: float) -> DA:
        return self + other

    def __sub__(self, other: Union[DA, float]) -> DA:
        """
        Compute the difference with another DA object or a float.
        The result is copied in a new DA object.
        """
        out = DA()
        if isinstance(other, DA):
            core.Subtract(self, other, out)
        elif isinstance(other, (float, int)):
            core.SubtractDouble(self, other, out)
        else:
            return NotImplemented
        return out

    def __rsub__(self, other: float) -> DA:
        """
        Compute the difference of a float and a DA object.
        The result is copied in a new DA object.
        """
        out = DA()
        if isinstance(other, (float, int)):
            core.DoubleSubtract(self, other, out)
        else:
            return NotImplemented
        return out

    @overload
    def __mul__(self, other: Union[DA, float]) -> DA: ...

    @overload
    def __mul__(self, other: np.ndarray) -> daceypy.array: ...

    def __mul__(
        self, other: Union[DA, float, np.ndarray]
    ) -> Union[DA, daceypy.array]:
        """
        Compute the product with another DA object, a float or a NumPy array.
        The result is copied in a new DA object.
        """
        out = DA()
        if isinstance(other, DA):
            core.Multiply(self, other, out)
        elif isinstance(other, (float, int)):
            core.MultiplyDouble(self, other, out)
        elif isinstance(other, np.ndarray):
            return other.view(daceypy.array).__rmul__(self)
        else:
            return NotImplemented
        return out

    def __rmul__(self, other: float) -> DA:
        return self * other

    def __truediv__(self, other: Union[DA, float]) -> DA:
        """
        Compute the quotient with another DA object or a float.
        The result is copied in a new DA object.
        """
        out = DA()
        if isinstance(other, DA):
            core.Divide(self, other, out)
        elif isinstance(other, (float, int)):
            core.DivideDouble(self, other, out)
        else:
            return NotImplemented
        return out

    def __rtruediv__(self, other: float) -> DA:
        """
        Compute the division of a float and a DA object.
        The result is copied in a new DA object.
        """
        out = DA()
        if isinstance(other, (float, int)):
            core.DoubleDivide(self, other, out)
        else:
            return NotImplemented
        return out

    # *************************************************************************
    # *     Math routines
    # *************************************************************************

    def multiplyMonomials(self, other: DA, *, out: Optional[DA] = None) -> DA:
        """
        Multiply the DA vector with another DA vector monomial by monomial.
        This is the equivalent of coefficient-wise multiplication
        (like in DA addition).
        The result is copied in a new DA object if out is not given.

        Args:
            da: DA vector to multiply with coefficient-wise.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::multiplyMonomials(const DA &da)`
        """
        if out is None:
            out = DA()
        core.MultiplyMonomials(self, other, out)
        return out

    def divide(self, var: int, p: int = 1, *, out: Optional[DA] = None) -> DA:
        """
        Divide by independent variable var raised to power p.
        The result is copied in a new DA object if out is not given.

        Args:
            var: independent variable number to divide by.
            p: power of the independent variable, default 1.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::divide(const unsigned int var, const unsigned int p)`
        """
        if out is None:
            out = DA()
        core.DivideByVariable(self, var, p, out)
        return out

    @overload
    def deriv(self, i: int, *, out: Optional[DA] = None) -> DA:
        """
        Compute the derivative of a DA object with respect to variable i.
        The result is copied in a new DA object if out is not given.

        Args:
            i: variable with respect to which the derivative is calculated.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            ValueError
            DACEException

        Derived from C++:
            `DA DA::deriv(const unsigned int i)`
        """
        ...

    @overload
    def deriv(self, i: List[int], *, out: Optional[DA] = None) -> DA:
        """
        Compute the derivative of a DA object with respect to variables ind.
        The result is copied in a new DA object if out is not given.

        Args:
            ind: vector containing the number of derivatives to take for each
              independent variable. If ind has fewer entries than there are
              independent variables, the missing entries are assumed to be zero.
              If ind has more entries than there are independent variables,
              extra values are ignored.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::deriv(const std::vector<unsigned int> ind)`
        """
        ...

    def deriv(self, i: Union[int, List[int]], *, out: Optional[DA] = None) -> DA:
        if isinstance(i, int):
            if not i:
                raise ValueError("Argument i must be >= 0")
            if out is None:
                out = DA()
            core.Differentiate(i, self, out)
        elif isinstance(i, list):
            if out is None:
                out = self.copy()
            else:
                out.assign(self)

            size: int = min(len(i), self.getMaxVariables())

            for var_i, times in enumerate(i[:size], 1):
                for _ in range(times):
                    core.Differentiate(var_i, out, out)
        else:
            raise TypeError("Argument must be int or list of ints")
        return out

    @overload
    def integ(self, i: int, *, out: Optional[DA] = None) -> DA:
        """
        Compute the integral of a DA object with respect to variable i.
        The result is copied in a new DA object if out is not given.

        Args:
            i: variable with respect to which the integral is calculated.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            ValueError
            DACEException

        Derived from C++:
            `DA DA::integ(const unsigned int i)`
        """
        ...

    @overload
    def integ(self, i: List[int], *, out: Optional[DA] = None) -> DA:
        """
        Compute the integral of a DA object with respect to variables ind.
        The result is copied in a new DA object if out is not given.

        Args:
            ind: vector containing the number of integrals to take for each
              independent variable. If ind has fewer entries than there are
              independent variables, the missing entries are assumed to be zero.
              If ind has more entries than there are independent variables,
              extra values are ignored.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::integ(const std::vector<unsigned int> ind)`
        """
        ...

    def integ(self, i: Union[int, List[int]], *, out: Optional[DA] = None) -> DA:
        if isinstance(i, int):
            if not i:
                raise ValueError("Argument i must be >= 0")
            if out is None:
                out = DA()
            core.Integrate(i, self, out)
        elif isinstance(i, list):
            if out is None:
                out = self.copy()
            else:
                out.assign(self)

            size: int = min(len(i), self.getMaxVariables())

            for var_i, times in enumerate(i[:size], 1):
                for _ in range(times):
                    core.Integrate(var_i, out, out)
        else:
            raise TypeError("Argument must be int or list of ints")
        return out

    def trim(self, min: int, max: Optional[int] = None, *, out: Optional[DA] = None) -> DA:
        """
        Returns a DA object with all monomials of order
        less than min and greater than max removed.
        The result is copied in a new DA object if out is not given.

        Args:
            min: The minimum order to keep in the DA object.
            max: The maximum order to keep in the DA object,
              default DA.getMaxOrder().
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.


        Raises:
            DACEException

        Derived from C++:
            `DA DA::trim(const unsigned int min, const unsigned int max)`
        """
        if max is None:
            max = DA.getMaxOrder()
        if out is None:
            out = DA()
        core.Trim(self, min, max, out)
        return out

    def trunc(self, *, out: Optional[DA] = None) -> DA:
        """
        Truncate the constant part of a DA object to an integer.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.


        Raises:
            DACEException

        Derived from C++:
            `DA DA::trunc()`
        """
        if out is None:
            out = DA()
        core.Truncate(self, out)
        return out

    def round(self, *, out: Optional[DA] = None) -> DA:
        """
        Round the constant part of a DA object to an integer.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException
        """
        if out is None:
            out = DA()
        core.Round(self, out)
        return out

    def __mod__(self, other: float) -> DA:
        """
        Compute the floating-point remainder of c/p (c modulo p),
        where c is the constant part of the current DA object.
        The result is copied in a new DA object.

        Args:
            other: costant with respect to which the modulo function is computed.

        Returns:
            A new DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::mod(const double p)`
        """
        if not isinstance(other, float):
            return NotImplemented
        out = DA()
        core.Modulo(self, other, out)
        return out

    def __imod__(self, other: float) -> DA:
        """
        Compute the floating-point remainder of c/p (c modulo p),
        where c is the constant part of the current DA object.
        The result is stored in the current DA object.

        Args:
            other: costant with respect to which the modulo function is computed.

        Returns:
            Current DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::mod(const double p)`
        """
        if not isinstance(other, float):
            return NotImplemented
        core.Modulo(self, other, self)
        return self

    def __pow__(self, other: Union[float, DA]) -> DA:
        """
        Elevate a DA object to a given power.
        The result is copied in a new DA object.

        Args:
            other: power to which the DA object is raised.

        Returns:
            A new DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::pow(const int p)`
            `DA DA::pow(const double p)`
        """
        out = DA()
        if isinstance(other, int):
            core.Power(self, other, out)
        elif isinstance(other, float):
            core.PowerDouble(self, other, out)
        elif isinstance(other, DA):
            (other * self.log()).exp(out=out)
        else:
            return NotImplemented
        return out

    pow = __pow__

    def __ipow__(self, other: Union[float, DA]) -> DA:
        """
        Elevate a DA object to a given power.
        The result is stored in the current DA object.

        Args:
            other: power to which the DA object is raised.

        Returns:
            Current DA object containing the result of the operation.

        Raises:
            DACEException
        """
        if isinstance(other, int):
            core.Power(self, other, self)
        elif isinstance(other, float):
            core.PowerDouble(self, other, self)
        elif isinstance(other, DA):
            (other * self.log()).exp(out=self)
        else:
            return NotImplemented
        return self

    def __rpow__(self, other: float) -> DA:
        """
        Elevate a float to a DA object.
        The result is copied in a new DA object.

        Args:
            other: float to be elevated to the DA object.

        Returns:
            A new DA object containing the result of the operation.

        Raises:
            DACEException
        """
        return (self * np.log(other)).exp()

    def root(self, p: int = 2, *, out: Optional[DA] = None) -> DA:
        """
        Compute the p-th root of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA objecc.
            p: root to be computed (default 2).

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::root(const int p)`
        """
        if out is None:
            out = DA()
        core.Root(self, p, out)
        return out

    def minv(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the multiplicative inverse of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA objecc.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::minv()`
        """
        if out is None:
            out = DA()
        core.MultiplicativeInverse(self, out)
        return out

    def sqr(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the square of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA objecc.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::sqr()`
        """
        if out is None:
            out = DA()
        core.Square(self, out)
        return out

    def sqrt(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the square root of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA objecc.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::sqrt()`
        """
        if out is None:
            out = DA()
        core.SquareRoot(self, out)
        return out

    def isrt(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the inverse square root of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA objecc.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::isrt()`
        """
        if out is None:
            out = DA()
        core.InverseSquareRoot(self, out)
        return out

    def cbrt(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the cubic root of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA objecc.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::cbrt()`
        """
        if out is None:
            out = DA()
        core.CubicRoot(self, out)
        return out

    def icrt(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the inverse cubic root of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA objecc.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::icrt()`
        """
        if out is None:
            out = DA()
        core.InverseCubicRoot(self, out)
        return out

    def hypot(self, other: DA, *, out: Optional[DA] = None) -> DA:
        """
        Compute the hypotenuse (sqrt(a*a+b*b)) of a DA object
        and the given DA argument.
        The result is copied in a new DA object if out is not given.

        Args:
            other: DA object representing "b" in the formula.
            out: optional destination DA objecc.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::hypot(const DA &da)`
        """
        if out is None:
            out = DA()
        core.Hypotenuse(self, other, out)
        return out

    def exp(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the exponential of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::exp()`
        """
        if out is None:
            out = DA()
        core.Exponential(self, out)
        return out

    def log(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the natural logarithm of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::log()`
        """
        if out is None:
            out = DA()
        core.Logarithm(self, out)
        return out

    def logb(self, b: float = 10.0, *, out: Optional[DA] = None) -> DA:
        """
        Compute the logarithm of a DA object with respect to a given base.
        The result is copied in a new DA object if out is not given.

        Args:
            b: base with respect to which the logarithm is computed
              (default 10).
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::logb(const double b)`
        """
        if out is None:
            out = DA()
        core.LogarithmBase(self, b, out)
        return out

    def log10(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the 10 based logarithm of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::log10()`
        """
        if out is None:
            out = DA()
        core.Logarithm10(self, out)
        return out

    def log2(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the 2 based logarithm of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::log2()`
        """
        if out is None:
            out = DA()
        core.Logarithm2(self, out)
        return out

    def sin(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the sine of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::sin()`
        """
        if out is None:
            out = DA()
        core.Sine(self, out)
        return out

    def cos(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the cosine of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::cos()`
        """
        if out is None:
            out = DA()
        core.Cosine(self, out)
        return out

    def tan(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the tangent of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::tan()`
        """
        if out is None:
            out = DA()
        core.Tangent(self, out)
        return out

    def asin(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the arcsine of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::asin()`
        """
        if out is None:
            out = DA()
        core.ArcSine(self, out)
        return out

    arcsin = asin

    def acos(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the arccosine of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::acos()`
        """
        if out is None:
            out = DA()
        core.ArcCosine(self, out)
        return out

    arccos = acos

    def atan(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the arctangent of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::atan()`
        """
        if out is None:
            out = DA()
        core.ArcTangent(self, out)
        return out

    arctan = atan

    def atan2(self, other: DA, *, out: Optional[DA] = None) -> DA:
        """
        Compute the four-quadrant arctangent of Y/X.
        Y is the current DA object, whereas X is the given `other`.
        The result is copied in a new DA object if out is not given.

        Args:
            other: DA object.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::atan2(const DA &da)`
        """
        if out is None:
            out = DA()
        core.ArcTangent2(self, other, out)
        return out

    arctan2 = atan2

    def sinh(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the hyperbolic sine of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::sinh()`
        """
        if out is None:
            out = DA()
        core.HyperbolicSine(self, out)
        return out

    def cosh(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the hyperbolic cosine of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::cosh()`
        """
        if out is None:
            out = DA()
        core.HyperbolicCosine(self, out)
        return out

    def tanh(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the hyperbolic tangent of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::tanh()`
        """
        if out is None:
            out = DA()
        core.HyperbolicTangent(self, out)
        return out

    def asinh(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the hyperbolic arcsine of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::asinh()`
        """
        if out is None:
            out = DA()
        core.HyperbolicArcSine(self, out)
        return out

    arcsinh = asinh

    def acosh(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the hyperbolic arccosine of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::acosh()`
        """
        if out is None:
            out = DA()
        core.HyperbolicArcCosine(self, out)
        return out

    arccosh = acosh

    def atanh(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the hyperbolic arctangent of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::atanh()`
        """
        if out is None:
            out = DA()
        core.HyperbolicArcTangent(self, out)
        return out

    arctanh = atanh

    def erf(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the error function of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::erf()`
        """
        if out is None:
            out = DA()
        core.ErrorFunction(self, out)
        return out

    def erfc(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the complementary error function of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::erfc()`
        """
        if out is None:
            out = DA()
        core.ComplementaryErrorFunction(self, out)
        return out

    def BesselJFunction(self, n: int, *, out: Optional[DA] = None) -> DA:
        """
        Compute the n-th Bessel function of first type J_n of a DA object.
        The result is copied in a new DA object if out is not given.

        The DA must have non-negative constant part while the order is allowed to be negative.
        This function fails if the result is too large to be represented in double precision.

        Args:
            n: order of the Bessel function
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::BesselJFunction(const int n)`
        """
        if out is None:
            out = DA()
        core.BesselJFunction(self, n, out)
        return out

    def BesselYFunction(self, n: int, *, out: Optional[DA] = None) -> DA:
        """
        Compute the n-th Bessel function of second type Y_n of a DA object.
        The result is copied in a new DA object if out is not given.

        The DA must have non-negative constant part while the order is allowed to be negative.
        This function fails if the result is too large to be represented in double precision.

        Args:
            n: order of the Bessel function
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::BesselYFunction(const int n)`
        """
        if out is None:
            out = DA()
        core.BesselYFunction(self, n, out)
        return out

    def BesselIFunction(self, n: int, scaled: bool = False, *, out: Optional[DA] = None) -> DA:
        """
        Compute the n-th modified Bessel function of first type I_n of a DA object.
        The result is copied in a new DA object if out is not given.

        The DA must have non-negative constant part while the order is allowed to be negative.
        This function fails if the result is too large to be represented in double precision.

        Args:
            n: order of the Bessel function
            scaled: if True, the modified Bessel function is scaled
              by a factor exp(-x), i.e. exp(-x)I_n(x) is returned;
              default False.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::BesselIFunction(const int n, const bool scaled)`
        """
        if out is None:
            out = DA()
        core.BesselIFunction(self, n, scaled, out)
        return out

    def BesselKFunction(self, n: int, scaled: bool = False, *, out: Optional[DA] = None) -> DA:
        """
        Compute the n-th modified Bessel function of second type K_n of a DA object.
        The result is copied in a new DA object if out is not given.

        The DA must have non-negative constant part while the order is allowed to be negative.
        This function fails if the result is too large to be represented in double precision.

        Args:
            n: order of the Bessel function
            scaled: if True, the modified Bessel function is scaled
              by a factor exp(x), i.e. exp(x)K_n(x) is returned;
              default False.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::BesselKFunction(const int n, const bool scaled)`
        """
        if out is None:
            out = DA()
        core.BesselKFunction(self, n, scaled, out)
        return out

    def GammaFunction(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the Gamma function of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::GammaFunction()`
        """
        if out is None:
            out = DA()
        core.GammaFunction(self, out)
        return out

    def LogGammaFunction(self, *, out: Optional[DA] = None) -> DA:
        """
        Compute the Logarithmic Gamma function (i.e. the natural logarithm of Gamma) of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::LogGammaFunction()`
        """
        if out is None:
            out = DA()
        core.LogGammaFunction(self, out)
        return out

    def PsiFunction(self, n: int, *, out: Optional[DA] = None) -> DA:
        """
        Compute the n-th order Psi function, i.e. the (n+1)st derivative
        of the Logarithmic Gamma function, of a DA object.
        The result is copied in a new DA object if out is not given.

        Args:
            n: order of the Psi function (n >= 0).
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::PsiFunction(const unsigned int n)`
        """
        if out is None:
            out = DA()
        core.PsiFunction(self, n, out)
        return out

    # *************************************************************************
    # *    Norm and estimation routines
    # *************************************************************************

    def size(self) -> int:
        """
        Get the number of non-zero coefficients of a DA object.

        Returns:
            The number of non-zero coefficients of a DA object.

        Raises:
            DACEException

        Derived from C++:
            `unsigned int DA::size()`
        """
        return core.GetLength(self)

    def abs(self) -> float:
        """
        Compute the max norm of a DA object.

        Returns:
            A float corresponding to the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `double DA::abs()`
        """
        return core.AbsoluteValue(self)

    __abs__ = abs

    def norm(self, type_: int = 0) -> float:
        """
        Compute different types of norms for a DA object.

        Args:
            type_: type of norm to be computed. Possible norms are:
              0: Max norm (default),
              1: Sum norm,
              >1: Vector norm of given type.

        Returns:
            A double corresponding to the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `double DA::norm(const unsigned int type)`
        """
        return core.Norm(self, type_)

    def orderNorm(self, var: int = 0, type: int = 0) -> NDArray[np.double]:
        """
        Extract different types of order sorted norms from a DA object.

        Args:
            var: order
              0: Terms are sorted by their order (default),
              >0: Terms are sorted by the exponent of variable var.
            type: type of norm to be computed. Possible norms are:
              0: Max norm (default),
              1: Sum norm,
              >1: Vector norm of given type.

        Returns:
            A NumPy array corresponding to the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `std::vector<double> DA::orderNorm(const unsigned int var, const unsigned int type)`
        """
        v = np.empty(core.GetMaxOrder() + 1)
        address = cast(
            c_void_p(v.__array_interface__["data"][0]), c_double_p)
        core.OrderedNorm(self, var, type, address)
        return v

    def estimNorm(
        self, var: int = 0, type: int = 0, nc: Optional[int] = None,
        get_error: bool = False,
    ) -> Tuple[NDArray[np.double], Optional[NDArray[np.double]]]:
        """
        Estimate different types of order sorted norms for terms of a DA object
        up to a specified order. If estimation is not possible, zero is
        returned for all requested orders.

        Args:
            var: order
              0: Terms are sorted by their order (default),
              >0: Terms are sorted by the exponent of variable var.
            type: type of norm to be computed. Possible norms are:
              0: Max norm (default),
              1: Sum norm,
              >1: Vector norm of given type.
            nc: maximum order to be estimated (default order = Max order).
            get_error: if True, also the underestimation error is returned.

        Returns:
            A tuple with:
                A NumPy array corresponding to the result of the operation.
                A NumPy array with the amount by which the estimate
                  underestimates the actual ordered norm of the terms in the
                  polynomial up to the minimum of `nc` or the maximum
                  computation order
                  (if requested with `get_error`, otherwise None).

        Raises:
            DACEException

        Derived from C++:
            `std::vector<double> estimNorm(const unsigned int var = 0, const unsigned int type = 0, const unsigned int nc = DA::getMaxOrder()) const;`
            `std::vector<double> estimNorm(std::vector<double> &err, const unsigned int var = 0, const unsigned int type = 0, const unsigned int nc = DA::getMaxOrder()) const;`
        """
        if nc is None:
            nc = self.getMaxOrder()
        v = np.empty(nc + 1)
        v_address = cast(
            c_void_p(v.__array_interface__["data"][0]), c_double_p)
        if get_error:
            err = np.empty(min(nc, self.getMaxOrder()) + 1)
            err_address = cast(
                c_void_p(err.__array_interface__["data"][0]), c_double_p)
            core.Estimate(self, var, type, v_address, err_address, nc)
            return v, err
        else:
            core.Estimate(self, var, type, v_address, None, nc)
            return v, None

    def bound(self) -> Tuple[float, float]:
        """
        Compute lower and upper bounds of a DA object.

        Returns:
            An tuple containing both the lower and the upper bound
            of the DA object.

        Raises:
            DACEException

        Derived from C++:
            `Interval DA::bound()`
        """
        m_lb = c_double()
        m_ub = c_double()
        core.GetBounds(self, m_lb, m_ub)
        return m_lb.value, m_ub.value

    def convRadius(self, eps: float, type_: int = 1) -> float:
        """
        Estimate the convergence radius of the DA object.

        Args:
            eps: requested tolerance.
            type_: type of norm (sum norm is used as default).

        Returns:
            A double corresponding to the estimated convergence radius.

        Raises:
            DACEException

        Derived from C++:
            `double DA::convRadius(const double eps, const unsigned int type)`
        """
        ord = self.getTO()
        res, _ = self.estimNorm(0, type_, ord + 1)
        return np.power(eps / res[ord + 1], 1.0 / (ord + 1))

    # *************************************************************************
    # *     DACE polynomial evaluation routines
    # *************************************************************************

    def compile(self) -> _compiledDA.compiledDA:
        """
        Compile current DA object and create a compiledDA object.

        Returns:
            The compiled DA object.

        Raises:
            DACEException

        Derived from C++:
            `compiledDA DA::compile()`
        """
        return _compiledDA.compiledDA(self)

    def plug(self, var: int, val: float = 0.0, *, out: Optional[DA] = None) -> DA:
        """
        Partial evaluation of a DA object.
        In the DA object, variable var is replaced by the value val.
        The result is copied in a new DA object if out is not given.

        Args:
            var: variable number to be replaced.
            val: value by which to replace the variable (default 0.0).
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::plug(const unsigned int var, const double val)`
        """
        if out is None:
            out = DA()
        core.EvalVariable(self, var, val, out)
        return out

    def evalMonomials(self, other: DA) -> float:
        """
        Evaluates the DA vector using the coefficients in argument values
        as the values for each monomial.
        This is equivalent to a monomial-wise dot product of two DA vectors.

        Args:
            values: DA vector containing the values of each monomial.

        Returns:
            The result of the evaluation.

        Raises:
            DACEException

        See also:
            DA.multiplyMonomial

        Derived from C++:
            `double DA::evalMonomials(const DA &values)`
        """
        return core.EvalMonomials(self, other)

    def replaceVariable(self, from_: int = 0, to: int = 0, val: float = 1, *, out: Optional[DA] = None) -> DA:
        """
        Partial evaluation of a DA object. In the DA object, variable `from` is
        replaced by the value `val` times variable `to`.
        The result is copied in a new DA object if out is not given.

        Args:
            from: variable number to be replaced.
            to: variable number to be inserted instead.
            val: value by which to scale the inserted variable.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::replaceVariable(const unsigned int from, const unsigned int to, const double val)`
        """
        if out is None:
            out = DA()
        core.ReplaceVariable(self, from_, to, val, out)
        return out

    def scaleVariable(self, var: int = 0, val: float = 1.0, *, out: Optional[DA] = None) -> DA:
        """
        Scaling of an independent variable. In the DA object, variable `var` is
        replaced by the value `val` times `var`.
        The result is copied in a new DA object if out is not given.

        Args:
            var: variable number to be scaled
            val: value by which to scale the variable
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::scaleVariable(const unsigned int var, const double val)`
        """
        if out is None:
            out = DA()
        core.ScaleVariable(self, var, val, out)
        return out

    def translateVariable(self, var: int = 0, a: float = 1.0, c: float = 0.0, *, out: Optional[DA] = None) -> DA:
        """
        Affine translation of an independent variable.
        In the DA object, variable `var` is replaced by `a*var + c`.
        The result is copied in a new DA object if out is not given.

        Args:
            var: variable number to be translated.
            a: value by which to scale the variable.
            c: value by which to shift the variable.
            out: optional destination DA object.

        Returns:
            A new or the given out DA object containing the result of the operation.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::translateVariable(const unsigned int var, const double a, const double c)`
        """
        if out is None:
            out = DA()
        core.TranslateVariable(self, var, a, c, out)
        return out

    @overload
    def eval(self, args: Union[daceypy.array, List[DA]]) -> DA:  # type: ignore
        """
        Evaluate the compiledDA object using a DACEyPy array or a list of DA objects.

        Args:
            args: DACEyPy array or list of DA objects to use for the evaluation.

        Returns:
            Result of the evaluation as a DA object.

        Raises:
            DACEException

        Derived from C++:
            `DA DA::eval(const std::vector<DA> &args)`
        """
        ...

    @overload
    def eval(self, args: Union[NDArray[np.double], List[float]]) -> float:
        """
        Evaluate the compiledDA object using a NumPy array of doubles or a list of float.

        Args:
            args: NumPy array or list of float of doubles to use for the evaluation.

        Returns:
            Result of the evaluation as float.

        Raises:
            DACEException

        Derived from C++:
            `float DA::eval(const std::vector<float> &args)`
        """
        ...

    def eval(self, args: Union[List[float], List[DA], NDArray[np.double], daceypy.array]) -> Union[float, DA]:
        return self.compile().eval(args)[0]

    @overload
    def evalScalar(self, arg: float) -> float:
        """
        Evaluate the DA object with a single argument of type
        float and return the result.

        Args:
            arg:
              The value of the first independent DA variable to evaluate with.
              All remaining independent DA variables are assumed to be zero.

        Returns:
            Result of the evaluation as float.

        Derived from C++:
            `double DA::evalScalar(const double &arg)`
        """
        ...

    @overload
    def evalScalar(self, arg: DA) -> DA:
        """
        Evaluate the DA object with a single argument of type
        DA and return the result.

        Args:
            arg:
              The value of the first independent DA variable to evaluate with.
              All remaining independent DA variables are assumed to be zero.

        Returns:
            Result of the evaluation as a DA object.

        Derived from C++:
            `DA DA::evalScalar(const DA &arg)`
        """
        ...

    def evalScalar(self, arg: Union[float, DA]) -> Union[float, DA]:
        return self.compile().evalScalar(arg)[0]

    @overload
    def __call__(self, arg: float) -> float: ...

    @overload
    def __call__(self, arg: DA) -> DA: ...

    @overload
    def __call__(self, arg: NDArray[np.double]) -> float: ...

    @overload
    def __call__(self, arg: List[float]) -> float: ...

    @overload
    def __call__(self, arg: daceypy.array) -> DA: ...

    @overload
    def __call__(self, arg: List[DA]) -> DA: ...

    def __call__(
        self, arg: Union[
            float, DA, NDArray[np.double],
            List[float], daceypy.array, List[DA],
        ]
    ) -> Union[float, DA]:

        if isinstance(arg, (list, daceypy.array, np.ndarray)):
            return self.eval(arg)
        return self.evalScalar(arg)

    # *************************************************************************
    # *     DACE input/output routines
    # *************************************************************************

    def __str__(self) -> str:
        """
        Convert DA object to string.

        Returns:
            A string.

        Raises:
            DACEException

        Derived from C++:
            `std::string DA::toString()`
        """
        # initialize 2D char array
        nstr = c_uint(core.GetMaxMonomials() + 2)
        ss = create_string_buffer(nstr.value * DACE_STRLEN)

        # call core.write
        core.Write(self, ss, nstr)

        # copy from char array to string
        return b"\n".join(
            ss.raw[i:i + DACE_STRLEN]
            for i in range(0, nstr.value * DACE_STRLEN, DACE_STRLEN)
        ).replace(b"\x00", b"").decode()

    toString = __str__

    def __bytes__(self) -> bytes:
        """
        Convert a DA object to bytes.

        Returns:
            Bytes representation of the DA object.

        Raises:
            DACEException
        """

        len_ = c_uint()
        core.ExportBlob(self, None, len_)
        dst = create_string_buffer(len_.value)
        core.ExportBlob(self, dst, len_)
        return dst.raw

    def __getstate__(self) -> bytes:
        return bytes(self)

    __setstate__ = __init__

    def __repr__(self) -> str:
        return f"<DA object of size {self.size()}>"

    @classmethod
    def fromNumber(cls, value: Union[float, int, DA]) -> DA:
        """
        Convert an integer or a float to a DA object.
        If it is already a DA object, it is returned unchanged.

        Args:
            value: object to be converted.

        Returns:
            A DA object representing the given value.

        Raises:
            DACEException
        """
        if isinstance(value, (float, cls)):
            return cls(value)
        if isinstance(value, int):
            return cls(float(value))
        raise ValueError(f"Invalid value of type {type(value)} in input array")

    # *************************************************************************
    # *     Caching routines (introduced in DACEyPy)
    # *************************************************************************

    class _DACEDA_cache_context_manager:

        def __enter__(self):
            DA.cache_enable()

        def __exit__(self, *args):
            DA.cache_disable()

    @classmethod
    def cache_manager(cls):
        """
        Get a context manager to be used in a `with` statement.

        All operations within the managed block will reuse the DACEDA C structs.
        This allows to run the code faster.

        Example:
            `with DA.cache_manager():`
        """
        return cls._DACEDA_cache_context_manager()

    @classmethod
    def cache_enable(cls):
        """
        Enable caching of DACEDA C structs.

        All subsequent operations will reuse the DACEDA C structs.
        This allows to run the code faster.
        """
        cls._DACEDA_cache_enabled = True

    @classmethod
    def cache_disable(cls):
        """
        Disable caching of DACEDA C structs and clear the cache.
        """
        cls._DACEDA_cache_enabled = True
        cls._DACEDA_cache.clear()
