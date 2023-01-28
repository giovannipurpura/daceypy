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

from ctypes import (POINTER, Structure, c_bool, c_char_p, c_double, c_int,
                    c_uint, c_void_p, cdll)
from inspect import getsourcefile
from pathlib import Path
from typing import Callable

from . import _DACEException
from .get_platform import get_platform

sourcefile = getsourcefile(lambda: 0)
assert sourcefile is not None
libfolder = Path(sourcefile).resolve().parent / "lib"
system, machine = get_platform()

try:
    library = next(libfolder.glob(f"dace_{system}-{machine}*"))
except StopIteration:
    raise Exception(
        f"DACEyPy does not support this architecture ({system} {machine}).\n"
        "Supported architectures:\n" + "\n".join(
            " - " + lib.stem[5:].replace("-", " ")
            for lib in libfolder.glob(f"dace_*")))

DAlib = cdll.LoadLibrary(str(library))


class dmonomial(Structure):
    __slots__ = "cc", "ii"
    __fields__ = [("cc", c_double), ("ii", c_uint)]


class DACEDA(Structure):

    __slots__ = "len", "max", "dmonomial"

    _fields_ = [
        ("len", c_uint),
        ("max", c_uint),
        ("dmonomial", POINTER(dmonomial)),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        AllocateDA(self, 0)

    def __del__(self):
        FreeDA(self)


c_int_p = POINTER(c_int)
c_uint_p = POINTER(c_uint)
c_double_p = POINTER(c_double)
DACEDA_p = POINTER(DACEDA)
DACEDA_p_p = POINTER(DACEDA_p)


def corefunction(argtypes=[], restype=None, errcheck: bool = True):
    """
    Provides a function decorator that is used to get the corresponding
    function from the DA core dynamic-link library.
    """

    def decorator(fun: Callable) -> Callable:
        function = getattr(DAlib, "dace" + fun.__name__)
        function.restype = restype
        function.argtypes = argtypes

        if not errcheck:
            # just return the function
            return function

        # wrap the function to automatically check for errors
        def with_errcheck(*args):
            res = function(*args)
            if GetError():
                raise _DACEException.DACEException
            return res

        return with_errcheck

    return decorator


# *****************************************************************************
# *     DACE initialization and state related routines
# *****************************************************************************


@corefunction([c_uint, c_uint])
def Initialize(no: int, nv: int) -> None:
    """`void daceInitialize(unsigned int no, unsigned int nv);`"""
    ...


@corefunction()
def InitializeThread() -> None:
    """`void daceInitializeThread();`"""
    ...


@corefunction()
def CleanupThread() -> None:
    """`void daceCleanupThread();`"""
    ...


@corefunction([c_int_p, c_int_p, c_int_p])
def GetVersion(imaj, imin, ipat) -> None:
    """`void daceGetVersion(int REF(imaj), int REF(imin), int REF(ipat));`"""
    ...


@corefunction([c_double], c_double)
def SetEpsilon(deps: float) -> float:
    """`double daceSetEpsilon(const double deps);`"""
    ...


@corefunction([], c_double)
def GetEpsilon() -> float:
    """`double daceGetEpsilon();`"""
    ...


@corefunction([], c_double)
def GetMachineEpsilon() -> float:
    """`double daceGetMachineEpsilon();`"""
    ...


@corefunction([], c_uint)
def GetMaxOrder() -> int:
    """`unsigned int daceGetMaxOrder();`"""
    ...


@corefunction([], c_uint)
def GetMaxVariables() -> int:
    """`unsigned int daceGetMaxVariables();`"""
    ...


@corefunction([], c_uint)
def GetMaxMonomials() -> int:
    """`unsigned int daceGetMaxMonomials();`"""
    ...


@corefunction([], c_uint)
def GetTruncationOrder() -> int:
    """`unsigned int daceGetTruncationOrder();`"""
    ...


@corefunction([c_uint], c_uint)
def SetTruncationOrder(fnot: int) -> int:
    """`unsigned int daceSetTruncationOrder(const unsigned int fnot);`"""
    ...


# *****************************************************************************
# *     DACE error state routine
# *****************************************************************************


@corefunction([], c_uint, errcheck=False)
def GetError() -> int:
    """`unsigned int daceGetError();`"""
    ...


@corefunction([], c_uint, errcheck=False)
def GetErrorX() -> int:
    """`unsigned int daceGetErrorX();`"""
    ...


@corefunction([], c_uint, errcheck=False)
def GetErrorYY() -> int:
    """`unsigned int daceGetErrorYY();`"""
    ...


@corefunction([], c_char_p, errcheck=False)
def GetErrorFunName() -> bytes:
    """`const char* daceGetErrorFunName();`"""
    ...


@corefunction([], c_char_p, errcheck=False)
def GetErrorMSG() -> bytes:
    """`const char* daceGetErrorMSG();`"""
    ...


@corefunction(errcheck=False)
def ClearError() -> None:
    """`void daceClearError();`"""
    ...


# *****************************************************************************
# *     DACE memory handling routines
# *****************************************************************************


@corefunction([DACEDA_p, c_uint])
def AllocateDA(inc, len: int) -> None:
    """`void daceAllocateDA(DACEDA REF(inc), const unsigned int len);`"""
    ...


@corefunction([DACEDA_p])
def FreeDA(inc) -> None:
    """`void daceFreeDA(DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p])
def InvalidateDA(inc) -> None:
    """`void daceInvalidateDA(DACEDA REF(inc));`"""
    ...


@corefunction()
def MemoryDump() -> None:
    """`void daceMemoryDump();`"""
    ...


# *****************************************************************************
# *     DACE variable creation routines
# *****************************************************************************


@corefunction([DACEDA_p, c_uint, c_double])
def CreateVariable(ina, i: int, ckon: float) -> None:
    """`void daceCreateVariable(DACEDA REF(ina), const unsigned int i, const double ckon);`"""
    ...


@corefunction([DACEDA_p, c_uint_p, c_double])
def CreateMonomial(ina, jj, ckon: float) -> None:
    """`void daceCreateMonomial(DACEDA REF(ina), const unsigned int jj[], const double ckon);`"""
    ...


@corefunction([DACEDA_p, c_double])
def CreateConstant(ina, ckon: float) -> None:
    """`void daceCreateConstant(DACEDA REF(ina), const double ckon);`"""
    ...


@corefunction([DACEDA_p, c_double])
def CreateFilled(ina, ckon: float) -> None:
    """`void daceCreateFilled(DACEDA REF(ina), const double ckon);`"""
    ...


@corefunction([DACEDA_p, c_double])
def CreateRandom(ina, cm: float) -> None:
    """`void daceCreateRandom(DACEDA REF(ina), const double cm);`"""
    ...


# *****************************************************************************
# *     DACE coefficient access routines
# *****************************************************************************


@corefunction([DACEDA_p], c_double)
def GetConstant(ina) -> float:
    """`double daceGetConstant(const DACEDA REF(ina));`"""
    ...


@corefunction([DACEDA_p, c_double_p])
def GetLinear(ina, c) -> None:
    """`void daceGetLinear(const DACEDA REF(ina), double c[]);`"""
    ...


@corefunction([DACEDA_p, c_uint_p], c_double)
def GetCoefficient(ina, jj) -> float:
    """`double daceGetCoefficient(const DACEDA REF(ina), const unsigned int jj[]);`"""
    ...


@corefunction([DACEDA_p, c_uint], c_double)
def GetCoefficient0(ina, ic: int) -> float:
    """`double daceGetCoefficient0(const DACEDA REF(ina), const unsigned int ic);`"""
    ...


@corefunction([DACEDA_p, c_uint_p, c_double])
def SetCoefficient(ina, jj, cjj: float) -> None:
    """`void daceSetCoefficient(DACEDA REF(ina), const unsigned int jj[], const double cjj);`"""
    ...


@corefunction([DACEDA_p, c_uint, c_double])
def SetCoefficient0(ina, ic: int, cjj: float) -> None:
    """`void daceSetCoefficient0(DACEDA REF(ina), const unsigned int ic, const double cjj);`"""
    ...


@corefunction([DACEDA_p, c_uint, c_uint_p, c_double_p])
def GetCoefficientAt(ina, npos: int, jj, cjj) -> None:
    """`void daceGetCoefficientAt(const DACEDA REF(ina), const unsigned int npos, unsigned int jj[], double REF(cjj));`"""
    ...


@corefunction([DACEDA_p], c_uint)
def GetLength(ina) -> int:
    """`unsigned int daceGetLength(const DACEDA REF(ina));`"""
    ...


# *****************************************************************************
# *     DACE DA copying and filtering
# *****************************************************************************


@corefunction([DACEDA_p, DACEDA_p])
def Copy(ina, inb) -> None:
    """`void daceCopy(const DACEDA REF(ina), DACEDA REF(inb));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def CopyFiltering(ina, inb) -> None:
    """`void daceCopyFiltering(const DACEDA REF(ina), DACEDA REF(inb));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p, DACEDA_p])
def Filter(ina, inb, inc) -> None:
    """`void daceFilter(const DACEDA REF(ina), DACEDA REF(inb), const DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_uint, c_uint, DACEDA_p])
def Trim(ina, imin: int, imax: int, inc) -> None:
    """`void daceTrim(const DACEDA REF(ina), const unsigned int imin, const unsigned int imax, DACEDA REF(inc));`"""
    ...


# *****************************************************************************
# *     Basic DACE arithmetic operations
# *****************************************************************************


@corefunction([DACEDA_p, c_double, DACEDA_p, c_double, DACEDA_p])
def WeightedSum(ina, afac: float, inb, bfac: float, inc) -> None:
    """`void daceWeightedSum(const DACEDA REF(ina), const double afac, const DACEDA REF(inb), const double bfac, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p, DACEDA_p])
def Add(ina, inb, inc) -> None:
    """`void daceAdd(const DACEDA REF(ina), const DACEDA REF(inb), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p, DACEDA_p])
def Subtract(ina, inb, inc) -> None:
    """`void daceSubtract(const DACEDA REF(ina), const DACEDA REF(inb), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p, DACEDA_p])
def Multiply(ina, inb, inc) -> None:
    """`void daceMultiply(const DACEDA REF(ina), const DACEDA REF(inb), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p, DACEDA_p])
def MultiplyMonomials(ina, inb, inc) -> None:
    """`void daceMultiplyMonomials(const DACEDA REF(ina), const DACEDA REF(inb), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p, DACEDA_p])
def Divide(ina, inb, inc) -> None:
    """`void daceDivide(const DACEDA REF(ina), const DACEDA REF(inb), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def Square(ina, inb) -> None:
    """`void daceSquare(const DACEDA REF(ina), DACEDA REF(inb));`"""
    ...


@corefunction([DACEDA_p, c_double, DACEDA_p])
def AddDouble(ina, ckon: float, inb) -> None:
    """`void daceAddDouble(const DACEDA REF(ina), const double ckon, DACEDA REF(inb));`"""
    ...


@corefunction([DACEDA_p, c_double, DACEDA_p])
def DoubleSubtract(ina, ckon: float, inb) -> None:
    """`void daceDoubleSubtract(const DACEDA REF(ina), const double ckon, DACEDA REF(inb));`"""
    ...


@corefunction([DACEDA_p, c_double, DACEDA_p])
def SubtractDouble(ina, ckon: float, inb) -> None:
    """`void daceSubtractDouble(const DACEDA REF(ina), const double ckon, DACEDA REF(inb));`"""
    ...


@corefunction([DACEDA_p, c_double, DACEDA_p])
def MultiplyDouble(ina, ckon: float, inb) -> None:
    """`void daceMultiplyDouble(const DACEDA REF(ina), const double ckon, DACEDA REF(inb));`"""
    ...


@corefunction([DACEDA_p, c_double, DACEDA_p])
def DivideDouble(ina, ckon: float, inb) -> None:
    """`void daceDivideDouble(const DACEDA REF(ina), const double ckon, DACEDA REF(inb));`"""
    ...


@corefunction([DACEDA_p, c_double, DACEDA_p])
def DoubleDivide(ina, ckon: float, inb) -> None:
    """`void daceDoubleDivide(const DACEDA REF(ina), const double ckon, DACEDA REF(inb));`"""
    ...


@corefunction([DACEDA_p, c_uint, c_uint, DACEDA_p])
def DivideByVariable(ina, var: int, p: int, inc) -> None:
    """`void daceDivideByVariable(const DACEDA REF(ina), const unsigned int var, const unsigned int p, DACEDA REF(inc));`"""
    ...


@corefunction([c_uint, DACEDA_p, DACEDA_p])
def Differentiate(idif: int, ina, inc) -> None:
    """`void daceDifferentiate(const unsigned int idif, const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([c_uint, DACEDA_p, DACEDA_p])
def Integrate(iint: int, ina, inc) -> None:
    """`void daceIntegrate(const unsigned int iint, const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


# *****************************************************************************
# *     DACE intrinsic function routines
# *****************************************************************************


@corefunction([DACEDA_p, DACEDA_p])
def Truncate(ina, inc) -> None:
    """`void daceTruncate(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def Round(ina, inc) -> None:
    """`void daceRound(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_double, DACEDA_p])
def Modulo(ina, p: float, inc) -> None:
    """`void daceModulo(const DACEDA REF(ina), const double p, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_double, DACEDA_p])
def PowerDouble(ina, p: float, inc) -> None:
    """`void dacePowerDouble(const DACEDA REF(ina), const double p, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_int, DACEDA_p])
def Power(ina, np: int, inc) -> None:
    """`void dacePower(const DACEDA REF(ina), const int np, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_int, DACEDA_p])
def Root(ina, np: int, inc) -> None:
    """`void daceRoot(const DACEDA REF(ina), const int np, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def MultiplicativeInverse(*args) -> None:
    """`void daceMultiplicativeInverse(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def SquareRoot(ina, inc) -> None:
    """`void daceSquareRoot(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def InverseSquareRoot(ina, inc) -> None:
    """`void daceInverseSquareRoot(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def CubicRoot(ina, inc) -> None:
    """`void daceCubicRoot(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def InverseCubicRoot(ina, inc) -> None:
    """`void daceInverseCubicRoot(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p, DACEDA_p])
def Hypotenuse(ina, inb, inc) -> None:
    """`void daceHypotenuse(const DACEDA REF(ina), const DACEDA REF(inb), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def Exponential(ina, inc) -> None:
    """`void daceExponential(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def Logarithm(ina, inc) -> None:
    """`void daceLogarithm(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_double, DACEDA_p])
def LogarithmBase(ina, b: float, inc) -> None:
    """`void daceLogarithmBase(const DACEDA REF(ina), const double b, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def Logarithm10(ina, inc) -> None:
    """`void daceLogarithm10(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def Logarithm2(ina, inc) -> None:
    """`void daceLogarithm2(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def Sine(ina, inc) -> None:
    """`void daceSine(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def Cosine(ina, inc) -> None:
    """`void daceCosine(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def Tangent(ina, inc) -> None:
    """`void daceTangent(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def ArcSine(ina, inc) -> None:
    """`void daceArcSine(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def ArcCosine(ina, inc) -> None:
    """`void daceArcCosine(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def ArcTangent(ina, inc) -> None:
    """`void daceArcTangent(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p, DACEDA_p])
def ArcTangent2(*args) -> None:
    """`void daceArcTangent2(const DACEDA REF(ina), const DACEDA REF(inb), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def HyperbolicSine(ina, inc) -> None:
    """`void daceHyperbolicSine(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def HyperbolicCosine(ina, inc) -> None:
    """`void daceHyperbolicCosine(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def HyperbolicTangent(ina, inc) -> None:
    """`void daceHyperbolicTangent(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def HyperbolicArcSine(ina, inc) -> None:
    """`void daceHyperbolicArcSine(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def HyperbolicArcCosine(ina, inc) -> None:
    """`void daceHyperbolicArcCosine(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def HyperbolicArcTangent(ina, inc) -> None:
    """`void daceHyperbolicArcTangent(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def ErrorFunction(ina, inc) -> None:
    """`void daceErrorFunction(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def ComplementaryErrorFunction(ina, inc) -> None:
    """`void daceComplementaryErrorFunction(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_int, c_bool, DACEDA_p])
def BesselIFunction(ina, n: int, scaled: bool, inc) -> None:
    """`void daceBesselIFunction(const DACEDA REF(ina), const int n, const bool scaled, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_int, DACEDA_p])
def BesselJFunction(ina, n: int, inc) -> None:
    """`void daceBesselJFunction(const DACEDA REF(ina), const int n, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_int, c_bool, DACEDA_p])
def BesselKFunction(ina, n: int, scaled: bool, inc) -> None:
    """`void daceBesselKFunction(const DACEDA REF(ina), const int n, const bool scaled, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_int, DACEDA_p])
def BesselYFunction(ina, n: int, inc) -> None:
    """`void daceBesselYFunction(const DACEDA REF(ina), const int n, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def LogGammaFunction(ina, inc) -> None:
    """`void daceLogGammaFunction(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, DACEDA_p])
def GammaFunction(ina, inc) -> None:
    """`void daceGammaFunction(const DACEDA REF(ina), DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_uint, DACEDA_p])
def PsiFunction(ina, n: int, inc) -> None:
    """`void dacePsiFunction(const DACEDA REF(ina), const unsigned int n, DACEDA REF(inc));`"""
    ...


# *****************************************************************************
# *     DACE norm and norm estimation routines
# *****************************************************************************


@corefunction([DACEDA_p], c_double)
def AbsoluteValue(ina) -> float:
    """`double daceAbsoluteValue(const DACEDA REF(ina));`"""
    ...


@corefunction([DACEDA_p, c_uint], c_double)
def Norm(ina, ityp: int) -> float:
    """`double daceNorm(const DACEDA REF(ina), const unsigned int ityp);`"""
    ...


@corefunction([DACEDA_p, c_uint, c_uint, c_double_p])
def OrderedNorm(ina, ivar: int, ityp: int, onorm) -> None:
    """`void daceOrderedNorm(const DACEDA REF(ina), const unsigned int ivar, const unsigned int ityp, double onorm[]);`"""
    ...


@corefunction([DACEDA_p, c_uint, c_uint, c_double_p, c_double_p, c_uint])
def Estimate(ina, ivar: int, ityp: int, c, err, nc: int) -> None:
    """`void daceEstimate(const DACEDA REF(ina), const unsigned int ivar, const unsigned int ityp, double c[], double err[], const unsigned int nc);`"""
    ...


@corefunction([DACEDA_p, c_double_p, c_double_p])
def GetBounds(ina, alo, aup) -> None:
    """`void daceGetBounds(const DACEDA REF(ina), double REF(alo), double REF(aup));`"""
    ...


# *****************************************************************************
# *     DACE polynomial evaluation routines
# *****************************************************************************


@corefunction([DACEDA_p, DACEDA_p], c_double)
def EvalMonomials(ina, inb) -> float:
    """`double daceEvalMonomials(const DACEDA REF(ina), const DACEDA REF(inb));`"""
    ...


@corefunction([DACEDA_p, c_uint, c_uint, c_double, DACEDA_p])
def ReplaceVariable(ina, from_: int, to: int, val: float, inc) -> None:
    """`void daceReplaceVariable(const DACEDA REF(ina), const unsigned int from, const unsigned int to, const double val, DACEDA REF(inc));`""",
    ...


@corefunction([DACEDA_p, c_uint, c_double, DACEDA_p])
def EvalVariable(ina, nvar: int, val: float, inc) -> None:
    """`void daceEvalVariable(const DACEDA REF(ina), const unsigned int nvar, const double val, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_uint, c_double, DACEDA_p])
def ScaleVariable(ina, nvar: int, val: float, inc) -> None:
    """`void daceScaleVariable(const DACEDA REF(ina), const unsigned int nvar, const double val, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p, c_uint, c_double, c_double, DACEDA_p])
def TranslateVariable(ina, nvar: int, a: float, c: float, inc) -> None:
    """`void daceTranslateVariable(const DACEDA REF(ina), const unsigned int nvar, const double a, const double c, DACEDA REF(inc));`"""
    ...


@corefunction([DACEDA_p_p, c_uint, c_double_p, c_uint_p, c_uint_p, c_uint_p])
def EvalTree(das, count: int, ac, nterm, nvar, nord) -> None:
    """`void daceEvalTree(const DACEDA *das[], const unsigned int count, double ac[], unsigned int REF(nterm), unsigned int REF(nvar), unsigned int REF(nord));`"""
    ...


# *****************************************************************************
# *     DACE input/output routines
# *****************************************************************************


@corefunction([DACEDA_p, c_char_p, c_uint_p])
def Write(ina, strs, nstrs) -> None:
    """`void daceWrite(const DACEDA REF(ina), char *strs, unsigned int REF(nstrs));`"""
    ...


@corefunction([DACEDA_p, c_char_p, c_uint])
def Read(ina, strs, nstrs) -> None:
    """`void daceRead(DACEDA REF(ina), char *strs, unsigned int nstrs);`"""
    ...


@corefunction([DACEDA_p])
def Print(ina) -> None:
    """`void dacePrint(const DACEDA REF(ina));`"""
    ...


@corefunction([DACEDA_p, c_void_p, c_uint_p], c_uint)
def ExportBlob(ina, blob, size) -> int:
    """`unsigned int daceExportBlob(const DACEDA REF(ina), void *blob, unsigned int REF(size));`"""
    ...


@corefunction([c_void_p], c_uint)
def BlobSize(blob) -> int:
    """`unsigned int daceBlobSize(const void *blob);`"""
    ...


@corefunction([c_void_p, DACEDA_p])
def ImportBlob(blob, inc) -> None:
    """`void daceImportBlob(const void *blob, DACEDA REF(inc));`"""
    ...


# *****************************************************************************
# *     DACE miscellaneous routines
# *****************************************************************************


@corefunction([], c_double)
def Random() -> float:
    """`double daceRandom();`"""
    ...
