"""
## DACEyPy operators

This submodule defines some useful mathematical operators that can be imported
and applied directly to floats, NumPy arrays, DA scalars and DACEyPy arrays.
"""


import numpy as np

import daceypy as dp

_original_round = round


class ArgTypeException(TypeError):
    def __init__(self, x) -> None:
        if isinstance(x, (list, tuple)):
            msg = f"Invalid argument type: {type(x)} of {type(x[0])}"
        else:
            msg = f"Invalid argument type: {type(x)}"
        super().__init__(msg)


def DA(x):
    if isinstance(x, (dp.DA, float, int)):
        return dp.DA(x)
    raise ArgTypeException(x)


def array(x):
    if isinstance(x, (list, tuple, dp.array, np.ndarray)):
        return dp.array(x)
    raise ArgTypeException(x)


def round(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).round()
        if isinstance(x[0], (float, int)):
            return np.array(x).round()
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.round()
    if isinstance(x, (float, int)):
        return _original_round(x)
    raise ArgTypeException(x)


def root(x, p=2):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).root(p)
        if isinstance(x[0], (float, int)):
            return np.power(np.array(x), 1 / p)
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.root(p)
    if isinstance(x, (float, int)):
        return np.power(x, 1 / p)
    raise ArgTypeException(x)


def minv(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).minv()
        if isinstance(x[0], (float, int)):
            return 1 / np.array(x)
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.minv()
    if isinstance(x, (float, int)):
        return 1 / x
    raise ArgTypeException(x)


def sqr(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).sqr()
        if isinstance(x[0], (float, int)):
            return np.power(np.array(x), 2)
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.sqr()
    if isinstance(x, (float, int)):
        return np.power(x, 2)
    raise ArgTypeException(x)


def sqrt(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).sqrt()
        if isinstance(x[0], (float, int)):
            return np.sqrt(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.sqrt()
    if isinstance(x, (float, int)):
        return np.sqrt(x)
    raise ArgTypeException(x)


def isrt(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).isrt()
        if isinstance(x[0], (float, int)):
            return 1 / np.array(x)
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.isrt()
    if isinstance(x, (float, int)):
        return 1 / np.sqrt(x)
    raise ArgTypeException(x)


def cbrt(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).cbrt()
        if isinstance(x[0], (float, int)):
            return np.power(np.array(x), 1 / 3)
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.cbrt()
    if isinstance(x, (float, int)):
        return np.power(x, 1 / 3)
    raise ArgTypeException(x)


def icrt(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).icrt()
        if isinstance(x[0], (float, int)):
            return np.power(np.array(x), -1 / 3)
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.icrt()
    if isinstance(x, (float, int)):
        return np.power(x, -1 / 3)
    raise ArgTypeException(x)


def hypot(x, other):
    if isinstance(x, dp.DA):
        return x.hypot(other)
    if isinstance(x, (float, int)):
        return np.hypot(x, other)
    raise ArgTypeException(x)


def exp(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).exp()
        if isinstance(x[0], (float, int)):
            return np.exp(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.exp()
    if isinstance(x, (float, int)):
        return np.exp(x)
    raise ArgTypeException(x)


def log(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).log()
        if isinstance(x[0], (float, int)):
            return np.log(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.log()
    if isinstance(x, (float, int)):
        return np.log(x)
    raise ArgTypeException(x)


def logb(x, b=10.0):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).logb(b)
        if isinstance(x[0], (float, int)):
            return np.log(np.array(x)) / np.log(b)
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.logb(b)
    if isinstance(x, (float, int)):
        return np.log(x) / np.log(b)
    raise ArgTypeException(x)


def log10(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).log10()
        if isinstance(x[0], (float, int)):
            return np.log10(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.log10()
    if isinstance(x, (float, int)):
        return np.log10(x)
    raise ArgTypeException(x)


def log2(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).log2()
        if isinstance(x[0], (float, int)):
            return np.log2(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.log2()
    if isinstance(x, (float, int)):
        return np.log2(x)
    raise ArgTypeException(x)


def sin(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).sin()
        if isinstance(x[0], (float, int)):
            return np.sin(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.sin()
    if isinstance(x, (float, int)):
        return np.sin(x)
    raise ArgTypeException(x)


def cos(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).cos()
        if isinstance(x[0], (float, int)):
            return np.cos(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.cos()
    if isinstance(x, (float, int)):
        return np.cos(x)
    raise ArgTypeException(x)


def tan(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).tan()
        if isinstance(x[0], (float, int)):
            return np.tan(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.tan()
    if isinstance(x, (float, int)):
        return np.tan(x)
    raise ArgTypeException(x)


def asin(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).asin()
        if isinstance(x[0], (float, int)):
            return np.arcsin(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.asin()
    if isinstance(x, (float, int)):
        return np.arcsin(x)
    raise ArgTypeException(x)


def acos(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).acos()
        if isinstance(x[0], (float, int)):
            return np.arccos(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.acos()
    if isinstance(x, (float, int)):
        return np.arccos(x)
    raise ArgTypeException(x)


def atan(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).atan()
        if isinstance(x[0], (float, int)):
            return np.arctan(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.atan()
    if isinstance(x, (float, int)):
        return np.arctan(x)
    raise ArgTypeException(x)


def atan2(x, other):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).atan2(other)
        if isinstance(x[0], (float, int)):
            return np.arctan2(np.array(x), np.array(other))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.atan2(other)
    if isinstance(x, (float, int)):
        return np.arctan2(x, other)
    raise ArgTypeException(x)


def sinh(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).sinh()
        if isinstance(x[0], (float, int)):
            return np.sinh(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.sinh()
    if isinstance(x, (float, int)):
        return np.sinh(x)
    raise ArgTypeException(x)


def cosh(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).cosh()
        if isinstance(x[0], (float, int)):
            return np.cosh(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.cosh()
    if isinstance(x, (float, int)):
        return np.cosh(x)
    raise ArgTypeException(x)


def tanh(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).tanh()
        if isinstance(x[0], (float, int)):
            return np.tanh(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.tanh()
    if isinstance(x, (float, int)):
        return np.tanh(x)
    raise ArgTypeException(x)


def asinh(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).asinh()
        if isinstance(x[0], (float, int)):
            return np.arcsinh(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.asinh()
    if isinstance(x, (float, int)):
        return np.arcsinh(x)
    raise ArgTypeException(x)


def acosh(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).acosh()
        if isinstance(x[0], (float, int)):
            return np.arccosh(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.acosh()
    if isinstance(x, (float, int)):
        return np.arccosh(x)
    raise ArgTypeException(x)


def atanh(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).atanh()
        if isinstance(x[0], (float, int)):
            return np.arctanh(np.array(x))
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.atanh()
    if isinstance(x, (float, int)):
        return np.arctanh(x)
    raise ArgTypeException(x)


def erf(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).erf()
        if isinstance(x[0], (float, int)):
            return dp.array(x).erf().cons()
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.erf()
    if isinstance(x, (float, int)):
        return dp.DA(x).erf().cons()
    raise ArgTypeException(x)


def erfc(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).erfc()
        if isinstance(x[0], (float, int)):
            return dp.array(x).erfc().cons()
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.erfc()
    if isinstance(x, (float, int)):
        return dp.DA(x).erfc().cons()
    raise ArgTypeException(x)


def GammaFunction(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).GammaFunction()
        if isinstance(x[0], (float, int)):
            return dp.array(x).GammaFunction().cons()
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.GammaFunction()
    if isinstance(x, (float, int)):
        return dp.DA(x).GammaFunction().cons()
    raise ArgTypeException(x)


def LogGammaFunction(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).LogGammaFunction()
        if isinstance(x[0], (float, int)):
            return dp.array(x).LogGammaFunction().cons()
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.LogGammaFunction()
    if isinstance(x, (float, int)):
        return dp.DA(x).LogGammaFunction().cons()
    raise ArgTypeException(x)


def PsiFunction(x, n):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).PsiFunction(n)
        if isinstance(x[0], (float, int)):
            return dp.array(x).PsiFunction(n).cons()
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.PsiFunction(n)
    if isinstance(x, (float, int)):
        return dp.DA(x).PsiFunction(n).cons()
    raise ArgTypeException(x)


def cons(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], dp.DA):
            return dp.array(x).cons()
        raise ArgTypeException(x)
    if isinstance(x, (dp.DA, dp.array)):
        return x.cons()
    raise ArgTypeException(x)


def vnorm(x):
    if isinstance(x, dp.array):
        return x.vnorm()
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x)
    raise ArgTypeException(x)
