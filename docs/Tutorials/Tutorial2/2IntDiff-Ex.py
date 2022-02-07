
import daceypy_import_helper  # noqa: F401

from math import pi
from typing import Union, overload

from daceypy import DA
from daceypy.op import cos, sin, sqrt  # can be used on DA and on floats


# Exercise 2.1.1: derivatives
@overload
def somb(x: DA, y: DA) -> DA: ...  # only for typechecking


@overload
def somb(x: float, y: float) -> float: ...  # only for typechecking


def somb(x: Union[DA, float], y: Union[DA, float]) -> Union[DA, float]:
    """Sombrero function"""
    r = sqrt(x ** 2 + y ** 2)
    s = sin(r) / r
    return s


def ex2_1_1():

    x0 = 2.0
    y0 = 3.0
    x = DA(1)
    y = DA(2)

    # expand sombrero function around (x0, y0)
    func = somb(x0 + x, y0 + y)

    # compute the derivative using DA
    dadx = func.deriv(1).cons()
    dady = func.deriv(2).cons()
    dadxx = func.deriv(1).deriv(1).cons()
    dadxy = func.deriv(1).deriv(2).cons()
    dadyy = func.deriv(2).deriv(2).cons()
    dadxxx = func.deriv(1).deriv(1).deriv(1).cons()

    # compute the derivatives using finite differences
    h = 1e-3
    dx = (somb(x0 + h, y0) - somb(x0 - h, y0)) / (2.0 * h)
    dy = (somb(x0, y0 + h) - somb(x0, y0 - h)) / (2.0 * h)
    dxx = (
        somb(x0 + 2.0 * h, y0)
        - 2.0 * somb(x0, y0)
        + somb(x0 - 2.0 * h, y0)) / (4.0 * h * h)
    dxy = (
        somb(x0 + h, y0 + h)
        - somb(x0 - h, y0 + h)
        - somb(x0 + h, y0 - h)
        + somb(x0 - h, y0 - h)) / (4.0 * h * h)
    dyy = (
        somb(x0, y0 + 2.0 * h)
        - 2.0 * somb(x0, y0)
        + somb(x0, y0 - 2.0 * h)) / (4.0 * h * h)
    dxxx = (
        somb(x0 + 3.0 * h, y0)
        - 3.0 * somb(x0 + h, y0)
        + 3.0 * somb(x0 - h, y0)
        - somb(x0 - 3.0 * h, y0)) / (8.0 * h * h * h)

    print("Exercise 2.1.1: Numerical derivatives\n")
    print(f"d/dx:    {abs(dadx - dx)}")
    print(f"d/dy:    {abs(dady - dy)}")
    print(f"d/dxx:   {abs(dadxx - dxx)}")
    print(f"d/dxy:   {abs(dadxy - dxy)}")
    print(f"d/dyy:   {abs(dadyy - dyy)}")
    print(f"d/dxxx:  {abs(dadxxx - dxxx)}\n")


# Exercise 2.1.2: indefinite integral
def ex2_1_2():

    x = DA(1)
    func = (1.0 / (1 + x.sqr())).integ(1)  # DA integral
    integral = x.atan()  # analytical integral DA expanded
    print(f"Exercise 2.1.2: Indefinite integral\n{func - integral}\n")


# Exercise 2.1.3: expand the error function
def ex2_1_3():
    t = DA(1)
    erf = 2.0/sqrt(pi)*(-t.sqr()).exp().integ(1)  # error function erf(x)
    print(f"Exercise 2.1.3: Error function\n{erf}\n")


# Exercise 2.2.1: DA based Newton solver
def f_DA(x: DA):
    return x * x.sin() + x.cos()


def f_float(x: float):
    return x * sin(x) + cos(x)


def ex2_2_1(x0: float) -> float:

    DA.pushTO(1)  # for this Newton solver we only need first derivatives

    err = 1e-14
    x = DA(1)
    i = 0

    flag = True
    while flag:
        func = f_DA(x0 + x)  # expand f around x0
        x0 -= func.cons()/func.deriv(1).cons()  # Newton step
        i += 1
        flag = abs(func.cons()) > err and i < 1000

    print("Exercise 2.2.1: DA Newton solver\n")
    print(f"root x0:           {x0}")
    print(f"residue at f(x0):  {abs(f_float(x0))}")
    print(f"Newton iterations: {i}\n")

    # don't forget to reset computation order to old value
    # for following computations
    DA.popTO()
    return x0


# Exercise 2.2.2: expand the error function around x0
def ex2_2_2(x0: float):
    t = x0 + DA(1)
    erf = 2.0 / sqrt(pi) * (-t.sqr()).exp().integ(1)  # error function erf(x)
    print(f"Exercise 2.2.2: Shifted indefinite integral\n{erf}")


def main():

    DA.init(30, 2)

    ex2_1_1()
    ex2_1_2()
    ex2_1_3()

    ex2_2_1(3.6)
    ex2_2_2(1.0)


if __name__ == "__main__":
    main()
