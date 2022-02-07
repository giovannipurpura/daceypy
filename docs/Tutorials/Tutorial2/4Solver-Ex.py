import daceypy_import_helper  # noqa: F401

from typing import Union, overload

from inspect import getsourcefile
from pathlib import Path

from daceypy import DA
from daceypy.op import cos, sin, sqrt


sourcefile = getsourcefile(lambda: 0)
assert sourcefile is not None
thisfolder = Path(sourcefile).resolve().parent

order = 10


def Nf(x, p):
    return x - (x * x - p) / (2 * x)


# Exercise 4.1.3: Naive Newton
def ex4_1_3():

    tol = 1e-14       # tolerance
    p0 = 4.0          # expansion point
    x0 = 1.0          # initial guess
    p = p0 + DA(1)    # DA parameter
    x = DA(x0)        # DA initial guess
    xp = DA()

    i = 0

    flag = True
    while flag:
        xp.assign(x)
        x = Nf(xp, p)
        i += 1
        flag = abs(xp - x) > tol and i < 1000

    print(f"Exercise 4.1.3: Naive Newton\n{x}\n{sqrt(p) - x}")
    print(f"Number of iterations: {i}\n")


# Exercise 4.1.4: complicated parameters
def ex4_1_4():
    p0 = 0.0
    x0 = 1.0       # x0 must now satisfy f(x0,cos(p0))=0
    p = cos(p0 + DA(1))
    x = x0

    i = 1
    while i <= order:
        x = Nf(x, p)
        i *= 2

    print(f"Exercise 4.1.3: Naive Newton\n{x}\n{sqrt(p) - x}\n")


# Exercise 4.2.1: Full DA Newton solver
def ex4_2_1(p0: float):
    tol = 1e-14
    x0 = p0 / 2.0   # x0 is just some initial guess
    i = 0

    # double precision computation => fast
    flag = True
    while flag:
        xp = x0
        x0 = Nf(xp, p0)
        i += 1
        flag = abs(xp-x0) > tol and i < 1000

    # DA computation => slow
    p = p0 + DA(1)
    x = x0
    i = 1
    while i <= order:
        x = Nf(x, p)
        i *= 2

    print(f"Exercise 4.2.1: Full DA Newton\n{x}\n{sqrt(p) - x}\n")


# Exercise 4.2.2 & 4.2.3: Kepler's equation solver
DA_or_float = Union[DA, float]


@overload
def NKep(E: DA, M: DA, ecc: DA) -> DA: ...  # only for typechecking


@overload
def NKep(E: float, M: float, ecc: float) -> float: ...  # only for typechecking


def NKep(E: DA_or_float, M: DA_or_float, ecc: DA_or_float) -> DA_or_float:
    return E - (E - ecc * sin(E) - M) / (1 - ecc * cos(E))


# double precision Kepler solver
def Kepler(M: float, ecc: float) -> float:

    tol = 1e-14
    E0 = M
    i = 0

    flag = True
    while flag:
        Ep = E0
        E0 = NKep(Ep, M, ecc)
        i += 1
        flag = abs(Ep - E0) > tol and i < 1000

    return E0


def ex4_2_2(M0: float, ecc0: float):

    M = M0 + DA(1)
    E = DA(Kepler(M0, ecc0))     # reference solution
    ecc = DA(ecc0)               # keep eccentricity constant (4.2.2)
    # ecc = ecc0 + 0.1*DA(2)     # also expand w.r.t. eccentricity (4.2.3)

    i = 1
    while i <= order:
        E = NKep(E, M, ecc)
        i *= 2

    print("Exercise 4.2.2: Expansion of the Anomaly")
    print(f"Resulting expansion:\n{E}")
    print(f"Residual error:\n{E-ecc*sin(E)-M}")

    # sample the resulting polynomial over M0+-3 rad
    with (thisfolder / "ex4_2_2.dat").open("w") as file:
        for i in range(-300, 300):
            file.write(
                f"{M0 + i / 100.0}    "
                f"{E.evalScalar(i / 100.0)}    "
                f"{Kepler(M0 + i / 100.0, ecc0)}\n")

    # gnuplot command:
    # plot 'ex4_2_2.dat' u ($1*180/pi):($2*180/pi) w l t 'DA',
    # 'ex4_2_2.dat' u ($1*180/pi):($3*180/pi) w l t 'pointwise'


def main():

    DA.init(order, 2)  # init with maximum computation order

    ex4_1_3()
    ex4_1_4()

    ex4_2_1(9.0)
    ex4_2_2(0.0, 0.5)


if __name__ == "__main__":
    main()
