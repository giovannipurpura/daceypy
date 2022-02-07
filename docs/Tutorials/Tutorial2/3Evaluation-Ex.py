import daceypy_import_helper  # noqa: F401

from inspect import getsourcefile
from math import pi, sin, sqrt
from pathlib import Path

import numpy as np
from daceypy import DA

sourcefile = getsourcefile(lambda: 0)
assert sourcefile is not None
thisfolder = Path(sourcefile).resolve().parent

val1 = 1.685401585899429  # reference value of the integral over -1, +1
val2 = 1.990644530037905  # reference value of the integral over -2, +2


# Exercise 3.1.1: plot a 1D polynomial
def ex3_1_1():
    x0 = 0.0          # expansion point
    N = 100           # number of points in grid
    hw = 2.0          # length of grid in each direction from expansion point

    x = DA(1) + x0
    func = (-x*x).exp()

    with (thisfolder / "ex3_1_1.dat").open("w") as file:
        for i in range(N):
            xx = -hw + i * 2.0 * hw / (N - 1)  # point on the grid on [-hw,hw]
            # note: this is not an efficient way to
            # repeatedly evaluate the same polynomial
            rda = func.evalScalar(xx)
            xx += x0  # add expansion point x0 for double evaluation
            rdouble = np.exp(-xx*xx)
            file.write(f"{xx}   {rda}   {rdouble}\n")

    # gnuplot command: plot 'ex3_1_1.dat' u 1:2 w l, 'ex3_1_1.dat' u 1:3 w l
    # or for the error: plot 'ex3_1_1.dat' u 1:($2-$3) w l


# Exercise 3.1.2: plot a 2D polynomial
def somb_DA(x: DA, y: DA) -> DA:
    r = (x*x + y*y).sqrt()
    return r.sin() / r


def somb_float(x: float, y: float) -> float:
    r = sqrt(x*x + y*y)
    try:
        return sin(r) / r
    except ZeroDivisionError:
        return float("nan")


def ex3_1_2():
    x0 = 1.0  # expansion point x
    y0 = 1.0  # expansion point y
    N = 50  # number of points in grid

    x = DA(1) + x0
    y = DA(2) + y0
    func = somb_DA(x, y)
    arg = np.zeros(2)  # vector holding two doubles

    with (thisfolder / "ex3_1_2.dat").open("w") as file:
        for i in range(N):
            # x coordinate on the grid on [-1, 1]
            arg[0] = -1.0 + i * 2.0 / (N - 1)
            for j in range(N):
                # y coordinate on the grid on [-1, 1]
                arg[1] = -1.0 + j*2.0 / (N - 1)
                # note: this is not an efficient way
                # to repeatedly evaluate the same polynomial
                rda = func.eval(arg)
                rdouble = somb_float(x0 + arg[0], y0 + arg[1])
                file.write(f"{arg[0]}   {arg[1]}   {rda}   {rdouble}\n")
            file.write("\n")   # empty line between lines of data for gnuplot

    # gnuplot command:
    #   splot 'ex3_1_2.dat' u 1:2:3 w l, 'ex3_1_2.dat' u 1:2:4 w l
    # or for the error: splot 'ex3_1_2.dat' u 1:2:($3-$4) w l


# Exercise 3.1.3: Sinusitis
def ex3_1_3():
    DA.pushTO(10)

    x = DA(1)
    sinda = x.sin()
    res1 = (x + 2).sin()  # compute directly sin(2+DA(1))
    res2 = sinda.evalScalar(x + 2)    # evaluate expansion of sine with 2+DA(1)

    print("Exercise 3.1.3: Sinusitis")
    print(res1 - res2)

    DA.popTO()


# Exercise 3.1.4: Gauss integral I
def ex3_1_4():
    with (thisfolder / "ex3_1_4.dat").open("w") as file:
        for order in range(1, 41):
            DA.setTO(order)  # limit the computation order
            t = DA(1)
            # error function erf(x)
            erf = 2.0/sqrt(pi)*(-t.sqr()).exp().integ(1)
            res = erf.evalScalar(1.0) - erf.evalScalar(-1.0)
            file.write(f"{order}   {res}   {np.log10(abs(res-val1))}\n")
    # gnuplot command: plot 'ex3_1_4.dat'u 1:2 w l
    # or for the error: plot 'ex3_1_4.dat'u 1:3 w l


# Exercise 3.2.1 & 3.2.2: Gauss integral II
def gaussInt(a: float, b: float) -> float:
    """compute integral of Gaussian on interval [a,b]"""
    t = (a + b) / 2.0 + DA(1)  # expand around center point
    func = 2.0 / sqrt(pi) * (-t.sqr()).exp().integ(1)
    # evaluate over -+ half width
    return func.evalScalar((b - a)/2.0) - func.evalScalar(-(b - a) / 2.0)


def ex3_2_1():
    hw = 2.0  # half-width of the interval to integrate on, i.e. [-hw,hw]
    with (thisfolder / "ex3_2_1.dat").open("w") as file:
        DA.pushTO(9)
        for n in range(1, 31):
            res = 0.0
            for i in range(1, n + 1):
                ai = -hw + (i-1)*2.0*hw/n
                ai1 = -hw + i*2.0*hw/n
                res += gaussInt(ai, ai1)
            file.write(f"{n}   {res}   {np.log10(abs(res-val2))}\n")
        DA.popTO()
        # compare to single expansion at full computation order
        res = gaussInt(-hw, hw)
        file.write(f"\n1   {res}   {np.log10(abs(res - val2))}\n")
    # gnuplot command: plot 'ex3_2_1.dat'u 1:3 w lp


def main():
    DA.init(40, 2)  # init with maximum computation order

    ex3_1_1()
    ex3_1_2()
    ex3_1_3()
    ex3_1_4()

    ex3_2_1()


if __name__ == "__main__":
    main()
