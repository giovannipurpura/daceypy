import daceypy_import_helper  # noqa: F401

from inspect import getsourcefile
from pathlib import Path

import numpy as np
from daceypy import DA, array
from daceypy.op import cos, exp, sin, sqr


sourcefile = getsourcefile(lambda: 0)
assert sourcefile is not None
thisfolder = Path(sourcefile).resolve().parent

order = 20


# Exercise 5.1.1: tangents and normals
def f(x):
    return -1/3*(sqr(x[0])+sqr(x[1])/2)+exp(x[0]/2+x[1]/4)


def ex5_1_1():
    s1 = DA(1)
    s2 = DA(2)
    surf = array([s1, s2, f([s1, s2])])

    t1 = surf.deriv(1)
    t2 = surf.deriv(2)
    n = t1.cross(t2)

    print(f"Exercise 5.1.1: tangents and normals\n{t1}{t2}{n}")


# Exercise 5.1.2: (Uncontrolled) Equations of motion of the inverted pendulum
def ode_pendulum(x, y):

    # constants
    L = 1.0    # length of pendulum (m)
    m = 0.1    # weight of balanced object (kg)
    M = 0.4    # weight of cart (kg)
    g = 9.81   # gravity acceleration constant on earth (kg*m/s^2)

    # variables
    sint = sin(x)  # sine of theta
    cost = cos(y)  # cosine of theta

    # Equations of motion
    return y, ((M+m)*g*sint-m*L*sqr(y)*sint*cost)/((M+m)*L+m*L*sqr(cost))


def ex5_1_2():
    x, y = 1.0, 0.0
    print("Exercise 5.1.2: Equations of Motion")
    print(ode_pendulum(x, y))


# Exercise 5.2.1: Solar flux
def ex5_2_1():
    s1 = DA(1)
    s2 = DA(2)
    surf = array([s1, s2, f([s1, s2])])
    # normalizing these helps keep the coefficents small
    # and prevents roundoff errors
    t1 = surf.deriv(1).normalize()
    t2 = surf.deriv(2).normalize()
    n = t1.cross(t2).normalize()    # normalized surface normal
    sun = [0.0, 0.0, 1.0]           # sun direction
    flux = n.dot(sun)               # solar flux on the surface

    # Output results
    print("Exercise 5.2.1: Solar flux")
    print(flux)
    N = 30
    arg = np.array([0.0, 0.0])
    res = np.array([0.0, 0.0, 0.0])
    with (thisfolder / "ex5_2_1.dat").open("w") as file:
        for i in range(-N, N + 1):
            arg[0] = i / N
            for j in range(-N, N + 1):
                arg[1] = j / N
                res = surf.eval(arg)
                file.write(
                    f"{res[0]}    {res[1]}    {f(res)}    {flux.eval(arg)}\n")
            file.write("\n")


# Exercise 5.2.2: Area
def ex5_2_2():
    x = DA(1)
    t = DA(2)
    res = array([DA(1), ((1.0-x*x)*(t+1.0)+(x*x*x-x)*(1.0-t))/2.0])

    print("Exercise 5.2.2: Area")
    print(res)


def main():
    DA.init(order, 2)      # init with maximum computation order

    ex5_1_1()
    ex5_1_2()

    ex5_2_1()
    ex5_2_2()


if __name__ == "__main__":
    main()
