import daceypy_import_helper  # noqa: F401

from time import perf_counter

from inspect import getsourcefile
from io import TextIOWrapper
from math import ceil
from pathlib import Path
from typing import Callable, List, Union, overload

import numpy as np
from daceypy import DA, array
from daceypy.op import cos, sin, sqr, sqrt, vnorm
from numpy.typing import NDArray


sourcefile = getsourcefile(lambda: 0)
assert sourcefile is not None
thisfolder = Path(sourcefile).resolve().parent


# Exercise 6.1.1: The Mid-point rule integrator
@overload
def midpoint(
    x0: array, t0: float, t1: float,
    f: Callable[[array, float], array]
) -> array: ...


@overload
def midpoint(
    x0: NDArray[np.double], t0: float, t1: float,
    f: Callable[[NDArray[np.double], float], NDArray[np.double]]
) -> NDArray[np.double]: ...


def midpoint(x0, t0, t1, f):
    hmax = 0.005
    steps = ceil((t1 - t0) / hmax)
    h = (t1 - t0) / steps
    t = t0

    for _ in range(steps):
        xmid = x0 + 0.5 * h * f(x0, t)
        x0 = x0 + h * f(xmid, t + 0.5 * h)
        t += h

    return x0


# Exercise 6.1.2: Model of the (uncontrolled) pendulum
# x = ( theta, theta_dot )
# since the motion for x decouples we ignore it here
@overload
def pendulumRHS(x: array, t: float) -> array: ...


@overload
def pendulumRHS(x: NDArray[np.double], t: float) -> NDArray[np.double]: ...


def pendulumRHS(x, t):
    # pendulum constants
    L = 0.61       # m
    m = 0.21       # kg
    M = 0.4926     # kg
    g = 9.81       # kg*m/s^2

    sint = sin(x[0])
    cost = cos(x[0])

    return np.array([
        x[1],
        ((M+m)*g*sint-m*L*sqr(x[1])*sint*cost)/((M+m)*L+m*L*sqr(cost)),
    ]).view(type(x))


def ex6_1_2():
    dt = 0.05

    with (thisfolder / "ex6_1_2.dat").open("w") as file:
        xdb = np.array([0.0, 0.2])  # initial condition (double)
        t = 0
        for _ in range(100):
            # propagate forward for dt seconds
            xdb = midpoint(xdb, t, t + dt, pendulumRHS)
            file.write(f"{t}   {xdb[0]}   {xdb[1]}\n")
            t += dt
        file.write("\n\n")

        xDA = array([0.0, 0.2 + 0.04 * DA(1)])  # initial condition (DA)
        t = 0
        for _ in range(100):
            # propagate forward for dt seconds
            xDA = midpoint(xDA, t, t + dt, pendulumRHS)
            file.write(
                f"{t}"
                f"   {xDA[0].evalScalar(-1.0)}   {xDA[0].evalScalar(+1.0)}\n"
                f"   {xDA[1].evalScalar(-1.0)}   {xDA[1].evalScalar(+1.0)}\n"
            )
            t += dt
        file.write("\n")

    print(f"Exercise 6.1.2: Model of the (uncontrolled) pendulum\n{xDA}")


# Exercise 6.1.3: Set propagation
# the right hand side

@overload
def f(x: array, t: float) -> array: ...


@overload
def f(x: NDArray[np.double], t: float) -> NDArray[np.double]: ...


def f(x, t):
    alpha = 0.1
    res = np.array([-x[1], x[0]]).view(type(x))
    return (1.0 + alpha * vnorm(x)) * res


# convenience routine to evaluate and plot
def plot(x: array, t: float, N: int, file: TextIOWrapper):

    arg = np.array([0.0, 0.0])
    for i in range(-N, N+1):
        arg[0] = i / N
        for j in range(-N, N+1):
            arg[1] = j / N
            res = x.eval(arg)
            file.write(f"{t}    {res[0]}    {res[1]}\n")
        file.write("\n")
    file.write("\n")


def ex6_1_3():
    pi = 3.141592653589793
    dt = 2.0 * pi / 6.0

    with (thisfolder / "ex6_1_3.dat").open("w") as file:
        x = array([2.0 + DA(1), DA(2)])    # initial condition box
        t = 0.0
        plot(x, t, 7, file)

        for i in range(6):
            x = midpoint(x, t, t + dt, f)  # propagate forward for dt seconds
            t += dt
            plot(x, t, 7, file)

    print(f"Exercise 6.1.3: Set propagation{x}")


# Exercise 6.1.4: State Transition Matrix
def ex6_1_4():
    pi = 3.141592653589793
    x = array([1.0 + DA(1), 1.0 + DA(2)])  # initial condition around (1,1)
    x = midpoint(x, 0, 2 * pi, f)

    print("Exercise 6.1.4: State Transition Matrix")
    print(f"{x[0].deriv(1).cons()}    {x[0].deriv(2).cons()}")
    print(f"{x[1].deriv(1).cons()}    {x[1].deriv(2).cons()}")


# Exercise 6.1.5: Parameter dependence
# the right hand side
def fParam(x: array, t: float) -> array:
    alpha = 0.05 + 0.05 * DA(1)     # parameter, now it's a DA
    res = array([-x[1], x[0]])
    return (1.0 + alpha * x.vnorm()) * res


def ex6_1_5():
    pi = 3.141592653589793
    x = array([1.0, 1.0])  # initial condition (1,1)
    x = midpoint(x, 0, 2*pi, fParam)

    with (thisfolder / "ex6_1_5.dat").open("w") as file:

        file.write("1 1\n\n\n")
        for i in range(21):
            file.write(
                f"{x[0].evalScalar(-1.0 + i/10.0)}   "
                f"{x[1].evalScalar(-1.0 + i/10.0)}\n")

    print("Exercise 6.1.5: Parameter dependence")


# Exercise 6.2.1: 3/8 rule RK4 integrator
@overload
def rk4(
    x0: array, t0: float, t1: float,
    f: Callable[[array, float], array]
) -> array: ...


@overload
def rk4(
    x0: NDArray[np.double], t0: float, t1: float,
    f: Callable[[NDArray[np.double], float], NDArray[np.double]]
) -> NDArray[np.double]: ...


def rk4(x0, t0, t1, f):

    hmax = 0.01
    steps = ceil((t1 - t0) / hmax)
    h = (t1 - t0) / steps
    t = t0

    for _ in range(steps):
        k1 = f(x0, t)
        k2 = f(x0 + h*k1/3.0, t + h/3.0)
        k3 = f(x0 + h*(-k1/3.0 + k2), t + 2.0*h/3.0)
        k4 = f(x0 + h*(k1 - k2 + k3), t + h)
        x0 = x0 + h * (k1 + 3*k2 + 3*k3 + k4)/8.0
        t += h

    return x0


# Exercise 6.2.2: Artsy Set Propagation
def ex6_2_2():
    pi = 3.141592653589793
    dt = 2.0*pi/6.0

    with (thisfolder / "ex6_2_2.dat").open("w") as file:

        # initial condition (c.f. 5Vectors-Ex.cpp)
        x = array([
            DA(1),
            (
                (1.0-DA(1)*DA(1)) *
                (DA(2)+1.0) +
                (DA(1)*DA(1)*DA(1)-DA(1)) *
                (1.0-DA(2))
            ) / 2.0
        ])
        t = 0
        plot(x, t, 7, file)

        for i in range(6):
            x = midpoint(x, t, t + dt, f)  # propagate forward for dt seconds
            t += dt
            plot(x, t, 7, file)

    print(f"Exercise 6.2.2: Artsy Set propagation\n{x}")


# Exercise 6.2.3: CR3BP
@overload
def CR3BP(x: array, t: float) -> array: ...


@overload
def CR3BP(x: NDArray[np.double], t: float) -> NDArray[np.double]: ...


def CR3BP(
    x: Union[array, NDArray[np.double]], t: float
) -> Union[array, NDArray[np.double]]:

    MU = 0.30404234e-5
    d1 = sqrt(sqr(x[0] + MU) + sqr(x[1]) + sqr(x[2]))
    d1 = 1.0 / (d1*d1*d1)    # first distance
    d2 = sqrt(sqr(x[0] + MU - 1.0) + sqr(x[1]) + sqr(x[2]))
    d2 = 1.0 / (d2*d2*d2)    # second distance

    res = np.array([
        x[3], x[4], x[5],
        x[0] + 2.0*x[4] - d1*(1-MU)*(x[0]+MU) - d2*MU*(x[0]+MU-1.0),
        x[1] - 2.0*x[3] - d1*(1-MU)*x[1] - d2*MU*x[1],
        - d1*(1-MU)*x[2] - d2*MU*x[2],
    ]).view(type(x))

    return res


def ex6_2_3():
    T = 3.05923
    x0 = array([0.9888426847, 0, 0.0011210277, 0, 0.0090335498, 0])
    x = x0 + array.identity()

    DA.pushTO(1)   # only first order computation needed

    x = rk4(x, 0, T, CR3BP)

    print("Exercise 6.2.3: CR3BP STM")
    for i in range(6):
        for j in range(1, 7):
            print(x[i].deriv(j).cons())

    DA.popTO()


# Exercise 6.2.4: Set propagation revisited
def ex6_2_4():

    pi = 3.141592653589793
    dt = 2.0 * pi / 6.0

    with (thisfolder / "ex6_2_4.dat").open("w") as file:

        # initial condition box, in polar coordinates
        x = array([cos(0.3*DA(2))*(2.0 + DA(1)), sin(0.3*DA(2))*(2.0 + DA(1))])
        t = 0
        plot(x, t, 40, file)

        for i in range(6):
            x = midpoint(x, t, t + dt, f)   # propagate forward for dt seconds
            t += dt
            plot(x, t, 40, file)

    print(f"Exercise 6.2.4: Set propagation revisited\n{x}")


# Exercise 6.2.5: The State Transition Matrix reloaded
def ex6_2_5():
    pi = 3.141592653589793
    # initial condition (1,1) plus DA identity
    # (but in DA(2) and DA(3) as DA(1) is already used for alpha!)
    x = array([1.0 + DA(2), 1.0 + DA(3)])
    x = midpoint(x, 0, 2 * pi, fParam)

    # we want to evaluate the derivatives at (alpha,0,0),
    # so keep DA(1) and replace DA(2) and DA(3) by zero
    arg = array([DA(1), 0.0, 0.0])

    print("Exercise 6.2.5: The State Transition Matrix reloaded")
    print(f"{x[0].deriv(2).eval(arg)}{x[0].deriv(3).eval(arg)}")
    print(f"{x[1].deriv(2).eval(arg)}{x[1].deriv(3).eval(arg)}")


def execute_and_time(f: Callable):
    print(f"Executing {f.__name__} ...")
    t0 = perf_counter()
    f()
    print(f"{f.__name__} executed in {perf_counter() - t0} s")


def main():
    DA.init(15, 6)  # init with maximum computation order

    DA.cache_enable()  # optional, should speed up the process

    exercises: List[Callable[[], None]] = [
        ex6_1_2, ex6_1_3, ex6_1_4, ex6_1_5,
        ex6_2_2, ex6_2_3, ex6_2_4, ex6_2_5,
    ]

    for ex in exercises:
        print("\n--------------------------------------")
        print(f"---- Executing {ex.__name__} ...")
        t0 = perf_counter()
        ex()
        print(f"{ex.__name__} executed in {perf_counter() - t0} s")
        print("--------------------------------------\n")


if __name__ == "__main__":
    main()
