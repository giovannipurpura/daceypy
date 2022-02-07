import daceypy_import_helper  # noqa: F401

from inspect import getsourcefile
from math import ceil
from pathlib import Path
from typing import Callable, List, Tuple, overload

import numpy as np
from daceypy import DA, array
from daceypy.op import cos, sin, sqr
from numpy.typing import NDArray


sourcefile = getsourcefile(lambda: 0)
assert sourcefile is not None
thisfolder = Path(sourcefile).resolve().parent


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


# Exercise 7.1.1: Model of the controlled pendulum
# x = ( theta, theta_dot, u )
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
        x[2] + ((M+m)*g*sint-m*L*sqr(x[1])*sint*cost)/((M+m)*L+m*L*sqr(cost)),
        0.0,
    ]).view(type(x))


# Exercise 7.1.2: Tuning the PID simulator (double)
def ex7_1_2():

    Kp, Ti, Td = 8.0, 3.0, 0.3   # PID parameters
    setPt = 0.0                  # Set point
    dt = 0.05                    # Controller time step (50 ms)
    umax = 10.0  # noqa: F841    # maximum control (Exercise 7.2.1)

    with (thisfolder / "ex7_1_2.dat").open("w") as file:
        # PID controller variables
        intErr = 0.0
        lastErr = 0.0

        t = 0.0
        x = np.array([0.1, 0.0, 0.0])  # Initial condition (u(0)=x[2]=0)

        # propagate the model for 100 sec
        while t < 100.0 and abs(x[0]) < 1.5:

            # compute the PID control at this controller time step
            err = setPt - x[0]
            derr = (err - lastErr) / dt
            intErr += lastErr * dt
            lastErr = err
            x[2] = Kp * (err + Td * derr + intErr / Ti)
            # prevent control saturation (Exercise 7.2.1)
    #        x[2] = tanh(x[2]/2.0/umax)*umax*2.0;
    #        x[2] = max(min(x[2],umax),-umax);

            # output and propagate one time step
            file.write(f"{t}   {x[0]}   {x[2]}\n")
            x = rk4(x, t, t + dt, pendulumRHS)
            t += dt

        print(f"Final angle theta:\n{x}")
        if abs(x[0]) > 1.5:
            print(f"WHOOPSY: Fell over after {t} seconds.")


# Exercise 7.1.3: PID simulator (DA)
def ex7_1_3():

    Kp, Ti, Td = 8.0, 3.0, 0.3   # PID parameters
    setPt = 0.0                  # Set point
    dt = 0.05                    # Controller time step (50 ms)
    umax = 10.0  # noqa: F841    # maximum control (Exercise 7.2.1)

    with (thisfolder / "ex7_1_3.dat").open("w") as file:
        # PID controller variables
        intErr = DA(0.0)
        lastErr = DA(0.0)

        t = 0.0

        x = array([0.1 + 0.1 * DA(1), 0.0, 0.0])  # Initial condition

        # propagate the model state for 100 sec
        while t < 40.0 and abs(x[0].cons()) < 1.5:
            # compute the PID control
            err = setPt - x[0]
            derr = (err - lastErr) / dt
            intErr += lastErr * dt
            lastErr = err
            x[2] = Kp * (err + Td * derr + intErr / Ti)
            # prevent control saturation (Exercise 7.2.1)
    #        x[2] = tanh(x[2]/umax)*umax;

            # output and propagate one time step (Exercise 7.1.4)
            bx: Tuple[float, float] = x[0].bound()
            bu: Tuple[float, float] = x[2].bound()

            file.write(
                f"{t}"
                f"   {x[0].cons()}   {bx[1]}   {bx[0]}"
                f"   {x[0].evalScalar(-1.0)}   {x[0].evalScalar(1.0)}"
                f"   {x[2].cons()}   {bu[1]}   {bu[0]}"
                f"   {x[2].evalScalar(-1.0)}   {x[2].evalScalar(1.0)}\n"
            )
            x = rk4(x, t, t + dt, pendulumRHS)
            t += dt

        print(f"Final angle theta:{x}")
        if abs(x[0].cons()) > 1.5:
            print(f"WHOOPSY: Fell over after {t} seconds.")


# Exercise 7.1.5: Bounding
def ex7_1_5():

    x = DA(1)
    y = DA(2)
    func = sin(x / 2) / (2 + cos(y / 2 + x * x))

    a: List[float]  # these is a list because tuples are immutable
    b: Tuple[float, float]
    c: List[float]  # these is a list because tuples are immutable

    # bound by rasterizing
    arg: List[float] = [0.0, 0.0]
    a = [9999999.0, -9999999.0]
    c = [9999999.0, -9999999.0]
    for i in range(-10, 10):
        arg[0] = i / 10.0
        for j in range(-10, 10):
            arg[1] = j / 10.0
            # polynomial expansion
            r = func.eval(arg)
            a[0] = min(a[0], r)
            a[1] = max(a[1], r)
            # actual function
            r = sin(arg[0] / 2) / (2 + cos(arg[1] / 2 + arg[0] * arg[0]))
            c[0] = min(c[0], r)
            c[1] = max(c[1], r)

    # DA bounding
    b = func.bound()

    print(f"func:\n{func}")
    print("Bounds:")
    print(f"DA bound:       [{b[0]}, {b[1]}]")
    print(f"DA raster:      [{a[0]}, {a[1]}]")
    print(f"double raster:  [{c[0]}, {c[1]}]\n")


# Exercise 7.2.2: PID simulator with uncertain mass (DA)

# Model of controlled pendulum with uncertain mass
# x = ( theta, theta_dot, u )
@overload
def pendulumRHSmass(x: array, t: float) -> array: ...


@overload
def pendulumRHSmass(x: NDArray[np.double], t: float) -> NDArray[np.double]: ...


def pendulumRHSmass(x, t):
    # pendulum constants
    L = 0.61                      # m
    m = 0.21 * (1 + 0.1 * DA(1))  # kg
    M = 0.4926                    # kg
    g = 9.81                      # kg*m/s^2

    sint = sin(x[0])
    cost = cos(x[0])

    return np.array([
        x[1],
        x[2] + ((M+m)*g*sint-m*L*sqr(x[1])*sint*cost)/((M+m)*L+m*L*sqr(cost)),
        0.0,
    ]).view(type(x))


def ex7_2_2():

    Kp, Ti, Td = 8.0, 3.0, 0.3   # PID parameters
    setPt = 0.0                  # Set point
    dt = 0.05                    # Controller time step (50 ms)
    umax = 10.0  # noqa: F841    # maximum control (Exercise 7.2.1)

    with (thisfolder / "ex7_2_2.dat").open("w") as file:

        # PID controller variables
        intErr = DA(0.0)
        lastErr = DA(0.0)

        t = 0.0

        x = array([0.1 + 0.1 * DA(1), 0.0, 0.0])  # Initial condition

        # propagate the model state for 100 sec
        while t < 40.0 and abs(x[0].cons()) < 1.5:
            # compute the PID control
            err = setPt - x[0]
            derr = (err - lastErr) / dt
            intErr += lastErr * dt
            lastErr = err
            x[2] = Kp * (err + Td * derr + intErr / Ti)
            # prevent control saturation (Exercise 7.2.1)
    #        x[2] = tanh(x[2]/umax)*umax;

            # output and propagate one time step (Exercise 7.1.4)
            bx: Tuple[float, float] = x[0].bound()
            bu: Tuple[float, float] = x[2].bound()

            file.write(
                f"{t}"
                f"   {x[0].cons()}   {bx[1]}   {bx[0]}"
                f"   {x[0].evalScalar(-1.0)}   {x[0].evalScalar(1.0)}"
                f"   {x[2].cons()}   {bu[1]}   {bu[0]}"
                f"   {x[2].evalScalar(-1.0)}   {x[2].evalScalar(1.0)}\n"
            )
            x = rk4(x, t, t+dt, pendulumRHSmass)
            t += dt

        print(f"Final angle theta:{x}")
        if abs(x[0].cons()) > 1.5:
            print(f"WHOOPSY: Fell over after {t} seconds.")


def main():

    DA.init(10, 2)

    ex7_1_2()
    ex7_1_3()
    ex7_1_5()

    ex7_2_2()


if __name__ == "__main__":
    main()
