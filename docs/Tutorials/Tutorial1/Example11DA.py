import daceypy_import_helper  # noqa: F401

from math import ceil
from time import perf_counter

import numpy as np
from daceypy import DA, array

mu = 398600.0  # km^3/s^2


def TBP(x: array, t: float) -> array:
    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc = -mu * pos / (r ** 3)
    dx = vel.concat(acc)
    return dx


def euler(x: array, t0: float, t1: float) -> array:
    hmax = 0.1
    steps = ceil((t1 - t0) / hmax)
    h = (t1 - t0) / steps
    t = t0

    x = x.copy()
    for _ in range(steps):
        x += h * TBP(x, t)
        t += h

    return x


def main():

    DA.init(3, 6)

    # Set initial conditions
    ecc = 0.5

    x0 = array.identity(6)
    x0[0] += 6678.0  # 300 km altitude
    x0[4] += np.sqrt(mu / 6678.0 * (1 + ecc))

    # integrate for half the orbital period
    a = 6678.0 / (1 - ecc)

    T0 = perf_counter()
    with DA.cache_manager():  # optional, for efficiency
        xf = euler(x0, 0.0, np.pi * np.sqrt(a**3 / mu))
    T1 = perf_counter()

    print(f"Initial conditions:\n{x0}\n")
    print(f"Final conditions:\n{xf}\n")
    print(f"Initial conditions (cons. part):\n{x0.cons()}\n")
    print(f"Final conditions: (cons. part)\n{xf.cons()}\n")

    # Evaluate for a displaced initial condition
    Deltax0 = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])  # km

    print(f"Displaced Initial condition:\n{x0.cons() + Deltax0}\n")
    print(f"Displaced Final condition:\n{xf.eval(Deltax0)}\n")

    print(f"Info: time required for integration = {T1 - T0} s")


if __name__ == "__main__":
    main()
