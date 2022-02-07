import daceypy_import_helper  # noqa: F401

import numpy as np
from daceypy import DA


def main():

    DA.init(10, 2)

    tol = 1.0e-12

    mu = 1.0

    a = DA(1.0)
    e = DA(0.5)
    t = np.pi/2.0

    M = ((mu / (a ** 3))).sqrt() * t  # real at this stage (i.e. constant DA)

    EccAn = M.copy()  # first guess

    err = abs(EccAn - e * EccAn.sin() - M)

    # Newton's method for the reference solution
    while err > tol:
        EccAn -= (EccAn - e * EccAn.sin() - M) / (1 - e * EccAn.cos())
        err = abs(EccAn - e * EccAn.sin() - M)

    print(f"Reference solution: E = {EccAn.cons()}\n")

    a += DA(1)
    e += DA(2)

    M = ((mu / (a ** 3))).sqrt() * t  # now M is a DA (with a non const part)

    # Newton's method for the Taylor expansion of the solution
    i = 1
    while i <= 10:
        EccAn -= (EccAn - e * EccAn.sin() - M) / (1 - e * EccAn.cos())
        i *= 2

    print(f"Taylor expansion of E\n{EccAn}\n")

    print("Let's verify it is the Taylor expansion of the solution:")
    print("Evaluate (E - e*sin(E) - M) in DA")

    print(EccAn - e*EccAn.sin() - M)


if __name__ == "__main__":
    main()
