import daceypy_import_helper  # noqa: F401

import numpy as np
from daceypy import DA


def ErrFunc(x: DA) -> DA:
    # Note: this is not the classical erf
    return 1.0 / np.sqrt(2.0 * np.pi) * (-x.sqr() / 2).exp()


def main():

    # initialize DACE for 20th-order computations in 1 variable
    DA.init(24, 1)

    x = DA(1)

    # compute Taylor expansion of 1/sqrt(2 * pi) * exp(-x^2/2)
    y = ErrFunc(x)

    # compute the Taylor expansion of the indefinite integral of
    # 1/sqrt(2 * pi) * exp(-x^2/2)
    Inty = y.integ(1)

    # compute int_{-1}^{+1} 1/sqrt(2 * pi) * exp(-x^2/2) dx
    value = Inty.evalScalar(1.0) - Inty.evalScalar(-1.0)

    print("int_{-1}^{+1} 1/sqrt(2 * pi) * exp(-x^2/2) dx")
    print("Exact result: 0.682689492137")
    print(f"Approx. result: {value}")
    print(f"Equivalent using DACE: {DA(0.5).sqrt().erf().cons()}")


if __name__ == "__main__":
    main()
