import daceypy_import_helper  # noqa: F401

from daceypy import DA


def main():

    # Initialize DACE for 1st-order computations in 1 variable
    DA.init(1, 1)

    # Initialize x as DA around 3
    x = 3 + DA(1)

    print(f"x\n{x}\n")

    # Evaluate f(x) = 1/(x+1/x)
    f = 1 / (x + 1 / x)

    print(f"f(x) = 1/(x+1/x)\n{f}\n")


if __name__ == "__main__":
    main()
