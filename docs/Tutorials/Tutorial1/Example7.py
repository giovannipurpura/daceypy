import daceypy_import_helper  # noqa: F401

from daceypy import DA, array


def somb(x: array) -> DA:
    norm_x = x.vnorm()
    return norm_x.sin() / norm_x


def main():

    # initialize DACE for 10th-order computations in 2 variables
    DA.init(10, 2)

    print("Initialize x as two-dim DA vector around (2,3)\n\n")

    x = array([2 + DA(1), 3 + DA(2)])

    print(f"x\n{x}\n")

    print("Evaluate sombrero function\n")

    # Evaluate sombrero function
    z = somb(x)

    print(f"z\n{z}\n")


if __name__ == "__main__":
    main()
