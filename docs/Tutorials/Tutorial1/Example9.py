import daceypy_import_helper  # noqa: F401

from daceypy import DA, array


def main():

    DA.init(10, 1)

    x = DA(1)

    # Compute Taylor expansion of sin(x)
    y = x.sin()

    # Invert Taylor polynomial
    inv_y: DA = array([y]).invert()[0]

    # Compare with asin(x)
    print(f"Polynomial inversion of sin(x)\n{inv_y}\n")

    print(f"asin(x)\n{x.asin()}")


if __name__ == "__main__":
    main()
