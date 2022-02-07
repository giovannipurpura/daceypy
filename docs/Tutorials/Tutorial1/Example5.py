import daceypy_import_helper  # noqa: F401

from daceypy import DA


def main():

    DA.init(20, 1)

    x = DA(1)

    y = x.sin()

    # compute Taylor expansion of d[sin(x)]/dx
    dy = y.deriv(1)

    # print d[sin(x)]/dx and cos(x) to compare
    print(f"d[sin(x)]/dx\n{dy}\n")
    print(f"cos(x)\n{x.cos()}\n")

    # compute Taylor expansion of int[sin(x)dx]
    int_y = y.integ(1)

    # print int[sin(x)dx] and -cos(x) to compare
    print(f"int[sin(x)dx]\n{int_y}\n")
    print(f"-cos(x)\n{-x.cos()}\n")


if __name__ == "__main__":
    main()
