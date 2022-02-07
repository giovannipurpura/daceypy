import daceypy_import_helper  # noqa: F401

from daceypy import DA


def main():

    # initialize DACE for 20th-order computations in 1 variable
    DA.init(20, 1)

    # initialize x as DA
    x = DA(1)

    # compute y = sin(x)
    y = x.sin()

    # print x and y to screen
    print("x\n" + str(x) + "\n")
    print("y = sin(x)\n" + str(y) + "\n")


if __name__ == "__main__":
    main()
