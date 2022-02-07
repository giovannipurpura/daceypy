import daceypy_import_helper  # noqa: F401

from daceypy import DA


def main():

    DA.init(20, 1)

    x = DA(1)

    # compute and print sin(x)^2
    y1 = x.sin().sqr()
    print("sin(x)^2\n" + str(y1) + "\n")

    # compute and print cos(x)^2
    y2 = x.cos().sqr()
    print("cos(x)^2\n" + str(y2) + "\n")

    # compute and print sin(x)^2+cos(x)^2
    s = y1 + y2
    print("sin(x)^2+cos(x)^2\n" + str(s) + "\n")


if __name__ == "__main__":
    main()
