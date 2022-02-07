import daceypy_import_helper  # noqa: F401

from daceypy import DA


def main():

    DA.init(20, 1)

    x = DA(1)

    # Compute [cos(x)-1]
    y = x.cos() - 1

    print(f"[cos(x)-1]\n{y}\n")

    # Compute [cos(x)-1]^11
    z = y ** 11

    print(f"[cos(x)-1]^11\n{z}\n")


if __name__ == "__main__":
    main()
