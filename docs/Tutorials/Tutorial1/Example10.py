import daceypy_import_helper  # noqa: F401

from math import pi

from daceypy import DA, array


def main():

    DA.init(10, 3)

    # initialize cyl
    cyl = array([100.0 + DA(1), DA(2) * pi/180.0, DA(3)])

    # initialize cart and compute transformation
    cart = array([
        cyl[0] * cyl[1].cos(),
        cyl[0] * cyl[1].sin(),
        cyl[2],
    ])

    # subtract constant part to build DirMap
    DirMap: array = cart - cart.cons()  # type: ignore

    print(f"Direct map: from cylindric to cartesian (DirMap)\n{DirMap}\n\n")

    # Invert DirMap to obtain InvMap
    InvMap = DirMap.invert()

    print(f"Inverse map: from cartesian to cylindric (InvMap)\n{InvMap}\n\n")

    # Verification
    print(
        "Concatenate the direct and inverse map: "
        "(DirMap) o (InvMap) = DirMap(InvMap) = I\n\n")
    print(DirMap.eval(InvMap))


if __name__ == "__main__":
    main()
