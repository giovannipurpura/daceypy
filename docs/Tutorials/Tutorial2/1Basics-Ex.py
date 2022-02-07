import daceypy_import_helper  # noqa: F401

from daceypy import DA


# Exercise 1.1.2: first steps
def ex1_1_2():
    x = DA(1)
    func = 3 * (x + 3) - x + 1 - (x + 8)
    print(f"Exercise 1.1.2: First steps\n{func}\n")


# Exercise 1.1.3: different expansion point
def ex1_1_3():
    x = DA(1)
    func = (1.0 + x).sin()
    print(f"Exercise 1.1.3: Different expansion point\n{func}\n")


# Exercise 1.1.4: a higher power
def ex1_1_4():
    x = DA(1)
    func = x.sin()
    res = DA(1.0)  # this makes res a constant function P(x) = 1.0
    for i in range(11):
        res *= func
    print(f"Exercise 1.1.4: A higher power\n{res}\n")


# Exercise 1.1.5: two arguments
def ex1_1_5(x: DA, y: DA) -> DA:
    return (1.0 + x*x + y*y).sqrt()


# Exercise 1.2.1: identity crisis
def ex1_2_1():
    x = DA(1)
    s2 = x.sin() * x.sin()
    c2 = x.cos() * x.cos()
    print(f"Exercise 1.2.1: Identity crisis\n{s2}\n{c2}\n{s2+c2}\n")


# Exercise 1.2.2: Breaking bad
def ex1_2_2(x: DA, y: DA):
    r = (x*x + y*y).sqrt()
    return r.sin() / r


def main():

    DA.init(10, 2)

    x = DA(1)
    y = DA(2)

    ex1_1_2()
    ex1_1_3()
    ex1_1_4()
    print(f"Exercise 1.1.5: Two arguments\n{ex1_1_5(x, y)}\n")

    ex1_2_1()
    try:
        print("Exercise 1.2.2: Breaking bad\n")
        print(ex1_2_2(x, y))
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
