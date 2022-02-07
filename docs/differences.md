# Differences with respect to DACE C++

DACEyPy replicates most of the features and functions of DACE in C++,
yet there are some differences.

## Arrays

Instead of the AlgebraicVector and the AlgebraicMatrix classes,
DACEyPy has only one class, daceypy.array.

The daceypy.array class inherits from NumPy's ndarray,
therefore it can be used mostly in the same way and supports
vectors, matrices or arrays with higher numbers of dimensions.

## Operator functions

The main mathematical operators (e.g., `sin`, `sqr`, `sqrt`) are implemented
as functions in the daceypy.op submodule.

Most functions can be applied to DA objects, DACEyPy arrays,
numbers and NumPy arrays.

## New or modified operators

## Evaluation

In DACEyPy, DA objects and DACEyPy array support the call syntax:
`res = da_scalar(x)`
`res = da_array(x)`

The argument can be a scalar float or DA
(in that case, it is substituted as the first DA variable and the other are set to zero)
or a list / array of numbers or DA
(in that case, the values are substituted to the variables following their order).

### Power

In DACEyPy, DA objects can be used both as base and exponent:

```python
x = 1 + DA(1)  # DA variable

a = x ** 2  # DA to number
b = 2 ** x  # number to DA
c = x ** x  # DA to DA
```

## DA cache

In DACEyPy, it is possible to avoid that DACEDA structs are deallocated
each time a DA object is destroyed. When a new DA object is created,
the last cached DACEDA struct is recycled.

This can speed up the computations, since it avoids memory deallocation and
reallocation, that require time to be performed.

To enable and disable this feature, use the methods:
`daceypy.DA.cache_enable()` and `daceypy.DA.cache_disable()`.

As an alternative, you can use a context manager:
`with daceypy.DA.cache_manager(): ...`

## Output variable

For the same efficiency considerations on which the DA cache is based, it is
possible to directly assign the result of many operators to an existing DA object.

To do so, use the `out` keyword-only argument of the DA methods
(e.g. `x.sin(out=y)` to assign the sine of x to y)
or the in-place variant of the operators (e.g. `x += y`).

## From text

It is possible to directly parse text expression using DACEyPy.

To do so, use the `.fromText` classmethod in `daceypy.DA` or `daceypy.array`.

All math operators are supported, and it is possible to define a custom alphabet
using a sequence or a dictionary as second argument.
The default is `("x", "y", "z", "xx", "yy", "zz")`.
When a sequence is used, the names are used in order.
When a dictionary is used, the values are substituted when the keys are found.

IMPORTANT: do not use this function if you do not trust the input,
since it may allow to run arbitrary Python code.

```python
x = DA.fromText("sqrt(x + 100)")  # use default alphabet
arr = array.fromText("sin([x, y+1, z+2, xx+3, yy+4, zz+5])")
y = DA.fromText("a + b * c", ["a", "b", "c"])  # use sequence
z = DA.fromText("α / β", {"α": DA(1) + 5, "β": DA(2) + 1})  # use dict
```