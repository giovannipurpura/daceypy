"""
In the following example, some DA scalars and arrays are created
in different ways and compared.
"""

from daceypy import DA, array
from daceypy.op import sin

# --- Initialize DACE ---

DA.init(3, 6)  # order = 3, num. of vars. = 6

# --- Scalar example ---

# Create a simple DA variable
# (when the arg. is int, it is interpreted as the n-th DA variable:
# to have a constant DA object of value one, use DA(1.0))
x = DA(1)

# Compute and print sin(x)
sin_x = sin(x)  # alternative: x.sin()
print(sin_x)

# --- Array example ---

# Create a vector with DA variables from 1 to 6
arr = array.identity(6)

# Add constant parts
arr += [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# Compute and print sin(arr)
sin_arr = sin(arr)  # alternative: arr.sin()
print(sin_arr)

# --- Text parser example ---

# Parse text (by default, the vars are called x, y, z, xx, yy, zz)
# but a different alphabet can be passed as second argument,
# either as a list of strings (converted to DA vars following their order)
# or as a dict str->DAobj (every str will be converted to the obj.)
sin_x_2 = DA.fromText("sin(x)")

# Compute the difference with respect to the variable created before
# (should be zero, i.e. `ALL COEFFICIENTS ZERO`)
print(sin_x_2 - sin_x)

# Parse text (by default, the vars are called x, y, z, xx, yy, zz)
# but a different alphabet can be passed as second argument
sin_arr_2 = array.fromText("sin([x, y+1, z+2, xx+3, yy+4, zz+5])")

# Compute the difference with respect to the variable created before
# (should be all zeros, i.e. `ALL COEFFICIENTS ZERO`)
print(sin_arr_2 - sin_arr)
