"""Contract tests for the interaction between ``daceypy.array`` and numpy.

These are *contract* tests in the sense of the review methodology: they do not
check that a computed number is right, but that the type ``daceypy.array``
presents to the outside world -- its MRO, its metaclass-provided repr, its
behaviour as an ``ndarray`` subclass -- is the one the library promises.

Regression guard for the ``TypeError: metaclass conflict`` raised on import
with numpy >= 2.5, where ``numpy.typing.NDArray`` became a PEP 695
``typing.TypeAliasType`` and could no longer be used as a base class.
"""

from __future__ import annotations

import numpy as np

from daceypy import DA, array


def test_array_is_an_ndarray_subclass():
    assert issubclass(array, np.ndarray)


def test_array_mro_is_exactly_array_ndarray_object():
    """The base list must resolve to plain ``np.ndarray`` and nothing else.

    Spelling the base as ``NDArray[np.object_]`` used to resolve to exactly
    this MRO; any future change to the base must preserve it, since the whole
    class relies on ``ndarray`` semantics (``view``, ``__array_finalize__``).
    """
    assert array.__mro__ == (array, np.ndarray, object)


def test_prettytype_metaclass_is_applied():
    """``PrettyType`` must still take effect on the class object."""
    assert repr(array) == "daceypy.array"
    assert array.__module__ == "daceypy"


def test_array_holds_object_dtype(da):
    a = array([DA(1), DA(2)])
    assert a.dtype == np.object_


def test_array_construction_promotes_numbers_to_da(da):
    """Plain numbers must come out as DA objects, not as floats."""
    a = array([1.0, DA(1)])
    assert all(isinstance(el, DA) for el in a)


def test_slicing_preserves_the_subclass(da):
    a = array.identity()
    assert type(a[:1]) is array


def test_identity_has_one_entry_per_variable(da_init):
    da_init(4, 3)
    assert array.identity().shape == (3,)
