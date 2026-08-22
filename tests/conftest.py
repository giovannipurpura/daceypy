"""Shared fixtures for the DACEyPy test suite.

The DACE core keeps the polynomial order and the number of variables in
*global* state, set once by :meth:`daceypy.DA.init`.  Re-initialising it
invalidates every DA object created before the call, so tests must never share
DA objects across an ``init``.  The fixtures below make the initialisation
explicit and per-test, which keeps the tests independent of execution order.
"""

from __future__ import annotations

import pytest

from daceypy import DA


@pytest.fixture
def da():
    """Initialise the DACE core with a small, general-purpose setting.

    Order 10 with 2 variables is enough for the analytic identities checked in
    the value tests while staying fast.
    """
    DA.init(10, 2)
    return DA


@pytest.fixture
def da_init():
    """Return a callable initialising the core with an explicit order/nvar.

    Use this in tests that need a specific number of variables, e.g. the ADS
    tests, which depend on ``DA.getMaxVariables()``.
    """

    def _init(order: int, nvar: int):
        DA.init(order, nvar)
        return DA

    return _init
