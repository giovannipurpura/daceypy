"""Value tests for the DA scalar type.

These check that the *numbers* the core computes are right, by comparing
Taylor expansions against closed-form results that are known independently of
DACEyPy (analytic derivatives, series identities, inverse-function round
trips).
"""

from __future__ import annotations

import math

import pytest

from daceypy import DA


def test_constant_part(da):
    assert (3.0 + DA(1)).cons() == pytest.approx(3.0)


def test_taylor_coefficients_of_exp_are_one_over_factorial(da):
    f = DA(1).exp()
    for n in range(DA.getMaxOrder() + 1):
        assert f.getCoefficient([n, 0]) == pytest.approx(1.0 / math.factorial(n))


def test_taylor_coefficients_of_sin(da):
    """sin(x) = x - x^3/3! + x^5/5! - ... ; even coefficients vanish."""
    f = DA(1).sin()
    for n in range(DA.getMaxOrder() + 1):
        expected = 0.0 if n % 2 == 0 else (-1) ** ((n - 1) // 2) / math.factorial(n)
        assert f.getCoefficient([n, 0]) == pytest.approx(expected, abs=1e-14)


def test_pythagorean_identity_is_exact_to_machine_precision(da):
    """sin^2 + cos^2 == 1 must hold coefficient by coefficient."""
    x = DA(1)
    residual = x.sin() ** 2 + x.cos() ** 2 - 1.0
    assert abs(residual.cons()) < 1e-14
    assert residual.norm() < 1e-13


def test_log_is_the_inverse_of_exp(da):
    x = DA(1)
    residual = (1.0 + x).exp().log() - (1.0 + x)
    assert residual.norm() < 1e-13


def test_derivative_of_a_monomial(da):
    """d/dx x^3 == 3 x^2, checked on the coefficients."""
    d = (DA(1) ** 3).deriv(1)
    assert d.getCoefficient([2, 0]) == pytest.approx(3.0)
    assert d.getCoefficient([3, 0]) == pytest.approx(0.0)


def test_deriv_and_integ_round_trip(da):
    """Integrating a derivative restores the function up to its constant.

    The polynomial is kept well below the computation order and depends on a
    single variable, so that neither truncation at the top order nor the
    vanishing of the other variable's terms interferes with the round trip.
    """
    f = 5.0 + 2.0 * DA(1) + DA(1) ** 3
    g = f.deriv(1).integ(1)
    residual = g - (f - f.cons())
    assert residual.norm() < 1e-13


def test_evaluation_matches_the_analytic_function_near_the_origin(da):
    """A truncated expansion must reproduce the true value near expansion point."""
    f = DA(1).exp()
    for point in (0.0, 0.01, 0.1):
        assert f.evalScalar(point) == pytest.approx(math.exp(point), rel=1e-9)


def test_multiplication_truncates_at_the_maximum_order(da):
    """Products beyond the computation order must be dropped, not kept."""
    order = DA.getMaxOrder()
    high = DA(1) ** order
    assert (high * DA(1)).norm() == pytest.approx(0.0, abs=1e-14)


def test_getMaxOrder_and_getMaxVariables_reflect_init(da_init):
    da_init(5, 3)
    assert DA.getMaxOrder() == 5
    assert DA.getMaxVariables() == 3
