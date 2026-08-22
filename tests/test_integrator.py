"""Tests for the Runge-Kutta integrators.

Value tests compare the propagated state against a closed-form solution of the
harmonic oscillator; contract tests pin down when the integrator decides it has
reached the final time, which is where a purely relative criterion used to
divide by zero.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from daceypy import array, integrator


def _harmonic(x, t):
    """x'' = -x, written as a first-order system. Solution: (cos t, -sin t)."""
    return np.array([x[1], -x[0]]).view(type(x))


def _make(t1: float, t2: float, stateType=np.ndarray) -> integrator:
    it = integrator(stateType=stateType)
    it.f = _harmonic
    it.loadTime(t1, t2)
    it.loadTol(1e-12, 1e-12)
    it.loadStepSize(0.01, 1.0, 1e-9)
    return it


def _exact(dt: float) -> np.ndarray:
    return np.array([math.cos(dt), -math.sin(dt)])


@pytest.mark.parametrize("t1, t2", [(0.0, 1.0), (0.0, 6.0), (1.0, 4.0)])
def test_propagation_matches_the_analytic_solution(da, t1, t2):
    out = _make(t1, t2).propagate(np.array([1.0, 0.0]), t1, t2)
    assert out == pytest.approx(_exact(t2 - t1), abs=1e-9)


def test_backward_propagation_matches_the_analytic_solution(da):
    out = _make(2.0, -3.0).propagate(np.array([1.0, 0.0]), 2.0, -3.0)
    assert out == pytest.approx(_exact(-5.0), abs=1e-9)


def test_forward_then_backward_returns_to_the_initial_state(da):
    x0 = np.array([1.0, 0.0])
    fwd = _make(0.0, 3.0).propagate(x0, 0.0, 3.0)
    back = _make(3.0, 0.0).propagate(fwd, 3.0, 0.0)
    assert back == pytest.approx(x0, abs=1e-8)


def test_propagation_to_time_zero_terminates(da):
    """A final time of exactly zero must be a legal target.

    The termination test used to be the purely relative
    ``abs(1 - t / tf) <= eps``, which raises ZeroDivisionError for ``tf == 0``.
    """
    out = _make(-1.0, 0.0).propagate(np.array([1.0, 0.0]), -1.0, 0.0)
    assert out == pytest.approx(_exact(1.0), abs=1e-9)


def test_propagation_over_a_zero_length_interval_is_a_no_op(da):
    x0 = np.array([1.0, 0.0])
    for t in (0.0, 5.0):
        out = _make(t, t).propagate(x0.copy(), t, t)
        assert out == pytest.approx(x0)


def test_propagation_far_from_the_origin_terminates(da):
    """``t0`` much larger than the span must not make termination unreachable.

    The tolerance has to stay scaled to at least ``abs(tf)``, since the final
    step only lands within one ulp of ``tf``.
    """
    out = _make(100.0, 101.0).propagate(np.array([1.0, 0.0]), 100.0, 101.0)
    assert out == pytest.approx(_exact(1.0), abs=1e-9)


def test_final_time_is_reached_exactly(da):
    it = _make(-2.0, 0.0)
    it.propagate(np.array([1.0, 0.0]), -2.0, 0.0)
    assert it._tout == pytest.approx(0.0, abs=1e-15)


def test_da_propagation_constant_part_matches_the_pointwise_result(da):
    """Propagating a DA state must reproduce the pointwise result at its centre."""
    x0 = np.array([1.0, 0.0])
    out_da = _make(0.0, 2.0, stateType=array).propagate(
        array(x0) + 0.01 * array.identity(), 0.0, 2.0)
    assert np.asarray(out_da.cons(), dtype=float) == pytest.approx(
        _exact(2.0), abs=1e-7)
