"""Tests for the ADS (Adaptive Domain Splitting) primitives.

The split-budget tests are *contract* tests: they check that the number of
splits a domain is allowed to undergo matches the number the API documents,
independently of whether the resulting Taylor maps are numerically good.

Note on initialisation: ``array.getTruncationErrors`` estimates the norm of
the terms at order ``getTO() + 1``, so the ADS machinery requires the current
truncation order to be strictly below the maximum order the core was
initialised with. The fixtures below therefore init at ``order + 1`` and push
the truncation order down to ``order``.
"""

from __future__ import annotations

import numpy as np
import pytest

from daceypy import ADS, DA, DACEException, array


@pytest.fixture
def ads_core():
    """Init the core so that ADS truncation-error estimation is possible."""
    DA.init(7, 2)
    DA.pushTO(6)
    yield DA
    DA.popTO()


def _nonlinear_map(d: ADS) -> ADS:
    """A transformation with genuine high-order content, so that splits occur.

    Both components must expand into *many* monomials: the bundled DACE core
    cannot estimate the norm of a polynomial with only a handful of terms (see
    ``test_estimNorm_fails_on_polynomials_with_very_few_monomials``), which the
    ADS split criterion relies on.
    """
    b = d.box
    return ADS(
        array([(b[0] + b[1]).sin() * b[1].exp(), (b[0] * 0.5 + b[1]).exp()]),
        d.nsplit,
    )


def test_a_fresh_domain_can_split(ads_core):
    d = ADS(array.identity())
    assert d.canSplit(1) is True


def test_canSplit_still_allows_a_split_at_exactly_the_budget(ads_core):
    """A domain whose split count equals N_max is still allowed to split once.

    This pins down deliberate behaviour, not an oversight. ``canSplit`` is
    queried *before* the split is performed and uses a non-strict comparison,
    so a domain can reach ``N_max + 1`` splits. The strict comparison used
    until v1.1.0 made DACEyPy disagree with the legacy C++ ADS implementation
    (issue #6) and was changed on purpose in v1.2.0.
    """
    n_max = 3
    d = ADS(array.identity(), nsplit=[1, 1, 2])
    assert int(np.sum(d.countSplits())) == n_max
    assert d.canSplit(n_max) is True


def test_canSplit_is_false_past_the_budget(ads_core):
    d = ADS(array.identity(), nsplit=[1, 1, 2, 2])
    assert d.canSplit(3) is False


def test_canSplit_with_a_zero_budget_allows_the_first_split(ads_core):
    """Consequence of the same C++-parity convention, spelled out."""
    assert ADS(array.identity()).canSplit(0) is True


def test_countSplits_counts_per_direction(ads_core):
    d = ADS(array.identity(), nsplit=[1, -1, 2])
    assert list(d.countSplits()) == [2, 1]


def test_eval_stops_one_split_past_the_budget(ads_core):
    """End-to-end guard on the C++-parity convention documented above.

    ``N_max + 1`` is the real ceiling; anything beyond it is a regression.
    """
    n_max = 3
    domains = ADS.eval(
        [ADS(array.identity())], 1e-8, n_max, _nonlinear_map,
        log_fun=lambda *args: None,
    )
    assert domains
    assert max(int(np.sum(d.countSplits())) for d in domains) <= n_max + 1


def test_split_returns_two_domains_recording_opposite_halves(ads_core):
    left, right = ADS(array.identity()).split(0)
    assert left.nsplit == [-1]
    assert right.nsplit == [+1]


def test_split_halves_cover_the_original_domain(ads_core):
    """Evaluating the halves at their own centres must give -0.5 and +0.5."""
    left, right = ADS(array.identity()).split(0)
    assert left.box[0].evalScalar(0.0) == pytest.approx(-0.5)
    assert right.box[0].evalScalar(0.0) == pytest.approx(+0.5)


def test_split_leaves_the_other_direction_untouched(ads_core):
    left, _ = ADS(array.identity()).split(0)
    assert left.box[1].getCoefficient([0, 1]) == pytest.approx(1.0)


def test_log_fun_is_called_with_several_arguments(ads_core):
    """``log_fun`` follows the ``print`` signature, not a single-string one."""
    calls = []

    ADS.eval(
        [ADS(array.identity())], 1e-2, 1, _nonlinear_map,
        log_fun=lambda *args: calls.append(args),
    )
    assert calls
    assert any(len(c) > 1 for c in calls), \
        "log_fun must be documented and annotated as variadic"


@pytest.mark.xfail(
    raises=DACEException,
    reason=(
        "Bundled DACE core (commit 2bda904, 2022-10-09) cannot estimate the "
        "norm of a polynomial with very few monomials. Fixed upstream by "
        "'Fix norm estimation to not return nan for cases of 0 or 1 "
        "monomials' (dacelib/dace#46), not yet part of the bundled build. "
        "Remove this xfail once the libraries are rebuilt."
    ),
    strict=True,
)
def test_estimNorm_fails_on_polynomials_with_very_few_monomials(ads_core):
    DA(1).estimNorm(0, 0, DA.getTO() + 1)
