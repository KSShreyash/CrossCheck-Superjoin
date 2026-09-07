from factlayer.models import Fact
from factlayer.reconcile.verify import verify

FY24 = ("2023-04-01", "2024-03-31")


def _f(value, quals=None, unit="INR", period=FY24):
    f = Fact(1, "Delhivery", "revenue", None, None, None, "FY24",
             qualifiers=quals or {})
    f.canon_value, f.canon_unit = value, unit
    f.period_start, f.period_end = period
    return f


def test_true_scale_claim_is_confirmed():
    # The realistic case: the text omitted the scale word, so normalisation left
    # the pair a hundred-fold apart and the model proposes the missing factor.
    # (A pair that already agrees never reaches the model - the rules settle it.)
    ok, note = verify(_f(8.142e10), _f(8.141538e8),
                      {"kind": "scale", "factor": 100}, tol=1e-3)
    assert ok and "factor of 100" in note


def test_false_scale_claim_is_rejected():
    # standalone vs consolidated revenue: no scale factor reconciles these
    ok, note = verify(_f(7.454082e10), _f(8.141538e10),
                      {"kind": "scale", "factor": 100}, tol=1e-3)
    assert not ok and "do not reconcile" in note


def test_basis_claim_requires_basis_to_actually_differ():
    ok, _ = verify(_f(7.4e10, {"basis": "standalone"}),
                   _f(8.1e10, {"basis": "consolidated"}), {"kind": "basis"}, tol=1e-3)
    assert ok
    ok2, note = verify(_f(7.4e10, {"basis": "consolidated"}),
                       _f(8.1e10, {"basis": "consolidated"}),
                       {"kind": "basis"}, tol=1e-3)
    assert not ok2 and "basis" in note


def test_period_claim_checks_the_normalised_dates():
    ok, _ = verify(_f(8.1e10), _f(7.2e10, period=("2022-04-01", "2023-03-31")),
                   {"kind": "period"}, tol=1e-3)
    assert ok
    ok2, note = verify(_f(8.1e10), _f(7.2e10), {"kind": "period"}, tol=1e-3)
    assert not ok2 and "period" in note


def test_unknown_transform_is_not_verifiable():
    ok, note = verify(_f(1.0), _f(2.0), {"kind": "vibes"}, tol=1e-3)
    assert not ok and "unrecognised" in note


def test_missing_transform_is_not_verifiable():
    ok, note = verify(_f(1.0), _f(2.0), None, tol=1e-3)
    assert not ok and "no transform" in note
