from factlayer.models import Fact
from factlayer.reconcile.rules import qualifier_diff, rule_verdict

FY24 = ("2023-04-01", "2024-03-31")
FY26 = ("2025-04-01", "2026-03-31")


def _f(value, unit="INR", period=FY24, quals=None, claim="measurement"):
    f = Fact(1, "Delhivery", "revenue", None, None, None, "FY24",
             qualifiers=quals or {}, claim_type=claim)
    f.canon_value, f.canon_unit = value, unit
    f.period_start, f.period_end = period
    return f


def test_same_value_same_qualifiers_corroborates():
    v, _ = rule_verdict(_f(8.142e10), _f(8.1415e10), tol=1e-3)
    assert v == "corroborates"


def test_same_value_different_basis_is_caveated():
    v, diff = rule_verdict(_f(8.142e10, quals={"basis": "consolidated"}),
                           _f(8.1415e10, quals={"basis": "standalone"}), tol=1e-3)
    assert v == "corroborates_with_caveat"
    assert "basis" in diff


def test_different_value_one_qualifier_differs_is_reconcilable():
    v, diff = rule_verdict(_f(7.454082e10, quals={"basis": "standalone"}),
                           _f(8.141538e10, quals={"basis": "consolidated"}), tol=1e-3)
    assert v == "reconcilable"
    assert list(diff) == ["basis"]


def test_different_value_same_qualifiers_is_contradiction_candidate():
    v, _ = rule_verdict(_f(6.5, "PERCENT"), _f(6.6, "PERCENT"), tol=1e-3)
    assert v == "contradiction_candidate"


def test_different_period_is_a_difference():
    _, diff = rule_verdict(_f(8.1e10), _f(7.2e10, period=("2022-04-01", "2023-03-31")),
                           tol=1e-3)
    assert "period" in diff


def test_attribute_facts_always_go_to_the_model():
    v, _ = rule_verdict(_f(None, None, claim="attribute"),
                        _f(None, None, claim="attribute"), tol=1e-3)
    assert v == "needs_model"


def test_unknown_period_is_not_a_contradiction():
    # An absent period means unknown, not "same period as the other fact".
    # Treating None == None as a match manufactured 1,480 false contradictions
    # across two starter documents, 45% of all pairs.
    a = _f(8.1e10, period=(None, None))
    b = _f(7.2e10, period=(None, None))
    assert rule_verdict(a, b, tol=1e-3)[0] == "insufficient_context"


def test_one_known_period_is_still_insufficient():
    a = _f(8.1e10)
    b = _f(7.2e10, period=(None, None))
    assert rule_verdict(a, b, tol=1e-3)[0] == "insufficient_context"


def test_agreeing_values_corroborate_even_without_periods():
    # the period guard only blocks claiming a contradiction, not agreement
    a = _f(8.142e10, period=(None, None))
    b = _f(8.1415e10, period=(None, None))
    assert rule_verdict(a, b, tol=1e-3)[0] == "corroborates"


def test_incomparable_units_are_unrelated_not_disputed():
    a, b = _f(8.1e10, "INR"), _f(6.5, "PERCENT")
    assert rule_verdict(a, b, tol=1e-3)[0] == "unrelated"


def test_known_periods_still_reach_a_contradiction():
    # the RBI vs IMF growth case must survive the guard above
    a = _f(6.5, "PERCENT", period=FY26)
    b = _f(6.6, "PERCENT", period=FY26)
    assert rule_verdict(a, b, tol=1e-3)[0] == "contradiction_candidate"


def test_qualifier_diff_ignores_keys_absent_from_both():
    a = _f(8.1e10, quals={"basis": "consolidated"})
    b = _f(8.1e10, quals={"basis": "consolidated"})
    assert qualifier_diff(a, b) == {}
