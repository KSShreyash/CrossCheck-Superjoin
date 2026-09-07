import pytest
from factlayer.normalize.units import normalize_value, values_agree

@pytest.mark.parametrize("raw,unit,expected,cu", [
    ("81,415.38", "₹ in Million", 8.141538e10, "INR"),
    ("8,142", "₹ Cr", 8.142e10, "INR"),
    ("74,540.82", "Rs. million", 7.454082e10, "INR"),
    ("6.5", "per cent", 6.5, "PERCENT"),
    ("1.4", "Mn Tons", 1.4e6, "TONNE"),
    ("740", "Mn", 7.4e8, "COUNT"),
    ("31", "days", 31.0, "DAYS"),
])
def test_normalises_scale_and_unit(raw, unit, expected, cu):
    value, canon_unit = normalize_value(raw, unit)
    assert canon_unit == cu
    assert value == pytest.approx(expected, rel=1e-9)

def test_parenthesised_value_is_negative():
    value, _ = normalize_value("(452)", "₹ Cr")
    assert value == pytest.approx(-4.52e9)

def test_leading_minus_is_negative():
    value, _ = normalize_value("-1,008", "Cr")
    assert value == pytest.approx(-1.008e10)

def test_crore_and_million_reconcile_within_tolerance():
    a, _ = normalize_value("8,142", "₹ Cr")
    b, _ = normalize_value("81,415.38", "₹ million")
    assert values_agree(a, b, tol=1e-3)

def test_clearly_different_values_do_not_agree():
    a, _ = normalize_value("74,540.82", "₹ million")
    b, _ = normalize_value("81,415.38", "₹ million")
    assert not values_agree(a, b, tol=1e-3)

def test_unparseable_returns_none():
    assert normalize_value("substantially higher", None) is None
