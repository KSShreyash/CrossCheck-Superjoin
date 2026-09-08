import pytest
from factlayer.normalize.units import normalize_value, values_agree

@pytest.mark.parametrize("raw,unit,expected,cu", [
    ("81,415.38", "₹ in Million", 8.141538e10, "INR"),
    ("8,142", "₹ Cr", 8.142e10, "INR"),
    ("74,540.82", "Rs. million", 7.454082e10, "INR"),
    ("6.5", "per cent", 6.5, "PERCENT"),
    ("1.4", "Mn Tons", 1.4e6, "TONNE"),
    ("740", "Mn", 7.4e8, "UNKNOWN"),
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


def test_mangled_rupee_glyph_in_the_unit_is_still_rupees():
    # models transcribe the rupee sign inconsistently; "(Rs in Million)" in the
    # source arrives as "I in Million". Left as COUNT it would never match the
    # same figure printed in crore.
    for unit in ["I in Million", "I million", "` in Crore", "i in million"]:
        out = normalize_value("81,415.38", unit)
        assert out is not None and out[1] == "INR", unit


def test_a_lone_letter_without_a_scale_word_is_not_a_currency():
    value, unit = normalize_value("42", "I")
    assert unit == "COUNT"


def test_currency_falls_back_to_the_source_quote():
    # a table cell prints a bare number; the quote it came from carries the sign
    out = normalize_value("81,415.38", None,
                          context="stood at \u20b9 81,415.38 million as against")
    assert out == pytest.approx((8.141538e10, "INR"), rel=1e-9) or out[1] == "INR"


def test_context_never_overrides_an_explicit_unit():
    # percentages must not become rupees just because the sentence mentions money
    out = normalize_value("6.5", "per cent",
                          context="revenue of \u20b9 81,415.38 million grew 6.5 per cent")
    assert out == (6.5, "PERCENT")


def test_a_bare_scale_word_is_an_unknown_dimension_not_a_count():
    # "(Rs in Million)" declared in a table header and dropped by the model
    # leaves "million": a magnitude with no dimension.
    assert normalize_value("81,415.38", "million")[1] == "UNKNOWN"
    assert normalize_value("8,142", "Cr")[1] == "UNKNOWN"
    # a real dimension word still reads as a count
    assert normalize_value("740", "Mn shipments")[1] == "COUNT"


def test_unknown_units_stay_comparable():
    from factlayer.normalize.units import units_compatible
    assert units_compatible("INR", "UNKNOWN")
    assert units_compatible("UNKNOWN", "PERCENT")
    assert units_compatible("INR", "INR")
    assert not units_compatible("INR", "PERCENT")
    assert not units_compatible("TONNE", "INR")


def test_case_one_survives_inconsistent_unit_reporting():
    # the real failure: the deck reports "Rs. Cr" and the annual report reports
    # a bare "million", so a strict unit gate would refuse to compare them
    deck = normalize_value("8,142", "Rs. Cr")
    report = normalize_value("81,415.38", "million")
    from factlayer.normalize.units import units_compatible
    assert units_compatible(deck[1], report[1])
    assert values_agree(deck[0], report[0], tol=1e-3)


def test_significant_figures_reads_printed_precision():
    from factlayer.normalize.units import significant_figures
    assert significant_figures("1.4") == 2
    assert significant_figures("1,429") == 4
    assert significant_figures("76") == 2
    assert significant_figures("758") == 3
    assert significant_figures("0.9") == 1
    assert significant_figures(None) is None
    assert significant_figures("n/a") is None


def test_the_same_figure_printed_at_different_precision_agrees():
    # the earnings deck rounds to two figures where the annual report gives four
    deck, _ = normalize_value("1.4", "Mn Tons")
    report, _ = normalize_value("1,429", "K tonnes")
    assert not values_agree(deck, report, tol=1e-3), "a flat tolerance cannot"
    assert values_agree(deck, report, tol=1e-3, a_raw="1.4", b_raw="1,429")

    a, _ = normalize_value("76", "Rs Cr")
    b, _ = normalize_value("758", "Mn")
    assert values_agree(a, b, tol=1e-3, a_raw="76", b_raw="758")


def test_a_real_disagreement_survives_the_precision_check():
    # 0.9% against 1.6% is not a rounding of one another at any precision
    assert not values_agree(0.9, 1.6, tol=1e-3, a_raw="0.9", b_raw="1.6")
    assert not values_agree(6.5, 6.6, tol=1e-3, a_raw="6.5", b_raw="6.6")
    assert not values_agree(7.454082e10, 8.141538e10, tol=1e-3,
                            a_raw="74,540.82", b_raw="81,415.38")
