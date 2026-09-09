from ..models import Fact
from ..normalize.units import units_compatible, values_agree


# Qualifiers describe the measurement; provenance describes who reported it.
PROVENANCE_KEYS = {"source", "publisher", "attribution", "reported_by",
                   "reporter", "author", "institution"}


def measurement_diff(diff: dict[str, tuple]) -> dict[str, tuple]:
    """The part of a qualifier difference that bears on comparability."""
    return {k: v for k, v in diff.items() if k not in PROVENANCE_KEYS}


def qualifier_diff(a: Fact, b: Fact) -> dict[str, tuple]:
    """Which recorded qualifiers differ between two facts."""
    diff: dict[str, tuple] = {}
    for key in set(a.qualifiers) | set(b.qualifiers):
        left, right = a.qualifiers.get(key), b.qualifiers.get(key)
        if left == right:
            continue
        # a qualifier present on one side only is unknown, not contradicted
        if left is None or right is None:
            continue
        diff[key] = (left, right)
    if a.entity_id and b.entity_id and a.entity_id != b.entity_id:
        # a real difference in what is measured, so record it rather than drop it
        diff["subject"] = (a.entity_id, b.entity_id)
    if (a.period_start, a.period_end) != (b.period_start, b.period_end):
        diff["period"] = ((a.period_start, a.period_end),
                          (b.period_start, b.period_end))
    if not units_compatible(a.canon_unit, b.canon_unit):
        diff["unit"] = (a.canon_unit, b.canon_unit)
    return diff


def rule_verdict(a: Fact, b: Fact, tol: float) -> tuple[str, dict]:
    """Decide what can be decided without the model."""
    diff = qualifier_diff(a, b)

    if a.claim_type == "attribute" or b.claim_type == "attribute" \
       or a.canon_value is None or b.canon_value is None:
        return "needs_model", diff
    if not units_compatible(a.canon_unit, b.canon_unit):
        # a rupee figure and a percentage are not in disagreement
        return "unrelated", diff
    if values_agree(a.canon_value, b.canon_value, tol,
                    a.value_raw, b.value_raw):
        return ("corroborates" if not diff else "corroborates_with_caveat"), diff

    # values differ, and calling that a contradiction needs both periods
    if a.period_start is None or b.period_start is None:
        return "insufficient_context", diff

    material = measurement_diff(diff)
    if list(material) == ["period"]:
        # different years reporting different numbers is reporting, not conflict
        return "different_period", diff
    if len(material) == 1:
        return "reconcilable", diff
    if not material:
        # nothing about the measurement distinguishes them, so surface it
        return "contradiction_candidate", diff
    return "needs_model", diff
