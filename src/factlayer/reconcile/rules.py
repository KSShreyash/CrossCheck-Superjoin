from ..models import Fact
from ..normalize.units import values_agree


def qualifier_diff(a: Fact, b: Fact) -> dict[str, tuple]:
    """Which recorded qualifiers differ between two facts.

    No qualifier key is hard-coded: any key present on either side with a
    different value counts, so new kinds of distinction introduced by an
    unfamiliar document are picked up without changing this code.
    """
    diff: dict[str, tuple] = {}
    for key in set(a.qualifiers) | set(b.qualifiers):
        left, right = a.qualifiers.get(key), b.qualifiers.get(key)
        if left != right:
            diff[key] = (left, right)
    if (a.period_start, a.period_end) != (b.period_start, b.period_end):
        diff["period"] = ((a.period_start, a.period_end),
                          (b.period_start, b.period_end))
    if a.canon_unit != b.canon_unit:
        diff["unit"] = (a.canon_unit, b.canon_unit)
    return diff


def rule_verdict(a: Fact, b: Fact, tol: float) -> tuple[str, dict]:
    """Decide what can be decided without the model."""
    diff = qualifier_diff(a, b)

    if a.claim_type == "attribute" or b.claim_type == "attribute" \
       or a.canon_value is None or b.canon_value is None:
        return "needs_model", diff
    if a.canon_unit != b.canon_unit:
        # a rupee figure and a percentage are not in disagreement
        return "unrelated", diff
    if values_agree(a.canon_value, b.canon_value, tol):
        return ("corroborates" if not diff else "corroborates_with_caveat"), diff

    # Values differ. Calling that a contradiction asserts the two facts are
    # comparable, and that cannot be asserted without knowing both periods.
    # An absent period is unknown, NOT "the same period as the other one":
    # treating None == None as a match manufactured 1,480 false contradictions
    # across two starter documents, 45% of all pairs.
    if a.period_start is None or b.period_start is None:
        return "insufficient_context", diff

    if len(diff) == 1:
        return "reconcilable", diff
    if not diff:
        return "contradiction_candidate", diff
    return "needs_model", diff
