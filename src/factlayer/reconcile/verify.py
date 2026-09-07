from ..models import Fact
from ..normalize.units import values_agree

# transforms whose truth is a question about qualifiers rather than arithmetic
_QUALIFIER_KINDS = {"basis", "vintage", "period", "scope", "segment"}


def verify(a: Fact, b: Fact, claimed: dict | None,
           tol: float) -> tuple[bool, str]:
    """Re-derive a transformation the model claimed, independently.

    The model is allowed to propose why two facts reconcile; it is not taken
    at its word. A scale claim has to survive the arithmetic, and a claim that
    some qualifier explains the gap has to survive checking that the qualifier
    genuinely differs.
    """
    if not claimed or "kind" not in claimed:
        return False, "no transform claimed; nothing to verify"
    kind = claimed["kind"]

    if kind == "scale":
        try:
            factor = float(claimed.get("factor") or 1)
        except (TypeError, ValueError):
            return False, f"scale factor is not a number: {claimed.get('factor')!r}"
        if a.canon_value is None or b.canon_value is None:
            return False, "missing canonical values"
        if values_agree(a.canon_value * factor, b.canon_value, tol) or \
           values_agree(a.canon_value, b.canon_value * factor, tol):
            return True, f"values reconcile under a factor of {factor:g}"
        return False, (f"values do not reconcile under a factor of {factor:g}: "
                       f"{a.canon_value:g} vs {b.canon_value:g}")

    if kind in _QUALIFIER_KINDS:
        if kind == "period":
            differs = (a.period_start, a.period_end) != (b.period_start, b.period_end)
        else:
            differs = a.qualifiers.get(kind) != b.qualifiers.get(kind)
        if differs:
            return True, f"{kind} genuinely differs between the two facts"
        return False, f"{kind} was claimed to differ but both facts agree on it"

    if kind == "none":
        return True, "no transform needed"
    return False, f"unrecognised transform kind: {kind}"
