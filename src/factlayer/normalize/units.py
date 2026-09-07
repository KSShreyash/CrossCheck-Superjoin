import re

SCALES = {
    "thousand": 1e3, "k": 1e3,
    "lakh": 1e5, "lac": 1e5,
    "million": 1e6, "mn": 1e6, "mio": 1e6,
    "crore": 1e7, "cr": 1e7,
    "billion": 1e9, "bn": 1e9,
    "trillion": 1e12, "tn": 1e12,
}
_CURRENCY = [("INR", r"(₹|rs\.?|inr|rupee)"), ("USD", r"(\$|usd|us\s?dollar)")]
_NUM = re.compile(r"-?\(?\s*-?[\d,]*\.?\d+\s*\)?")


def _parse_number(raw: str | None) -> float | None:
    if raw is None:
        return None
    m = _NUM.search(str(raw))
    if not m:
        return None
    token = m.group(0).strip()
    # accounting notation puts losses in parentheses
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").replace(",", "").strip()
    try:
        value = float(token)
    except ValueError:
        return None
    return -value if negative else value


def _scale_from(text: str) -> float:
    # longest first so "crore" is not shadowed by "cr"
    for word, mult in sorted(SCALES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{word}s?\b", text):
            return mult
    return 1.0


def normalize_value(value_raw: str | None, unit_raw: str | None
                    ) -> tuple[float, str] | None:
    """Reduce a printed value and its unit to a magnitude in a base unit.

    Returns None when there is no number to parse. Percentages keep their
    printed magnitude; everything else is multiplied by its scale word so
    that "8,142 Cr" and "81,415.38 million" land on the same number.
    """
    number = _parse_number(value_raw)
    if number is None:
        return None
    blob = f"{value_raw or ''} {unit_raw or ''}".lower()
    scale = _scale_from(blob)

    if re.search(r"(per cent|percent|%|percentage point|bps)", blob):
        return number, "PERCENT"
    if re.search(r"\b(ton|tonne|mt)s?\b", blob):
        return number * scale, "TONNE"
    if re.search(r"\bdays?\b", blob):
        return number * scale, "DAYS"
    for code, pattern in _CURRENCY:
        if re.search(pattern, blob):
            return number * scale, code
    return number * scale, "COUNT"


def values_agree(a: float | None, b: float | None, tol: float) -> bool:
    """Relative comparison, so printed rounding does not read as disagreement."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    return denom > 0 and abs(a - b) / denom <= tol
