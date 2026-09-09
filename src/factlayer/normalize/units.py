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
_SCALE_WORDS = re.compile(
    r"\b(" + "|".join(sorted(SCALES, key=len, reverse=True)) + r")s?\b|\bin\b", re.I)


def units_compatible(a: str | None, b: str | None) -> bool:
    """Two facts are unit-incompatible only when both units are known."""
    if not a or not b or a == "UNKNOWN" or b == "UNKNOWN":
        return True
    return a == b


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


# in a unit field, a lone letter before a scale word is a mangled currency
_MANGLED_RUPEE = re.compile(r"(^|[\s(])[i`¹¦](?=\s*(in\s+)?"
                            r"(million|mn|crore|cr|lakh|billion|bn|thousand)\b)",
                            re.I)


def normalize_value(value_raw: str | None, unit_raw: str | None,
                    context: str | None = None) -> tuple[float, str] | None:
    """Reduce a printed value and its unit to a magnitude in a base unit."""
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
    if unit_raw and _MANGLED_RUPEE.search(unit_raw):
        return number * scale, "INR"
    if context:
        source = str(context).lower()
        for code, pattern in _CURRENCY:
            if re.search(pattern, source):
                return number * scale, code

    # strip scale words: nothing left means the dimension is unknown, not a count
    residue = _SCALE_WORDS.sub(" ", f" {unit_raw or ''} ".lower())
    residue = re.sub(r"[^a-z]+", "", residue)
    if not residue:
        return number * scale, "UNKNOWN"
    return number * scale, "COUNT"


def significant_figures(raw: str | None) -> int | None:
    """How precisely a number was printed, in significant figures."""
    if raw is None:
        return None
    digits = re.sub(r"[^\d.]", "", str(raw))
    if not digits or not any(ch.isdigit() for ch in digits):
        return None
    if "." in digits:
        stripped = digits.replace(".", "").lstrip("0")
        return len(stripped) or None
    stripped = digits.strip("0")
    return len(stripped) or 1


def _round_to(value: float, figures: int) -> float:
    if value == 0 or figures <= 0:
        return 0.0
    from math import floor, log10
    exponent = floor(log10(abs(value)))
    return round(value, -(exponent - figures + 1))


def values_agree(a: float | None, b: float | None, tol: float,
                 a_raw: str | None = None, b_raw: str | None = None) -> bool:
    """Whether two canonical values report the same quantity."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    if denom > 0 and abs(a - b) / denom <= tol:
        return True

    fa, fb = significant_figures(a_raw), significant_figures(b_raw)
    if fa and fb:
        figures = min(fa, fb)
        if figures >= 1 and _round_to(a, figures) == _round_to(b, figures):
            return True
    return False
