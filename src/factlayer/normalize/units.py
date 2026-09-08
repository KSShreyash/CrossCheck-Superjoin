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
    """Two facts are unit-incompatible only when both units are known.

    UNKNOWN means the dimension was not recorded, which is a gap in what we
    read rather than evidence that the two measure different things.
    """
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


# A unit field is a constrained vocabulary, so a lone letter or stray mark
# sitting where a currency symbol belongs is a mangled glyph rather than a
# word. Models transcribe the rupee sign as "I" or a backtick often enough
# that "I in Million" would otherwise normalise to a bare count and never
# match the same figure printed in crore.
_MANGLED_RUPEE = re.compile(r"(^|[\s(])[i`¹¦](?=\s*(in\s+)?"
                            r"(million|mn|crore|cr|lakh|billion|bn|thousand)\b)",
                            re.I)


def normalize_value(value_raw: str | None, unit_raw: str | None,
                    context: str | None = None) -> tuple[float, str] | None:
    """Reduce a printed value and its unit to a magnitude in a base unit.

    Returns None when there is no number to parse. Percentages keep their
    printed magnitude; everything else is multiplied by its scale word so
    that "8,142 Cr" and "81,415.38 million" land on the same number.

    `context` is the verbatim source span the fact came from. It is consulted
    only for the currency, and only when the unit field does not name one:
    the quote is real document text, so it is the more reliable witness when
    a symbol has been dropped or garbled on the way through the model.
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
    if unit_raw and _MANGLED_RUPEE.search(unit_raw):
        return number * scale, "INR"
    if context:
        source = str(context).lower()
        for code, pattern in _CURRENCY:
            if re.search(pattern, source):
                return number * scale, code

    # Strip the scale words and punctuation. If nothing is left, the unit named
    # a magnitude and no dimension - "(Rs in Million)" declared in a table
    # header and dropped on the way out, say. That is an unknown dimension, not
    # a count of things, and the distinction matters: the same revenue printed
    # as "81,415.38 million" in one document and "Rs 8,142 Cr" in another must
    # still be comparable. Unknown is not the same as different.
    residue = _SCALE_WORDS.sub(" ", f" {unit_raw or ''} ".lower())
    residue = re.sub(r"[^a-z]+", "", residue)
    if not residue:
        return number * scale, "UNKNOWN"
    return number * scale, "COUNT"


def values_agree(a: float | None, b: float | None, tol: float) -> bool:
    """Relative comparison, so printed rounding does not read as disagreement."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    return denom > 0 and abs(a - b) / denom <= tol
