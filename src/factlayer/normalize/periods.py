import re
from calendar import monthrange
from datetime import date

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def _fy(end_year: int) -> tuple[str, str, str]:
    """Indian fiscal year: FY24 is the year ending 31 March 2024."""
    return (f"{end_year - 1}-04-01", f"{end_year}-03-31", "fiscal_year")


def _q(quarter: int, fy_end_year: int) -> tuple[str, str, str]:
    starts = {1: (fy_end_year - 1, 4), 2: (fy_end_year - 1, 7),
              3: (fy_end_year - 1, 10), 4: (fy_end_year, 1)}
    y, m = starts[quarter]
    end_m = m + 2
    last = monthrange(y, end_m)[1]
    return (date(y, m, 1).isoformat(), date(y, end_m, last).isoformat(), "quarter")


def _expand(yy: int) -> int:
    return yy if yy > 100 else (2000 + yy)


def normalize_period(period_raw: str | None) -> tuple[str, str, str] | None:
    """Parse a printed period into (start, end, kind)."""
    if not period_raw:
        return None
    t = re.sub(r"\s+", " ", str(period_raw)).strip().lower()

    m = re.search(r"q([1-4])\s*(?:of\s*)?fy\s*'?(\d{2,4})", t)
    if m:
        return _q(int(m.group(1)), _expand(int(m.group(2))))

    # FY2025/26 and FY2025-26: the second component names the ending year
    m = re.search(r"fy\s*'?(\d{4})\s*[/-]\s*(\d{2,4})", t)
    if m:
        tail = int(m.group(2))
        return _fy(_expand(tail) if tail < 100 else tail)

    m = re.search(r"fy\s*'?(\d{2,4})\b", t)
    if m:
        return _fy(_expand(int(m.group(1))))

    m = re.search(r"\b(20\d{2})\s*[-/]\s*(\d{2})\b", t)
    if m:
        return _fy(_expand(int(m.group(2))))

    m = re.search(r"(as (?:on|at|of))\s+(\w+)\s+(\d{1,2}),?\s*(\d{4})", t)
    if m and m.group(2) in MONTHS:
        d = date(int(m.group(4)), MONTHS[m.group(2)], int(m.group(3))).isoformat()
        return (d, d, "instant")

    m = re.search(r"year ended\s+(\w+)\s+(\d{1,2}),?\s*(\d{4})", t)
    if m and m.group(1) in MONTHS:
        month, year = MONTHS[m.group(1)], int(m.group(3))
        if month == 3:
            return _fy(year)
        start = date(year - 1, month, 1).isoformat()
        end = date(year, month, monthrange(year, month)[1]).isoformat()
        return (start, end, "part_year")

    m = re.search(r"\bcy\s*(20\d{2})\b", t) or re.fullmatch(r"(20\d{2})", t)
    if m:
        y = int(m.group(1))
        return (f"{y}-01-01", f"{y}-12-31", "calendar_year")

    return None
