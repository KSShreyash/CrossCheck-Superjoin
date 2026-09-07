import pytest
from factlayer.normalize.periods import normalize_period

@pytest.mark.parametrize("raw,start,end,kind", [
    ("FY24", "2023-04-01", "2024-03-31", "fiscal_year"),
    ("FY2024", "2023-04-01", "2024-03-31", "fiscal_year"),
    ("FY25", "2024-04-01", "2025-03-31", "fiscal_year"),
    ("2024-25", "2024-04-01", "2025-03-31", "fiscal_year"),
    ("FY2025/26", "2025-04-01", "2026-03-31", "fiscal_year"),
    ("year ended March 31, 2024", "2023-04-01", "2024-03-31", "fiscal_year"),
    ("Q4 FY24", "2024-01-01", "2024-03-31", "quarter"),
    ("Q1 FY25", "2024-04-01", "2024-06-30", "quarter"),
    ("as on March 31, 2024", "2024-03-31", "2024-03-31", "instant"),
    ("CY2025", "2025-01-01", "2025-12-31", "calendar_year"),
])
def test_period_parsing(raw, start, end, kind):
    assert normalize_period(raw) == (start, end, kind)

def test_imf_and_indian_notation_agree():
    assert normalize_period("FY2025/26") == normalize_period("FY26")

def test_unparseable_returns_none():
    assert normalize_period("recently") is None
