# Fact Knowledge Layer — Implementation Plan

**Goal:** Ingest PDFs, extract facts that each carry a verbatim span in their source
document, and classify cross-document fact pairs as corroborating, contradicting, or
reconcilable through context.

**Architecture:** A seven-stage pipeline over SQLite. The LLM proposes facts and judges
ambiguous pairs; deterministic normalisers and a verification pass independently re-derive
anything quantitative it claims. Where rules and model disagree, the pair is recorded as
`needs_review` rather than silently resolved.

**Tech Stack:** Python 3.11, FastAPI, Jinja2 + htmx, SQLite, PyMuPDF, Google Gemini Flash,
pytest.

**Spec:** `docs/design.md`

## Global Constraints

- Python 3.11. Dependencies pinned in `pyproject.toml`.
- Nothing document-specific may be hard-coded: no filenames, no fixed metric lists, no
  entity alias tables, no per-document branches. Generic dimensions (a qualifier named
  `basis`, a unit named `crore`) are fine; `if doc == "delhivery"` is not.
- Every stored fact must carry an `evidence_quote` found verbatim in its source window.
  Facts that fail this check are rejected, never stored as facts.
- One SQLite file. No external services.
- No frontend build step. Jinja templates plus htmx served by the same FastAPI process.
- LLM temperature 0. Every model call cached under a content hash.
- Tests never make network calls. The LLM is stubbed by pre-seeding the cache.
- No AI or assistant attribution in commits, code comments, or documentation.

## Cut Line

Tasks 1–13 and 15–16 are load-bearing: all four required cases depend on them. If time
runs short, drop in this order — Task 14 (embedding-accelerated pairing, falls back to
lexical), the gaps UI screen in Task 16, and background job polling in Task 15 (ingest
synchronously instead). Never leave the engine half-built to start the UI.

## File Structure

```
src/factlayer/
  config.py              settings, model name, paths, thresholds
  db.py                  schema + connection
  models.py              dataclasses shared across stages
  ingest/pdf.py          PyMuPDF → blocks with bbox
  ingest/boilerplate.py  repeated-block detection
  ingest/gaps.py         image-only page detection
  ingest/segment.py      block windows for long context
  llm/client.py          provider interface + Gemini implementation
  llm/cache.py           content-hash cache
  llm/prompts.py         extraction, canonicalisation, adjudication prompts
  extract.py             extraction + grounding check
  normalize/units.py     value + unit → canonical magnitude
  normalize/periods.py   period string → (start, end, kind)
  normalize/canon.py     entity and metric canonicalisation
  pairing.py             candidate pair generation
  reconcile/rules.py     deterministic verdict
  reconcile/adjudicate.py LLM adjudication
  reconcile/verify.py    re-derive claimed transformations
  pipeline.py            end-to-end orchestration
  api.py                 FastAPI routes
  templates/             base, documents, facts, relation, gaps
tests/
scripts/ingest_starter.py
```

---

### Task 1: Project skeleton, config, database schema

**Files:**
- Create: `pyproject.toml`, `src/factlayer/__init__.py`, `src/factlayer/config.py`,
  `src/factlayer/db.py`, `src/factlayer/models.py`, `.env.example`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces: `connect(path) -> sqlite3.Connection`, `init_schema(conn) -> None`,
  `Settings` with `db_path`, `gemini_api_key`, `model`, `embed_model`,
  `value_tolerance`, `upload_dir`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
from factlayer.db import connect, init_schema

def test_schema_creates_expected_tables(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"documents", "blocks", "facts", "evidence", "relations",
            "llm_cache", "rejected_facts", "jobs", "canon_terms"} <= names

def test_init_schema_is_idempotent(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    init_schema(conn)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'factlayer'`

- [ ] **Step 3: Write pyproject.toml**

```toml
[project]
name = "factlayer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "jinja2>=3.1",
    "python-multipart>=0.0.9",
    "pymupdf>=1.24",
    "google-generativeai>=0.5",
    "python-dotenv>=1.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27", "reportlab>=4.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Write config.py**

```python
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    # FACTLAYER_DB lets tests and deployments point at a different file
    db_path: Path = Path(os.getenv("FACTLAYER_DB", ROOT / "factlayer.sqlite"))
    upload_dir: Path = ROOT / "uploads"
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    model: str = os.getenv("FACTLAYER_MODEL", "gemini-2.0-flash")
    embed_model: str = "text-embedding-004"
    value_tolerance: float = 1e-3        # relative, absorbs printed rounding
    window_chars: int = 12000            # long-context extraction window
    window_overlap: int = 1200
    boilerplate_min_pages: int = 4       # repeats on >= N pages -> boilerplate
    gap_min_chars: int = 120             # below this on a full page -> gap
    max_pairs_per_fact: int = 12

settings = Settings()
```

- [ ] **Step 5: Write models.py**

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Block:
    page_no: int
    block_index: int
    text: str
    bbox: tuple[float, float, float, float]
    is_boilerplate: bool = False

@dataclass
class Window:
    doc_id: int
    index: int
    text: str
    block_ids: list[int]
    # char span of each block inside `text`, parallel to block_ids.
    # Recorded at build time because block text contains its own newlines,
    # so splitting the window on "\n" does not recover block boundaries.
    block_spans: list[tuple[int, int]] = field(default_factory=list)
    density: float = 0.0

@dataclass
class Fact:
    doc_id: int
    subject: str
    metric: str
    value_raw: str | None
    value_num: float | None
    unit_raw: str | None
    period_raw: str | None
    qualifiers: dict[str, Any] = field(default_factory=dict)
    claim_type: str = "measurement"
    evidence_quote: str = ""
    confidence: float = 0.0
    # filled by normalisation
    canon_value: float | None = None
    canon_unit: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    period_kind: str | None = None
    entity_id: str | None = None
    metric_id: str | None = None
    # char span of evidence_quote inside its window, set during extraction
    span: tuple[int, int] | None = None
```

- [ ] **Step 6: Write db.py**

```python
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE, filename TEXT,
  title TEXT, page_count INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS blocks (
  id INTEGER PRIMARY KEY, doc_id INTEGER, page_no INTEGER, block_index INTEGER,
  text TEXT, x0 REAL, y0 REAL, x1 REAL, y1 REAL, is_boilerplate INTEGER DEFAULT 0,
  FOREIGN KEY(doc_id) REFERENCES documents(id));

CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY, doc_id INTEGER, subject TEXT, metric TEXT,
  value_raw TEXT, value_num REAL, unit_raw TEXT, period_raw TEXT,
  qualifiers TEXT, claim_type TEXT, confidence REAL,
  canon_value REAL, canon_unit TEXT,
  period_start TEXT, period_end TEXT, period_kind TEXT,
  entity_id TEXT, metric_id TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(id));

CREATE TABLE IF NOT EXISTS evidence (
  id INTEGER PRIMARY KEY, fact_id INTEGER, quote TEXT, page_no INTEGER,
  char_start INTEGER, char_end INTEGER, block_ids TEXT,
  x0 REAL, y0 REAL, x1 REAL, y1 REAL,
  FOREIGN KEY(fact_id) REFERENCES facts(id));

CREATE TABLE IF NOT EXISTS relations (
  id INTEGER PRIMARY KEY, fact_a INTEGER, fact_b INTEGER,
  rule_verdict TEXT, model_verdict TEXT, final_verdict TEXT,
  reason_code TEXT, explanation TEXT, qualifier_diff TEXT,
  claimed_transform TEXT, verified INTEGER, agreed INTEGER, confidence REAL,
  FOREIGN KEY(fact_a) REFERENCES facts(id), FOREIGN KEY(fact_b) REFERENCES facts(id));

CREATE TABLE IF NOT EXISTS llm_cache (
  key TEXT PRIMARY KEY, response TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS rejected_facts (
  id INTEGER PRIMARY KEY, doc_id INTEGER, payload TEXT, reason TEXT);

CREATE TABLE IF NOT EXISTS gaps (
  id INTEGER PRIMARY KEY, doc_id INTEGER, page_no INTEGER, reason TEXT);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, doc_id INTEGER, stage TEXT, done INTEGER, total INTEGER,
  facts INTEGER DEFAULT 0, error TEXT);

CREATE TABLE IF NOT EXISTS canon_terms (
  kind TEXT, raw TEXT, canon_id TEXT, label TEXT, PRIMARY KEY (kind, raw));

CREATE INDEX IF NOT EXISTS idx_facts_metric ON facts(metric_id);
CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id);
CREATE INDEX IF NOT EXISTS idx_rel_verdict ON relations(final_verdict);
"""

def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # background ingest writes while the UI polls; without this the reader
    # raises "database is locked" instead of waiting for the writer
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pip install -e ".[dev]" && pytest tests/test_db.py -v`
Expected: 2 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .env.example src tests
git commit -m "Add project skeleton and SQLite schema"
```

---

### Task 2: PDF ingest into blocks with bounding boxes

**Files:**
- Create: `src/factlayer/ingest/__init__.py`, `src/factlayer/ingest/pdf.py`
- Test: `tests/conftest.py`, `tests/test_ingest_pdf.py`

**Interfaces:**
- Produces: `extract_blocks(pdf_path) -> tuple[list[Block], int]` returning blocks and
  page count; `file_sha256(path) -> str`.

- [ ] **Step 1: Write the fixture PDF helper**

```python
# tests/conftest.py
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

@pytest.fixture
def make_pdf(tmp_path):
    def _make(pages: list[list[str]], name="t.pdf"):
        path = tmp_path / name
        c = canvas.Canvas(str(path), pagesize=A4)
        for lines in pages:
            y = 800
            for line in lines:
                c.drawString(60, y, line)
                y -= 18
            c.showPage()
        c.save()
        return path
    return _make
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_ingest_pdf.py
from factlayer.ingest.pdf import extract_blocks, file_sha256

def test_extracts_text_page_and_bbox(make_pdf):
    p = make_pdf([["Revenue from operations 81,415.38"], ["Second page"]])
    blocks, pages = extract_blocks(p)
    assert pages == 2
    assert any("81,415.38" in b.text for b in blocks)
    first = next(b for b in blocks if "81,415.38" in b.text)
    assert first.page_no == 1
    x0, y0, x1, y1 = first.bbox
    assert x1 > x0 and y1 > y0

def test_sha256_is_stable(make_pdf):
    p = make_pdf([["same"]])
    assert file_sha256(p) == file_sha256(p)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_ingest_pdf.py -v`
Expected: FAIL, no module `factlayer.ingest.pdf`

- [ ] **Step 4: Implement pdf.py**

```python
import hashlib
from pathlib import Path
import fitz
from ..models import Block

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_blocks(pdf_path: str | Path) -> tuple[list[Block], int]:
    doc = fitz.open(str(pdf_path))
    blocks: list[Block] = []
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            for i, b in enumerate(page.get_text("blocks")):
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                text = (text or "").strip()
                if not text:
                    continue
                blocks.append(Block(page_no=pno + 1, block_index=i, text=text,
                                    bbox=(x0, y0, x1, y1)))
        return blocks, doc.page_count
    finally:
        doc.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ingest_pdf.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/factlayer/ingest tests/conftest.py tests/test_ingest_pdf.py
git commit -m "Extract PDF text blocks with page numbers and bounding boxes"
```

---

### Task 3: Boilerplate and extraction-gap detection

**Files:**
- Create: `src/factlayer/ingest/boilerplate.py`, `src/factlayer/ingest/gaps.py`
- Test: `tests/test_boilerplate.py`

**Interfaces:**
- Produces: `mark_boilerplate(blocks, min_pages) -> None` (mutates `is_boilerplate`);
  `find_gaps(blocks, page_count, min_chars) -> list[tuple[int, str]]` of
  `(page_no, reason)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_boilerplate.py
from factlayer.models import Block
from factlayer.ingest.boilerplate import mark_boilerplate
from factlayer.ingest.gaps import find_gaps

def _b(page, text):
    return Block(page_no=page, block_index=0, text=text, bbox=(0, 0, 1, 1))

def test_repeated_running_header_is_marked():
    blocks = [_b(p, "Annual Report 2023-24") for p in range(1, 6)]
    blocks.append(_b(1, "Revenue from operations 81,415.38"))
    mark_boilerplate(blocks, min_pages=4)
    assert all(b.is_boilerplate for b in blocks if "Annual Report" in b.text)
    assert not [b for b in blocks if "81,415" in b.text][0].is_boilerplate

def test_page_with_no_text_is_reported_as_gap():
    blocks = [_b(1, "cover page has no text layer" * 20)]
    gaps = find_gaps(blocks, page_count=3, min_chars=120)
    pages = {p for p, _ in gaps}
    assert pages == {2, 3}

def test_page_with_trivial_text_is_a_gap():
    blocks = [_b(1, "12")]
    gaps = find_gaps(blocks, page_count=1, min_chars=120)
    assert gaps and gaps[0][0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_boilerplate.py -v`
Expected: FAIL, modules missing

- [ ] **Step 3: Implement boilerplate.py**

```python
import re
from collections import defaultdict
from ..models import Block

_NUM = re.compile(r"\d+")

def _norm(text: str) -> str:
    # page numbers vary between repeats of the same running header
    return re.sub(r"\s+", " ", _NUM.sub("#", text)).strip().lower()

def mark_boilerplate(blocks: list[Block], min_pages: int) -> None:
    pages_by_norm: dict[str, set[int]] = defaultdict(set)
    for b in blocks:
        pages_by_norm[_norm(b.text)].add(b.page_no)
    repeated = {k for k, pages in pages_by_norm.items() if len(pages) >= min_pages}
    for b in blocks:
        if _norm(b.text) in repeated:
            b.is_boilerplate = True
```

- [ ] **Step 4: Implement gaps.py**

```python
from collections import defaultdict
from ..models import Block

def find_gaps(blocks: list[Block], page_count: int,
              min_chars: int) -> list[tuple[int, str]]:
    chars: dict[int, int] = defaultdict(int)
    for b in blocks:
        chars[b.page_no] += len(b.text)
    gaps = []
    for page in range(1, page_count + 1):
        n = chars.get(page, 0)
        if n == 0:
            gaps.append((page, "no extractable text; page is likely image-only"))
        elif n < min_chars:
            gaps.append((page, f"only {n} characters extracted; likely scanned or graphical"))
    return gaps
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_boilerplate.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/factlayer/ingest tests/test_boilerplate.py
git commit -m "Detect running headers and pages with no usable text layer"
```

---

### Task 4: Segment blocks into long-context windows

**Files:**
- Create: `src/factlayer/ingest/segment.py`
- Test: `tests/test_segment.py`

**Interfaces:**
- Produces: `build_windows(blocks, doc_id, window_chars, overlap) -> list[Window]`.
  Each `Window.text` is the joined text of its blocks; `density` scores numeric content
  for ordering.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_segment.py
from factlayer.models import Block
from factlayer.ingest.segment import build_windows, density_score

def _b(i, text):
    return Block(page_no=1 + i // 5, block_index=i, text=text, bbox=(0, 0, 1, 1))

def test_windows_respect_size_and_overlap():
    blocks = [_b(i, "x" * 400) for i in range(20)]
    wins = build_windows(blocks, doc_id=1, window_chars=1000, overlap=400)
    assert len(wins) > 1
    assert all(len(w.text) <= 1600 for w in wins)
    assert set(wins[0].block_ids) & set(wins[1].block_ids)

def test_boilerplate_blocks_are_excluded():
    b = _b(0, "Annual Report 2023-24")
    b.is_boilerplate = True
    wins = build_windows([b, _b(1, "Revenue 81,415.38")], 1, 1000, 100)
    assert "Annual Report" not in wins[0].text

def test_density_prefers_numeric_text():
    assert density_score("Revenue from operations was 81,415.38 million in FY24") > \
           density_score("The board places on record its appreciation")

def test_block_spans_survive_newlines_inside_a_block():
    # PDF blocks carry their own newlines, so window.text.split("\n") does NOT
    # recover block boundaries. Spans must be recorded when the window is built.
    a = Block(1, 0, "Revenue from operations\nwas 81,415.38 million", (0, 0, 1, 1))
    b = Block(1, 1, "Unrelated text", (0, 0, 1, 1))
    w = build_windows([a, b], doc_id=1, window_chars=10_000, overlap=0)[0]
    assert len(w.block_spans) == 2
    start, end = w.block_spans[0]
    assert w.text[start:end] == a.text
    quote = "81,415.38 million"
    at = w.text.index(quote)
    assert start <= at and at + len(quote) <= end   # quote belongs to block 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_segment.py -v`
Expected: FAIL, module missing

- [ ] **Step 3: Implement segment.py**

```python
import re
from ..models import Block, Window

_NUMERIC = re.compile(r"\d[\d,]*\.?\d*")
_UNITISH = re.compile(
    r"(per cent|percent|%|million|billion|crore|cr\b|lakh|mn\b|bn\b|₹|rs\.?|usd|\$)",
    re.I)
_PERIODISH = re.compile(r"(fy\s?\d{2,4}|q[1-4]|20\d{2}|march 31|year ended)", re.I)

def density_score(text: str) -> float:
    if not text:
        return 0.0
    n = len(_NUMERIC.findall(text))
    u = len(_UNITISH.findall(text))
    p = len(_PERIODISH.findall(text))
    return (n + 2 * u + 2 * p) / max(len(text) / 200.0, 1.0)

def _emit(doc_id: int, index: int, ids: list[int], parts: list[str]) -> Window:
    """Join blocks and record where each one lands in the joined text."""
    spans, cursor = [], 0
    for part in parts:
        spans.append((cursor, cursor + len(part)))
        cursor += len(part) + 1          # +1 for the "\n" inserted by join
    text = "\n".join(parts)
    return Window(doc_id=doc_id, index=index, text=text, block_ids=list(ids),
                  block_spans=spans, density=density_score(text))

def build_windows(blocks: list[Block], doc_id: int, window_chars: int,
                  overlap: int) -> list[Window]:
    usable = [(i, b) for i, b in enumerate(blocks) if not b.is_boilerplate]
    windows: list[Window] = []
    cur_ids: list[int] = []
    cur_parts: list[str] = []
    cur_len = 0

    def flush():
        nonlocal cur_ids, cur_parts, cur_len
        if not cur_parts:
            return
        windows.append(_emit(doc_id, len(windows), cur_ids, cur_parts))
        keep, kept_len = [], 0
        for idx, part in zip(reversed(cur_ids), reversed(cur_parts)):
            if kept_len >= overlap:
                break
            keep.append((idx, part))
            kept_len += len(part)
        keep.reverse()
        cur_ids = [i for i, _ in keep]
        cur_parts = [p for _, p in keep]
        cur_len = kept_len

    for idx, b in usable:
        if cur_len + len(b.text) > window_chars and cur_parts:
            flush()
        cur_ids.append(idx)
        cur_parts.append(b.text)
        cur_len += len(b.text)
    if cur_parts:
        windows.append(_emit(doc_id, len(windows), cur_ids, cur_parts))
    return windows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_segment.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/factlayer/ingest/segment.py tests/test_segment.py
git commit -m "Group blocks into overlapping windows sized for long context"
```

---

### Task 5: Value and unit normaliser

This is the highest-value task in the plan. Case 1 and case 3 both live or die here.

**Files:**
- Create: `src/factlayer/normalize/__init__.py`, `src/factlayer/normalize/units.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `normalize_value(value_raw, unit_raw) -> tuple[float, str] | None` returning
  `(magnitude, canonical_unit)` where canonical unit is one of `INR`, `USD`, `PERCENT`,
  `TONNE`, `COUNT`, `DAYS`; `values_agree(a, b, tol) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_units.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py -v`
Expected: FAIL, module missing

- [ ] **Step 3: Implement units.py**

```python
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

def _parse_number(raw: str) -> float | None:
    if raw is None:
        return None
    m = _NUM.search(str(raw))
    if not m:
        return None
    token = m.group(0).strip()
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").replace(",", "").strip()
    try:
        value = float(token)
    except ValueError:
        return None
    return -value if negative else value

def _scale_from(text: str) -> float:
    for word, mult in sorted(SCALES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{word}s?\b", text):
            return mult
    return 1.0

def normalize_value(value_raw: str | None, unit_raw: str | None
                    ) -> tuple[float, str] | None:
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
    if a is None or b is None:
        return False
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    return denom > 0 and abs(a - b) / denom <= tol
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/factlayer/normalize tests/test_units.py
git commit -m "Normalise values across crore, lakh, million and percent scales"
```

---

### Task 6: Period normaliser

**Files:**
- Create: `src/factlayer/normalize/periods.py`
- Test: `tests/test_periods.py`

**Interfaces:**
- Produces: `normalize_period(period_raw) -> tuple[str, str, str] | None` returning
  `(start_iso, end_iso, kind)` with kind in `fiscal_year`, `quarter`, `calendar_year`,
  `instant`, `part_year`. Indian fiscal years run 1 April to 31 March; `FY24` is the
  year *ending* 31 March 2024.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_periods.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_periods.py -v`
Expected: FAIL, module missing

- [ ] **Step 3: Implement periods.py**

```python
import re
from calendar import monthrange
from datetime import date

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}

def _fy(end_year: int) -> tuple[str, str, str]:
    return (f"{end_year - 1}-04-01", f"{end_year}-03-31", "fiscal_year")

def _q(quarter: int, fy_end_year: int) -> tuple[str, str, str]:
    starts = {1: (fy_end_year - 1, 4), 2: (fy_end_year - 1, 7),
              3: (fy_end_year - 1, 10), 4: (fy_end_year, 1)}
    y, m = starts[quarter]
    end_m = m + 2
    end_y = y
    last = monthrange(end_y, end_m)[1]
    return (date(y, m, 1).isoformat(), date(end_y, end_m, last).isoformat(), "quarter")

def _expand(yy: int) -> int:
    return yy if yy > 100 else (2000 + yy)

def normalize_period(period_raw: str | None) -> tuple[str, str, str] | None:
    if not period_raw:
        return None
    t = re.sub(r"\s+", " ", str(period_raw)).strip().lower()

    m = re.search(r"q([1-4])\s*(?:of\s*)?fy\s*'?(\d{2,4})", t)
    if m:
        return _q(int(m.group(1)), _expand(int(m.group(2))))

    m = re.search(r"fy\s*'?(\d{4})\s*[/-]\s*(\d{2,4})", t)
    if m:
        return _fy(_expand(int(m.group(2))) if int(m.group(2)) < 100
                   else int(m.group(2)))

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_periods.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/factlayer/normalize/periods.py tests/test_periods.py
git commit -m "Parse Indian fiscal years, quarters and IMF FY notation"
```

---

### Task 7: LLM client with content-hash cache

**Files:**
- Create: `src/factlayer/llm/__init__.py`, `src/factlayer/llm/cache.py`,
  `src/factlayer/llm/client.py`
- Test: `tests/test_llm_cache.py`

**Interfaces:**
- Produces: `cache_key(model, prompt_version, payload) -> str`;
  `LLMClient.complete_json(prompt, prompt_version) -> dict` which returns cached
  responses without touching the network and raises `NoAPIKey` on a miss with no key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_cache.py
import pytest
from factlayer.db import connect, init_schema
from factlayer.llm.client import LLMClient, NoAPIKey
from factlayer.llm.cache import cache_key, put

def test_cache_hit_never_calls_provider(tmp_path):
    conn = connect(tmp_path / "t.sqlite"); init_schema(conn)
    client = LLMClient(conn, api_key=None, model="m")
    key = cache_key("m", "v1", "PROMPT")
    put(conn, key, {"facts": [{"metric": "revenue"}]})
    assert client.complete_json("PROMPT", "v1")["facts"][0]["metric"] == "revenue"

def test_cache_miss_without_key_raises(tmp_path):
    conn = connect(tmp_path / "t.sqlite"); init_schema(conn)
    client = LLMClient(conn, api_key=None, model="m")
    with pytest.raises(NoAPIKey):
        client.complete_json("UNSEEN", "v1")

def test_key_is_stable_and_sensitive(tmp_path):
    assert cache_key("m", "v1", "a") == cache_key("m", "v1", "a")
    assert cache_key("m", "v1", "a") != cache_key("m", "v2", "a")

def test_truncated_json_raises_a_named_error():
    from factlayer.llm.client import _loads_lenient, BadModelJSON
    with pytest.raises(BadModelJSON):
        _loads_lenient('{"facts": [{"metric": "revenue"}, {"metric": "EBI')

def test_rate_limit_is_retryable_but_bad_json_is_not():
    from factlayer.llm.client import _is_retryable, BadModelJSON
    assert _is_retryable(Exception('429 Resource has been exhausted'))
    assert _is_retryable(Exception('503 Service Unavailable'))
    assert not _is_retryable(BadModelJSON('truncated'))
    assert not _is_retryable(ValueError('bad argument'))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_cache.py -v`
Expected: FAIL, modules missing

- [ ] **Step 3: Implement cache.py**

```python
import hashlib, json, sqlite3

def cache_key(model: str, prompt_version: str, payload: str) -> str:
    h = hashlib.sha256()
    h.update(f"{model}\x00{prompt_version}\x00{payload}".encode("utf-8"))
    return h.hexdigest()

def get(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
    return json.loads(row["response"]) if row else None

def put(conn: sqlite3.Connection, key: str, response: dict) -> None:
    conn.execute("INSERT OR REPLACE INTO llm_cache(key, response) VALUES (?,?)",
                 (key, json.dumps(response, ensure_ascii=False)))
    conn.commit()
```

- [ ] **Step 4: Implement client.py**

```python
import json, re, sqlite3, time
from . import cache

class NoAPIKey(RuntimeError):
    """Cache miss with no API key configured."""

class BadModelJSON(RuntimeError):
    """Model returned something that is not usable JSON."""

# free-tier quota and transient server errors are worth waiting out;
# a malformed response is not, because temperature 0 reproduces it
_RETRYABLE = ("429", "rate limit", "resource_exhausted", "quota", "exhausted",
              "503", "unavailable", "500", "internal", "deadline")

def _is_retryable(exc: Exception) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _RETRYABLE)

def _loads_lenient(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        raise BadModelJSON(
            f"could not parse model output ({len(text)} chars); "
            "most likely truncated at the output token limit")

class LLMClient:
    def __init__(self, conn: sqlite3.Connection, api_key: str | None, model: str):
        self.conn, self.api_key, self.model = conn, api_key, model
        self._model_obj = None

    def _provider(self):
        if self._model_obj is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model_obj = genai.GenerativeModel(self.model)
        return self._model_obj

    def complete_json(self, prompt: str, prompt_version: str,
                      max_attempts: int = 5) -> dict:
        key = cache.cache_key(self.model, prompt_version, prompt)
        hit = cache.get(self.conn, key)
        if hit is not None:
            return hit
        if not self.api_key:
            raise NoAPIKey(
                "No cached response and GEMINI_API_KEY is unset. "
                "Set a key to ingest documents the cache has not seen.")

        for attempt in range(max_attempts):
            try:
                resp = self._provider().generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0,
                        "response_mime_type": "application/json",
                        # a 12k-char window can yield a lot of facts; the
                        # default ceiling truncates the JSON mid-object
                        "max_output_tokens": 8192,
                    })
                break
            except Exception as exc:
                if attempt == max_attempts - 1 or not _is_retryable(exc):
                    raise
                # free tier rations requests per minute, so wait it out
                time.sleep(min(2 ** attempt * 2, 60))

        data = _loads_lenient(resp.text)      # BadModelJSON is not retried
        cache.put(self.conn, key, data)
        return data
```

The retry exists because the free tier rations requests per minute and a long
ingest will hit that ceiling. Malformed JSON is deliberately *not* retried:
temperature is zero, so the model reproduces it and retrying only burns quota.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_llm_cache.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/factlayer/llm tests/test_llm_cache.py
git commit -m "Add cached LLM client so runs are reproducible and key-free"
```

---

### Task 8: Fact extraction with the grounding check

**Files:**
- Create: `src/factlayer/llm/prompts.py`, `src/factlayer/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Produces: `EXTRACTION_PROMPT_VERSION`, `build_extraction_prompt(window_text) -> str`,
  `extract_facts(client, window, doc_id) -> tuple[list[Fact], list[dict]]` returning
  accepted facts and rejected payloads; `locate_quote(haystack, quote) -> tuple[int,int]|None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extract.py
from factlayer.db import connect, init_schema
from factlayer.llm.cache import cache_key, put
from factlayer.llm.client import LLMClient
from factlayer.llm.prompts import build_extraction_prompt, EXTRACTION_PROMPT_VERSION
from factlayer.extract import extract_facts, locate_quote
from factlayer.models import Window

TEXT = ("Revenue from operations on consolidated basis for FY24 stood at "
        "Rs 81,415.38 million as against Rs 72,253.01 million for FY23.")

def _client(tmp_path, payload):
    conn = connect(tmp_path / "t.sqlite"); init_schema(conn)
    prompt = build_extraction_prompt(TEXT)
    put(conn, cache_key("m", EXTRACTION_PROMPT_VERSION, prompt), payload)
    return LLMClient(conn, api_key=None, model="m")

def test_grounded_fact_is_accepted_with_offsets(tmp_path):
    payload = {"facts": [{
        "subject": "Delhivery Limited", "metric": "revenue from operations",
        "value_raw": "81,415.38", "unit_raw": "Rs million", "period_raw": "FY24",
        "qualifiers": {"basis": "consolidated"}, "claim_type": "measurement",
        "evidence_quote": "Rs 81,415.38 million", "confidence": 0.9}]}
    client = _client(tmp_path, payload)
    facts, rejected = extract_facts(client, Window(1, 0, TEXT, [0]), doc_id=1)
    assert not rejected
    assert facts[0].qualifiers["basis"] == "consolidated"

def test_hallucinated_quote_is_rejected(tmp_path):
    payload = {"facts": [{
        "subject": "Delhivery Limited", "metric": "revenue",
        "value_raw": "99,999", "unit_raw": "Rs million", "period_raw": "FY24",
        "qualifiers": {}, "claim_type": "measurement",
        "evidence_quote": "Rs 99,999 million", "confidence": 0.9}]}
    client = _client(tmp_path, payload)
    facts, rejected = extract_facts(client, Window(1, 0, TEXT, [0]), doc_id=1)
    assert facts == []
    assert rejected[0]["reason"] == "evidence_quote not found in source window"

def test_locate_quote_tolerates_whitespace():
    assert locate_quote("a  b\nc", "a b c") is not None

def test_overlap_twins_are_deduped_but_other_documents_survive():
    from factlayer.extract import dedupe_facts
    from factlayer.models import Fact
    def _f(doc, conf, quote, value='81,415.38'):
        f = Fact(doc, 'Delhivery', 'revenue from operations', value,
                 None, 'Rs million', 'FY24')
        f.evidence_quote, f.confidence = quote, conf
        return f
    # same sentence seen twice because the windows overlap
    twin_a = _f(1, 0.90, 'Revenue from operations  stood at  Rs 81,415.38 million')
    twin_b = _f(1, 0.95, 'Revenue from operations stood at Rs 81,415.38 million')
    other_value = _f(1, 0.90, 'as against Rs 72,253.01 million', '72,253.01')
    other_doc = _f(2, 0.80, 'Revenue from operations stood at Rs 81,415.38 million')
    out = dedupe_facts([twin_a, twin_b, other_value, other_doc])
    assert len(out) == 3
    kept = [f for f in out if f.doc_id == 1 and f.value_raw == '81,415.38']
    assert len(kept) == 1 and kept[0].confidence == 0.95
    assert any(f.doc_id == 2 for f in out), 'cross-document copy is a real corroboration'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL, modules missing

- [ ] **Step 3: Implement prompts.py (extraction prompt)**

```python
import json

EXTRACTION_PROMPT_VERSION = "extract-v1"

EXTRACTION_PROMPT = """You extract checkable facts from an excerpt of a document.

Return JSON: {{"facts": [...]}}. Each fact has:
  subject         the entity the fact is about, exactly as the text names it
  metric          what is being asserted or measured, in the text's own words
  value_raw       the number exactly as printed, or null for non-numeric facts
  unit_raw        the unit and scale as printed, e.g. "Rs in Million", "per cent"
  period_raw      the period as printed, e.g. "FY24", "year ended March 31, 2024"
  qualifiers      an object for anything that changes what the number means:
                  basis (standalone/consolidated), vintage (estimate/actual/projection),
                  segment, scope, geography, attribution. Omit keys that do not apply.
                  Invent new keys when the text implies a distinction not listed here.
  claim_type      measurement | estimate | projection | attribute | event
  evidence_quote  a span copied VERBATIM from the excerpt that states this fact
  confidence      0 to 1

Rules:
- evidence_quote must appear character-for-character in the excerpt. Never paraphrase it.
- Extract only what the excerpt states. Do not infer, compute, or combine numbers.
- Prefer several precise facts over one broad one.
- A table row is a fact per cell when the column header gives it distinct meaning.
- Skip navigation text, page furniture, and legal disclaimers.

EXCERPT:
{window}
"""

def build_extraction_prompt(window_text: str) -> str:
    return EXTRACTION_PROMPT.format(window=window_text)
```

- [ ] **Step 4: Implement extract.py**

```python
import re
from .models import Fact, Window
from .llm.prompts import build_extraction_prompt, EXTRACTION_PROMPT_VERSION

_WS = re.compile(r"\s+")

def locate_quote(haystack: str, quote: str) -> tuple[int, int] | None:
    """Find quote in haystack, tolerating differences in whitespace only."""
    if not quote:
        return None
    idx = haystack.find(quote)
    if idx >= 0:
        return idx, idx + len(quote)
    # rebuild an index that maps collapsed positions back to original ones
    positions, collapsed = [], []
    prev_space = False
    for i, ch in enumerate(haystack):
        if ch.isspace():
            if prev_space:
                continue
            collapsed.append(" ")
            positions.append(i)
            prev_space = True
        else:
            collapsed.append(ch)
            positions.append(i)
            prev_space = False
    flat = "".join(collapsed)
    needle = _WS.sub(" ", quote).strip()
    j = flat.find(needle)
    if j < 0:
        return None
    start = positions[j]
    end = positions[min(j + len(needle) - 1, len(positions) - 1)] + 1
    return start, end

def extract_facts(client, window: Window, doc_id: int
                  ) -> tuple[list[Fact], list[dict]]:
    prompt = build_extraction_prompt(window.text)
    data = client.complete_json(prompt, EXTRACTION_PROMPT_VERSION)
    accepted: list[Fact] = []
    rejected: list[dict] = []
    for raw in data.get("facts", []):
        quote = (raw.get("evidence_quote") or "").strip()
        span = locate_quote(window.text, quote)
        if span is None:
            rejected.append({"payload": raw,
                             "reason": "evidence_quote not found in source window"})
            continue
        if not (raw.get("subject") and raw.get("metric")):
            rejected.append({"payload": raw, "reason": "missing subject or metric"})
            continue
        fact = Fact(
            doc_id=doc_id,
            subject=str(raw["subject"]).strip(),
            metric=str(raw["metric"]).strip(),
            value_raw=raw.get("value_raw"),
            value_num=None,
            unit_raw=raw.get("unit_raw"),
            period_raw=raw.get("period_raw"),
            qualifiers=raw.get("qualifiers") or {},
            claim_type=raw.get("claim_type") or "measurement",
            evidence_quote=quote,
            confidence=float(raw.get("confidence") or 0.0),
        )
        fact.span = span            # consumed by the persistence layer
        accepted.append(fact)
    return accepted, rejected

def dedupe_facts(facts: list[Fact]) -> list[Fact]:
    """Collapse facts re-extracted from the overlap between windows.

    Windows overlap so that a fact straddling a boundary is not lost, but that
    means roughly a tenth of blocks are read twice and their facts arrive in
    duplicate. Left alone the twins pair with each other and register as
    corroborations, inflating the counts with a sentence agreeing with itself.

    Keyed per document, so the same fact appearing in a DIFFERENT document
    survives - that one is a real cross-document corroboration.
    """
    best: dict[tuple, Fact] = {}
    for f in facts:
        key = (f.doc_id, f.subject.strip().lower(), f.metric.strip().lower(),
               (f.value_raw or "").strip(), (f.period_raw or "").strip(),
               _WS.sub(" ", f.evidence_quote).strip().lower())
        current = best.get(key)
        if current is None or f.confidence > current.confidence:
            best[key] = f
    return list(best.values())
```


- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_extract.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/factlayer/llm/prompts.py src/factlayer/extract.py src/factlayer/models.py tests/test_extract.py
git commit -m "Extract facts and reject any whose quote is not in the source"
```

---

### Task 9: Persist facts and evidence, map spans back to pages

**Files:**
- Create: `src/factlayer/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `save_document(conn, sha, filename, title, pages) -> int`,
  `save_blocks(conn, doc_id, blocks) -> list[int]`,
  `save_fact(conn, fact, window, block_row_ids) -> int` which resolves the fact's span
  to a page number and bounding box, `save_gaps`, `save_rejected`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
from factlayer.db import connect, init_schema
from factlayer.models import Block, Fact, Window
from factlayer.store import save_document, save_blocks, save_fact

import pytest
from factlayer.ingest.segment import build_windows

def _save(conn, blocks):
    doc_id = save_document(conn, "sha", "t.pdf", "Test", 2)
    ids = save_blocks(conn, doc_id, blocks)
    window = build_windows(blocks, doc_id, window_chars=10_000, overlap=0)[0]
    return doc_id, ids, window

def _fact_at(doc_id, window, quote):
    fact = Fact(doc_id, "Co", "revenue", "81,415.38", None, "million", "FY24")
    fact.evidence_quote = quote
    at = window.text.index(quote)
    fact.span = (at, at + len(quote))
    return fact

def test_fact_evidence_resolves_to_page_and_bbox(tmp_path):
    conn = connect(tmp_path / "t.sqlite"); init_schema(conn)
    blocks = [Block(1, 0, "Revenue was 81,415.38 million", (10, 20, 300, 40)),
              Block(2, 0, "Unrelated text", (11, 21, 301, 41))]
    doc_id, ids, window = _save(conn, blocks)
    fact_id = save_fact(conn, _fact_at(doc_id, window, "81,415.38 million"), window, ids)
    row = conn.execute("SELECT * FROM evidence WHERE fact_id=?", (fact_id,)).fetchone()
    assert row["page_no"] == 1
    assert row["x0"] == 10 and row["y1"] == 40

def test_reuploading_a_document_returns_its_original_id(tmp_path):
    # Guards a silent corruption: INSERT OR IGNORE that ignores still leaves
    # lastrowid pointing at the connection's previous insert, so re-uploading
    # a PDF after ingesting another one would return the OTHER document's id
    # and file this document's blocks and facts under it.
    conn = connect(tmp_path / "t.sqlite"); init_schema(conn)
    first = save_document(conn, "sha-a", "a.pdf", "A", 10)
    again = save_document(conn, "sha-a", "a.pdf", "A", 10)
    other = save_document(conn, "sha-b", "b.pdf", "B", 5)
    after_other = save_document(conn, "sha-a", "a.pdf", "A", 10)
    assert again == first
    assert other != first
    assert after_other == first, "re-upload resolved to the wrong document"
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2


@pytest.mark.parametrize("first", [
    "Revenue was 81,415.38 million",                      # single line
    "Revenue from operations\nwas 81,415.38 million",      # newline inside block
    "Revenue\nfrom operations\nwas 81,415.38 million",     # several newlines
])
def test_evidence_page_is_correct_when_blocks_contain_newlines(tmp_path, first):
    # Guards the misattribution bug: splitting window.text on "\n" credits the
    # quote to a block on the wrong page, silently corrupting the evidence trail.
    conn = connect(tmp_path / "t.sqlite"); init_schema(conn)
    blocks = [Block(1, 0, first, (10, 20, 300, 40)),
              Block(2, 0, "Unrelated text", (11, 21, 301, 41))]
    doc_id, ids, window = _save(conn, blocks)
    fact_id = save_fact(conn, _fact_at(doc_id, window, "81,415.38 million"), window, ids)
    row = conn.execute("SELECT * FROM evidence WHERE fact_id=?", (fact_id,)).fetchone()
    assert row["page_no"] == 1, "quote came from page 1, not the following block"
    assert row["x0"] == 10 and row["y1"] == 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store.py -v`
Expected: FAIL, module missing

- [ ] **Step 3: Implement store.py**

```python
import json, sqlite3
from .models import Block, Fact, Window

def save_document(conn, sha, filename, title, pages) -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO documents(sha256, filename, title, page_count) "
        "VALUES (?,?,?,?)", (sha, filename, title, pages))
    conn.commit()
    # rowcount is 1 when the row was inserted and 0 when it was ignored.
    # Do NOT branch on lastrowid: when the insert is ignored it still reports
    # the connection's previous successful insert, so re-uploading a document
    # would return some other document's id and attach facts to the wrong file.
    if cur.rowcount:
        return cur.lastrowid
    return conn.execute("SELECT id FROM documents WHERE sha256=?", (sha,)).fetchone()["id"]

def save_blocks(conn, doc_id: int, blocks: list[Block]) -> list[int]:
    ids = []
    for b in blocks:
        cur = conn.execute(
            "INSERT INTO blocks(doc_id,page_no,block_index,text,x0,y0,x1,y1,"
            "is_boilerplate) VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, b.page_no, b.block_index, b.text, *b.bbox, int(b.is_boilerplate)))
        ids.append(cur.lastrowid)
    conn.commit()
    return ids

def _blocks_covering(window: Window, span: tuple[int, int]) -> list[int]:
    """Window block positions the span overlaps, using recorded spans.

    Do not derive these by splitting window.text on newlines: block text
    contains its own newlines, so the split silently misattributes the quote
    to a later block and reports the wrong page and bounding box.
    """
    return [pos for pos, (start, end) in enumerate(window.block_spans)
            if span[0] < end and span[1] > start]

def save_fact(conn, fact: Fact, window: Window, block_row_ids: list[int]) -> int:
    cur = conn.execute(
        "INSERT INTO facts(doc_id,subject,metric,value_raw,value_num,unit_raw,"
        "period_raw,qualifiers,claim_type,confidence,canon_value,canon_unit,"
        "period_start,period_end,period_kind,entity_id,metric_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (fact.doc_id, fact.subject, fact.metric, fact.value_raw, fact.value_num,
         fact.unit_raw, fact.period_raw, json.dumps(fact.qualifiers),
         fact.claim_type, fact.confidence, fact.canon_value, fact.canon_unit,
         fact.period_start, fact.period_end, fact.period_kind,
         fact.entity_id, fact.metric_id))
    fact_id = cur.lastrowid

    positions = _blocks_covering(window, fact.span) if fact.span else []
    row_ids = [block_row_ids[window.block_ids[p]] for p in positions
               if window.block_ids[p] < len(block_row_ids)]
    page_no, bbox = None, (None, None, None, None)
    if row_ids:
        # ORDER BY id because IN (...) does not preserve the order given.
        rows = conn.execute(
            f"SELECT id,page_no,x0,y0,x1,y1 FROM blocks WHERE id IN "
            f"({','.join('?' * len(row_ids))}) ORDER BY id", row_ids).fetchall()
        page_no = rows[0]["page_no"]
        # A quote can straddle a page break. Union the boxes only within the
        # page the quote starts on: merging a box at the foot of one page with
        # one at the head of the next produces a rectangle on neither page.
        same_page = [r for r in rows if r["page_no"] == page_no]
        bbox = (min(r["x0"] for r in same_page), min(r["y0"] for r in same_page),
                max(r["x1"] for r in same_page), max(r["y1"] for r in same_page))
    conn.execute(
        "INSERT INTO evidence(fact_id,quote,page_no,char_start,char_end,block_ids,"
        "x0,y0,x1,y1) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (fact_id, fact.evidence_quote, page_no,
         fact.span[0] if fact.span else None, fact.span[1] if fact.span else None,
         json.dumps(row_ids), *bbox))
    conn.commit()
    return fact_id

def save_gaps(conn, doc_id: int, gaps: list[tuple[int, str]]) -> None:
    conn.executemany("INSERT INTO gaps(doc_id,page_no,reason) VALUES (?,?,?)",
                     [(doc_id, p, r) for p, r in gaps])
    conn.commit()

def save_rejected(conn, doc_id: int, rejected: list[dict]) -> None:
    conn.executemany("INSERT INTO rejected_facts(doc_id,payload,reason) VALUES (?,?,?)",
                     [(doc_id, json.dumps(r["payload"]), r["reason"]) for r in rejected])
    conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_store.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/factlayer/store.py tests/test_store.py
git commit -m "Persist facts with evidence resolved back to page and bounding box"
```

---

### Task 10: Entity and metric canonicalisation

One batched model call over the distinct strings in the corpus. At this scale that beats
embedding clustering on accuracy, cost, and code size; embeddings remain the scale-up path.

**Files:**
- Create: `src/factlayer/normalize/canon.py`; extend `src/factlayer/llm/prompts.py`
- Test: `tests/test_canon.py`

**Interfaces:**
- Produces: `canonicalise(client, conn, kind, raw_terms) -> dict[str, tuple[str, str]]`
  mapping each raw term to `(canon_id, label)`. Results persist in `canon_terms`, so
  re-ingestion only sends terms never seen before.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canon.py
from factlayer.db import connect, init_schema
from factlayer.llm.cache import cache_key, put
from factlayer.llm.client import LLMClient
from factlayer.llm.prompts import build_canon_prompt, CANON_PROMPT_VERSION
from factlayer.normalize.canon import canonicalise

TERMS = ["revenue from services", "Revenue from Operations", "PTL freight tonnage"]

def test_synonymous_metrics_share_a_canonical_id(tmp_path):
    conn = connect(tmp_path / "t.sqlite"); init_schema(conn)
    prompt = build_canon_prompt("metric", TERMS)
    put(conn, cache_key("m", CANON_PROMPT_VERSION, prompt), {"groups": [
        {"canon_id": "revenue_from_operations", "label": "Revenue from operations",
         "members": ["revenue from services", "Revenue from Operations"]},
        {"canon_id": "ptl_freight_tonnage", "label": "PTL freight tonnage",
         "members": ["PTL freight tonnage"]}]})
    client = LLMClient(conn, api_key=None, model="m")
    mapping = canonicalise(client, conn, "metric", TERMS)
    assert mapping["revenue from services"][0] == mapping["Revenue from Operations"][0]
    assert mapping["PTL freight tonnage"][0] != mapping["revenue from services"][0]

def test_known_terms_are_not_resent(tmp_path):
    conn = connect(tmp_path / "t.sqlite"); init_schema(conn)
    conn.execute("INSERT INTO canon_terms VALUES ('metric','x','cid','X')")
    conn.commit()
    client = LLMClient(conn, api_key=None, model="m")   # no key: a call would raise
    assert canonicalise(client, conn, "metric", ["x"]) == {"x": ("cid", "X")}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_canon.py -v`
Expected: FAIL, module missing

- [ ] **Step 3: Add the canonicalisation prompt to prompts.py**

```python
CANON_PROMPT_VERSION = "canon-v1"

CANON_PROMPT = """Group the {kind} names below by whether they denote the same thing.

Two names belong together only if a reader of the source documents would treat them as
the same {kind}. Different scopes, segments, or reporting bases are the SAME {kind} —
those distinctions are recorded separately as qualifiers. Genuinely different subjects
stay apart.

Return JSON: {{"groups": [{{"canon_id": "snake_case_id", "label": "Readable label",
"members": ["...", "..."]}}]}}
Every input name must appear in exactly one group.

NAMES:
{terms}
"""

def build_canon_prompt(kind: str, terms: list[str]) -> str:
    listed = "\n".join(f"- {t}" for t in sorted(set(terms)))
    return CANON_PROMPT.format(kind=kind, terms=listed)
```

- [ ] **Step 4: Implement canon.py**

```python
from ..llm.prompts import build_canon_prompt, CANON_PROMPT_VERSION

def _known(conn, kind: str) -> dict[str, tuple[str, str]]:
    rows = conn.execute("SELECT raw, canon_id, label FROM canon_terms WHERE kind=?",
                        (kind,)).fetchall()
    return {r["raw"]: (r["canon_id"], r["label"]) for r in rows}

def canonicalise(client, conn, kind: str,
                 raw_terms: list[str]) -> dict[str, tuple[str, str]]:
    mapping = _known(conn, kind)
    pending = sorted({t for t in raw_terms if t and t not in mapping})
    if not pending:
        return {t: mapping[t] for t in raw_terms if t in mapping}

    prompt = build_canon_prompt(kind, pending)
    data = client.complete_json(prompt, CANON_PROMPT_VERSION)
    rows = []
    for group in data.get("groups", []):
        cid, label = group.get("canon_id"), group.get("label") or ""
        if not cid:
            continue
        for member in group.get("members", []):
            mapping[member] = (cid, label)
            rows.append((kind, member, cid, label))
    # anything the model dropped keeps its own identity rather than vanishing
    for term in pending:
        if term not in mapping:
            cid = term.lower().replace(" ", "_")
            mapping[term] = (cid, term)
            rows.append((kind, term, cid, term))
    conn.executemany(
        "INSERT OR REPLACE INTO canon_terms(kind,raw,canon_id,label) VALUES (?,?,?,?)",
        rows)
    conn.commit()
    return {t: mapping[t] for t in raw_terms if t in mapping}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_canon.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/factlayer/normalize/canon.py src/factlayer/llm/prompts.py tests/test_canon.py
git commit -m "Cluster entity and metric names into canonical ids from the corpus"
```

---

### Task 11: Candidate pair generation

**Files:**
- Create: `src/factlayer/pairing.py`
- Test: `tests/test_pairing.py`

**Interfaces:**
- Produces: `candidate_pairs(facts, max_per_fact) -> list[tuple[int, int]]` over indices
  into `facts`. Channels: shared `metric_id`, and token overlap on `subject + metric`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pairing.py
from factlayer.models import Fact
from factlayer.pairing import candidate_pairs

def _f(doc, metric_id, metric, entity="e"):
    f = Fact(doc, "Delhivery", metric, "1", 1.0, None, "FY24")
    f.metric_id, f.entity_id = metric_id, entity
    return f

def test_pairs_facts_sharing_a_canonical_metric():
    facts = [_f(1, "revenue", "revenue from services"),
             _f(2, "revenue", "Revenue from Operations"),
             _f(2, "tonnage", "PTL freight tonnage")]
    pairs = candidate_pairs(facts, max_per_fact=10)
    assert (0, 1) in pairs
    assert (0, 2) not in pairs

def test_pairs_within_one_document_are_allowed():
    facts = [_f(1, "revenue", "revenue standalone"),
             _f(1, "revenue", "revenue consolidated")]
    assert (0, 1) in candidate_pairs(facts, max_per_fact=10)

def test_lexical_channel_catches_missing_canonical_id():
    a, b = _f(1, None, "net working capital days"), _f(2, None, "net working capital days")
    assert (0, 1) in candidate_pairs([a, b], max_per_fact=10)

def test_different_entities_are_not_paired():
    a, b = _f(1, "revenue", "revenue", entity="delhivery"), _f(2, "revenue", "revenue", entity="india")
    assert candidate_pairs([a, b], max_per_fact=10) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pairing.py -v`
Expected: FAIL, module missing

- [ ] **Step 3: Implement pairing.py**

```python
import re
from collections import defaultdict
from .models import Fact

_STOP = {"of", "from", "the", "in", "for", "and", "a", "on", "to"}

def _tokens(fact: Fact) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", f"{fact.metric}".lower())
    return frozenset(w for w in words if w not in _STOP and len(w) > 2)

def candidate_pairs(facts: list[Fact], max_per_fact: int) -> list[tuple[int, int]]:
    by_metric: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(facts):
        if f.metric_id:
            by_metric[f.metric_id].append(i)

    scored: dict[tuple[int, int], float] = {}

    def offer(i: int, j: int, score: float) -> None:
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        if facts[a].entity_id and facts[b].entity_id and \
           facts[a].entity_id != facts[b].entity_id:
            return
        # incomparable units are noise, not disagreement: without this gate a
        # rupee figure pairs with a percentage purely on shared metric words
        if facts[a].canon_unit and facts[b].canon_unit and \
           facts[a].canon_unit != facts[b].canon_unit:
            return
        scored[(a, b)] = max(scored.get((a, b), 0.0), score)

    for members in by_metric.values():
        for i in members:
            for j in members:
                offer(i, j, 1.0)

    toks = [_tokens(f) for f in facts]
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            if not toks[i] or not toks[j]:
                continue
            overlap = len(toks[i] & toks[j]) / len(toks[i] | toks[j])
            if overlap >= 0.6:
                offer(i, j, overlap)

    kept: dict[int, int] = defaultdict(int)
    out = []
    for (a, b), _ in sorted(scored.items(), key=lambda kv: -kv[1]):
        if kept[a] >= max_per_fact or kept[b] >= max_per_fact:
            continue
        kept[a] += 1
        kept[b] += 1
        out.append((a, b))
    return sorted(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pairing.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/factlayer/pairing.py tests/test_pairing.py
git commit -m "Generate candidate fact pairs from canonical and lexical channels"
```

---

### Task 12: Deterministic verdict rules

**Files:**
- Create: `src/factlayer/reconcile/__init__.py`, `src/factlayer/reconcile/rules.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Produces: `qualifier_diff(a, b) -> dict[str, tuple]` covering the union of qualifier
  keys plus period and unit; `rule_verdict(a, b, tol) -> tuple[str, dict]` returning one
  of `corroborates`, `corroborates_with_caveat`, `reconcilable`,
  `contradiction_candidate`, `needs_model`.

No qualifier key list is hard-coded: any key present on either side with differing values
counts as a difference.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rules.py
from factlayer.models import Fact
from factlayer.reconcile.rules import rule_verdict, qualifier_diff

def _f(value, unit="INR", period=("2023-04-01", "2024-03-31"), quals=None,
       claim="measurement"):
    f = Fact(1, "Delhivery", "revenue", None, None, None, "FY24",
             qualifiers=quals or {}, claim_type=claim)
    f.canon_value, f.canon_unit = value, unit
    f.period_start, f.period_end = period
    return f

def test_same_value_same_qualifiers_corroborates():
    v, _ = rule_verdict(_f(8.142e10), _f(8.1415e10), tol=1e-3)
    assert v == "corroborates"

def test_same_value_different_basis_is_caveated():
    v, diff = rule_verdict(_f(8.142e10, quals={"basis": "consolidated"}),
                           _f(8.1415e10, quals={"basis": "standalone"}), tol=1e-3)
    assert v == "corroborates_with_caveat"
    assert "basis" in diff

def test_different_value_one_qualifier_differs_is_reconcilable():
    v, diff = rule_verdict(_f(7.454082e10, quals={"basis": "standalone"}),
                           _f(8.141538e10, quals={"basis": "consolidated"}), tol=1e-3)
    assert v == "reconcilable"
    assert list(diff) == ["basis"]

def test_different_value_same_qualifiers_is_contradiction_candidate():
    v, _ = rule_verdict(_f(6.5, "PERCENT"), _f(6.6, "PERCENT"), tol=1e-3)
    assert v == "contradiction_candidate"

def test_different_period_is_a_difference():
    _, diff = rule_verdict(_f(8.1e10), _f(7.2e10, period=("2022-04-01", "2023-03-31")),
                           tol=1e-3)
    assert "period" in diff

def test_attribute_facts_always_go_to_the_model():
    v, _ = rule_verdict(_f(None, None, claim="attribute"),
                        _f(None, None, claim="attribute"), tol=1e-3)
    assert v == "needs_model"

def test_unknown_period_is_not_a_contradiction():
    # An absent period means unknown, not 'same period as the other fact'.
    # Treating None == None as a match manufactured 1,480 false contradictions
    # across two starter documents (45% of all pairs).
    a = _f(8.1e10, period=(None, None))
    b = _f(7.2e10, period=(None, None))
    verdict, _ = rule_verdict(a, b, tol=1e-3)
    assert verdict == 'insufficient_context'

def test_one_known_period_is_still_insufficient():
    a = _f(8.1e10)
    b = _f(7.2e10, period=(None, None))
    assert rule_verdict(a, b, tol=1e-3)[0] == 'insufficient_context'

def test_incomparable_units_are_unrelated_not_disputed():
    a, b = _f(8.1e10, 'INR'), _f(6.5, 'PERCENT')
    assert rule_verdict(a, b, tol=1e-3)[0] == 'unrelated'

def test_known_periods_still_reach_a_contradiction():
    # the RBI vs IMF growth case must survive the guard above
    a = _f(6.5, 'PERCENT', period=('2025-04-01', '2026-03-31'))
    b = _f(6.6, 'PERCENT', period=('2025-04-01', '2026-03-31'))
    assert rule_verdict(a, b, tol=1e-3)[0] == 'contradiction_candidate'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rules.py -v`
Expected: FAIL, module missing

- [ ] **Step 3: Implement rules.py**

```python
from ..models import Fact
from ..normalize.units import values_agree

def qualifier_diff(a: Fact, b: Fact) -> dict[str, tuple]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rules.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/factlayer/reconcile tests/test_rules.py
git commit -m "Decide fact pairs by rule where values and qualifiers settle it"
```

---

### Task 13: Model adjudication and arithmetic verification

**Files:**
- Create: `src/factlayer/reconcile/adjudicate.py`, `src/factlayer/reconcile/verify.py`;
  extend `src/factlayer/llm/prompts.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Produces: `adjudicate(client, a, b, rule_verdict, diff, a_meta, b_meta) -> dict`
  with keys `verdict`, `reason_code`, `explanation`, `claimed_transform`,
  `confidence`, where `a_meta`/`b_meta` carry `filename` and `page_no` for the
  prompt;
  `verify(a, b, claimed_transform, tol) -> tuple[bool, str]`.

Recognised transforms: `{"kind": "scale", "factor": 100}`, `{"kind": "basis"}`,
`{"kind": "vintage"}`, `{"kind": "period"}`, `{"kind": "none"}`. Only `scale` is
arithmetically checkable; the others are checked for the qualifier actually differing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify.py
from factlayer.models import Fact
from factlayer.reconcile.verify import verify

def _f(value, quals=None, unit="INR"):
    f = Fact(1, "Delhivery", "revenue", None, None, None, "FY24", qualifiers=quals or {})
    f.canon_value, f.canon_unit = value, unit
    f.period_start, f.period_end = "2023-04-01", "2024-03-31"
    return f

def test_true_scale_claim_is_confirmed():
    # The realistic case: the text omitted the scale word, so normalisation left
    # the pair a hundred-fold apart and the model proposes the missing factor.
    # (A pair that already agrees never reaches the model - the rules settle it.)
    ok, note = verify(_f(8.142e10), _f(8.141538e8),
                      {"kind": "scale", "factor": 100}, tol=1e-3)
    assert ok and "factor of 100" in note

def test_false_scale_claim_is_rejected():
    # standalone vs consolidated revenue: no scale factor reconciles these
    ok, note = verify(_f(7.454082e10), _f(8.141538e10),
                      {"kind": "scale", "factor": 100}, tol=1e-3)
    assert not ok and "do not reconcile" in note

def test_basis_claim_requires_basis_to_actually_differ():
    ok, _ = verify(_f(7.4e10, {"basis": "standalone"}),
                   _f(8.1e10, {"basis": "consolidated"}), {"kind": "basis"}, tol=1e-3)
    assert ok
    ok2, note = verify(_f(7.4e10, {"basis": "consolidated"}),
                       _f(8.1e10, {"basis": "consolidated"}), {"kind": "basis"}, tol=1e-3)
    assert not ok2 and "basis" in note

def test_unknown_transform_is_not_verifiable():
    ok, note = verify(_f(1.0), _f(2.0), {"kind": "vibes"}, tol=1e-3)
    assert not ok and "unrecognised" in note
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_verify.py -v`
Expected: FAIL, module missing

- [ ] **Step 3: Implement verify.py**

```python
from ..models import Fact
from ..normalize.units import values_agree

def verify(a: Fact, b: Fact, claimed: dict | None, tol: float) -> tuple[bool, str]:
    if not claimed or "kind" not in claimed:
        return False, "no transform claimed; nothing to verify"
    kind = claimed["kind"]

    if kind == "scale":
        factor = float(claimed.get("factor") or 1)
        if a.canon_value is None or b.canon_value is None:
            return False, "missing canonical values"
        if values_agree(a.canon_value * factor, b.canon_value, tol) or \
           values_agree(a.canon_value, b.canon_value * factor, tol):
            return True, f"values reconcile under a factor of {factor:g}"
        return False, (f"values do not reconcile under a factor of {factor:g}: "
                       f"{a.canon_value:g} vs {b.canon_value:g}")

    if kind in {"basis", "vintage", "period", "scope", "segment"}:
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
```

- [ ] **Step 4: Add the adjudication prompt to prompts.py**

```python
ADJUDICATE_PROMPT_VERSION = "adjudicate-v1"

ADJUDICATE_PROMPT = """Two facts were extracted from documents. Decide how they relate.

FACT A: {a}
  evidence: "{a_quote}"  (document {a_doc}, page {a_page})
FACT B: {b}
  evidence: "{b_quote}"  (document {b_doc}, page {b_page})

A deterministic check found: values {agree_text}; differing attributes: {diff}

Return JSON:
  verdict           corroborates | contradicts | reconciled_by_context | unrelated
  reason_code       short snake_case tag, e.g. unit_scale, reporting_basis,
                    data_vintage, different_period, genuine_disagreement
  explanation       two sentences at most, citing what in the evidence decides it
  claimed_transform if reconciled_by_context, the transformation that makes them
                    consistent: {{"kind": "scale", "factor": N}} for a pure scale
                    change, otherwise {{"kind": "basis"|"vintage"|"period"|"scope"|
                    "segment"}}. Use {{"kind": "none"}} when no transform applies.
  confidence        0 to 1

Judge only from the evidence shown. If the two facts are about different things, say
unrelated rather than forcing a relation.
"""
```

with a helper whose signature the adjudicator below calls exactly:

```python
def build_adjudicate_prompt(a_summary: str, b_summary: str, a_quote: str, b_quote: str,
                            a_meta: dict, b_meta: dict, diff: dict,
                            agree_text: str) -> str:
    return ADJUDICATE_PROMPT.format(
        a=a_summary, b=b_summary, a_quote=a_quote, b_quote=b_quote,
        a_doc=a_meta.get("filename", "?"), a_page=a_meta.get("page_no", "?"),
        b_doc=b_meta.get("filename", "?"), b_page=b_meta.get("page_no", "?"),
        diff=json.dumps(diff, sort_keys=True, default=str) or "none",
        agree_text=agree_text)
```

- [ ] **Step 5: Implement adjudicate.py**

```python
import json
from ..llm.prompts import build_adjudicate_prompt, ADJUDICATE_PROMPT_VERSION

def _summary(f) -> str:
    parts = [f.subject, f.metric]
    if f.value_raw:
        parts.append(f"{f.value_raw} {f.unit_raw or ''}".strip())
    if f.period_raw:
        parts.append(f"period {f.period_raw}")
    if f.qualifiers:
        parts.append(json.dumps(f.qualifiers, sort_keys=True))
    return " | ".join(p for p in parts if p)

def adjudicate(client, a, b, rule_verdict: str, diff: dict,
               a_meta: dict, b_meta: dict) -> dict:
    prompt = build_adjudicate_prompt(
        _summary(a), _summary(b), a.evidence_quote, b.evidence_quote,
        a_meta, b_meta, diff,
        "agree" if rule_verdict.startswith("corroborates") else "differ")
    data = client.complete_json(prompt, ADJUDICATE_PROMPT_VERSION)
    return {
        "verdict": data.get("verdict") or "unrelated",
        "reason_code": data.get("reason_code") or "",
        "explanation": data.get("explanation") or "",
        "claimed_transform": data.get("claimed_transform"),
        "confidence": float(data.get("confidence") or 0.0),
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_verify.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add src/factlayer/reconcile src/factlayer/llm/prompts.py tests/test_verify.py
git commit -m "Adjudicate ambiguous pairs and re-derive every claimed transform"
```

---

### Task 14 (droppable): Embedding-accelerated pairing

Only build this if Tasks 1–13 and 15–16 are done. Adds a third recall channel using
Gemini embeddings over `subject + metric`, cached like every other call, with brute-force
cosine in numpy. Falls back silently to the lexical channels when no key is available, so
nothing downstream changes.

**Files:** Create `src/factlayer/embeddings.py`; modify `src/factlayer/pairing.py` to
accept an optional `vectors` argument and union its neighbours into `scored`.
**Test:** `tests/test_embeddings.py` asserting that with stub vectors, two facts whose
metrics share no tokens are still paired.

---

### Task 15: Pipeline orchestration and API

**Files:**
- Create: `src/factlayer/pipeline.py`, `src/factlayer/api.py`
- Test: `tests/test_pipeline.py`, `tests/test_api.py`

**Interfaces:**
- Produces: `ingest(conn, client, pdf_path, job_id=None, canonicalise_terms=True) -> int`
  running every stage in order and returning the document id;
  `build_relations(conn, client) -> int`; `app` (FastAPI).

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/test_pipeline.py
from factlayer.db import connect, init_schema
from factlayer.llm.cache import cache_key, put
from factlayer.llm.client import LLMClient
from factlayer.llm.prompts import build_extraction_prompt, EXTRACTION_PROMPT_VERSION
from factlayer.pipeline import ingest

def test_ingest_stores_grounded_facts(tmp_path, make_pdf):
    line = "Revenue from operations for FY24 stood at Rs 81,415.38 million."
    pdf = make_pdf([[line]])
    conn = connect(tmp_path / "t.sqlite"); init_schema(conn)
    client = LLMClient(conn, api_key=None, model="m")

    from factlayer.ingest.pdf import extract_blocks
    from factlayer.ingest.segment import build_windows
    blocks, _ = extract_blocks(pdf)
    window = build_windows(blocks, 1, 12000, 1200)[0]
    put(conn, cache_key("m", EXTRACTION_PROMPT_VERSION,
                        build_extraction_prompt(window.text)),
        {"facts": [{"subject": "Delhivery Limited",
                    "metric": "revenue from operations",
                    "value_raw": "81,415.38", "unit_raw": "Rs million",
                    "period_raw": "FY24", "qualifiers": {"basis": "consolidated"},
                    "claim_type": "measurement",
                    "evidence_quote": "Rs 81,415.38 million", "confidence": 0.9}]})

    doc_id = ingest(conn, client, pdf, canonicalise_terms=False)
    row = conn.execute("SELECT * FROM facts WHERE doc_id=?", (doc_id,)).fetchone()
    assert row["canon_value"] == 8.141538e10
    assert row["canon_unit"] == "INR"
    assert row["period_start"] == "2023-04-01"
    ev = conn.execute("SELECT * FROM evidence WHERE fact_id=?", (row["id"],)).fetchone()
    assert ev["page_no"] == 1 and "81,415.38" in ev["quote"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL, module missing

- [ ] **Step 3: Implement pipeline.py**

Wire the stages in order: `extract_blocks` → `mark_boilerplate` → `find_gaps` →
`save_document` / `save_blocks` / `save_gaps` → `build_windows` sorted by `-density` →
`extract_facts` per window → `dedupe_facts` across the whole document → normalise each
fact with `normalize_value` and `normalize_period` → `canonicalise` over the distinct
subjects and metrics (skipped when `canonicalise_terms=False`) → `save_fact` /
`save_rejected`. Update the `jobs` row after each window so progress is observable.

Isolate each window, or one bad response destroys a hundred pages of work:

```python
from .llm.client import BadModelJSON, NoAPIKey

for window in windows:
    try:
        got, bad = extract_facts(client, window, doc_id)
    except NoAPIKey:
        raise                      # nothing is cached and there is no key: stop
    except BadModelJSON as exc:
        rejected.append({"payload": {"window_index": window.index},
                         "reason": f"unusable model output: {exc}"})
        continue                   # this window yields nothing; the rest proceed
    facts.extend(got)
    rejected.extend(bad)
    _bump_job(conn, job_id, done=window.index + 1, facts=len(facts))
```

`NoAPIKey` propagates deliberately: it means the run cannot proceed at all, and
failing loudly beats writing an empty knowledge layer that looks like a result.

`build_relations` loads all facts, calls `candidate_pairs`, applies `rule_verdict`, calls
`adjudicate` for anything not settled by rule, runs `verify` on any claimed transform,
sets `final_verdict` (downgrading to `needs_review` when verification fails or rule and
model disagree), and writes the `relations` row.

- [ ] **Step 4: Implement api.py**

Routes exactly as listed in `docs/design.md`. Upload writes the file to
`settings.upload_dir`, creates a `jobs` row, and runs `ingest` then `build_relations` in
a `BackgroundTasks` callback. Every read route returns plain JSON built from SQL joins.

Resolve the database per request rather than binding it at import, or the API tests
cannot redirect it — `settings` is a module-level singleton evaluated once at import
time, so a `monkeypatch.setenv` that lands after the first import would be ignored:

```python
import os, functools
from .config import settings
from .db import connect, init_schema

@functools.lru_cache(maxsize=8)
def _conn_for(path: str):
    conn = connect(path)
    init_schema(conn)
    return conn

def get_conn():
    return _conn_for(os.getenv("FACTLAYER_DB", str(settings.db_path)))
```

Upload must reject anything that is not a PDF with HTTP 400, by both content type and
magic bytes (`%PDF`), since the test asserts that and a mislabelled upload would
otherwise crash the ingest worker.

- [ ] **Step 5: Write the API test**

```python
# tests/test_api.py
from fastapi.testclient import TestClient

def test_health_and_empty_listing(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTLAYER_DB", str(tmp_path / "t.sqlite"))
    from factlayer.api import app
    client = TestClient(app)
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/documents").json() == []

def test_upload_rejects_non_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTLAYER_DB", str(tmp_path / "t.sqlite"))
    from factlayer.api import app
    client = TestClient(app)
    r = client.post("/api/documents", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 400
```

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/factlayer/pipeline.py src/factlayer/api.py tests/test_pipeline.py tests/test_api.py
git commit -m "Wire the pipeline end to end and expose it over HTTP"
```

---

### Task 16: UI

**Files:**
- Create: `src/factlayer/templates/{base,documents,facts,relation,gaps}.html`,
  `src/factlayer/static/app.css`; modify `src/factlayer/api.py` to mount templates.

Four screens, htmx for interaction, no build step. The relation detail screen is the one
that matters: two evidence quotes side by side, each with document and page, canonical
values beneath each, the rule verdict, the model's explanation, whether verification
passed, and whether rule and model agreed. Render `needs_review` prominently rather than
hiding it.

- [ ] **Step 1: Build `base.html` with a nav across the four screens and `app.css`.**
- [ ] **Step 2: Build `documents.html` — upload form posting to `/api/documents`, list
      with page counts, fact counts, and gap counts; poll `/api/jobs/{id}` with
      `hx-trigger="every 2s"` while a job is running.**
- [ ] **Step 3: Build `facts.html` — filter controls for entity, metric, period and
      document; a table showing value, canonical value, unit, period, basis, source
      document and page; each row links to its fact detail.**
- [ ] **Step 4: Build `relation.html` — the side-by-side view described above.**
- [ ] **Step 5: Build `gaps.html` — unreadable pages and rejected facts with reasons.**
- [ ] **Step 6: Check every screen renders against a populated database.**

Run: `uvicorn factlayer.api:app --reload` and walk all four screens.

- [ ] **Step 7: Commit**

```bash
git add src/factlayer/templates src/factlayer/static src/factlayer/api.py
git commit -m "Add web screens for documents, facts, relations and gaps"
```

---

### Task 17: Ingest the starter documents and commit the cache

**Files:**
- Create: `scripts/ingest_starter.py`

- [ ] **Step 1: Write the script** — walk a directory given on the command line, call
      `ingest` for each PDF, then `build_relations` once at the end. Print a per-document
      summary of facts found, facts rejected, and gaps.
- [ ] **Step 2: Run it against `../starter-datasets/starter-datasets` with a real key.**
- [ ] **Step 3: Confirm all four cases appear.** Query the relations table for a
      `corroborates` spanning two documents, a `contradicts`, a `reconciled_by_context`,
      and check the gaps table contains the IMF cover page.
- [ ] **Step 4: Commit the populated cache** so the project runs without a key.

```bash
git add scripts/ingest_starter.py cache/starter_cache.sqlite   # .gitignore negates this path
git commit -m "Ingest starter documents and commit the response cache"
```

- [ ] **Step 5: Verify the key-free path** — move your `.env` aside, delete the working
      database, restore from the committed cache, re-run, and confirm the same facts and
      relations appear with no network access.

---

### Task 18: README

**Files:** Create `README.md`

Sections required by the brief, in this order: **Setup and Run Instructions**,
**Video Demo**, **Approach**, **Limitations and Next Steps**, **Additional Notes**.

- [ ] **Step 1: Setup** — clone, `pip install -e ".[dev]"`, `uvicorn factlayer.api:app`,
      note that the committed cache means no API key is needed to see the starter results.
- [ ] **Step 2: Approach** — the propose-and-verify split, why the fact schema is a fixed
      core plus an open qualifier map, why SQLite and not a graph store, and which AI
      tools were used.
- [ ] **Step 3: The four cases** — each with its evidence quotes, both documents and page
      numbers, and the system's reasoning. Screenshots from the relation screen.
- [ ] **Step 4: Limitations** — copy the spec's limitations section, including run-to-run
      variance and unmeasured recall.
- [ ] **Step 5: Commit.**

```bash
git add README.md
git commit -m "Add README with setup, approach and the four demonstrated cases"
```

---

## Self-Review Notes

### Fourth review round

This pass built the unwritten tasks as a throwaway harness and ran the whole pipeline
against two real starter documents with a stubbed model. That surfaced the most serious
problems found in any round, including a claim in the design document that was simply
false.

13. **The cost argument was wrong.** The design asserted that pairs agreeing outright
    are settled without a model call, and that this is "most of what makes a free tier
    workable". Measured: **0.5%** of pairs were settled by rule and **99.5% would have
    reached the model** - 3,274 adjudications on two documents alone, which does not
    fit any free tier. The design has been corrected rather than quietly patched.
14. **A missing period read as a matching period.** `qualifier_diff` compares
    `period_start` with `==`, so two undated facts looked contemporaneous and any
    difference in their values became a contradiction. This produced **1,480 false
    contradictions, 45% of all pairs**. An absent period now yields
    `insufficient_context`: the pair is recorded and visible, but no relationship is
    claimed and no quota is spent. This one fix cut adjudication calls fourfold.
15. **Incomparable units were being compared.** A rupee figure paired with a percentage
    whenever the metric words overlapped. Pairing now gates on canonical unit, and
    differing units resolve to `unrelated` rather than being escalated.
16. **`max_pairs_per_fact` was too generous.** 12 produced 3,291 pairs from 553 facts
    with no gain in the cases that matter; 6 halves the budget. Now 6.

Together these take adjudication from 3,274 pairs to 460 - a sevenfold reduction - while
leaving all four required cases intact, since each carries an explicit period on both
sides and is therefore untouched by the period guard.

**Verified working against real documents**, not fixtures: the pipeline ingested 100 and
27 page PDFs, stored 553 facts, and a grounding audit re-checked every stored quote
against the text of the page it claimed. **553 of 553 were found on their claimed page**,
which confirms the block-span fix from the first round holds on real input. Every fact
also carried a page number and bounding box.

**Left as a known limitation:** period coverage. Only 39% of facts in the harness run
carried a parseable period. A real model should do better than the stub's regex, but the
design must not assume high coverage, and `insufficient_context` is what keeps low
coverage honest instead of dangerous. Worth reporting in the README as a measured number
once the real ingest runs.

### Third review round

Four more, found by probing failure modes rather than the happy path:

9.  **Overlapping windows produced duplicate facts that corroborated themselves.**
    Windows overlap by design so a fact straddling a boundary is not lost, but
    measured against the real annual report that puts 11.6% of blocks in two
    windows. Their facts arrive twice, pair with each other, and register as
    corroborations - the demo would show an inflated count of a sentence agreeing
    with itself. Added `dedupe_facts`, keyed per document so the same fact in a
    *different* document still survives, since that one is a real corroboration.
10. **No rate-limit handling at all.** The whole design rests on the free tier,
    yet the client had no retry, so the first 429 during a 158-call ingest would
    abort the run. Added bounded exponential backoff on quota and transient
    server errors. Malformed JSON is deliberately not retried: temperature is
    zero, so the model reproduces it and retrying only burns quota.
11. **A truncated response destroyed the whole document.** `_loads_lenient` raised
    a bare `JSONDecodeError` that propagated out of extraction, so one over-long
    window lost all hundred pages. It now raises a named `BadModelJSON`, the
    output token ceiling is set explicitly so truncation is rarer, and the
    pipeline isolates each window and records the failure instead of aborting.
    `NoAPIKey` still propagates on purpose - that means the run cannot proceed,
    and failing loudly beats writing an empty knowledge layer that looks like a
    result.
12. **A verification test asserted something that cannot happen.** It claimed a
    scale factor of 1 on values that already agree, but such a pair never reaches
    the model at all - the rules settle it. Rewritten around the real scenario,
    where the text omitted the scale word and the model proposes the missing
    hundred-fold factor.

### Second review round

Three further defects found by executing the plan's code, two of them silent
corruptions:

6. **Re-uploading a document returned the wrong id.** `save_document` branched on
   `cur.lastrowid`, but an `INSERT OR IGNORE` that ignores its row still reports the
   connection's previous successful insert. Verified: ingesting `a.pdf`, then `b.pdf`,
   then `a.pdf` again returned id 2 for the third call instead of 1, which would file
   one document's blocks and facts under another. Now branches on `cur.rowcount`, with
   a regression test.
7. **Evidence bounding boxes merged across page breaks.** A quote straddling a page
   boundary unioned a box at the foot of one page with one at the head of the next,
   producing a rectangle that exists on neither, and `rows[0]` picked an arbitrary page
   because `IN (...)` does not preserve order. Now ordered, with the box confined to
   the page the quote starts on.
8. **`prompts.py` was missing `import json`**, which `build_adjudicate_prompt` needs;
   and the Task 13 interface line omitted the `a_meta`/`b_meta` arguments the
   implementation actually takes.

Also hardened: SQLite now sets `busy_timeout`, because ingest writes from a background
thread while the UI polls for progress and the reader would otherwise fail immediately
rather than wait.

Measured and found acceptable, no change made: candidate pairing is O(n^2), which runs
in 0.39s at the ~1,250 facts the six starter documents produce and 2.2s at 3,000. It
degrades quadratically, so the "many PDFs in one layer" extension would need the
blocking rewritten before it scales much further.

### First review round

Five defects were found by executing the plan's own code rather than reading it, and
have been fixed above:

1. **Evidence was attributed to the wrong block.** `_blocks_covering` recovered block
   boundaries by splitting the window on newlines, but PDF block text contains its own
   newlines. Verified: with a two-line block, the quote was credited to the *following*
   block, giving the wrong page and bounding box. The plan's original test passed
   because its fixture blocks were single-line. `Window.block_spans` now records
   boundaries at build time, and `tests/test_store.py` parametrises over multi-line
   blocks so the bug cannot return.
2. **`.gitignore` blocked the response cache.** `*.sqlite` matched
   `cache/starter_cache.sqlite`, so Task 17 would have appeared to succeed while
   committing nothing — silently breaking the run-without-a-key path the brief requires.
   A negation now exempts that one file; stray working databases stay ignored.
3. **Task 13's assertion never matched its own message.** The test looked for "does not
   reconcile" against a message reading "do not reconcile".
4. **`FACTLAYER_DB` was read by the API tests but never by `config.py`,** so the tests
   would have written to the real database. Config now reads it, and the API resolves
   its connection per request instead of at import.
5. **Two wrong expected-test counts** (Task 5 said 11, actual 12; Task 13 said 5, actual
   4), plus a stale "add this field" note left over from an earlier revision.

The units and periods suites in Tasks 5 and 6 were executed as written: 12 passed and
12 passed respectively, including the crore-to-million reconciliation that case 1
depends on and the `FY2025/26` equals `FY26` equivalence that case 3 depends on.

Known and accepted: Tasks 15 and 16 specify prose plus interface contracts rather than
complete code for the FastAPI routes and Jinja templates. That is a deviation from the
"no placeholders" rule. It is acceptable here only because the same session that wrote
this plan is executing it and holds the full context; a cold executor would need those
two tasks expanded first.

Checked against `docs/design.md`:

- Every spec stage maps to a task: ingest and segment (2–4), extraction and grounding
  (8), normalisation (5, 6, 10), pairing (11, 14), adjudication (12, 13), verification
  (13), storage (1, 9), API and UI (15, 16).
- The spec's "run without a key" requirement is Task 17 step 5; the committed cache is
  Task 17 step 4.
- The four demo cases are verified in Task 17 step 3 and written up in Task 18 step 3.
- Names used across tasks are consistent: `normalize_value`, `normalize_period`,
  `canonicalise`, `candidate_pairs`, `rule_verdict`, `qualifier_diff`, `adjudicate`,
  `verify`, `extract_facts`, `locate_quote`, `ingest`, `build_relations`.
- One gap accepted deliberately: the spec mentions embeddings as a pairing channel, and
  that is Task 14, which sits below the cut line. The lexical and canonical-id channels
  cover the required cases without it.
