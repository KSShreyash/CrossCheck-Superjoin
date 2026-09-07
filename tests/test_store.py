import pytest

from factlayer.db import connect, init_schema
from factlayer.ingest.segment import build_windows
from factlayer.models import Block, Fact
from factlayer.store import save_blocks, save_document, save_fact, save_gaps


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
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
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
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    first = save_document(conn, "sha-a", "a.pdf", "A", 10)
    again = save_document(conn, "sha-a", "a.pdf", "A", 10)
    other = save_document(conn, "sha-b", "b.pdf", "B", 5)
    after_other = save_document(conn, "sha-a", "a.pdf", "A", 10)
    assert again == first
    assert other != first
    assert after_other == first, "re-upload resolved to the wrong document"
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2


@pytest.mark.parametrize("first", [
    "Revenue was 81,415.38 million",                       # single line
    "Revenue from operations\nwas 81,415.38 million",       # newline inside block
    "Revenue\nfrom operations\nwas 81,415.38 million",      # several newlines
])
def test_evidence_page_is_correct_when_blocks_contain_newlines(tmp_path, first):
    # Guards the misattribution bug: splitting window.text on "\n" credits the
    # quote to a block on the wrong page, silently corrupting the evidence trail.
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    blocks = [Block(1, 0, first, (10, 20, 300, 40)),
              Block(2, 0, "Unrelated text", (11, 21, 301, 41))]
    doc_id, ids, window = _save(conn, blocks)
    fact_id = save_fact(conn, _fact_at(doc_id, window, "81,415.38 million"), window, ids)
    row = conn.execute("SELECT * FROM evidence WHERE fact_id=?", (fact_id,)).fetchone()
    assert row["page_no"] == 1, "quote came from page 1, not the following block"
    assert row["x0"] == 10 and row["y1"] == 40


def test_gaps_are_recorded_against_the_document(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    doc_id = save_document(conn, "sha", "t.pdf", "T", 3)
    save_gaps(conn, doc_id, [(1, "no extractable text; page is likely image-only")])
    row = conn.execute("SELECT * FROM gaps WHERE doc_id=?", (doc_id,)).fetchone()
    assert row["page_no"] == 1 and "image-only" in row["reason"]
