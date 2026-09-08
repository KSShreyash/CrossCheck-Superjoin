from factlayer.db import connect, init_schema
from factlayer.ingest.pdf import extract_blocks
from factlayer.ingest.segment import build_windows
from factlayer.llm.cache import cache_key, put
from factlayer.llm.client import LLMClient
from factlayer.llm.prompts import EXTRACTION_PROMPT_VERSION, build_extraction_prompt
from factlayer.pipeline import build_relations, ingest

LINE = "Revenue from operations for FY24 stood at Rs 81,415.38 million."
OTHER = "Revenue from operations for FY24 stood at Rs 8,142 Cr."


def _seed(conn, pdf, facts):
    blocks, _ = extract_blocks(pdf)
    window = build_windows(blocks, 1, 12000, 1200)[0]
    put(conn, cache_key("m", EXTRACTION_PROMPT_VERSION,
                        build_extraction_prompt(window.text)), {"facts": facts})


def _fact(value, unit, quote, basis="consolidated"):
    return {"subject": "Delhivery Limited", "metric": "revenue from operations",
            "value_raw": value, "unit_raw": unit, "period_raw": "FY24",
            "qualifiers": {"basis": basis}, "claim_type": "measurement",
            "evidence_quote": quote, "confidence": 0.9}


def test_ingest_stores_grounded_and_normalised_facts(tmp_path, make_pdf):
    pdf = make_pdf([[LINE]])
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    client = LLMClient(conn, api_key=None, model="m")
    _seed(conn, pdf, [_fact("81,415.38", "Rs million", "Rs 81,415.38 million")])

    doc_id = ingest(conn, client, pdf, canonicalise_terms=False)
    row = conn.execute("SELECT * FROM facts WHERE doc_id=?", (doc_id,)).fetchone()
    assert row["canon_value"] == 8.141538e10
    assert row["canon_unit"] == "INR"
    assert row["period_start"] == "2023-04-01"
    ev = conn.execute("SELECT * FROM evidence WHERE fact_id=?", (row["id"],)).fetchone()
    assert ev["page_no"] == 1 and "81,415.38" in ev["quote"]


def test_reingesting_the_same_pdf_does_not_duplicate(tmp_path, make_pdf):
    pdf = make_pdf([[LINE]])
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    client = LLMClient(conn, api_key=None, model="m")
    _seed(conn, pdf, [_fact("81,415.38", "Rs million", "Rs 81,415.38 million")])

    first = ingest(conn, client, pdf, canonicalise_terms=False)
    second = ingest(conn, client, pdf, canonicalise_terms=False)
    assert first == second
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 1


def test_cross_document_corroboration_is_found_by_rule(tmp_path, make_pdf):
    # crore in one document, million in the other: the rule layer settles it
    # without the model, so this runs with no key and no adjudication cache
    a = make_pdf([[LINE]], name="ar.pdf")
    b = make_pdf([[OTHER]], name="deck.pdf")
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    client = LLMClient(conn, api_key=None, model="m")
    _seed(conn, a, [_fact("81,415.38", "Rs million", "Rs 81,415.38 million")])
    _seed(conn, b, [_fact("8,142", "Rs Cr", "Rs 8,142 Cr")])

    for pdf in (a, b):
        ingest(conn, client, pdf, canonicalise_terms=False)
    # both facts share metric wording, so the lexical channel pairs them
    for fid in (1, 2):
        conn.execute("UPDATE facts SET metric_id='rev', entity_id='dl' WHERE id=?",
                     (fid,))
    conn.commit()

    written = build_relations(conn, client)
    assert written == 1
    rel = conn.execute("SELECT * FROM relations").fetchone()
    assert rel["final_verdict"] == "corroborates"
    assert rel["model_verdict"] is None, "rules settled it, no model call needed"


def test_relations_are_idempotent(tmp_path, make_pdf):
    a = make_pdf([[LINE]], name="ar.pdf")
    b = make_pdf([[OTHER]], name="deck.pdf")
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    client = LLMClient(conn, api_key=None, model="m")
    _seed(conn, a, [_fact("81,415.38", "Rs million", "Rs 81,415.38 million")])
    _seed(conn, b, [_fact("8,142", "Rs Cr", "Rs 8,142 Cr")])
    for pdf in (a, b):
        ingest(conn, client, pdf, canonicalise_terms=False)
    conn.execute("UPDATE facts SET metric_id='rev', entity_id='dl'")
    conn.commit()

    build_relations(conn, client)
    build_relations(conn, client)
    build_relations(conn, client)
    assert conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0] == 1


class _TruncatingModel:
    """Fails on the whole window, succeeds on each half.

    Mimics a dense window whose facts overflow the output token limit.
    """

    def __init__(self, whole_text):
        self.whole = whole_text
        self.calls = []

    def complete_json(self, prompt, version, **kw):
        from factlayer.llm.client import BadModelJSON
        excerpt = prompt.split("EXCERPT:", 1)[-1]
        self.calls.append(len(excerpt))
        if self.whole.strip() in excerpt:
            raise BadModelJSON("truncated at the output token limit")
        facts = []
        for value, quote in (("81,415.38", "Rs 81,415.38 million"),
                             ("74,540.82", "Rs 74,540.82 million")):
            if quote in excerpt:
                facts.append({"subject": "Delhivery Limited",
                              "metric": "revenue from operations",
                              "value_raw": value, "unit_raw": "Rs million",
                              "period_raw": "FY24", "qualifiers": {},
                              "claim_type": "measurement",
                              "evidence_quote": quote, "confidence": 0.9})
        return {"facts": facts}


def test_a_truncated_window_is_recovered_by_splitting(tmp_path, make_pdf):
    # two lines land on two different pages, so a misattributed span after the
    # split would show up immediately as the wrong page number
    pdf = make_pdf([["Consolidated revenue was Rs 81,415.38 million for FY24."],
                    ["Standalone revenue was Rs 74,540.82 million for FY24."]])
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)

    from factlayer.ingest.pdf import extract_blocks as _eb
    from factlayer.ingest.segment import build_windows as _bw
    blocks, _ = _eb(pdf)
    whole = _bw(blocks, 1, 12000, 1200)[0].text
    model = _TruncatingModel(whole)

    doc_id = ingest(conn, model, pdf, canonicalise_terms=False)
    rows = conn.execute(
        "SELECT f.value_raw, e.page_no, e.quote FROM facts f "
        "JOIN evidence e ON e.fact_id = f.id WHERE f.doc_id=? ORDER BY f.id",
        (doc_id,)).fetchall()

    assert len(rows) == 2, "both halves recovered instead of the window being lost"
    by_value = {r["value_raw"]: r for r in rows}
    assert by_value["81,415.38"]["page_no"] == 1
    assert by_value["74,540.82"]["page_no"] == 2
    for r in rows:
        page_text = " ".join(x["text"] for x in conn.execute(
            "SELECT text FROM blocks WHERE doc_id=? AND page_no=?",
            (doc_id, r["page_no"])))
        assert r["quote"] in page_text, "quote must be on the page it claims"


def _counting_model(payload_by_text=None):
    class M:
        def __init__(self):
            self.extract_calls = 0
            self.canon_calls = 0

        def complete_json(self, prompt, version, **kw):
            if version.startswith("canon"):
                self.canon_calls += 1
                import re as _re
                names = _re.findall(r"^- (.+)$", prompt.split("NAMES:", 1)[-1], _re.M)
                return {"groups": [{"canon_id": "c" + str(i), "label": n,
                                    "members": [n]} for i, n in enumerate(names)]}
            self.extract_calls += 1
            excerpt = prompt.split("EXCERPT:", 1)[-1]
            quote = "Rs 81,415.38 million"
            if quote not in excerpt:
                return {"facts": []}
            return {"facts": [{"subject": "Delhivery Limited",
                               "metric": "revenue from operations",
                               "value_raw": "81,415.38", "unit_raw": "Rs million",
                               "period_raw": "FY24", "qualifiers": {},
                               "claim_type": "measurement",
                               "evidence_quote": quote, "confidence": 0.9}]}
    return M()


def test_max_windows_caps_the_number_of_extraction_calls(tmp_path, make_pdf):
    # a document long enough to need several windows
    # text must genuinely vary: identical lines across pages are correctly
    # treated as running headers and filtered out before windowing
    words = ("freight parcel tonnage revenue margin expense reserve inflation "
             "deficit export credit growth capacity network hub gateway").split()
    pages = [[" ".join(words[(i + j + k) % len(words)] for k in range(11))
              + f" was Rs 1,2{j}3.45 million in FY24."
              for j in range(38)] for i in range(14)]
    pdf = make_pdf(pages)
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)

    from factlayer.ingest.pdf import extract_blocks as _eb
    from factlayer.ingest.segment import build_windows as _bw
    blocks, _ = _eb(pdf)
    available = len(_bw(blocks, 1, 12000, 1200))

    model = _counting_model()
    ingest(conn, model, pdf, canonicalise_terms=False, max_windows=2)
    assert model.extract_calls == min(2, available)
    assert available >= 2, "fixture must produce more windows than the cap"


def test_corpus_canonicalisation_costs_two_calls_not_two_per_document(tmp_path,
                                                                     make_pdf):
    from factlayer.pipeline import canonicalise_corpus
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    model = _counting_model()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        pdf = make_pdf([["Revenue for FY24 stood at Rs 81,415.38 million."]], name=name)
        ingest(conn, model, pdf, canonicalise_terms=False)
    assert model.canon_calls == 0, "per-document canonicalisation must be off"

    canonicalise_corpus(conn, model)
    assert model.canon_calls == 2, "one call for entities, one for metrics"
    rows = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE entity_id IS NOT NULL "
        "AND metric_id IS NOT NULL").fetchone()[0]
    assert rows == conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]


def test_an_interrupted_ingest_is_retried_not_skipped(tmp_path, make_pdf):
    # Blocks are written before the model is called. A run cut short by a spent
    # quota leaves blocks with no facts, and the document must not then look
    # complete for ever.
    pdf = make_pdf([[LINE]])
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)

    class Failing:
        def complete_json(self, prompt, version, **kw):
            raise RuntimeError("quota gone")

    try:
        ingest(conn, Failing(), pdf, canonicalise_terms=False)
    except RuntimeError:
        pass
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0

    # a later run with working access must actually read it
    client = LLMClient(conn, api_key=None, model="m")
    _seed(conn, pdf, [_fact("81,415.38", "Rs million", "Rs 81,415.38 million")])
    ingest(conn, client, pdf, canonicalise_terms=False)
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    # and blocks are not duplicated by the retry
    assert conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 1
