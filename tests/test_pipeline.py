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
