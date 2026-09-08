from factlayer.db import connect, init_schema
from factlayer.extract import dedupe_facts, extract_facts, locate_quote
from factlayer.llm.cache import cache_key, put
from factlayer.llm.client import LLMClient
from factlayer.llm.prompts import EXTRACTION_PROMPT_VERSION, build_extraction_prompt
from factlayer.models import Fact, Window

TEXT = ("Revenue from operations on consolidated basis for FY24 stood at "
        "Rs 81,415.38 million as against Rs 72,253.01 million for FY23.")


def _client(tmp_path, payload):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    prompt = build_extraction_prompt(TEXT)
    put(conn, cache_key("m", EXTRACTION_PROMPT_VERSION, prompt), payload)
    return LLMClient(conn, api_key=None, model="m")


def _window():
    return Window(1, 0, TEXT, [0], [(0, len(TEXT))])


def test_grounded_fact_is_accepted_with_offsets(tmp_path):
    payload = {"facts": [{
        "subject": "Delhivery Limited", "metric": "revenue from operations",
        "value_raw": "81,415.38", "unit_raw": "Rs million", "period_raw": "FY24",
        "qualifiers": {"basis": "consolidated"}, "claim_type": "measurement",
        "evidence_quote": "Rs 81,415.38 million", "confidence": 0.9}]}
    client = _client(tmp_path, payload)
    facts, rejected = extract_facts(client, _window(), doc_id=1)
    assert not rejected
    assert facts[0].qualifiers["basis"] == "consolidated"
    start, end = facts[0].span
    assert TEXT[start:end] == "Rs 81,415.38 million"


def test_hallucinated_quote_is_rejected(tmp_path):
    payload = {"facts": [{
        "subject": "Delhivery Limited", "metric": "revenue",
        "value_raw": "99,999", "unit_raw": "Rs million", "period_raw": "FY24",
        "qualifiers": {}, "claim_type": "measurement",
        "evidence_quote": "Rs 99,999 million", "confidence": 0.9}]}
    client = _client(tmp_path, payload)
    facts, rejected = extract_facts(client, _window(), doc_id=1)
    assert facts == []
    assert rejected[0]["reason"] == "evidence_quote not found in source window"


def test_fact_without_subject_or_metric_is_rejected(tmp_path):
    payload = {"facts": [{
        "subject": "", "metric": "", "value_raw": "81,415.38",
        "unit_raw": "Rs million", "period_raw": "FY24", "qualifiers": {},
        "claim_type": "measurement",
        "evidence_quote": "Rs 81,415.38 million", "confidence": 0.9}]}
    client = _client(tmp_path, payload)
    facts, rejected = extract_facts(client, _window(), doc_id=1)
    assert facts == []
    assert rejected[0]["reason"] == "missing subject or metric"


def test_locate_quote_tolerates_whitespace():
    assert locate_quote("a  b\nc", "a b c") is not None
    assert locate_quote("plain text", "absent") is None


def test_overlap_twins_are_deduped_but_other_documents_survive():
    def _f(doc, conf, quote, value="81,415.38"):
        f = Fact(doc, "Delhivery", "revenue from operations", value,
                 None, "Rs million", "FY24")
        f.evidence_quote, f.confidence = quote, conf
        return f

    # the same sentence seen twice because the windows overlap
    twin_a = _f(1, 0.90, "Revenue from operations  stood at  Rs 81,415.38 million")
    twin_b = _f(1, 0.95, "Revenue from operations stood at Rs 81,415.38 million")
    other_value = _f(1, 0.90, "as against Rs 72,253.01 million", "72,253.01")
    other_doc = _f(2, 0.80, "Revenue from operations stood at Rs 81,415.38 million")

    out = dedupe_facts([twin_a, twin_b, other_value, other_doc])
    assert len(out) == 3
    kept = [f for f in out if f.doc_id == 1 and f.value_raw == "81,415.38"]
    assert len(kept) == 1 and kept[0].confidence == 0.95
    assert any(f.doc_id == 2 for f in out), "cross-document copy is a real corroboration"


def test_numeric_and_odd_json_types_do_not_crash_extraction(tmp_path):
    # JSON has numbers, and a model asked for "the number exactly as printed"
    # will sometimes return 6.5 rather than "6.5". Every downstream string
    # operation would then fail on a whole window of otherwise good facts.
    payload = {"facts": [{
        "subject": "India", "metric": "real GDP growth",
        "value_raw": 6.5, "unit_raw": "per cent", "period_raw": 2026,
        "qualifiers": ["not", "an", "object"], "claim_type": None,
        "evidence_quote": "Rs 81,415.38 million", "confidence": "0.9"}]}
    client = _client(tmp_path, payload)
    facts, rejected = extract_facts(client, _window(), doc_id=1)
    assert not rejected
    f = facts[0]
    assert f.value_raw == "6.5" and f.period_raw == "2026"
    assert f.qualifiers == {}, "a non-object qualifiers field is discarded"
    assert f.claim_type == "measurement"
    dedupe_facts(facts)          # must not raise
