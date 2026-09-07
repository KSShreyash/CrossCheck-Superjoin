from factlayer.db import connect, init_schema
from factlayer.llm.cache import cache_key, put
from factlayer.llm.client import LLMClient
from factlayer.llm.prompts import ADJUDICATE_PROMPT_VERSION, build_adjudicate_prompt
from factlayer.models import Fact
from factlayer.reconcile.adjudicate import adjudicate, summarise

A_META = {"filename": "ar24.pdf", "page_no": 42}
B_META = {"filename": "deck.pdf", "page_no": 5}


def _f(value_raw, quals, quote):
    f = Fact(1, "Delhivery Limited", "revenue from operations", value_raw, None,
             "Rs million", "FY24", qualifiers=quals)
    f.evidence_quote = quote
    return f


def _seed(tmp_path, a, b, diff, payload):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    prompt = build_adjudicate_prompt(summarise(a), summarise(b), a.evidence_quote,
                                     b.evidence_quote, A_META, B_META, diff, "differ")
    put(conn, cache_key("m", ADJUDICATE_PROMPT_VERSION, prompt), payload)
    return LLMClient(conn, api_key=None, model="m")


def test_model_verdict_and_transform_are_returned(tmp_path):
    a = _f("74,540.82", {"basis": "standalone"}, "Rs 74,540.82 million")
    b = _f("81,415.38", {"basis": "consolidated"}, "Rs 81,415.38 million")
    diff = {"basis": ("standalone", "consolidated")}
    client = _seed(tmp_path, a, b, diff, {
        "verdict": "reconciled_by_context", "reason_code": "reporting_basis",
        "explanation": "One figure is standalone and the other consolidated.",
        "claimed_transform": {"kind": "basis"}, "confidence": 0.9})
    out = adjudicate(client, a, b, "reconcilable", diff, A_META, B_META)
    assert out["verdict"] == "reconciled_by_context"
    assert out["claimed_transform"] == {"kind": "basis"}
    assert out["confidence"] == 0.9


def test_unrecognised_verdict_is_not_coerced_into_a_real_one(tmp_path):
    a = _f("1", {}, "one")
    b = _f("2", {}, "two")
    client = _seed(tmp_path, a, b, {}, {"verdict": "definitely_a_contradiction!!"})
    out = adjudicate(client, a, b, "contradiction_candidate", {}, A_META, B_META)
    assert out["verdict"] == "unrelated"


def test_summary_carries_qualifiers_and_period():
    f = _f("81,415.38", {"basis": "consolidated"}, "q")
    s = summarise(f)
    assert "Delhivery Limited" in s and "period FY24" in s and "consolidated" in s
