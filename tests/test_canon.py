from factlayer.db import connect, init_schema
from factlayer.llm.cache import cache_key, put
from factlayer.llm.client import LLMClient
from factlayer.llm.prompts import CANON_PROMPT_VERSION, build_canon_prompt
from factlayer.normalize.canon import canonicalise

TERMS = ["revenue from services", "Revenue from Operations", "PTL freight tonnage"]


def test_synonymous_metrics_share_a_canonical_id(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    prompt = build_canon_prompt("metric", TERMS, {})
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
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    conn.execute("INSERT INTO canon_terms VALUES ('metric','x','cid','X')")
    conn.commit()
    client = LLMClient(conn, api_key=None, model="m")   # no key: a call would raise
    assert canonicalise(client, conn, "metric", ["x"]) == {"x": ("cid", "X")}


def test_a_later_document_is_offered_the_existing_groups(tmp_path):
    # Documents arrive one at a time, so a term from the second document must be
    # able to join a group created by the first. If the prompt does not carry the
    # existing groups, the same metric gets two canonical ids and nothing ever
    # pairs across documents.
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    conn.execute("INSERT INTO canon_terms VALUES "
                 "('metric','revenue from services','revenue_from_ops','Revenue')")
    conn.commit()
    prompt = build_canon_prompt("metric", ["Revenue from Operations"],
                                {"revenue_from_ops": "Revenue"})
    assert "revenue_from_ops" in prompt, "existing group must reach the model"
    put(conn, cache_key("m", CANON_PROMPT_VERSION, prompt), {"groups": [
        {"canon_id": "revenue_from_ops", "label": "Revenue",
         "members": ["Revenue from Operations"]}]})
    client = LLMClient(conn, api_key=None, model="m")
    mapping = canonicalise(client, conn, "metric",
                           ["revenue from services", "Revenue from Operations"])
    assert mapping["Revenue from Operations"][0] == mapping["revenue from services"][0]


def test_a_term_the_model_drops_keeps_its_own_identity(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    prompt = build_canon_prompt("metric", ["net working capital days"], {})
    put(conn, cache_key("m", CANON_PROMPT_VERSION, prompt), {"groups": []})
    client = LLMClient(conn, api_key=None, model="m")
    mapping = canonicalise(client, conn, "metric", ["net working capital days"])
    assert mapping["net working capital days"][0] == "net_working_capital_days"
