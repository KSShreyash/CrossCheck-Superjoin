import pytest

from factlayer.db import connect, init_schema
from factlayer.llm.cache import cache_key, put
from factlayer.llm.client import LLMClient, NoAPIKey


def test_cache_hit_never_calls_provider(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    client = LLMClient(conn, api_key=None, model="m")
    key = cache_key("m", "v1", "PROMPT")
    put(conn, key, {"facts": [{"metric": "revenue"}]})
    assert client.complete_json("PROMPT", "v1")["facts"][0]["metric"] == "revenue"


def test_cache_miss_without_key_raises(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    client = LLMClient(conn, api_key=None, model="m")
    with pytest.raises(NoAPIKey):
        client.complete_json("UNSEEN", "v1")


def test_key_is_stable_and_sensitive():
    assert cache_key("m", "v1", "a") == cache_key("m", "v1", "a")
    assert cache_key("m", "v1", "a") != cache_key("m", "v2", "a")
    assert cache_key("m", "v1", "a") != cache_key("n", "v1", "a")


def test_truncated_json_raises_a_named_error():
    from factlayer.llm.client import BadModelJSON, _loads_lenient
    with pytest.raises(BadModelJSON):
        _loads_lenient('{"facts": [{"metric": "revenue"}, {"metric": "EBI')


def test_fenced_json_is_recovered():
    from factlayer.llm.client import _loads_lenient
    assert _loads_lenient('```json\n{"facts": []}\n```') == {"facts": []}


def test_rate_limit_is_retryable_but_bad_json_is_not():
    from factlayer.llm.client import BadModelJSON, _is_retryable
    assert _is_retryable(Exception("429 Resource has been exhausted"))
    assert _is_retryable(Exception("503 Service Unavailable"))
    assert not _is_retryable(BadModelJSON("truncated"))
    assert not _is_retryable(ValueError("bad argument"))
