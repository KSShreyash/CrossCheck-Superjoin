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


def test_array_wrapped_json_is_unwrapped():
    # some models return [{"facts": [...]}] instead of {"facts": [...]}
    from factlayer.llm.client import _loads_lenient
    assert _loads_lenient('[{"facts": [{"metric": "revenue"}]}]') == \
        {"facts": [{"metric": "revenue"}]}


def test_split_objects_are_merged_not_dropped():
    from factlayer.llm.client import _loads_lenient
    out = _loads_lenient('[{"facts": [{"a": 1}]}, {"facts": [{"b": 2}]}]')
    assert out["facts"] == [{"a": 1}, {"b": 2}]


DAILY = ('429 You exceeded your current quota. '
         'quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier" '
         'quota_value: 20')
PER_MINUTE = '429 Resource has been exhausted (e.g. check quota).'


def test_daily_quota_is_never_retried():
    # every attempt counts against the allowance, so retrying a daily limit
    # spends more requests to be told the same thing
    from factlayer.llm.client import _is_daily_quota, _is_retryable
    assert _is_daily_quota(Exception(DAILY))
    assert not _is_retryable(Exception(DAILY))
    # a transient limit still is
    assert not _is_daily_quota(Exception(PER_MINUTE))
    assert _is_retryable(Exception(PER_MINUTE))


def _rotating_client(tmp_path, behaviour):
    """behaviour: dict of (key, model) -> "ok" | "daily"."""
    from factlayer.db import connect as _c, init_schema as _i
    from factlayer.llm.client import LLMClient
    conn = _c(tmp_path / "r.sqlite")
    _i(conn)
    client = LLMClient(conn, "k1", "m1", models=["m1", "m2"],
                       api_keys=["k1", "k2"])
    calls = []

    class Resp:
        text = '{"facts": [{"metric": "revenue"}]}'

    def fake_provider(key, model):
        class P:
            def generate_content(self, prompt, generation_config=None):
                calls.append((key, model))
                if behaviour.get((key, model)) == "daily":
                    raise Exception(DAILY)
                return Resp()
        return P()

    client._provider = fake_provider
    return client, calls


def test_rotation_moves_past_an_exhausted_model(tmp_path):
    client, calls = _rotating_client(tmp_path, {("k1", "m1"): "daily"})
    out = client.complete_json("PROMPT", "v1")
    assert out["facts"][0]["metric"] == "revenue"
    assert calls == [("k1", "m1"), ("k1", "m2")], "one retry of m1, then move on"


def test_an_exhausted_pairing_is_not_tried_again(tmp_path):
    client, calls = _rotating_client(tmp_path, {("k1", "m1"): "daily"})
    client.complete_json("FIRST", "v1")
    calls.clear()
    client.complete_json("SECOND", "v1")
    assert ("k1", "m1") not in calls, "already known to be spent"


def test_all_pairings_exhausted_raises_a_clear_error(tmp_path):
    from factlayer.llm.client import DailyQuotaExhausted
    spent = {(k, m): "daily" for k in ("k1", "k2") for m in ("m1", "m2")}
    client, calls = _rotating_client(tmp_path, spent)
    with pytest.raises(DailyQuotaExhausted) as e:
        client.complete_json("PROMPT", "v1")
    assert "2 key(s) x 2 model(s)" in str(e.value)
    assert len(calls) == 4, "each pairing tried exactly once"


def test_a_cached_answer_under_any_rotated_model_is_reused(tmp_path):
    client, calls = _rotating_client(tmp_path, {})
    put(client.conn, cache_key("m2", "v1", "PROMPT"), {"facts": [{"m": 1}]})
    assert client.complete_json("PROMPT", "v1") == {"facts": [{"m": 1}]}
    assert calls == [], "rotation must not re-ask a question already paid for"
