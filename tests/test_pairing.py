from factlayer.models import Fact
from factlayer.pairing import candidate_pairs


def _f(doc, metric_id, metric, entity="e", unit="INR"):
    f = Fact(doc, "Delhivery", metric, "1", 1.0, None, "FY24")
    f.metric_id, f.entity_id, f.canon_unit = metric_id, entity, unit
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
    a = _f(1, None, "net working capital days")
    b = _f(2, None, "net working capital days")
    assert (0, 1) in candidate_pairs([a, b], max_per_fact=10)


def test_different_entities_are_not_paired():
    a = _f(1, "revenue", "revenue", entity="delhivery")
    b = _f(2, "revenue", "revenue", entity="india")
    assert candidate_pairs([a, b], max_per_fact=10) == []


def test_incomparable_units_are_not_paired():
    # a rupee figure and a percentage share metric words but are not comparable
    a = _f(1, "growth", "revenue growth", unit="INR")
    b = _f(2, "growth", "revenue growth", unit="PERCENT")
    assert candidate_pairs([a, b], max_per_fact=10) == []


def test_cap_limits_pairs_per_fact():
    facts = [_f(1, "revenue", "revenue from operations") for _ in range(10)]
    pairs = candidate_pairs(facts, max_per_fact=2)
    seen = {}
    for a, b in pairs:
        seen[a] = seen.get(a, 0) + 1
        seen[b] = seen.get(b, 0) + 1
    assert max(seen.values()) <= 2
