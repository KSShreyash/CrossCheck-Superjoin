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


def test_different_subjects_sharing_a_metric_are_surfaced_not_dropped():
    # "delhivery revenue" against "india revenue" is probably not a real
    # comparison, but canonicalisation cannot be trusted to tell that apart
    # from "India's GDP" against "real GDP". The pair is surfaced with the
    # subject difference recorded, and the model is allowed to call it
    # unrelated - which is a verdict, not a silent omission.
    a = _f(1, "revenue", "revenue", entity="delhivery")
    b = _f(2, "revenue", "revenue", entity="india")
    assert (0, 1) in candidate_pairs([a, b], max_per_fact=10)


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


def test_differing_subjects_still_pair_when_the_metric_matches():
    # canonicalisation names the thing measured, so one document says
    # "India's GDP" where another says "real GDP". Dropping the pair on that
    # basis hides the comparison entirely instead of judging it.
    a = _f(1, "growth", "real GDP growth", entity="indias_gdp", unit="PERCENT")
    b = _f(2, "growth", "growth at constant prices", entity="real_gdp",
           unit="PERCENT")
    assert (0, 1) in candidate_pairs([a, b], max_per_fact=10)


def test_unrelated_subjects_and_metrics_are_still_kept_apart():
    a = _f(1, "revenue", "revenue from operations", entity="delhivery")
    b = _f(2, "tonnage", "freight tonnage", entity="india")
    assert candidate_pairs([a, b], max_per_fact=10) == []


def test_an_unrecorded_unit_does_not_block_a_pair():
    # A bare "million" in a table whose header carried the currency normalises
    # to UNKNOWN. Comparing units with a plain inequality makes that look like
    # a different unit and silently drops the comparison - which is exactly
    # the revenue case the whole system is meant to catch.
    a = _f(1, "revenue", "Turnover", unit="UNKNOWN")
    b = _f(2, "revenue", "revenue from services", unit="INR")
    assert (0, 1) in candidate_pairs([a, b], max_per_fact=10)
