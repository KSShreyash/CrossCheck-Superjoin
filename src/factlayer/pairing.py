import re
from collections import defaultdict

from .models import Fact
from .normalize.units import units_compatible

_STOP = {"of", "from", "the", "in", "for", "and", "a", "on", "to"}


def _tokens(fact: Fact) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", str(fact.metric).lower())
    return frozenset(w for w in words if w not in _STOP and len(w) > 2)


def candidate_pairs(facts: list[Fact], max_per_fact: int) -> list[tuple[int, int]]:
    """Facts worth comparing, from the union of two recall channels.

    Comparing every pair is quadratic and mostly wasted, so pairs come from
    sharing a canonical metric or from overlapping metric wording. Union
    rather than intersection: each channel catches pairs the other misses.
    """
    by_metric: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(facts):
        if f.metric_id:
            by_metric[f.metric_id].append(i)

    scored: dict[tuple[int, int], float] = {}

    def offer(i: int, j: int, score: float) -> None:
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        # Differing subjects block a pair only when the metric differs too.
        # Canonicalisation names the thing measured, so one document says
        # "India's GDP" where another says "real GDP" and a strict identity
        # test drops the comparison before anything can look at it. When the
        # metric already matches, the subject difference is recorded and
        # judged rather than used to silently discard the pair.
        if facts[a].entity_id and facts[b].entity_id and \
           facts[a].entity_id != facts[b].entity_id and \
           facts[a].metric_id != facts[b].metric_id:
            return
        # incomparable units are noise, not disagreement: without this gate a
        # rupee figure pairs with a percentage purely on shared metric words
        if facts[a].canon_unit and facts[b].canon_unit and \
           facts[a].canon_unit != facts[b].canon_unit:
            return
        scored[(a, b)] = max(scored.get((a, b), 0.0), score)

    for members in by_metric.values():
        for i in members:
            for j in members:
                offer(i, j, 1.0)

    toks = [_tokens(f) for f in facts]
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            if not toks[i] or not toks[j]:
                continue
            overlap = len(toks[i] & toks[j]) / len(toks[i] | toks[j])
            if overlap >= 0.6:
                offer(i, j, overlap)

    kept: dict[int, int] = defaultdict(int)
    out = []
    for (a, b), _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0])):
        if kept[a] >= max_per_fact or kept[b] >= max_per_fact:
            continue
        kept[a] += 1
        kept[b] += 1
        out.append((a, b))
    return sorted(out)
