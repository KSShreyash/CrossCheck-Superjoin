import re

from .llm.prompts import EXTRACTION_PROMPT_VERSION, build_extraction_prompt
from .models import Fact, Window

_WS = re.compile(r"\s+")


def locate_quote(haystack: str, quote: str) -> tuple[int, int] | None:
    """Find quote in haystack, tolerating differences in whitespace only.

    The model copies from window text that still carries the PDF's own line
    breaks, so an otherwise verbatim quote often differs only in whitespace.
    Anything beyond that is treated as ungrounded and rejected.
    """
    if not quote:
        return None
    idx = haystack.find(quote)
    if idx >= 0:
        return idx, idx + len(quote)

    # map each character of a whitespace-collapsed copy back to its original index
    positions, collapsed = [], []
    prev_space = False
    for i, ch in enumerate(haystack):
        if ch.isspace():
            if prev_space:
                continue
            collapsed.append(" ")
            positions.append(i)
            prev_space = True
        else:
            collapsed.append(ch)
            positions.append(i)
            prev_space = False

    flat = "".join(collapsed)
    needle = _WS.sub(" ", quote).strip()
    j = flat.find(needle)
    if j < 0:
        return None
    start = positions[j]
    end = positions[min(j + len(needle) - 1, len(positions) - 1)] + 1
    return start, end


def extract_facts(client, window: Window, doc_id: int
                  ) -> tuple[list[Fact], list[dict]]:
    """Ask for facts, keep only those whose quote is really in the window."""
    prompt = build_extraction_prompt(window.text)
    data = client.complete_json(prompt, EXTRACTION_PROMPT_VERSION)
    accepted: list[Fact] = []
    rejected: list[dict] = []

    for raw in data.get("facts", []):
        quote = (raw.get("evidence_quote") or "").strip()
        span = locate_quote(window.text, quote)
        if span is None:
            rejected.append({"payload": raw,
                             "reason": "evidence_quote not found in source window"})
            continue
        if not (raw.get("subject") and raw.get("metric")):
            rejected.append({"payload": raw, "reason": "missing subject or metric"})
            continue
        fact = Fact(
            doc_id=doc_id,
            subject=str(raw["subject"]).strip(),
            metric=str(raw["metric"]).strip(),
            value_raw=raw.get("value_raw"),
            value_num=None,
            unit_raw=raw.get("unit_raw"),
            period_raw=raw.get("period_raw"),
            qualifiers=raw.get("qualifiers") or {},
            claim_type=raw.get("claim_type") or "measurement",
            evidence_quote=quote,
            confidence=float(raw.get("confidence") or 0.0),
        )
        fact.span = span
        accepted.append(fact)
    return accepted, rejected


def dedupe_facts(facts: list[Fact]) -> list[Fact]:
    """Collapse facts re-extracted from the overlap between windows.

    Windows overlap so that a fact straddling a boundary is not lost, but that
    puts about a tenth of blocks in two windows and their facts arrive twice.
    Left alone the twins pair with each other and register as corroborations,
    inflating the counts with a sentence agreeing with itself.

    Keyed per document, so the same fact appearing in a DIFFERENT document
    survives: that one is a real cross-document corroboration.
    """
    best: dict[tuple, Fact] = {}
    for f in facts:
        key = (f.doc_id, f.subject.strip().lower(), f.metric.strip().lower(),
               (f.value_raw or "").strip(), (f.period_raw or "").strip(),
               _WS.sub(" ", f.evidence_quote).strip().lower())
        current = best.get(key)
        if current is None or f.confidence > current.confidence:
            best[key] = f
    return list(best.values())
