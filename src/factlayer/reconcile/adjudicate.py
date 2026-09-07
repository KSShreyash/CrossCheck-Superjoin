import json

from ..llm.prompts import ADJUDICATE_PROMPT_VERSION, build_adjudicate_prompt
from ..models import Fact

_MODEL_VERDICTS = {"corroborates", "contradicts", "reconciled_by_context", "unrelated"}


def summarise(f: Fact) -> str:
    parts = [f.subject, f.metric]
    if f.value_raw:
        parts.append(f"{f.value_raw} {f.unit_raw or ''}".strip())
    if f.period_raw:
        parts.append(f"period {f.period_raw}")
    if f.qualifiers:
        parts.append(json.dumps(f.qualifiers, sort_keys=True, default=str))
    return " | ".join(p for p in parts if p)


def adjudicate(client, a: Fact, b: Fact, rule_verdict: str, diff: dict,
               a_meta: dict, b_meta: dict) -> dict:
    """Ask the model to judge a pair the rules could not settle."""
    prompt = build_adjudicate_prompt(
        summarise(a), summarise(b), a.evidence_quote, b.evidence_quote,
        a_meta, b_meta, diff,
        "agree" if rule_verdict.startswith("corroborates") else "differ")
    data = client.complete_json(prompt, ADJUDICATE_PROMPT_VERSION)

    verdict = str(data.get("verdict") or "").strip()
    if verdict not in _MODEL_VERDICTS:
        # an unrecognised verdict is not silently coerced into a real one
        verdict = "unrelated"
    return {
        "verdict": verdict,
        "reason_code": str(data.get("reason_code") or ""),
        "explanation": str(data.get("explanation") or ""),
        "claimed_transform": data.get("claimed_transform"),
        "confidence": float(data.get("confidence") or 0.0),
    }
