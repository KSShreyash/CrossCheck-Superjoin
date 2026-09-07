import json

EXTRACTION_PROMPT_VERSION = "extract-v1"
CANON_PROMPT_VERSION = "canon-v1"
ADJUDICATE_PROMPT_VERSION = "adjudicate-v1"


EXTRACTION_PROMPT = """You extract checkable facts from an excerpt of a document.

Return JSON: {{"facts": [...]}}. Each fact has:
  subject         the entity the fact is about, exactly as the text names it
  metric          what is being asserted or measured, in the text's own words
  value_raw       the number exactly as printed, or null for non-numeric facts
  unit_raw        the unit and scale as printed, e.g. "Rs in Million", "per cent"
  period_raw      the period as printed, e.g. "FY24", "year ended March 31, 2024"
  qualifiers      an object for anything that changes what the number means:
                  basis (standalone/consolidated), vintage (estimate/actual/projection),
                  segment, scope, geography, attribution. Omit keys that do not apply.
                  Invent new keys when the text implies a distinction not listed here.
  claim_type      measurement | estimate | projection | attribute | event
  evidence_quote  a span copied VERBATIM from the excerpt that states this fact
  confidence      0 to 1

Rules:
- evidence_quote must appear character-for-character in the excerpt. Never paraphrase it.
- Extract only what the excerpt states. Do not infer, compute, or combine numbers.
- period_raw matters more than almost anything else. A fact with no period cannot be
  compared against anything, so look above and around the number for the period it
  belongs to, including a table header or a section heading, before giving up.
- Prefer several precise facts over one broad one.
- A table row is a fact per cell when the column header gives it distinct meaning.
- Skip navigation text, page furniture, and legal disclaimers.

EXCERPT:
{window}
"""


CANON_PROMPT = """Group the {kind} names below by whether they denote the same thing.

Two names belong together only if a reader of the source documents would treat them as
the same {kind}. Different scopes, segments, or reporting bases are the SAME {kind} -
those distinctions are recorded separately as qualifiers. Genuinely different subjects
stay apart.

These groups already exist, from documents ingested earlier. If a name below belongs to
one of them, reuse that canon_id exactly. Reusing an existing id is what lets the same
{kind} be recognised across documents; minting a new id for something already known
silently stops those facts ever being compared. Create a new id only when nothing fits.

EXISTING GROUPS:
{existing}

Return JSON: {{"groups": [{{"canon_id": "snake_case_id", "label": "Readable label",
"members": ["...", "..."]}}]}}
Every input name must appear in exactly one group.

NAMES:
{terms}
"""


ADJUDICATE_PROMPT = """Two facts were extracted from documents. Decide how they relate.

FACT A: {a}
  evidence: "{a_quote}"  (document {a_doc}, page {a_page})
FACT B: {b}
  evidence: "{b_quote}"  (document {b_doc}, page {b_page})

A deterministic check found: values {agree_text}; differing attributes: {diff}

Return JSON:
  verdict           corroborates | contradicts | reconciled_by_context | unrelated
  reason_code       short snake_case tag, e.g. unit_scale, reporting_basis,
                    data_vintage, different_period, genuine_disagreement
  explanation       two sentences at most, citing what in the evidence decides it
  claimed_transform if reconciled_by_context, the transformation that makes them
                    consistent: {{"kind": "scale", "factor": N}} for a pure scale
                    change, otherwise {{"kind": "basis"|"vintage"|"period"|"scope"|
                    "segment"}}. Use {{"kind": "none"}} when no transform applies.
  confidence        0 to 1

Judge only from the evidence shown. If the two facts are about different things, say
unrelated rather than forcing a relation.
"""


def build_extraction_prompt(window_text: str) -> str:
    return EXTRACTION_PROMPT.format(window=window_text)


def build_canon_prompt(kind: str, terms: list[str],
                       existing: dict[str, str] | None = None) -> str:
    listed = "\n".join(f"- {t}" for t in sorted(set(terms)))
    known = ("\n".join(f"- {cid}: {label}" for cid, label in sorted(existing.items()))
             if existing else "(none yet - this is the first document)")
    return CANON_PROMPT.format(kind=kind, terms=listed, existing=known)


def build_adjudicate_prompt(a_summary: str, b_summary: str, a_quote: str, b_quote: str,
                            a_meta: dict, b_meta: dict, diff: dict,
                            agree_text: str) -> str:
    return ADJUDICATE_PROMPT.format(
        a=a_summary, b=b_summary, a_quote=a_quote, b_quote=b_quote,
        a_doc=a_meta.get("filename", "?"), a_page=a_meta.get("page_no", "?"),
        b_doc=b_meta.get("filename", "?"), b_page=b_meta.get("page_no", "?"),
        diff=json.dumps(diff, sort_keys=True, default=str) or "none",
        agree_text=agree_text)
