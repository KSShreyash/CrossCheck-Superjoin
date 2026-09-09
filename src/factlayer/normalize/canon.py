import re

from ..llm.prompts import CANON_PROMPT_VERSION, build_canon_prompt


def _slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_") or "unnamed"


def _known(conn, kind: str) -> dict[str, tuple[str, str]]:
    rows = conn.execute("SELECT raw, canon_id, label FROM canon_terms WHERE kind=?",
                        (kind,)).fetchall()
    return {r["raw"]: (r["canon_id"], r["label"]) for r in rows}


def canonicalise(client, conn, kind: str,
                 raw_terms: list[str]) -> dict[str, tuple[str, str]]:
    """Map each raw term to a canonical id, reusing groups already assigned."""
    mapping = _known(conn, kind)
    pending = sorted({t for t in raw_terms if t and t not in mapping})
    if not pending:
        return {t: mapping[t] for t in raw_terms if t in mapping}

    # offer existing groups, or a later document can never join one
    existing = {cid: label for cid, label in mapping.values()}
    prompt = build_canon_prompt(kind, pending, existing)
    data = client.complete_json(prompt, CANON_PROMPT_VERSION)

    rows = []
    for group in data.get("groups", []):
        cid, label = group.get("canon_id"), group.get("label") or ""
        if not cid:
            continue
        for member in group.get("members", []):
            mapping[member] = (cid, label)
            rows.append((kind, member, cid, label))

    # anything the model dropped keeps its own identity rather than vanishing
    for term in pending:
        if term not in mapping:
            mapping[term] = (_slug(term), term)
            rows.append((kind, term, _slug(term), term))

    conn.executemany(
        "INSERT OR REPLACE INTO canon_terms(kind,raw,canon_id,label) VALUES (?,?,?,?)",
        rows)
    conn.commit()
    return {t: mapping[t] for t in raw_terms if t in mapping}
