"""Wiring: PDF in, grounded facts and their relations out."""
import json
import sqlite3
from pathlib import Path

from .config import settings
from .extract import dedupe_facts, extract_facts
from .ingest.boilerplate import mark_boilerplate
from .ingest.gaps import find_gaps
from .ingest.pdf import extract_blocks, file_sha256
from .ingest.segment import build_windows, split_window
from .llm.client import BadModelJSON, NoAPIKey
from .models import Fact
from .normalize.canon import canonicalise
from .normalize.periods import normalize_period
from .normalize.units import normalize_value
from .pairing import candidate_pairs
from .reconcile.adjudicate import adjudicate
from .reconcile.rules import measurement_diff, rule_verdict
from .reconcile.verify import verify
from .store import (save_blocks, save_document, save_fact, save_gaps,
                    save_rejected)

# pairs settled by rule still need a final verdict or the API filter hides them
RULE_FINAL = {
    "corroborates": "corroborates",
    "corroborates_with_caveat": "corroborates",
    "insufficient_context": "insufficient_context",
    "unrelated": "unrelated",
    "different_period": "reconciled_by_context",
}


def _bump_job(conn, job_id, *, stage="extracting", done=0, total=None, facts=0):
    if not job_id:
        return                      # scripted ingest has no job row
    if total is None:
        conn.execute("UPDATE jobs SET stage=?, done=?, facts=? WHERE id=?",
                     (stage, done, facts, job_id))
    else:
        conn.execute("UPDATE jobs SET stage=?, done=?, total=?, facts=? WHERE id=?",
                     (stage, done, total, facts, job_id))
    conn.commit()


def _already_ingested(conn, doc_id: int) -> bool:
    """Whether extraction finished for this document."""
    row = conn.execute("SELECT ingest_complete FROM documents WHERE id=?",
                       (doc_id,)).fetchone()
    return bool(row and row["ingest_complete"])


def ingest(conn: sqlite3.Connection, client, pdf_path: str | Path,
           job_id: str | None = None, canonicalise_terms: bool = True,
           max_windows: int | None = None) -> int:
    """Ingest one PDF and return its document id. Re-ingesting is a no-op."""
    pdf_path = Path(pdf_path)
    blocks, pages = extract_blocks(pdf_path)
    mark_boilerplate(blocks, settings.boilerplate_min_pages)
    gaps = find_gaps(blocks, pages, settings.gap_min_chars)

    doc_id = save_document(conn, file_sha256(pdf_path), pdf_path.name,
                           pdf_path.stem, pages)
    if _already_ingested(conn, doc_id):
        # same bytes as a document already read through to the end
        _bump_job(conn, job_id, stage="already ingested", done=1, total=1)
        return doc_id

    # Clear everything the previous attempt left behind.
    conn.execute(
        "DELETE FROM relations WHERE fact_a IN "
        "(SELECT id FROM facts WHERE doc_id=?) OR fact_b IN "
        "(SELECT id FROM facts WHERE doc_id=?)", (doc_id, doc_id))
    conn.execute("DELETE FROM evidence WHERE fact_id IN "
                 "(SELECT id FROM facts WHERE doc_id=?)", (doc_id,))
    conn.execute("DELETE FROM facts WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM rejected_facts WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM blocks WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM gaps WHERE doc_id=?", (doc_id,))
    conn.commit()
    block_row_ids = save_blocks(conn, doc_id, blocks)
    save_gaps(conn, doc_id, gaps)

    windows = build_windows(blocks, doc_id, settings.window_chars,
                            settings.window_overlap)
    # densest first, so a rate limit costs the least valuable pages
    windows.sort(key=lambda w: -w.density)
    if max_windows is not None:
        windows = windows[:max_windows]
    _bump_job(conn, job_id, done=0, total=len(windows))

    facts: list[Fact] = []
    rejected: list[dict] = []
    window_of: dict[int, object] = {}
    skipped: list[int] = []

    def pull(window, allow_split=True):
        """Extract from one window, halving it once if the output truncated."""
        try:
            got, bad = extract_facts(client, window, doc_id)
            for f in got:
                window_of[id(f)] = window
            return got, bad
        except NoAPIKey:
            # This window was never cached and there is no key to read it with.
            skipped.append(window.index)
            return [], [{"payload": {"window_index": window.index},
                         "reason": "no cached response and no API key"}]
        except BadModelJSON as exc:
            halves = split_window(window) if allow_split else []
            if not halves:
                return [], [{"payload": {"window_index": window.index},
                             "reason": f"unusable model output: {exc}"}]
            # a dense window can overflow the output limit, so retry it in halves
            got, bad = [], []
            for half in halves:
                g, b = pull(half, allow_split=False)
                got.extend(g)
                bad.extend(b)
            return got, bad

    for n, window in enumerate(windows, start=1):
        got, bad = pull(window)
        facts.extend(got)
        rejected.extend(bad)
        _bump_job(conn, job_id, done=n, total=len(windows), facts=len(facts))

    facts = dedupe_facts(facts)

    for f in facts:
        nv = normalize_value(f.value_raw, f.unit_raw, f.evidence_quote)
        if nv:
            f.canon_value, f.canon_unit = nv
            f.value_num = nv[0]
        np_ = normalize_period(f.period_raw)
        if np_:
            f.period_start, f.period_end, f.period_kind = np_

    if canonicalise_terms and facts:
        _bump_job(conn, job_id, stage="canonicalising", done=len(windows),
                  total=len(windows), facts=len(facts))
        entities = canonicalise(client, conn, "entity", [f.subject for f in facts])
        metrics = canonicalise(client, conn, "metric", [f.metric for f in facts])
        for f in facts:
            f.entity_id = (entities.get(f.subject) or (None, None))[0]
            f.metric_id = (metrics.get(f.metric) or (None, None))[0]

    for f in facts:
        save_fact(conn, f, window_of[id(f)], block_row_ids)
    save_rejected(conn, doc_id, rejected)
    if not skipped:
        conn.execute("UPDATE documents SET ingest_complete=1 WHERE id=?", (doc_id,))
    conn.commit()
    _bump_job(conn, job_id, stage="done", done=len(windows), total=len(windows),
              facts=len(facts))
    return doc_id


def canonicalise_corpus(conn: sqlite3.Connection, client) -> int:
    """Canonicalise every stored subject and metric in one pass."""
    rows = conn.execute("SELECT DISTINCT subject, metric FROM facts").fetchall()
    if not rows:
        return 0
    subjects = sorted({r["subject"] for r in rows if r["subject"]})
    metrics = sorted({r["metric"] for r in rows if r["metric"]})
    entities = canonicalise(client, conn, "entity", subjects)
    metric_ids = canonicalise(client, conn, "metric", metrics)

    updated = 0
    for r in conn.execute("SELECT id, subject, metric FROM facts").fetchall():
        eid = (entities.get(r["subject"]) or (None, None))[0]
        mid = (metric_ids.get(r["metric"]) or (None, None))[0]
        conn.execute("UPDATE facts SET entity_id=?, metric_id=? WHERE id=?",
                     (eid, mid, r["id"]))
        updated += 1
    conn.commit()
    return updated


def load_facts(conn) -> tuple[list[Fact], list[int], dict[int, dict]]:
    """Every stored fact, its row id, and the source metadata for prompts."""
    rows = conn.execute(
        "SELECT f.*, d.filename, e.page_no FROM facts f "
        "JOIN documents d ON d.id = f.doc_id "
        "LEFT JOIN evidence e ON e.fact_id = f.id ORDER BY f.id").fetchall()
    facts, ids, meta = [], [], {}
    for r in rows:
        f = Fact(r["doc_id"], r["subject"], r["metric"], r["value_raw"],
                 r["value_num"], r["unit_raw"], r["period_raw"],
                 json.loads(r["qualifiers"] or "{}"), r["claim_type"])
        f.confidence = r["confidence"] or 0.0
        f.canon_value, f.canon_unit = r["canon_value"], r["canon_unit"]
        f.period_start, f.period_end = r["period_start"], r["period_end"]
        f.period_kind = r["period_kind"]
        f.entity_id, f.metric_id = r["entity_id"], r["metric_id"]
        facts.append(f)
        ids.append(r["id"])
        meta[r["id"]] = {"filename": r["filename"], "page_no": r["page_no"]}
    return facts, ids, meta


def _fetch_quote(conn, fact_id: int) -> str:
    row = conn.execute("SELECT quote FROM evidence WHERE fact_id=?",
                       (fact_id,)).fetchone()
    return row["quote"] if row else ""


def _rules_only_verdict(rule_verdict: str) -> tuple[str, str | None, str]:
    """What to record when no adjudication is available."""
    if rule_verdict == "contradiction_candidate":
        return ("contradicts", "genuine_disagreement",
                "Same metric, period and unit; the values differ and no recorded "
                "qualifier distinguishes them. Decided by rule alone; no model "
                "review.")
    return ("insufficient_context", None,
            "no adjudication available for this pair")


def _write_relation(conn, fact_a, fact_b, rule_v, model_v, final, reason,
                    explanation, diff, transform, verified, agreed, confidence):
    conn.execute(
        "INSERT OR REPLACE INTO relations(fact_a,fact_b,rule_verdict,"
        "model_verdict,final_verdict,reason_code,explanation,qualifier_diff,"
        "claimed_transform,verified,agreed,confidence) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (fact_a, fact_b, rule_v, model_v, final, reason, explanation,
         json.dumps(diff, default=str),
         json.dumps(transform) if transform else None,
         verified, agreed, confidence))


def build_relations(conn: sqlite3.Connection, client,
                    max_model_calls: int | None = None) -> int:
    """Classify every candidate pair; ask the model only where rules cannot."""
    facts, ids, meta = load_facts(conn)
    if len(facts) < 2:
        return 0
    for f, fid in zip(facts, ids):
        f.evidence_quote = _fetch_quote(conn, fid)

    pairs = candidate_pairs(facts, settings.max_pairs_per_fact)

    # spend a limited budget on the most informative pairs, not the first ones
    def priority(pair: tuple[int, int]) -> tuple:
        a, b = facts[pair[0]], facts[pair[1]]
        verdict, _ = rule_verdict(a, b, settings.value_tolerance)
        rank = {"contradiction_candidate": 0, "reconcilable": 1,
                "needs_model": 2}.get(verdict, 9)
        cross = 0 if a.doc_id != b.doc_id else 1
        # Crossing documents outranks the kind of disagreement.
        return (cross, rank, -(a.confidence + b.confidence))

    pairs = sorted(pairs, key=priority)
    calls = 0
    written = 0
    exhausted = False

    for i, j in pairs:
        a, b = facts[i], facts[j]
        rv, diff = rule_verdict(a, b, settings.value_tolerance)
        model_verdict = explanation = reason_code = None
        transform = None
        verified = agreed = None
        confidence = None

        if rv in RULE_FINAL:
            final = RULE_FINAL[rv]
            if rv == "different_period":
                reason_code = "different_period"
                explanation = ("The two facts cover different periods, so their "
                               "values are not in conflict.")
        elif exhausted or (max_model_calls is not None
                           and calls >= max_model_calls):
            # no adjudication available, whether the budget was spent or never offered
            final, reason_code, explanation = _rules_only_verdict(rv)
        else:
            calls += 1
            try:
                out = adjudicate(client, a, b, rv, diff,
                                 meta[ids[i]], meta[ids[j]])
            except Exception as exc:               # noqa: BLE001
                # running out mid-pass must not discard the pairs already classified
                exhausted = True
                final, reason_code, explanation = _rules_only_verdict(rv)
                explanation = f"{explanation} ({type(exc).__name__})"
                _write_relation(conn, ids[i], ids[j], rv, None, final,
                                reason_code, explanation, diff,
                                None, None, None, None)
                written += 1
                continue
            model_verdict = out["verdict"]
            reason_code = out["reason_code"]
            explanation = out["explanation"]
            transform = out["claimed_transform"]
            confidence = out["confidence"]
            final = model_verdict
            agreed = 1

            if model_verdict == "reconciled_by_context":
                ok, note = verify(a, b, transform, settings.value_tolerance)
                verified = 1 if ok else 0
                if not ok:
                    final, agreed = "needs_review", 0
                    explanation = f"{explanation} [unverified: {note}]"

            # the model contradicting arithmetic outright is not accepted
            if model_verdict == "corroborates" and rv not in (
                    "corroborates", "corroborates_with_caveat"):
                final, agreed = "needs_review", 0

            # a contradiction asserts comparability, so hold it when a qualifier differs
            material = measurement_diff(diff)
            if model_verdict == "contradicts" and material:
                final, agreed = "needs_review", 0
                explanation = (f"{explanation} [held: {', '.join(material)} differs, "
                               "so the two may not be comparable]")

        _write_relation(conn, ids[i], ids[j], rv, model_verdict, final,
                        reason_code, explanation, diff, transform, verified,
                        agreed, confidence)
        written += 1

    conn.commit()
    return written
