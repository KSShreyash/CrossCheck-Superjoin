"""Find and print the four cases the assignment asks to see.

Reads whatever has been ingested and pulls out the strongest example of each,
with both evidence quotes and the system's reasoning.

    python scripts/show_cases.py
"""
import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factlayer.config import settings          # noqa: E402
from factlayer.db import connect, init_schema  # noqa: E402

REL_SQL = """
SELECT r.*, da.filename AS a_doc, db.filename AS b_doc,
       fa.subject a_subject, fa.metric a_metric, fa.value_raw a_value,
       fa.unit_raw a_unit, fa.period_raw a_period, fa.canon_value a_canon,
       fa.canon_unit a_cunit, fa.qualifiers a_quals,
       fb.subject b_subject, fb.metric b_metric, fb.value_raw b_value,
       fb.unit_raw b_unit, fb.period_raw b_period, fb.canon_value b_canon,
       fb.canon_unit b_cunit, fb.qualifiers b_quals,
       ea.quote a_quote, ea.page_no a_page, eb.quote b_quote, eb.page_no b_page
FROM relations r
JOIN facts fa ON fa.id = r.fact_a
JOIN facts fb ON fb.id = r.fact_b
JOIN documents da ON da.id = fa.doc_id
JOIN documents db ON db.id = fb.doc_id
LEFT JOIN evidence ea ON ea.fact_id = fa.id
LEFT JOIN evidence eb ON eb.fact_id = fb.id
WHERE r.final_verdict = ?
"""


def wrap(text, indent="      "):
    return textwrap.fill(str(text or "").strip(), width=96,
                         initial_indent=indent, subsequent_indent=indent)


def show(row):
    print(f"    A  {row['a_doc']}  page {row['a_page']}")
    print(wrap(f"“{row['a_quote']}”"))
    print(f"       {row['a_metric']} = {row['a_value']} {row['a_unit'] or ''}"
          f"  period {row['a_period']}  {row['a_quals']}")
    if row["a_canon"] is not None:
        print(f"       normalised: {row['a_canon']:,.2f} {row['a_cunit']}")
    print()
    print(f"    B  {row['b_doc']}  page {row['b_page']}")
    print(wrap(f"“{row['b_quote']}”"))
    print(f"       {row['b_metric']} = {row['b_value']} {row['b_unit'] or ''}"
          f"  period {row['b_period']}  {row['b_quals']}")
    if row["b_canon"] is not None:
        print(f"       normalised: {row['b_canon']:,.2f} {row['b_cunit']}")
    print()
    print(f"    rule layer : {row['rule_verdict']}")
    if row["model_verdict"]:
        print(f"    model      : {row['model_verdict']}  ({row['reason_code']})")
        print(wrap(row["explanation"], "                 "))
    if row["claimed_transform"]:
        state = {1: "confirmed", 0: "REJECTED"}.get(row["verified"], "n/a")
        print(f"    verified   : {state}  transform {row['claimed_transform']}")
    if row["agreed"] == 0:
        print("    note       : rule and model disagree; held for review")


def pick(conn, verdict, cross_document=True, limit=1):
    rows = [r for r in conn.execute(REL_SQL, (verdict,))]
    if cross_document:
        cross = [r for r in rows if r["a_doc"] != r["b_doc"]]
        rows = cross or rows
    rows.sort(key=lambda r: -(r["confidence"] or 0))
    return rows[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(settings.db_path))
    args = ap.parse_args()
    if not Path(args.db).exists():
        print(f"no database at {args.db}; run scripts/ingest_starter.py first")
        return 1
    conn = connect(args.db)
    init_schema(conn)            # tolerate a database created by an older run

    total = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    if not total:
        print("no relations stored; run scripts/ingest_starter.py first")
        return 1

    cases = [
        ("CASE 1  a fact corroborated across documents, expressed differently",
         "corroborates", True),
        ("CASE 2  a genuine or likely contradiction", "contradicts", True),
        ("CASE 3  an apparent contradiction explained by context",
         "reconciled_by_context", False),
    ]
    missing = []
    for title, verdict, cross in cases:
        print("=" * 98)
        print(title)
        print("=" * 98)
        rows = pick(conn, verdict, cross_document=cross)
        if not rows:
            print("    none found\n")
            missing.append(verdict)
            continue
        for r in rows:
            show(r)
            print(f"    inspect: /relations/{r['id']}\n")

    print("=" * 98)
    print("CASE 4  extraction and reasoning failures, and how they are handled")
    print("=" * 98)
    for g in conn.execute(
            "SELECT d.filename, g.page_no, g.reason FROM gaps g "
            "JOIN documents d ON d.id = g.doc_id ORDER BY d.id, g.page_no"):
        print(f"    unreadable  {g['filename']} page {g['page_no']}: {g['reason']}")
    rej = conn.execute("SELECT COUNT(*) FROM rejected_facts").fetchone()[0]
    print(f"    ungrounded  {rej} proposed facts rejected: quote not found in source")
    review = conn.execute(
        "SELECT COUNT(*) FROM relations WHERE final_verdict='needs_review'"
    ).fetchone()[0]
    insuff = conn.execute(
        "SELECT COUNT(*) FROM relations WHERE final_verdict='insufficient_context'"
    ).fetchone()[0]
    print(f"    disputed    {review} pairs where rule and model disagree")
    print(f"    undecided   {insuff} pairs lacking the periods needed to compare")

    if missing:
        print(f"\nmissing verdicts: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
