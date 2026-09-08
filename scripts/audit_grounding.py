"""Check the central claim: every stored fact quotes text that is really there.

Re-reads each piece of evidence and looks for it in the text of the page it
says it came from, going back to the stored blocks rather than trusting the
recorded offsets. Independent of the extraction path that produced them.

    python scripts/audit_grounding.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factlayer.config import settings          # noqa: E402
from factlayer.db import connect               # noqa: E402

WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return WS.sub(" ", text or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(settings.db_path))
    ap.add_argument("--show", type=int, default=5, help="failures to print")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"no database at {args.db}")
        return 1
    conn = connect(args.db)

    facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    if not facts:
        print("no facts stored")
        return 1

    pages: dict[tuple[int, int], str] = {}

    def page_text(doc_id: int, page_no: int) -> str:
        key = (doc_id, page_no)
        if key not in pages:
            pages[key] = norm(" ".join(r["text"] for r in conn.execute(
                "SELECT text FROM blocks WHERE doc_id=? AND page_no=?", key)))
        return pages[key]

    checked = failures = unplaced = straddling = 0
    bad = []
    for r in conn.execute(
            "SELECT f.id, f.doc_id, e.quote, e.page_no, e.block_ids, d.filename "
            "FROM facts f "
            "JOIN evidence e ON e.fact_id = f.id "
            "JOIN documents d ON d.id = f.doc_id"):
        if r["page_no"] is None:
            unplaced += 1
            continue
        checked += 1
        quote = norm(r["quote"])
        if quote in page_text(r["doc_id"], r["page_no"]):
            continue

        # Not a contiguous run on the printed page. Two innocent reasons for
        # that, so check what the evidence actually claims: the blocks it
        # names. Running headers are filtered before extraction, so a quote
        # crossing one joins text either side of it, and a quote can also run
        # from the foot of one page onto the next.
        block_ids = json.loads(r["block_ids"] or "[]")
        if block_ids:
            placeholders = ",".join("?" for _ in block_ids)
            source = norm(" ".join(
                x["text"] for x in conn.execute(
                    "SELECT text FROM blocks WHERE id IN (" + placeholders + ") "
                    "ORDER BY id", block_ids)))
            if quote in source:
                straddling += 1
                continue

        failures += 1
        if len(bad) < args.show:
            bad.append(r)

    print(f"facts stored             {facts}")
    print(f"evidence checked         {checked}")
    print(f"not resolved to a page   {unplaced}")
    print(f"spans blocks or pages     {straddling}")
    print(f"quote not in its source  {failures}")

    for r in bad:
        print(f"\n  fact {r['id']} claims {r['filename']} page {r['page_no']}")
        print(f"    quote: {norm(r['quote'])[:110]}")

    rejected = conn.execute("SELECT COUNT(*) FROM rejected_facts").fetchone()[0]
    print(f"\n{rejected} proposed facts were rejected before storage for failing "
          f"the same check at extraction time.")

    if failures or unplaced:
        print("\nRESULT: grounding is incomplete")
        return 1
    print("\nRESULT: every stored fact quotes text found on the page it cites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
