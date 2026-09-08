"""Ingest a directory of PDFs and build the relations between them.

Ingest order is fixed and sorted. Canonicalisation is incremental, so its
prompt - and therefore its cache key - reflects what was ingested before it.
Replaying the committed cache requires the same order.

    python scripts/ingest_starter.py ../starter-datasets/starter-datasets
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factlayer.config import settings                       # noqa: E402
from factlayer.db import connect, init_schema                # noqa: E402
from factlayer.llm.client import LLMClient, NoAPIKey         # noqa: E402
from factlayer.pipeline import build_relations, ingest       # noqa: E402


def dry_run(pdfs) -> int:
    """Cost the run before spending any quota on it."""
    from factlayer.ingest.boilerplate import mark_boilerplate
    from factlayer.ingest.gaps import find_gaps
    from factlayer.ingest.pdf import extract_blocks
    from factlayer.ingest.segment import build_windows

    total_windows = total_pages = total_gaps = 0
    print(f"{'document':46} {'pages':>6} {'windows':>8} {'gaps':>5}")
    for pdf in pdfs:
        blocks, pages = extract_blocks(pdf)
        mark_boilerplate(blocks, settings.boilerplate_min_pages)
        gaps = find_gaps(blocks, pages, settings.gap_min_chars)
        windows = build_windows(blocks, 0, settings.window_chars,
                                settings.window_overlap)
        total_windows += len(windows)
        total_pages += pages
        total_gaps += len(gaps)
        print(f"{pdf.name[:46]:46} {pages:>6} {len(windows):>8} {len(gaps):>5}")

    print(f"\n{len(pdfs)} documents, {total_pages} pages, {total_gaps} unreadable")
    print(f"{total_windows} extraction calls, plus 2 canonicalisation calls per "
          f"document ({2 * len(pdfs)})")
    print(f"minimum before adjudication: ~{total_windows + 2 * len(pdfs)} requests")
    print("\nAdjudication depends on how many pairs the rules cannot settle, so it "
          "cannot be counted\nup front. Use --max-model-calls to cap it.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory to search for PDFs")
    ap.add_argument("--db", default=str(settings.db_path))
    ap.add_argument("--max-model-calls", type=int, default=None,
                    help="cap adjudication calls to stay inside a free tier")
    ap.add_argument("--limit", type=int, default=None,
                    help="ingest only the first N documents")
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many extraction calls this would cost, "
                         "without making any")
    args = ap.parse_args()

    pdfs = sorted(Path(args.root).rglob("*.pdf"), key=lambda p: str(p).lower())
    if not pdfs:
        print(f"no PDFs under {args.root}")
        return 1
    if args.limit:
        pdfs = pdfs[:args.limit]

    if args.dry_run:
        return dry_run(pdfs)

    conn = connect(args.db)
    init_schema(conn)
    client = LLMClient(conn, settings.gemini_api_key, settings.model)

    print(f"{len(pdfs)} documents -> {args.db}")
    if not settings.gemini_api_key:
        print("no GEMINI_API_KEY set: this will only work for cached responses\n")

    started = time.time()
    for n, pdf in enumerate(pdfs, start=1):
        print(f"[{n}/{len(pdfs)}] {pdf.name}", flush=True)
        t0 = time.time()
        try:
            doc_id = ingest(conn, client, pdf)
        except NoAPIKey as exc:
            print(f"    stopped: {exc}")
            return 2
        facts = conn.execute("SELECT COUNT(*) FROM facts WHERE doc_id=?",
                             (doc_id,)).fetchone()[0]
        dated = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE doc_id=? AND period_start IS NOT NULL",
            (doc_id,)).fetchone()[0]
        rejected = conn.execute("SELECT COUNT(*) FROM rejected_facts WHERE doc_id=?",
                                (doc_id,)).fetchone()[0]
        gaps = conn.execute("SELECT COUNT(*) FROM gaps WHERE doc_id=?",
                            (doc_id,)).fetchone()[0]
        print(f"    {facts} facts ({dated} dated), {rejected} rejected, "
              f"{gaps} unreadable pages, {time.time() - t0:.1f}s")

    print("\nbuilding relations...", flush=True)
    written = build_relations(conn, client, max_model_calls=args.max_model_calls)
    print(f"{written} relations in {time.time() - started:.1f}s total\n")

    for row in conn.execute("SELECT final_verdict, COUNT(*) n FROM relations "
                            "GROUP BY final_verdict ORDER BY n DESC"):
        print(f"  {row['final_verdict']:24} {row['n']:>6}")

    total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    dated = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE period_start IS NOT NULL").fetchone()[0]
    grounded = conn.execute(
        "SELECT COUNT(*) FROM evidence WHERE page_no IS NOT NULL").fetchone()[0]
    print(f"\n{total} facts, {grounded} resolved to a page, "
          f"{dated} carry a period ({100 * dated / max(total, 1):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
