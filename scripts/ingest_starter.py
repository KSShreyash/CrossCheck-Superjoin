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

# Windows defaults stdout to cp1252, which cannot encode the rupee sign these
# documents are full of. Redirecting output to a file would otherwise crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from factlayer.config import (key_list, model_list,          # noqa: E402
                              settings)
from factlayer.db import (connect, init_schema,               # noqa: E402
                          seed_from_shipped_cache)
from factlayer.llm.client import LLMClient, NoAPIKey         # noqa: E402
from factlayer.pipeline import (build_relations,               # noqa: E402
                                canonicalise_corpus, ingest)


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
    print(f"{total_windows} extraction calls, plus 2 for canonicalising the corpus")
    print(f"minimum before adjudication: ~{total_windows + 2} requests")
    print("\nThe Gemini free tier allows 20 requests per day per model, so reading "
          "every window\nof this corpus is not possible on one model in one day. "
          "--max-windows-per-doc N\nspends the budget on the N densest windows of "
          "each document; --max-model-calls\ncaps adjudication on top of that.")
    for n in (2, 3, 5):
        print(f"  --max-windows-per-doc {n}: ~{n * len(pdfs) + 2} requests before "
              f"adjudication")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=None,
                    help="directory to search for PDFs; defaults to the "
                         "starter-datasets folder bundled with the repository")
    ap.add_argument("--db", default=str(settings.db_path))
    ap.add_argument("--max-model-calls", type=int, default=None,
                    help="cap adjudication calls to stay inside a free tier")
    ap.add_argument("--limit", type=int, default=None,
                    help="ingest only the first N documents")
    ap.add_argument("--max-windows-per-doc", type=int, default=None,
                    help="read only the N densest windows of each document")
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many extraction calls this would cost, "
                         "without making any")
    args = ap.parse_args()

    root = Path(args.root) if args.root else (
        Path(__file__).resolve().parents[1] / "starter-datasets")
    pdfs = sorted(root.rglob("*.pdf"), key=lambda p: str(p).lower())
    if not pdfs:
        print(f"no PDFs under {root}")
        return 1
    if args.limit:
        pdfs = pdfs[:args.limit]

    if args.dry_run:
        return dry_run(pdfs)

    conn = connect(args.db)
    init_schema(conn)
    seeded = seed_from_shipped_cache(conn)
    if seeded:
        print(f"loaded {seeded} committed responses; documents already read "
              f"cost no requests")
    client = LLMClient(conn, settings.gemini_api_key, settings.model,
                       models=model_list(), api_keys=key_list())

    print(f"{len(pdfs)} documents -> {args.db}")
    if not settings.gemini_api_key:
        print("no GEMINI_API_KEY set: this will only work for cached responses\n")

    started = time.time()
    exhausted = False
    for n, pdf in enumerate(pdfs, start=1):
        if exhausted:
            print(f"[{n}/{len(pdfs)}] {pdf.name} - skipped, quota spent")
            continue
        print(f"[{n}/{len(pdfs)}] {pdf.name}", flush=True)
        t0 = time.time()
        try:
            doc_id = ingest(conn, client, pdf,
                            canonicalise_terms=False,
                            max_windows=args.max_windows_per_doc)
        except NoAPIKey as exc:
            print(f"    stopped: {exc}")
            return 2
        except Exception as exc:                   # noqa: BLE001
            # A daily quota running out mid-corpus must not discard the work
            # already done. Stop reading, keep what was extracted, and carry on
            # to the phases that can still run.
            print(f"    stopped reading: {type(exc).__name__}: {str(exc)[:100]}")
            exhausted = True
            continue
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

    print("\ncanonicalising the whole corpus in one pass...", flush=True)
    try:
        canonicalise_corpus(conn, client)
    except Exception as exc:                       # noqa: BLE001
        print(f"    canonicalisation incomplete: {type(exc).__name__}: "
              f"{str(exc)[:120]}")

    print("building relations...", flush=True)
    budget = 0 if exhausted else args.max_model_calls
    if exhausted:
        print("    quota spent, so relations are built by rule alone "
              "(no adjudication)")
    try:
        written = build_relations(conn, client, max_model_calls=budget)
    except Exception as exc:                       # noqa: BLE001
        print(f"    adjudication stopped: {type(exc).__name__}; "
              f"retrying by rule alone")
        written = build_relations(conn, client, max_model_calls=0)
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
