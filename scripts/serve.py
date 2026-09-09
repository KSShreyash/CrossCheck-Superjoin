"""Run the web interface without installing the package first.

    python scripts/serve.py
    python scripts/serve.py --port 8080
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    # PORT and HOST come from the environment on a hosted platform
    ap.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run:  pip install -e \".[dev]\"")
        return 1

    from factlayer.config import settings
    from factlayer.db import connect, init_schema

    conn = connect(settings.db_path)
    init_schema(conn)
    docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    conn.close()

    if docs:
        print(f"{docs} documents, {facts} facts already ingested")
    else:
        print("Nothing ingested yet. In another terminal, run:")
        print("    python scripts/ingest_starter.py")
        print("The page will tell you the same thing.")

    print(f"\nOpen http://{args.host}:{args.port}\n")
    uvicorn.run("factlayer.api:app", host=args.host, port=args.port,
                reload=args.reload, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
