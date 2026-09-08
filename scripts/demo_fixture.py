"""Build a tiny synthetic corpus and run the whole pipeline on it, offline.

This is a smoke test and a way to see the interface with data in it before any
API key exists. The PDFs are written here and the model is a fixed stub, so
nothing in this script touches the network or the starter documents.

    python scripts/demo_fixture.py
    FACTLAYER_DB=demo.sqlite uvicorn factlayer.api:app

The patterns mirror the real reconciliations: crore against million, standalone
against consolidated, and two institutions disagreeing over one period.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Windows defaults stdout to cp1252, which cannot encode the rupee sign these
# documents are full of. Redirecting output to a file would otherwise crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from reportlab.lib.pagesizes import A4                      # noqa: E402
from reportlab.pdfgen import canvas                          # noqa: E402

from factlayer.db import connect, init_schema                # noqa: E402
from factlayer.pipeline import build_relations, ingest       # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "demo_fixture"
DB = Path(__file__).resolve().parents[1] / "demo.sqlite"

DOCS = {
    "annual-report-sample.pdf": [
        "Financial Performance",
        "The revenue from operations on a consolidated basis for FY24 stood at",
        "Rs 81,415.38 million as against Rs 72,253.01 million for FY23.",
        "The revenue from operations on a standalone basis for FY24 stood at",
        "Rs 74,540.82 million as against Rs 66,586.61 million for FY23.",
    ],
    "earnings-deck-sample.pdf": [
        "FY24 highlights",
        "Rs 8,142 Cr FY24 revenue from services, up 12.7 per cent",
    ],
    "central-bank-sample.pdf": [
        "Assessment and Prospects",
        "Taking these factors into account, real GDP growth for 2025-26 is",
        "projected at 6.5 per cent, with risks evenly balanced.",
    ],
    "fund-staff-report-sample.pdf": [
        "Staff Report",
        "Real GDP is projected to grow at 6.6 percent in FY2025/26 before",
        "moderating thereafter.",
    ],
}

# (pattern, subject, metric, unit, period, qualifiers)
RULES = [
    (r"Rs 81,415\.38 million", "Delhivery Limited", "revenue from operations",
     "Rs million", "FY24", {"basis": "consolidated"}),
    (r"Rs 72,253\.01 million", "Delhivery Limited", "revenue from operations",
     "Rs million", "FY23", {"basis": "consolidated"}),
    (r"Rs 74,540\.82 million", "Delhivery Limited", "revenue from operations",
     "Rs million", "FY24", {"basis": "standalone"}),
    (r"Rs 66,586\.61 million", "Delhivery Limited", "revenue from operations",
     "Rs million", "FY23", {"basis": "standalone"}),
    (r"Rs 8,142 Cr", "Delhivery Limited", "revenue from services",
     "Rs Cr", "FY24", {"basis": "consolidated"}),
    (r"6\.5 per cent", "India", "real GDP growth",
     "per cent", "2025-26", {"vintage": "projection", "source": "central bank"}),
    (r"6\.6 percent", "India", "real GDP growth",
     "percent", "FY2025/26", {"vintage": "projection", "source": "fund staff"}),
]


class StubModel:
    """Deterministic stand-in. Never called for pairs the rules can settle."""

    def __init__(self):
        self.calls = 0

    def complete_json(self, prompt: str, version: str, **kw) -> dict:
        self.calls += 1
        if version.startswith("extract"):
            excerpt = prompt.split("EXCERPT:", 1)[-1]
            facts = []
            for pattern, subj, metric, unit, period, quals in RULES:
                m = re.search(pattern, excerpt)
                if not m:
                    continue
                quote = m.group(0)
                value = re.search(r"[\d,]+\.?\d*", quote).group(0)
                facts.append({"subject": subj, "metric": metric,
                              "value_raw": value, "unit_raw": unit,
                              "period_raw": period, "qualifiers": quals,
                              "claim_type": "measurement",
                              "evidence_quote": quote, "confidence": 0.9})
            return {"facts": facts}

        if version.startswith("canon"):
            names = re.findall(r"^- (.+)$",
                               prompt.split("NAMES:", 1)[-1], re.M)
            groups = {}
            for n in names:
                low = n.lower()
                if "revenue" in low:
                    cid, label = "revenue_from_operations", "Revenue from operations"
                elif "gdp" in low:
                    cid, label = "real_gdp_growth", "Real GDP growth"
                elif "delhivery" in low:
                    cid, label = "delhivery_limited", "Delhivery Limited"
                elif "india" in low:
                    cid, label = "india", "India"
                else:
                    cid, label = re.sub(r"\W+", "_", low).strip("_"), n
                groups.setdefault(cid, {"canon_id": cid, "label": label,
                                        "members": []})["members"].append(n)
            return {"groups": list(groups.values())}

        # adjudication
        if "standalone" in prompt and "consolidated" in prompt:
            return {"verdict": "reconciled_by_context",
                    "reason_code": "reporting_basis",
                    "explanation": "One figure consolidates the subsidiaries and the "
                                   "other reports the parent alone, so the two are "
                                   "measuring different scopes of the same business.",
                    "claimed_transform": {"kind": "basis"}, "confidence": 0.92}
        return {"verdict": "contradicts", "reason_code": "genuine_disagreement",
                "explanation": "Both figures describe the same measure over the same "
                               "period and no recorded attribute distinguishes them.",
                "claimed_transform": {"kind": "none"}, "confidence": 0.8}


def write_pdfs() -> list[Path]:
    OUT.mkdir(exist_ok=True)
    paths = []
    for name, lines in DOCS.items():
        path = OUT / name
        c = canvas.Canvas(str(path), pagesize=A4)
        y = 780
        for line in lines:
            c.drawString(60, y, line)
            y -= 22
        c.showPage()
        c.save()
        paths.append(path)
    return paths


def main() -> int:
    for suffix in ("", "-wal", "-shm"):
        Path(str(DB) + suffix).unlink(missing_ok=True)
    paths = write_pdfs()
    conn = connect(DB)
    init_schema(conn)
    model = StubModel()

    for p in sorted(paths, key=lambda x: x.name):
        doc_id = ingest(conn, model, p)
        n = conn.execute("SELECT COUNT(*) FROM facts WHERE doc_id=?",
                         (doc_id,)).fetchone()[0]
        print(f"  {p.name:32} {n} facts")

    written = build_relations(conn, model)
    print(f"\n{written} relations, {model.calls} stub model calls\n")
    for r in conn.execute("SELECT final_verdict, COUNT(*) n FROM relations "
                          "GROUP BY final_verdict ORDER BY n DESC"):
        print(f"  {r['final_verdict']:24} {r['n']}")
    print(f"\ndatabase: {DB}")
    print(f"view it:  FACTLAYER_DB={DB.name} uvicorn factlayer.api:app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
