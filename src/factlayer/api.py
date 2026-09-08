import functools
import json
import os
import sqlite3
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import key_list, model_list, settings
from .db import connect, init_schema, seed_from_shipped_cache
from .llm.client import LLMClient, NoAPIKey
from .pipeline import build_relations, ingest

HERE = Path(__file__).parent
app = FastAPI(title="Fact Knowledge Layer")
templates = Jinja2Templates(directory=str(HERE / "templates"))
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


@functools.lru_cache(maxsize=8)
def _conn_for(path: str) -> sqlite3.Connection:
    conn = connect(path)
    init_schema(conn)
    seed_from_shipped_cache(conn)
    return conn


def get_conn() -> sqlite3.Connection:
    # resolved per request, not bound at import, so tests can redirect it
    return _conn_for(os.getenv("FACTLAYER_DB", str(settings.db_path)))


def get_client(conn) -> LLMClient:
    return LLMClient(conn, settings.gemini_api_key, settings.model,
                     models=model_list(), api_keys=key_list())


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur]


# ---------------------------------------------------------------- ingestion

def _run_ingest(db_path: str, pdf_path: str, job_id: str) -> None:
    conn = _conn_for(db_path)
    try:
        ingest(conn, get_client(conn), pdf_path, job_id=job_id)
        build_relations(conn, get_client(conn))
    except NoAPIKey as exc:
        conn.execute("UPDATE jobs SET stage='failed', error=? WHERE id=?",
                     (str(exc), job_id))
        conn.commit()
    except Exception as exc:                       # noqa: BLE001 - surfaced to the UI
        conn.execute("UPDATE jobs SET stage='failed', error=? WHERE id=?",
                     (f"{type(exc).__name__}: {exc}", job_id))
        conn.commit()


@app.post("/api/documents")
async def upload(background: BackgroundTasks, file: UploadFile):
    data = await file.read()
    if not data[:5].startswith(b"%PDF"):
        raise HTTPException(status_code=400,
                            detail="not a PDF (missing %PDF header)")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / (file.filename or "upload.pdf")
    dest.write_bytes(data)

    conn = get_conn()
    job_id = uuid.uuid4().hex
    conn.execute("INSERT INTO jobs(id, stage, done, total) VALUES (?,?,?,?)",
                 (job_id, "queued", 0, 0))
    conn.commit()
    db_path = os.getenv("FACTLAYER_DB", str(settings.db_path))
    background.add_task(_run_ingest, db_path, str(dest), job_id)
    return {"job_id": job_id, "filename": dest.name}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    row = get_conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="no such job")
    return dict(row)


# ------------------------------------------------------------------- reads

@app.get("/api/documents")
def documents():
    return _rows(get_conn().execute(
        "SELECT d.*, "
        "(SELECT COUNT(*) FROM facts f WHERE f.doc_id=d.id) AS fact_count, "
        "(SELECT COUNT(*) FROM gaps g WHERE g.doc_id=d.id) AS gap_count, "
        "(SELECT COUNT(*) FROM rejected_facts r WHERE r.doc_id=d.id) AS rejected_count "
        "FROM documents d ORDER BY d.id"))


@app.get("/api/documents/{doc_id}/gaps")
def gaps(doc_id: int):
    conn = get_conn()
    return {
        "gaps": _rows(conn.execute(
            "SELECT page_no, reason FROM gaps WHERE doc_id=? ORDER BY page_no",
            (doc_id,))),
        "rejected": _rows(conn.execute(
            "SELECT payload, reason FROM rejected_facts WHERE doc_id=?", (doc_id,))),
    }


@app.get("/api/facts")
def facts(entity: str | None = None, metric: str | None = None,
          period: str | None = None, doc: int | None = None,
          q: str | None = None, limit: int = 200, offset: int = 0):
    where, args = [], []
    if entity:
        where.append("f.entity_id = ?")
        args.append(entity)
    if metric:
        where.append("f.metric_id = ?")
        args.append(metric)
    if period:
        where.append("f.period_start = ?")
        args.append(period)
    if doc:
        where.append("f.doc_id = ?")
        args.append(doc)
    if q:
        where.append("(f.subject LIKE ? OR f.metric LIKE ? OR e.quote LIKE ?)")
        args += [f"%{q}%"] * 3
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    args += [limit, offset]
    return _rows(get_conn().execute(
        "SELECT f.*, d.filename, e.quote, e.page_no FROM facts f "
        "JOIN documents d ON d.id=f.doc_id "
        "LEFT JOIN evidence e ON e.fact_id=f.id "
        f"{clause} ORDER BY f.id LIMIT ? OFFSET ?", args))


@app.get("/api/facts/{fact_id}")
def fact(fact_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT f.*, d.filename FROM facts f JOIN documents d ON d.id=f.doc_id "
        "WHERE f.id=?", (fact_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="no such fact")
    out = dict(row)
    out["qualifiers"] = json.loads(out.get("qualifiers") or "{}")
    ev = conn.execute("SELECT * FROM evidence WHERE fact_id=?", (fact_id,)).fetchone()
    out["evidence"] = dict(ev) if ev else None
    out["relations"] = _rows(conn.execute(
        "SELECT * FROM relations WHERE fact_a=? OR fact_b=?", (fact_id, fact_id)))
    return out


@app.get("/api/relations")
def relations(type: str | None = None, limit: int = 200, offset: int = 0):
    where, args = [], []
    if type:
        where.append("r.final_verdict = ?")
        args.append(type)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    args += [limit, offset]
    return _rows(get_conn().execute(
        "SELECT r.*, "
        "fa.subject AS a_subject, fa.metric AS a_metric, fa.value_raw AS a_value, "
        "fa.unit_raw AS a_unit, fa.period_raw AS a_period, da.filename AS a_doc, "
        "fb.subject AS b_subject, fb.metric AS b_metric, fb.value_raw AS b_value, "
        "fb.unit_raw AS b_unit, fb.period_raw AS b_period, db.filename AS b_doc "
        "FROM relations r "
        "JOIN facts fa ON fa.id=r.fact_a JOIN facts fb ON fb.id=r.fact_b "
        "JOIN documents da ON da.id=fa.doc_id JOIN documents db ON db.id=fb.doc_id "
        f"{clause} ORDER BY r.id LIMIT ? OFFSET ?", args))


@app.get("/api/relations/{rel_id}")
def relation(rel_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM relations WHERE id=?", (rel_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="no such relation")
    out = dict(row)
    out["qualifier_diff"] = json.loads(out.get("qualifier_diff") or "{}")
    if out.get("claimed_transform"):
        out["claimed_transform"] = json.loads(out["claimed_transform"])
    out["a"] = fact(row["fact_a"])
    out["b"] = fact(row["fact_b"])
    return out


@app.get("/api/stats")
def stats():
    conn = get_conn()

    def count(sql, *a):
        return conn.execute(sql, a).fetchone()[0]

    verdicts = {r["final_verdict"] or "unclassified": r["n"] for r in conn.execute(
        "SELECT final_verdict, COUNT(*) n FROM relations GROUP BY final_verdict")}
    return {
        "documents": count("SELECT COUNT(*) FROM documents"),
        "facts": count("SELECT COUNT(*) FROM facts"),
        "grounded_facts": count("SELECT COUNT(*) FROM evidence WHERE page_no IS NOT NULL"),
        "relations": count("SELECT COUNT(*) FROM relations"),
        "gaps": count("SELECT COUNT(*) FROM gaps"),
        "rejected": count("SELECT COUNT(*) FROM rejected_facts"),
        "with_period": count("SELECT COUNT(*) FROM facts WHERE period_start IS NOT NULL"),
        "verdicts": verdicts,
        "has_api_key": bool(settings.gemini_api_key),
    }


# --------------------------------------------------------------------- UI

@app.get("/", response_class=HTMLResponse)
def page_documents(request: Request):
    jobs = _rows(get_conn().execute(
        "SELECT * FROM jobs ORDER BY rowid DESC LIMIT 5"))
    running = any(j["stage"] not in ("done", "failed", "already ingested")
                  for j in jobs)
    return templates.TemplateResponse(
        request, "documents.html", { "stats": stats(),
                           "documents": documents(), "jobs": jobs,
                           "refresh": running})


@app.post("/upload")
async def upload_form(background: BackgroundTasks, file: UploadFile):
    """Browser form post: ingest, then send the user back to the list."""
    await upload(background, file)
    return RedirectResponse(url="/", status_code=303)


@app.get("/facts", response_class=HTMLResponse)
def page_facts(request: Request, q: str | None = None, doc: int | None = None):
    return templates.TemplateResponse(
        request, "facts.html", { "stats": stats(),
                       "facts": facts(q=q, doc=doc), "documents": documents(),
                       "q": q or "", "doc": doc})


@app.get("/relations", response_class=HTMLResponse)
def page_relations(request: Request, type: str | None = None):
    return templates.TemplateResponse(
        request, "relations.html", { "stats": stats(),
                           "relations": relations(type=type), "type": type or ""})


@app.get("/relations/{rel_id}", response_class=HTMLResponse)
def page_relation(request: Request, rel_id: int):
    return templates.TemplateResponse(
        request, "relation.html", { "stats": stats(),
                          "rel": relation(rel_id)})


@app.get("/gaps", response_class=HTMLResponse)
def page_gaps(request: Request):
    conn = get_conn()
    rows = _rows(conn.execute(
        "SELECT g.page_no, g.reason, d.filename, d.id AS doc_id "
        "FROM gaps g JOIN documents d ON d.id=g.doc_id ORDER BY d.id, g.page_no"))
    rejected = _rows(conn.execute(
        "SELECT r.payload, r.reason, d.filename FROM rejected_facts r "
        "JOIN documents d ON d.id=r.doc_id ORDER BY r.id LIMIT 200"))
    return templates.TemplateResponse(
        request, "gaps.html", { "stats": stats(),
                      "gaps": rows, "rejected": rejected})
