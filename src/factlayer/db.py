import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE, filename TEXT,
  title TEXT, page_count INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS blocks (
  id INTEGER PRIMARY KEY, doc_id INTEGER, page_no INTEGER, block_index INTEGER,
  text TEXT, x0 REAL, y0 REAL, x1 REAL, y1 REAL, is_boilerplate INTEGER DEFAULT 0,
  FOREIGN KEY(doc_id) REFERENCES documents(id));

CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY, doc_id INTEGER, subject TEXT, metric TEXT,
  value_raw TEXT, value_num REAL, unit_raw TEXT, period_raw TEXT,
  qualifiers TEXT, claim_type TEXT, confidence REAL,
  canon_value REAL, canon_unit TEXT,
  period_start TEXT, period_end TEXT, period_kind TEXT,
  entity_id TEXT, metric_id TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(id));

CREATE TABLE IF NOT EXISTS evidence (
  id INTEGER PRIMARY KEY, fact_id INTEGER, quote TEXT, page_no INTEGER,
  char_start INTEGER, char_end INTEGER, block_ids TEXT,
  x0 REAL, y0 REAL, x1 REAL, y1 REAL,
  FOREIGN KEY(fact_id) REFERENCES facts(id));

CREATE TABLE IF NOT EXISTS relations (
  id INTEGER PRIMARY KEY, fact_a INTEGER, fact_b INTEGER,
  rule_verdict TEXT, model_verdict TEXT, final_verdict TEXT,
  reason_code TEXT, explanation TEXT, qualifier_diff TEXT,
  claimed_transform TEXT, verified INTEGER, agreed INTEGER, confidence REAL,
  FOREIGN KEY(fact_a) REFERENCES facts(id), FOREIGN KEY(fact_b) REFERENCES facts(id));

CREATE TABLE IF NOT EXISTS llm_cache (
  key TEXT PRIMARY KEY, response TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS rejected_facts (
  id INTEGER PRIMARY KEY, doc_id INTEGER, payload TEXT, reason TEXT);

CREATE TABLE IF NOT EXISTS gaps (
  id INTEGER PRIMARY KEY, doc_id INTEGER, page_no INTEGER, reason TEXT);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, doc_id INTEGER, stage TEXT, done INTEGER, total INTEGER,
  facts INTEGER DEFAULT 0, error TEXT);

CREATE TABLE IF NOT EXISTS canon_terms (
  kind TEXT, raw TEXT, canon_id TEXT, label TEXT, PRIMARY KEY (kind, raw));

CREATE INDEX IF NOT EXISTS idx_facts_metric ON facts(metric_id);
CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id);
CREATE INDEX IF NOT EXISTS idx_rel_verdict ON relations(final_verdict);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # background ingest writes while the UI polls; without this the reader
    # raises "database is locked" instead of waiting for the writer
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
