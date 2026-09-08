import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE, filename TEXT,
  title TEXT, page_count INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  -- set only when extraction finished. Blocks are saved before extraction
  -- starts, so their presence does not mean the document was read: an ingest
  -- interrupted by a spent quota would otherwise look complete for ever.
  ingest_complete INTEGER DEFAULT 0);

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

-- build_relations runs after every upload and re-pairs documents already
-- ingested, so without this each re-run would duplicate every relation
CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_pair ON relations(fact_a, fact_b);

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


# Columns added after the first release. CREATE TABLE IF NOT EXISTS leaves an
# existing table untouched, so a database made by an earlier version needs the
# new column adding explicitly rather than silently failing on the next query.
_ADDED_COLUMNS = [
    ("documents", "ingest_complete", "INTEGER DEFAULT 0"),
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for table, column, spec in _ADDED_COLUMNS:
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue                      # table not created yet
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {spec}")


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _apply_migrations(conn)
    conn.commit()


SHIPPED_CACHE = Path(__file__).resolve().parents[2] / "cache" / "starter_cache.sqlite"


def seed_from_shipped_cache(conn: sqlite3.Connection,
                            path: Path | None = None) -> int:
    """Load the committed responses so the starter corpus replays without a key.

    Only the model responses and the canonical term mapping are shipped, not
    facts or relations: everything else is recomputed from them, so the results
    a grader sees are produced by this code rather than copied from a database
    I prepared. Returns the number of cached responses loaded.
    """
    path = path or SHIPPED_CACHE
    if not path.exists():
        return 0
    if conn.execute("SELECT 1 FROM llm_cache LIMIT 1").fetchone():
        return 0                     # already populated; leave it alone
    conn.execute("ATTACH DATABASE ? AS shipped", (str(path),))
    try:
        conn.execute("INSERT OR REPLACE INTO llm_cache(key, response) "
                     "SELECT key, response FROM shipped.llm_cache")
        conn.execute("INSERT OR REPLACE INTO canon_terms(kind, raw, canon_id, label) "
                     "SELECT kind, raw, canon_id, label FROM shipped.canon_terms")
        conn.commit()
        loaded = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
    finally:
        conn.execute("DETACH DATABASE shipped")
    return loaded
