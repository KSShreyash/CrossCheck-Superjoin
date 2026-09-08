from factlayer.db import connect, init_schema


def test_schema_creates_expected_tables(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"documents", "blocks", "facts", "evidence", "relations",
            "llm_cache", "rejected_facts", "jobs", "canon_terms"} <= names


def test_init_schema_is_idempotent(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_schema(conn)
    init_schema(conn)  # must not raise


def test_a_database_from_an_older_version_gains_new_columns(tmp_path):
    # CREATE TABLE IF NOT EXISTS leaves an existing table alone, so a column
    # added later has to be applied explicitly or every query using it fails
    import sqlite3
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(str(path))
    old.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, sha256 TEXT "
                "UNIQUE, filename TEXT, title TEXT, page_count INTEGER)")
    old.execute("INSERT INTO documents(sha256, filename) VALUES ('a','a.pdf')")
    old.commit()
    old.close()

    conn = connect(path)
    init_schema(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)")}
    assert "ingest_complete" in cols
    row = conn.execute("SELECT ingest_complete FROM documents").fetchone()
    assert row["ingest_complete"] == 0, "existing rows default to not ingested"


def test_shipped_cache_is_loaded_into_a_fresh_database(tmp_path):
    from factlayer.db import seed_from_shipped_cache
    shipped = tmp_path / "shipped.sqlite"
    src = connect(shipped)
    init_schema(src)
    src.execute("INSERT INTO llm_cache(key, response) VALUES ('k','{\"a\":1}')")
    src.execute("INSERT INTO canon_terms VALUES ('metric','x','cid','X')")
    src.commit()
    src.close()

    conn = connect(tmp_path / "work.sqlite")
    init_schema(conn)
    assert seed_from_shipped_cache(conn, shipped) == 1
    assert conn.execute("SELECT response FROM llm_cache").fetchone()[0] == '{"a":1}'
    assert conn.execute("SELECT canon_id FROM canon_terms").fetchone()[0] == "cid"


def test_seeding_leaves_an_already_populated_database_alone(tmp_path):
    from factlayer.db import seed_from_shipped_cache
    shipped = tmp_path / "shipped.sqlite"
    src = connect(shipped)
    init_schema(src)
    src.execute("INSERT INTO llm_cache(key, response) VALUES ('k','shipped')")
    src.commit()
    src.close()

    conn = connect(tmp_path / "work.sqlite")
    init_schema(conn)
    conn.execute("INSERT INTO llm_cache(key, response) VALUES ('mine','local')")
    conn.commit()
    assert seed_from_shipped_cache(conn, shipped) == 0
    assert conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 1
