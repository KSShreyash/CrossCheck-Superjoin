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
