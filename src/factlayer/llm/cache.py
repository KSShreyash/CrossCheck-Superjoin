import hashlib
import json
import sqlite3


def cache_key(model: str, prompt_version: str, payload: str) -> str:
    h = hashlib.sha256()
    h.update(f"{model}\x00{prompt_version}\x00{payload}".encode("utf-8"))
    return h.hexdigest()


def get(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
    return json.loads(row["response"]) if row else None


def put(conn: sqlite3.Connection, key: str, response: dict) -> None:
    conn.execute("INSERT OR REPLACE INTO llm_cache(key, response) VALUES (?,?)",
                 (key, json.dumps(response, ensure_ascii=False)))
    conn.commit()
