import json
import sqlite3

from .models import Block, Fact, Window


def save_document(conn: sqlite3.Connection, sha: str, filename: str,
                  title: str | None, pages: int) -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO documents(sha256, filename, title, page_count) "
        "VALUES (?,?,?,?)", (sha, filename, title, pages))
    conn.commit()
    # rowcount distinguishes insert from ignore; lastrowid does not
    if cur.rowcount:
        return cur.lastrowid
    return conn.execute("SELECT id FROM documents WHERE sha256=?",
                        (sha,)).fetchone()["id"]


def save_blocks(conn: sqlite3.Connection, doc_id: int,
                blocks: list[Block]) -> list[int]:
    ids = []
    for b in blocks:
        cur = conn.execute(
            "INSERT INTO blocks(doc_id,page_no,block_index,text,x0,y0,x1,y1,"
            "is_boilerplate) VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, b.page_no, b.block_index, b.text, *b.bbox,
             int(b.is_boilerplate)))
        ids.append(cur.lastrowid)
    conn.commit()
    return ids


def _blocks_covering(window: Window, span: tuple[int, int]) -> list[int]:
    """Window block positions the span overlaps, using recorded spans."""
    return [pos for pos, (start, end) in enumerate(window.block_spans)
            if span[0] < end and span[1] > start]


def save_fact(conn: sqlite3.Connection, fact: Fact, window: Window,
              block_row_ids: list[int]) -> int:
    cur = conn.execute(
        "INSERT INTO facts(doc_id,subject,metric,value_raw,value_num,unit_raw,"
        "period_raw,qualifiers,claim_type,confidence,canon_value,canon_unit,"
        "period_start,period_end,period_kind,entity_id,metric_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (fact.doc_id, fact.subject, fact.metric, fact.value_raw, fact.value_num,
         fact.unit_raw, fact.period_raw, json.dumps(fact.qualifiers),
         fact.claim_type, fact.confidence, fact.canon_value, fact.canon_unit,
         fact.period_start, fact.period_end, fact.period_kind,
         fact.entity_id, fact.metric_id))
    fact_id = cur.lastrowid

    positions = _blocks_covering(window, fact.span) if fact.span else []
    row_ids = [block_row_ids[window.block_ids[p]] for p in positions
               if window.block_ids[p] < len(block_row_ids)]
    page_no, bbox = None, (None, None, None, None)
    if row_ids:
        # ORDER BY id because IN (...) does not preserve the order given.
        placeholders = ",".join("?" for _ in row_ids)
        rows = conn.execute(
            "SELECT id,page_no,x0,y0,x1,y1 FROM blocks WHERE id IN "
            "(" + placeholders + ") ORDER BY id", row_ids).fetchall()
        page_no = rows[0]["page_no"]
        # a quote can straddle a page break, so keep the box on its starting page
        same_page = [r for r in rows if r["page_no"] == page_no]
        bbox = (min(r["x0"] for r in same_page), min(r["y0"] for r in same_page),
                max(r["x1"] for r in same_page), max(r["y1"] for r in same_page))

    conn.execute(
        "INSERT INTO evidence(fact_id,quote,page_no,char_start,char_end,block_ids,"
        "x0,y0,x1,y1) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (fact_id, fact.evidence_quote, page_no,
         fact.span[0] if fact.span else None,
         fact.span[1] if fact.span else None,
         json.dumps(row_ids), *bbox))
    conn.commit()
    return fact_id


def save_gaps(conn: sqlite3.Connection, doc_id: int,
              gaps: list[tuple[int, str]]) -> None:
    conn.executemany("INSERT INTO gaps(doc_id,page_no,reason) VALUES (?,?,?)",
                     [(doc_id, p, r) for p, r in gaps])
    conn.commit()


def save_rejected(conn: sqlite3.Connection, doc_id: int,
                  rejected: list[dict]) -> None:
    conn.executemany(
        "INSERT INTO rejected_facts(doc_id,payload,reason) VALUES (?,?,?)",
        [(doc_id, json.dumps(r["payload"], default=str), r["reason"])
         for r in rejected])
    conn.commit()
