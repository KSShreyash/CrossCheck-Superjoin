import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTLAYER_DB", str(tmp_path / "api.sqlite"))
    from factlayer.api import app
    return TestClient(app)


def test_stats_and_empty_listings(client):
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/documents").json() == []
    assert client.get("/api/relations").json() == []
    assert client.get("/api/facts").json() == []


def test_upload_rejects_non_pdf(client):
    r = client.post("/api/documents",
                    files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 400
    assert "PDF" in r.json()["detail"]


def test_missing_records_are_404(client):
    assert client.get("/api/facts/999").status_code == 404
    assert client.get("/api/relations/999").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_every_page_renders(client):
    for path in ("/", "/facts", "/relations", "/gaps"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "Fact Knowledge Layer" in r.text


def test_relation_page_renders_a_real_relation(client, tmp_path, monkeypatch):
    from factlayer.api import get_conn
    conn = get_conn()
    conn.execute("INSERT INTO documents(id,sha256,filename,page_count) "
                 "VALUES (1,'a','ar24.pdf',10),(2,'b','deck.pdf',5)")
    conn.execute(
        "INSERT INTO facts(id,doc_id,subject,metric,value_raw,unit_raw,period_raw,"
        "qualifiers,claim_type,canon_value,canon_unit,period_start,period_end) VALUES "
        "(1,1,'Delhivery','revenue from operations','81,415.38','Rs million','FY24',"
        "'{\"basis\":\"consolidated\"}','measurement',81415380000.0,'INR',"
        "'2023-04-01','2024-03-31'),"
        "(2,2,'Delhivery','revenue from services','8,142','Rs Cr','FY24',"
        "'{\"basis\":\"consolidated\"}','measurement',81420000000.0,'INR',"
        "'2023-04-01','2024-03-31')")
    conn.execute("INSERT INTO evidence(fact_id,quote,page_no) VALUES "
                 "(1,'Rs 81,415.38 million',42),(2,'Rs 8,142 Cr',5)")
    conn.execute(
        "INSERT INTO relations(id,fact_a,fact_b,rule_verdict,final_verdict,"
        "qualifier_diff) VALUES (1,1,2,'corroborates','corroborates','{}')")
    conn.commit()

    r = client.get("/relations/1")
    assert r.status_code == 200
    # both evidence quotes appear side by side, with their sources
    assert "Rs 81,415.38 million" in r.text
    assert "Rs 8,142 Cr" in r.text
    assert "ar24.pdf" in r.text and "deck.pdf" in r.text
    assert "corroborates" in r.text

    listing = client.get("/api/relations?type=corroborates").json()
    assert len(listing) == 1 and listing[0]["a_doc"] == "ar24.pdf"
