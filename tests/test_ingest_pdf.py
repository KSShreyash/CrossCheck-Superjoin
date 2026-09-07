from factlayer.ingest.pdf import extract_blocks, file_sha256


def test_extracts_text_page_and_bbox(make_pdf):
    p = make_pdf([["Revenue from operations 81,415.38"], ["Second page"]])
    blocks, pages = extract_blocks(p)
    assert pages == 2
    assert any("81,415.38" in b.text for b in blocks)
    first = next(b for b in blocks if "81,415.38" in b.text)
    assert first.page_no == 1
    x0, y0, x1, y1 = first.bbox
    assert x1 > x0 and y1 > y0


def test_sha256_is_stable(make_pdf):
    p = make_pdf([["same"]])
    assert file_sha256(p) == file_sha256(p)
