from factlayer.ingest.boilerplate import mark_boilerplate
from factlayer.ingest.gaps import find_gaps
from factlayer.models import Block


def _b(page, text):
    return Block(page_no=page, block_index=0, text=text, bbox=(0, 0, 1, 1))


def test_repeated_running_header_is_marked():
    blocks = [_b(p, "Annual Report 2023-24") for p in range(1, 6)]
    blocks.append(_b(1, "Revenue from operations 81,415.38"))
    mark_boilerplate(blocks, min_pages=4)
    assert all(b.is_boilerplate for b in blocks if "Annual Report" in b.text)
    assert not [b for b in blocks if "81,415" in b.text][0].is_boilerplate


def test_page_with_no_text_is_reported_as_gap():
    blocks = [_b(1, "cover page has no text layer" * 20)]
    gaps = find_gaps(blocks, page_count=3, min_chars=120)
    pages = {p for p, _ in gaps}
    assert pages == {2, 3}


def test_page_with_trivial_text_is_a_gap():
    blocks = [_b(1, "12")]
    gaps = find_gaps(blocks, page_count=1, min_chars=120)
    assert gaps and gaps[0][0] == 1
