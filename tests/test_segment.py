from factlayer.ingest.segment import build_windows, density_score, split_window
from factlayer.models import Block


def _b(i, text):
    return Block(page_no=1 + i // 5, block_index=i, text=text, bbox=(0, 0, 1, 1))


def test_windows_respect_size_and_overlap():
    blocks = [_b(i, "x" * 400) for i in range(20)]
    wins = build_windows(blocks, doc_id=1, window_chars=1000, overlap=400)
    assert len(wins) > 1
    assert all(len(w.text) <= 1600 for w in wins)
    assert set(wins[0].block_ids) & set(wins[1].block_ids)


def test_boilerplate_blocks_are_excluded():
    b = _b(0, "Annual Report 2023-24")
    b.is_boilerplate = True
    wins = build_windows([b, _b(1, "Revenue 81,415.38")], 1, 1000, 100)
    assert "Annual Report" not in wins[0].text


def test_density_prefers_numeric_text():
    assert density_score("Revenue from operations was 81,415.38 million in FY24") > \
           density_score("The board places on record its appreciation")


def test_block_spans_survive_newlines_inside_a_block():
    # PDF blocks carry their own newlines, so window.text.split("\n") does NOT
    # recover block boundaries. Spans must be recorded when the window is built.
    a = Block(1, 0, "Revenue from operations\nwas 81,415.38 million", (0, 0, 1, 1))
    b = Block(1, 1, "Unrelated text", (0, 0, 1, 1))
    w = build_windows([a, b], doc_id=1, window_chars=10_000, overlap=0)[0]
    assert len(w.block_spans) == 2
    start, end = w.block_spans[0]
    assert w.text[start:end] == a.text
    quote = "81,415.38 million"
    at = w.text.index(quote)
    assert start <= at and at + len(quote) <= end   # quote belongs to block 0


def test_split_window_halves_at_a_block_boundary():
    blocks = [Block(1, i, f"block {i} text\nwith a newline", (0, 0, 1, 1))
              for i in range(4)]
    w = build_windows(blocks, doc_id=1, window_chars=10_000, overlap=0)[0]
    halves = split_window(w)
    assert len(halves) == 2
    assert [len(h.block_ids) for h in halves] == [2, 2]
    # spans must be recomputed against each half's own text, not inherited
    for h in halves:
        for bid, (s, e) in zip(h.block_ids, h.block_spans):
            assert h.text[s:e] == blocks[bid].text
    assert halves[0].block_ids + halves[1].block_ids == w.block_ids


def test_split_window_declines_when_there_is_nothing_to_split():
    blocks = [Block(1, 0, "only one block", (0, 0, 1, 1))]
    w = build_windows(blocks, doc_id=1, window_chars=10_000, overlap=0)[0]
    assert split_window(w) == []
