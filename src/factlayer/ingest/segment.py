import re

from ..models import Block, Window

_NUMERIC = re.compile(r"\d[\d,]*\.?\d*")
_UNITISH = re.compile(
    r"(per cent|percent|%|million|billion|crore|cr\b|lakh|mn\b|bn\b|₹|rs\.?|usd|\$)",
    re.I)
_PERIODISH = re.compile(r"(fy\s?\d{2,4}|q[1-4]|20\d{2}|march 31|year ended)", re.I)


def density_score(text: str) -> float:
    if not text:
        return 0.0
    n = len(_NUMERIC.findall(text))
    u = len(_UNITISH.findall(text))
    p = len(_PERIODISH.findall(text))
    return (n + 2 * u + 2 * p) / max(len(text) / 200.0, 1.0)


def _emit(doc_id: int, index: int, ids: list[int], parts: list[str]) -> Window:
    """Join blocks and record where each one lands in the joined text."""
    spans, cursor = [], 0
    for part in parts:
        spans.append((cursor, cursor + len(part)))
        cursor += len(part) + 1          # +1 for the "\n" inserted by join
    text = "\n".join(parts)
    return Window(doc_id=doc_id, index=index, text=text, block_ids=list(ids),
                  block_spans=spans, density=density_score(text))


def build_windows(blocks: list[Block], doc_id: int, window_chars: int,
                  overlap: int) -> list[Window]:
    usable = [(i, b) for i, b in enumerate(blocks) if not b.is_boilerplate]
    windows: list[Window] = []
    cur_ids: list[int] = []
    cur_parts: list[str] = []
    cur_len = 0

    def flush():
        nonlocal cur_ids, cur_parts, cur_len
        if not cur_parts:
            return
        windows.append(_emit(doc_id, len(windows), cur_ids, cur_parts))
        keep, kept_len = [], 0
        for idx, part in zip(reversed(cur_ids), reversed(cur_parts)):
            if kept_len >= overlap:
                break
            keep.append((idx, part))
            kept_len += len(part)
        keep.reverse()
        cur_ids = [i for i, _ in keep]
        cur_parts = [p for _, p in keep]
        cur_len = kept_len

    for idx, b in usable:
        if cur_len + len(b.text) > window_chars and cur_parts:
            flush()
        cur_ids.append(idx)
        cur_parts.append(b.text)
        cur_len += len(b.text)
    if cur_parts:
        windows.append(_emit(doc_id, len(windows), cur_ids, cur_parts))
    return windows
