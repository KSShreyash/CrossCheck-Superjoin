import re
from collections import defaultdict

from ..models import Block

_NUM = re.compile(r"\d+")


def _norm(text: str) -> str:
    # page numbers vary between repeats of the same running header
    return re.sub(r"\s+", " ", _NUM.sub("#", text)).strip().lower()


def mark_boilerplate(blocks: list[Block], min_pages: int) -> None:
    pages_by_norm: dict[str, set[int]] = defaultdict(set)
    for b in blocks:
        pages_by_norm[_norm(b.text)].add(b.page_no)
    repeated = {k for k, pages in pages_by_norm.items() if len(pages) >= min_pages}
    for b in blocks:
        if _norm(b.text) in repeated:
            b.is_boilerplate = True
