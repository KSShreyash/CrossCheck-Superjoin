import hashlib
from pathlib import Path

import pymupdf

from ..models import Block


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_blocks(pdf_path: str | Path) -> tuple[list[Block], int]:
    doc = pymupdf.open(str(pdf_path))
    blocks: list[Block] = []
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            for i, b in enumerate(page.get_text("blocks")):
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                text = (text or "").strip()
                if not text:
                    continue
                blocks.append(Block(page_no=pno + 1, block_index=i, text=text,
                                    bbox=(x0, y0, x1, y1)))
        return blocks, doc.page_count
    finally:
        doc.close()
