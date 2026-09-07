from collections import defaultdict

from ..models import Block


def find_gaps(blocks: list[Block], page_count: int,
              min_chars: int) -> list[tuple[int, str]]:
    chars: dict[int, int] = defaultdict(int)
    for b in blocks:
        chars[b.page_no] += len(b.text)
    gaps = []
    for page in range(1, page_count + 1):
        n = chars.get(page, 0)
        if n == 0:
            gaps.append((page, "no extractable text; page is likely image-only"))
        elif n < min_chars:
            gaps.append((page,
                         f"only {n} characters extracted; likely scanned or graphical"))
    return gaps
