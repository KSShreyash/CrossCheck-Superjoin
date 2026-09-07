from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    page_no: int
    block_index: int
    text: str
    bbox: tuple[float, float, float, float]
    is_boilerplate: bool = False


@dataclass
class Window:
    doc_id: int
    index: int
    text: str
    block_ids: list[int]
    # char span of each block inside `text`, parallel to block_ids.
    # Recorded at build time because block text contains its own newlines,
    # so splitting the window on "\n" does not recover block boundaries.
    block_spans: list[tuple[int, int]] = field(default_factory=list)
    density: float = 0.0


@dataclass
class Fact:
    doc_id: int
    subject: str
    metric: str
    value_raw: str | None
    value_num: float | None
    unit_raw: str | None
    period_raw: str | None
    qualifiers: dict[str, Any] = field(default_factory=dict)
    claim_type: str = "measurement"
    evidence_quote: str = ""
    confidence: float = 0.0
    # filled by normalisation
    canon_value: float | None = None
    canon_unit: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    period_kind: str | None = None
    entity_id: str | None = None
    metric_id: str | None = None
    # char span of evidence_quote inside its window, set during extraction
    span: tuple[int, int] | None = None
