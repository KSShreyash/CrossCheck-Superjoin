import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # FACTLAYER_DB lets tests and deployments point at a different file
    db_path: Path = Path(os.getenv("FACTLAYER_DB", ROOT / "factlayer.sqlite"))
    upload_dir: Path = Path(os.getenv("FACTLAYER_UPLOADS", ROOT / "uploads"))
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    model: str = os.getenv("FACTLAYER_MODEL", "gemini-3.6-flash")
    # quota is counted per model, so several models stretch one key further
    model_rotation: str = os.getenv(
        "FACTLAYER_MODELS",
        "gemini-3.6-flash,gemini-3.7-flash,gemini-3.5-flash,gemini-3.1-flash-lite")
    # optional comma-separated additional keys
    api_key_rotation: str = os.getenv("GEMINI_API_KEYS", "")
    value_tolerance: float = 1e-3        # relative, absorbs printed rounding
    window_chars: int = 12000            # long-context extraction window
    window_overlap: int = 1200
    boilerplate_min_pages: int = 4       # repeats on >= N pages -> boilerplate
    gap_min_chars: int = 120             # below this on a full page -> gap
    # 12 gave 3,291 pairs from 553 facts with no gain; 6 halves the budget
    max_pairs_per_fact: int = 6


settings = Settings()


def _split(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def model_list() -> list[str]:
    """Models to rotate through, the configured default always first."""
    models = _split(settings.model_rotation)
    return [settings.model] + [m for m in models if m != settings.model]


def key_list() -> list[str]:
    """Keys to rotate through, the primary key always first."""
    keys = _split(settings.api_key_rotation)
    primary = [settings.gemini_api_key] if settings.gemini_api_key else []
    return primary + [k for k in keys if k not in primary]
