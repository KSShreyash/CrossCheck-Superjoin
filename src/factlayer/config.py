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
    upload_dir: Path = ROOT / "uploads"
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    model: str = os.getenv("FACTLAYER_MODEL", "gemini-3.6-flash")
    embed_model: str = "text-embedding-004"
    value_tolerance: float = 1e-3        # relative, absorbs printed rounding
    window_chars: int = 12000            # long-context extraction window
    window_overlap: int = 1200
    boilerplate_min_pages: int = 4       # repeats on >= N pages -> boilerplate
    gap_min_chars: int = 120             # below this on a full page -> gap
    # 12 produced 3,291 pairs from 553 facts on two starter documents with no
    # gain in the cases that matter; 6 halves the adjudication budget
    max_pairs_per_fact: int = 6


settings = Settings()
