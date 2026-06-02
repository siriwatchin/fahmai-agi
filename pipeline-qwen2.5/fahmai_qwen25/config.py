from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def load_env() -> None:
    if load_dotenv:
        load_dotenv()


@dataclass(frozen=True)
class Settings:
    pg_dsn: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    embed_model: str
    qwen_model_path: Path
    data_dir: Path
    questions_csv: Path
    output_dir: Path
    max_context_chars: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        return cls(
            pg_dsn=os.getenv("PG_DSN", ""),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "fahmai_public"),
            embed_model=os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base"),
            qwen_model_path=Path(os.getenv("QWEN_MODEL_PATH", "")),
            data_dir=Path(os.getenv("DATA_DIR", "")),
            questions_csv=Path(os.getenv("QUESTIONS_CSV", "")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "9000")),
        )

