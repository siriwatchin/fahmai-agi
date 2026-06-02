from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from database_tools import DbToolConfig, QdrantDatabaseTools


TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".tsv"}


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 160) -> list[str]:
    text = text.strip()
    if not text:
        return []
    step = max_chars - overlap
    return [text[i : i + max_chars].strip() for i in range(0, len(text), step) if text[i : i + max_chars].strip()]


def iter_records(data_dir: Path):
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(data_dir))
        if "question" in rel.lower() or ".ipynb_checkpoints" in rel:
            continue
        try:
            if p.suffix.lower() in TEXT_SUFFIXES:
                text = p.read_text(errors="ignore")
                for chunk_id, chunk in enumerate(chunk_text(text)):
                    yield {"text": chunk, "path": rel, "source": "text", "chunk": chunk_id}
            elif p.suffix.lower() == ".csv":
                df = pd.read_csv(p, nrows=50)
                yield {
                    "text": f"[CSV_SAMPLE] {rel}\n{df.to_string(index=False)}",
                    "path": rel,
                    "source": "csv_sample",
                    "chunk": 0,
                }
        except Exception as e:
            yield {"text": f"[READ_ERROR] {rel}\n{e}", "path": rel, "source": "error", "chunk": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--collection", default=None)
    args = ap.parse_args()

    cfg = DbToolConfig.from_env()
    qdrant = QdrantDatabaseTools(cfg.qdrant_url, cfg.qdrant_api_key, args.collection or cfg.qdrant_collection, cfg.embed_model)

    if args.recreate:
        print(qdrant.recreate_collection(args.collection or cfg.qdrant_collection))

    batch: list[dict[str, Any]] = []
    next_id = 0
    for rec in iter_records(args.data_dir):
        batch.append(rec)
        if len(batch) >= args.batch_size:
            print(qdrant.upsert_texts(batch, collection=args.collection, start_id=next_id))
            next_id += len(batch)
            batch = []
    if batch:
        print(qdrant.upsert_texts(batch, collection=args.collection, start_id=next_id))
        next_id += len(batch)
    print({"ok": True, "total": next_id})


if __name__ == "__main__":
    main()

