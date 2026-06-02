from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Settings
from .vector_tool import QdrantVectorTool


TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".tsv"}


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 160) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    step = max_chars - overlap
    for i in range(0, len(text), step):
        chunk = text[i : i + max_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def iter_records(data_dir: Path):
    for p in tqdm(list(data_dir.rglob("*")), desc="scan"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(data_dir))
        if "question" in rel.lower() or ".ipynb_checkpoints" in rel:
            continue
        try:
            if p.suffix.lower() in TEXT_SUFFIXES:
                text = p.read_text(errors="ignore")
                for idx, chunk in enumerate(chunk_text(text)):
                    yield {"text": chunk, "path": rel, "chunk": idx, "source": "text"}
            elif p.suffix.lower() == ".csv":
                df = pd.read_csv(p, nrows=40)
                text = f"[CSV_SAMPLE] {rel}\n{df.to_string(index=False)}"
                yield {"text": text, "path": rel, "chunk": 0, "source": "csv_sample"}
        except Exception as e:
            yield {"text": f"[READ_ERROR] {rel}\n{e}", "path": rel, "chunk": 0, "source": "error"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    settings = Settings.from_env()
    data_dir = args.data_dir or settings.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    tool = QdrantVectorTool(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        embed_model=settings.embed_model,
    )
    if args.recreate:
        tool.recreate_collection()

    batch = []
    total = 0
    for rec in iter_records(data_dir):
        batch.append(rec)
        if len(batch) >= args.batch_size:
            tool.upsert_texts(batch, batch_size=args.batch_size)
            total += len(batch)
            print("upserted", total)
            batch = []
    if batch:
        tool.upsert_texts(batch, batch_size=args.batch_size)
        total += len(batch)
    print("done", total)


if __name__ == "__main__":
    main()

