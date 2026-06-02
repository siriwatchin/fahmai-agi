from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def embed_query_prefix(text: str) -> str:
    return f"query: {text}"


def embed_passage_prefix(text: str) -> str:
    return f"passage: {text}"


@dataclass
class VectorSearchResult:
    score: float
    text: str
    payload: dict[str, Any]


class QdrantVectorTool:
    def __init__(self, url: str, api_key: str | None, collection: str, embed_model: str):
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key)
        self.encoder = SentenceTransformer(embed_model)

    @property
    def dim(self) -> int:
        return int(self.encoder.get_sentence_embedding_dimension())

    def recreate_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def upsert_texts(self, records: list[dict[str, Any]], batch_size: int = 64) -> None:
        point_id = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            texts = [embed_passage_prefix(r["text"]) for r in batch]
            vectors = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
            points = []
            for r, v in zip(batch, vectors):
                payload = dict(r)
                text = payload.pop("text")
                payload["text"] = text
                points.append(PointStruct(id=point_id, vector=v, payload=payload))
                point_id += 1
            self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int = 8, source: str | None = None) -> list[VectorSearchResult]:
        vector = self.encoder.encode([embed_query_prefix(query)], normalize_embeddings=True)[0].tolist()
        flt = None
        if source:
            flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )
        out = []
        for h in hits:
            payload = dict(h.payload or {})
            text = str(payload.get("text", ""))
            out.append(VectorSearchResult(score=float(h.score), text=text, payload=payload))
        return out

