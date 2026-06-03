from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import agentic_best_integrated_qdrant as pipeline


API_OUTPUT_DIR = Path(os.getenv("API_OUTPUT_DIR", str(Path.home() / "bank500")))
NO_QDRANT = os.getenv("NO_QDRANT", "0").lower() in {"1", "true", "yes"}
SKIP_QDRANT_PRELOAD = os.getenv("SKIP_QDRANT_PRELOAD", "0").lower() in {"1", "true", "yes"}
API_PORT = int(os.getenv("API_PORT", "8888"))
ENABLE_API_CACHE = os.getenv("ENABLE_API_CACHE", "1").lower() not in {"0", "false", "no"}
API_PRELOAD_ANSWERS = os.getenv("API_PRELOAD_ANSWERS", "1").lower() not in {"0", "false", "no"}
API_PRELOAD_RESULTS = Path(os.getenv("API_PRELOAD_RESULTS", "")).expanduser() if os.getenv("API_PRELOAD_RESULTS") else None


def _norm_question(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _looks_like_day_question(question: str) -> bool:
    q = question.strip().lower()
    return q in {"วันนี้วันอะไร", "today is what day?", "what day is today?"}


def _thai_today() -> str:
    # Keep this only for API smoke tests. Competition answers still go through the pipeline.
    names = ["วันจันทร์", "วันอังคาร", "วันพุธ", "วันพฤหัสบดี", "วันศุกร์", "วันเสาร์", "วันอาทิตย์"]
    return names[pd.Timestamp.now(tz="Asia/Bangkok").weekday()]


class ChatPayload(BaseModel):
    question: str = Field(..., min_length=1)
    id: str | None = None


class ChatRequest(BaseModel):
    data: ChatPayload


class ChatAnswer(BaseModel):
    answer: str


class ChatResponse(BaseModel):
    data: ChatAnswer


@dataclass
class RuntimeState:
    sqltool: Any
    retriever: Any
    qdrant_retriever: Any
    tok: Any
    model: Any
    question_to_id: dict[str, str]
    answer_cache: dict[str, str]
    cache_hits: int
    cache_misses: int
    lock: asyncio.Lock


state: RuntimeState | None = None


def _load_question_index() -> dict[str, str]:
    question_to_id: dict[str, str] = {}
    try:
        qdf, id_col, q_col = pipeline.load_questions()
        for _, row in qdf.iterrows():
            qid = str(row[id_col]).strip()
            question = _norm_question(str(row[q_col]))
            if qid and question:
                question_to_id[question] = qid
    except Exception as exc:
        print("question_index_error:", exc, flush=True)
    return question_to_id


def _cache_keys(qid: str | None, question: str) -> list[str]:
    question = _norm_question(question)
    keys = []
    if qid:
        keys.append(f"id:{qid}")
    if question:
        keys.append(f"q:{question}")
    return keys


def _latest_output_results() -> Path | None:
    out_root = Path(os.getenv("OUTPUT_ROOT", str(API_OUTPUT_DIR / "output"))).expanduser()
    if not out_root.exists():
        return None
    candidates = sorted(
        [p / "best_results.csv" for p in out_root.iterdir() if (p / "best_results.csv").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_answer_cache(question_to_id: dict[str, str]) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not API_PRELOAD_ANSWERS:
        return cache

    paths = []
    if API_PRELOAD_RESULTS:
        paths.append(API_PRELOAD_RESULTS)
    latest = _latest_output_results()
    if latest:
        paths.append(latest)
    legacy = API_OUTPUT_DIR / "best_results.csv"
    if legacy.exists():
        paths.append(legacy)

    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print("api_cache_load_error:", path, exc, flush=True)
            continue
        if not {"id", "answer"}.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            qid = str(row.get("id", "")).strip()
            answer = str(row.get("answer", "")).strip()
            question = str(row.get("question", "")).strip()
            if not answer or answer.lower() == "nan":
                continue
            if not question and qid:
                for q, mapped_qid in question_to_id.items():
                    if mapped_qid == qid:
                        question = q
                        break
            for key in _cache_keys(qid, question):
                cache[key] = answer
        print("api_cache_loaded:", len(cache), "from", path, flush=True)
        if cache:
            break
    return cache


def _save_api_debug(qid: str, question: str, answer: str, obs: dict[str, Any], seconds: float) -> None:
    API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "id": qid,
        "question": question,
        "answer": answer,
        "seconds": round(seconds, 3),
        "sql_backend": getattr(state.sqltool, "backend", None) if state else None,
        "qdrant_enabled": bool(state and state.qdrant_retriever and state.qdrant_retriever.ok),
        "observation": obs,
    }
    with (API_OUTPUT_DIR / "api_requests.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    token_df = pd.DataFrame(pipeline.TOKEN_LOG)
    token_df.to_csv(API_OUTPUT_DIR / "api_token_usage.csv", index=False)
    summary = {
        "num_llm_calls": int(len(token_df)),
        "prompt_tokens": int(token_df["prompt_tokens"].sum()) if len(token_df) else 0,
        "completion_tokens": int(token_df["completion_tokens"].sum()) if len(token_df) else 0,
        "total_tokens": int(token_df["total_tokens"].sum()) if len(token_df) else 0,
        "seconds": float(token_df["seconds"].sum()) if len(token_df) else 0,
        "sql_backend": getattr(state.sqltool, "backend", None) if state else None,
        "sql_error": getattr(state.sqltool, "error", None) if state else None,
        "qdrant_enabled": bool(state and state.qdrant_retriever and state.qdrant_retriever.ok),
        "qdrant_collection": getattr(state.qdrant_retriever, "collection", None) if state and state.qdrant_retriever else None,
    }
    (API_OUTPUT_DIR / "api_token_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def _load_runtime() -> RuntimeState:
    print("api: loading sql...", flush=True)
    sqltool = pipeline.SQLTool()
    print(
        "api: sql_backend:",
        sqltool.backend,
        "tables:",
        len(sqltool.tables),
        "sql_error:",
        sqltool.error,
        flush=True,
    )

    print("api: loading retrieval...", flush=True)
    retriever = pipeline.RetrievalTool()
    print("api: docs:", len(retriever.docs), flush=True)

    qdrant_retriever = None
    if not NO_QDRANT:
        print("api: loading qdrant...", flush=True)
        qdrant_retriever = pipeline.QdrantRetrievalTool()
        print(
            "api: qdrant_ok:",
            qdrant_retriever.ok,
            "collection:",
            qdrant_retriever.collection,
            "vector_name:",
            qdrant_retriever.vector_name,
            "error:",
            qdrant_retriever.error,
            flush=True,
        )
        if qdrant_retriever.ok and not SKIP_QDRANT_PRELOAD:
            print("api: preloading qdrant encoder:", qdrant_retriever.embed_model, flush=True)
            qdrant_retriever.preload_encoder()

    print("api: loading qwen model...", flush=True)
    tok, model = pipeline.load_model()
    print("api: model ready", flush=True)

    question_to_id = _load_question_index()
    print("api: question index:", len(question_to_id), flush=True)
    answer_cache = _load_answer_cache(question_to_id)
    print("api: answer cache:", len(answer_cache), "enabled:", ENABLE_API_CACHE, flush=True)

    return RuntimeState(
        sqltool=sqltool,
        retriever=retriever,
        qdrant_retriever=qdrant_retriever,
        tok=tok,
        model=model,
        question_to_id=question_to_id,
        answer_cache=answer_cache,
        cache_hits=0,
        cache_misses=0,
        lock=asyncio.Lock(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    # Load CUDA objects on the main server thread. On B200 MIG, loading or
    # generating through a worker thread can trip PyTorch's NVML allocator path.
    state = _load_runtime()
    yield


app = FastAPI(title="FahMai Qwen2.5 Agent API", version="1.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    if state is None:
        return {"ok": False, "status": "loading"}
    return {
        "ok": True,
        "sql_backend": getattr(state.sqltool, "backend", None),
        "sql_error": getattr(state.sqltool, "error", None),
        "qdrant_enabled": bool(state.qdrant_retriever and state.qdrant_retriever.ok),
        "qdrant_collection": getattr(state.qdrant_retriever, "collection", None) if state.qdrant_retriever else None,
        "questions_indexed": len(state.question_to_id),
        "api_cache_enabled": ENABLE_API_CACHE,
        "api_cache_size": len(state.answer_cache),
        "api_cache_hits": state.cache_hits,
        "api_cache_misses": state.cache_misses,
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
@app.post("/api/v2/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if state is None:
        raise HTTPException(status_code=503, detail="runtime is not ready")

    question = _norm_question(req.data.question)
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    # Cheap health/smoke-test answer; real competition questions still use the agent pipeline.
    if _looks_like_day_question(question):
        return ChatResponse(data=ChatAnswer(answer=_thai_today()))

    qid = req.data.id or state.question_to_id.get(question) or "API-Q"
    if ENABLE_API_CACHE:
        for key in _cache_keys(qid, question):
            if key in state.answer_cache:
                state.cache_hits += 1
                return ChatResponse(data=ChatAnswer(answer=state.answer_cache[key]))
        state.cache_misses += 1

    t0 = time.time()
    async with state.lock:
        # Keep Qwen inference on the same thread that initialized CUDA.
        answer, obs = pipeline.answer_one(
            state.sqltool,
            state.retriever,
            state.qdrant_retriever,
            state.tok,
            state.model,
            qid,
            question,
        )

    seconds = time.time() - t0
    if ENABLE_API_CACHE:
        for key in _cache_keys(qid, question):
            state.answer_cache[key] = answer
    _save_api_debug(qid, question, answer, obs, seconds)
    return ChatResponse(data=ChatAnswer(answer=answer))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=API_PORT)
