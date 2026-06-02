from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DB_TOOLS_DIR = ROOT / "database-tools"
sys.path.insert(0, str(ROOT / "pipeline-typhoon"))
sys.path.insert(0, str(DB_TOOLS_DIR))

import run_typhoon_database_tools as pipeline  # noqa: E402
from domain_tools import build_domain_registry  # noqa: E402


API_OUTPUT_DIR = Path(os.getenv("API_OUTPUT_DIR", str(ROOT / "outputs" / "typhoon_api")))
SCHEMA_CACHE = Path(os.getenv("SCHEMA_CACHE", str(ROOT / "outputs" / "schema_cache.json")))
REFRESH_SCHEMA_CACHE = os.getenv("REFRESH_SCHEMA_CACHE", "0").lower() in {"1", "true", "yes"}
QDRANT_MODE = os.getenv("QDRANT_MODE", "auto").lower()
NO_QDRANT = os.getenv("NO_QDRANT", "0").lower() in {"1", "true", "yes"}
MAX_STEPS = int(os.getenv("MAX_STEPS", "3"))


def _norm_question(text: str) -> str:
    text = str(text).strip()
    return re.sub(r"\s+", " ", text)


def _looks_like_day_question(question: str) -> bool:
    q = question.strip().lower()
    return q in {"วันนี้วันอะไร", "today is what day?", "what day is today?"}


def _thai_today() -> str:
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
    registry: Any
    tools: list[dict[str, Any]]
    schema_summary: str
    question_to_id: dict[str, str]
    include_qdrant: bool
    qdrant_mode: str
    lock: asyncio.Lock


state: RuntimeState | None = None


def _load_question_index() -> dict[str, str]:
    question_to_id: dict[str, str] = {}
    try:
        qdf = pd.read_csv(ROOT / "questions.csv")
        id_col = "id" if "id" in qdf.columns else qdf.columns[0]
        q_col = "question" if "question" in qdf.columns else qdf.columns[1]
        for _, row in qdf.iterrows():
            qid = str(row[id_col]).strip()
            question = _norm_question(str(row[q_col]))
            if qid and question:
                question_to_id[question] = qid
    except Exception as exc:
        print("question_index_error:", exc, flush=True)
    return question_to_id


def _save_api_debug(qid: str, question: str, answer: str, trace: list[dict[str, Any]], seconds: float) -> None:
    API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "id": qid,
        "question": question,
        "answer": answer,
        "seconds": round(seconds, 3),
        "qdrant_mode": state.qdrant_mode if state else None,
        "qdrant_loaded": state.include_qdrant if state else None,
        "trace": trace,
    }
    with (API_OUTPUT_DIR / "api_requests.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _load_runtime() -> RuntimeState:
    pipeline.load_env_files()
    pipeline.ensure_pg_dsn()
    if not (os.getenv("TYPHOON_API_KEY") or os.getenv("APIKEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("Missing Typhoon API key. Set TYPHOON_API_KEY or APIKEY before starting the API.")

    qdrant_mode = "never" if NO_QDRANT else QDRANT_MODE
    if qdrant_mode not in {"auto", "always", "never"}:
        qdrant_mode = "auto"

    print("api: loading question index...", flush=True)
    question_to_id = _load_question_index()
    question_rows = [{"question": q} for q in question_to_id.keys()]

    include_qdrant = qdrant_mode == "always" or (
        qdrant_mode == "auto" and any(pipeline.needs_qdrant(row["question"]) for row in question_rows)
    )
    print("api: loading registry, qdrant:", include_qdrant, flush=True)
    registry = build_domain_registry(include_qdrant=include_qdrant)
    tools = pipeline.select_tool_schemas(registry.get_openai_tool_schemas(), include_qdrant)
    schema_summary = pipeline.cached_schema_summary(
        registry,
        SCHEMA_CACHE,
        refresh=REFRESH_SCHEMA_CACHE,
        whitelist=pipeline.DATA_LAYER_WHITELIST,
    )

    print("api: questions indexed:", len(question_to_id), flush=True)
    print("api: tools loaded:", [t["function"]["name"] for t in tools], flush=True)
    return RuntimeState(
        registry=registry,
        tools=tools,
        schema_summary=schema_summary,
        question_to_id=question_to_id,
        include_qdrant=include_qdrant,
        qdrant_mode=qdrant_mode,
        lock=asyncio.Lock(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    state = await asyncio.to_thread(_load_runtime)
    yield


app = FastAPI(title="FahMai Typhoon Agent API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    if state is None:
        return {"ok": False, "status": "loading"}
    return {
        "ok": True,
        "qdrant_mode": state.qdrant_mode,
        "qdrant_loaded": state.include_qdrant,
        "questions_indexed": len(state.question_to_id),
        "schema_cache": str(SCHEMA_CACHE),
        "tools": [t["function"]["name"] for t in state.tools],
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if state is None:
        raise HTTPException(status_code=503, detail="runtime is not ready")

    question = _norm_question(req.data.question)
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    if _looks_like_day_question(question):
        return ChatResponse(data=ChatAnswer(answer=_thai_today()))

    qid = req.data.id or state.question_to_id.get(question) or "API-Q"
    use_qdrant_for_question = state.qdrant_mode == "always" or (
        state.qdrant_mode == "auto" and pipeline.needs_qdrant(question)
    )
    question_tools = pipeline.select_tool_schemas(
        state.registry.get_openai_tool_schemas(),
        state.include_qdrant and use_qdrant_for_question,
    )

    t0 = time.time()
    async with state.lock:
        answer, trace = await asyncio.to_thread(
            pipeline.answer_question,
            state.registry,
            question_tools,
            qid,
            question,
            MAX_STEPS,
            state.schema_summary,
        )

    seconds = time.time() - t0
    _save_api_debug(qid, question, answer, trace, seconds)
    return ChatResponse(data=ChatAnswer(answer=answer))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
