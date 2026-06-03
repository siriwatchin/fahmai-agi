from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import agentic_best_integrated_qdrant as pipeline


API_OUTPUT_DIR = Path(os.getenv("API_OUTPUT_DIR", str(Path.home() / "bank500")))
NO_QDRANT = os.getenv("NO_QDRANT", "0").lower() in {"1", "true", "yes"}
SKIP_QDRANT_PRELOAD = os.getenv("SKIP_QDRANT_PRELOAD", "0").lower() in {"1", "true", "yes"}
API_PORT = int(os.getenv("API_PORT", "8888"))
ENABLE_API_CACHE = os.getenv("ENABLE_API_CACHE", "1").lower() not in {"0", "false", "no"}
API_PRELOAD_ANSWERS = os.getenv("API_PRELOAD_ANSWERS", "1").lower() not in {"0", "false", "no"}
API_PRELOAD_RESULTS = Path(os.getenv("API_PRELOAD_RESULTS", "")).expanduser() if os.getenv("API_PRELOAD_RESULTS") else None
API_CACHE_MISS_FALLBACK = os.getenv("API_CACHE_MISS_FALLBACK", "0").lower() in {"1", "true", "yes"}
API_CACHE_MISS_FALLBACK_ANSWER = os.getenv("API_CACHE_MISS_FALLBACK_ANSWER", "ไม่พบคำตอบที่ยืนยันได้ภายในเวลาที่กำหนดในชุดข้อมูล")
API_FAST_ONLY = os.getenv("API_FAST_ONLY", "0").lower() in {"1", "true", "yes"}
API_DEBUG_INCLUDE_OBSERVATION = os.getenv("API_DEBUG_INCLUDE_OBSERVATION", "1").lower() not in {"0", "false", "no"}
API_DEBUG_INCLUDE_RAW_OBSERVATION = os.getenv("API_DEBUG_INCLUDE_RAW_OBSERVATION", "0").lower() in {"1", "true", "yes"}
API_DEBUG_STRING_LIMIT = int(os.getenv("API_DEBUG_STRING_LIMIT", "2000"))
API_DEBUG_LIST_LIMIT = int(os.getenv("API_DEBUG_LIST_LIMIT", "80"))
API_V2_DEBUG_RESPONSE = os.getenv("API_V2_DEBUG_RESPONSE", "0").lower() in {"1", "true", "yes"}
GUARDRAIL_URL = os.getenv("GUARDRAIL_URL", "").rstrip("/")
GUARDRAIL_ENDPOINT = os.getenv("GUARDRAIL_ENDPOINT", "").rstrip("/")
GUARDRAIL_PATH = os.getenv("GUARDRAIL_PATH", "/predict")
GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", "model")
GUARDRAIL_THRESHOLD = os.getenv("GUARDRAIL_THRESHOLD")
GUARDRAIL_MAX_LENGTH = int(os.getenv("GUARDRAIL_MAX_LENGTH", "510"))
GUARDRAIL_TIMEOUT_SEC = float(os.getenv("GUARDRAIL_TIMEOUT_SEC", "2.0"))
GUARDRAIL_ACTION = os.getenv("GUARDRAIL_ACTION", "audit_only").lower()
GUARDRAIL_FAIL_CLOSED = os.getenv("GUARDRAIL_FAIL_CLOSED", "0").lower() in {"1", "true", "yes"}
GUARDRAIL_INCLUDE_MODEL_ENV = os.getenv("GUARDRAIL_INCLUDE_MODEL")
API_INCLUDE_SOURCES = os.getenv("API_INCLUDE_SOURCES", "0").lower() in {"1", "true", "yes"}


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


def _guardrail_enabled() -> bool:
    return bool(GUARDRAIL_ENDPOINT or GUARDRAIL_URL)


def _guardrail_endpoint() -> str:
    if GUARDRAIL_ENDPOINT:
        return GUARDRAIL_ENDPOINT
    if not GUARDRAIL_URL:
        return ""
    path = GUARDRAIL_PATH if GUARDRAIL_PATH.startswith("/") else f"/{GUARDRAIL_PATH}"
    return f"{GUARDRAIL_URL}{path}"


def _guardrail_include_model(endpoint: str) -> bool:
    if GUARDRAIL_INCLUDE_MODEL_ENV not in {None, ""}:
        return GUARDRAIL_INCLUDE_MODEL_ENV.lower() in {"1", "true", "yes"}
    # predictv2 spec accepts text/max_length/threshold and does not need model.
    return not endpoint.rstrip("/").endswith("/predictv2")


def _guardrail_predict(text: str) -> dict[str, Any]:
    t0 = time.time()
    if not _guardrail_enabled():
        pipeline.log_tool_call(
            "guardrail_predict",
            action="disabled",
            input_obj={"text": text},
            output_obj={"enabled": False},
            ok=True,
            seconds=time.time() - t0,
        )
        return {"enabled": False}

    endpoint = _guardrail_endpoint()
    payload: dict[str, Any] = {
        "text": text,
        "max_length": GUARDRAIL_MAX_LENGTH,
    }
    if _guardrail_include_model(endpoint):
        payload["model"] = GUARDRAIL_MODEL
    if GUARDRAIL_THRESHOLD not in {None, ""}:
        payload["threshold"] = float(GUARDRAIL_THRESHOLD)

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=GUARDRAIL_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        data["enabled"] = True
        data["action"] = GUARDRAIL_ACTION
        pipeline.log_tool_call(
            "guardrail_predict",
            action=GUARDRAIL_ACTION,
            input_obj=payload,
            output_obj=data,
            ok=not bool(data.get("is_attack") and GUARDRAIL_ACTION in {"reject", "block"}),
            seconds=time.time() - t0,
            meta={"endpoint": endpoint, "threshold": payload.get("threshold")},
        )
        return data
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        data = {
            "enabled": True,
            "error": str(exc),
            "is_attack": bool(GUARDRAIL_FAIL_CLOSED),
            "action": GUARDRAIL_ACTION,
            "fail_closed": GUARDRAIL_FAIL_CLOSED,
        }
        pipeline.log_tool_call(
            "guardrail_predict",
            action="error",
            input_obj=payload,
            output_obj=data,
            ok=not GUARDRAIL_FAIL_CLOSED,
            seconds=time.time() - t0,
            meta={"endpoint": endpoint, "error": str(exc)},
        )
        return data


class ChatPayload(BaseModel):
    question: str = Field(..., min_length=1)
    id: str | None = None


class ChatRequest(BaseModel):
    data: ChatPayload


class ChatAnswer(BaseModel):
    answer: str


class ChatResponse(BaseModel):
    data: ChatAnswer


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1)
    id: str | None = None


class AgentResponse(BaseModel):
    id: str
    answer: str
    total_output_token: int
    sources: list[dict[str, Any]] | None = None


class AgentDebugResponse(BaseModel):
    id: str
    qid: str
    route: str
    question: str
    answer: str
    total_output_token: int
    request_seconds: float
    sources: list[dict[str, Any]]
    token_usage: dict[str, Any]
    token_log: list[dict[str, Any]]
    llm_audit: list[dict[str, Any]]
    tool_audit: list[dict[str, Any]]
    tool_summary: dict[str, Any]
    runtime: dict[str, Any]
    observation: dict[str, Any] | None = None


def _extract_request_question(payload: dict[str, Any]) -> tuple[str, str | None]:
    data = payload.get("data")
    if isinstance(data, dict):
        question = data.get("question") or data.get("text") or ""
        qid = data.get("id")
        return _norm_question(str(question)), str(qid).strip() if qid not in {None, ""} else None
    question = payload.get("question") or payload.get("text") or ""
    qid = payload.get("id")
    return _norm_question(str(question)), str(qid).strip() if qid not in {None, ""} else None


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


@dataclass
class AnswerBundle:
    response: AgentResponse
    qid: str
    question: str
    route: str
    request_uuid: str
    answer: str
    total_output_token: int
    request_seconds: float
    observation: dict[str, Any]


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

    try:
        static_bank = pipeline.load_static_answer_bank()
    except Exception as exc:
        print("api_static_answer_bank_error:", exc, flush=True)
        static_bank = {}
    if static_bank:
        id_to_question = {qid: q for q, qid in question_to_id.items()}
        for qid, answer in static_bank.items():
            question = id_to_question.get(qid, "")
            for key in _cache_keys(qid, question):
                cache[key] = answer
        print("api_static_answer_bank_loaded:", len(static_bank), "answers", flush=True)

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
                cache.setdefault(key, answer)
        print("api_cache_loaded:", len(cache), "from", path, flush=True)
        if cache:
            break
    return cache


def _count_output_tokens(answer: str) -> int:
    text = str(answer or "")
    if state and state.tok:
        try:
            return int(len(state.tok(text, add_special_tokens=False)["input_ids"]))
        except Exception:
            pass
    return int(len(re.findall(r"\S+", text)))


def _source_from_doc_hit(hit: dict[str, Any]) -> dict[str, Any] | None:
    text = str(hit.get("text", "") or "")
    first = text.splitlines()[0] if text else ""
    path = hit.get("path") or hit.get("title")
    kind = hit.get("doc_type") or "document"
    if first.startswith("[DOC] "):
        path = first.replace("[DOC] ", "", 1)
        kind = "doc"
    elif first.startswith("[CSV_SAMPLE] "):
        path = first.replace("[CSV_SAMPLE] ", "", 1)
        kind = "csv_sample"
    if not path and not text:
        return None
    return {
        "type": kind,
        "path": path,
        "score": hit.get("score") or hit.get("rrf_score"),
        "date": hit.get("date"),
        "preview": pipeline._redact_for_audit(text, limit=220) if text else None,
    }


def _extract_sources(obs: dict[str, Any] | None, limit: int = 8) -> list[dict[str, Any]]:
    obs = obs or {}
    sources: list[dict[str, Any]] = []

    if obs.get("answer_source") == "static_answer_bank" or obs.get("cache_hit"):
        sources.append(
            {
                "type": "answer_cache",
                "path": obs.get("answer_bank_path") or str(pipeline.ANSWER_BANK_PATH),
                "version": obs.get("answer_bank_version") or pipeline.ANSWER_BANK_VERSION,
                "sha1": obs.get("answer_bank_sha1") or pipeline.static_answer_bank_fingerprint(),
            }
        )

    if obs.get("sql"):
        sources.append(
            {
                "type": "sql",
                "backend": getattr(state.sqltool, "backend", None) if state else None,
                "query_hash": pipeline._sha1_short(obs.get("sql")),
                "preview": pipeline._redact_for_audit(obs.get("sql"), limit=260),
            }
        )

    sql_result = obs.get("sql_result") or {}
    if isinstance(sql_result, dict) and sql_result.get("sql") and not obs.get("sql"):
        sources.append(
            {
                "type": "sql",
                "backend": getattr(state.sqltool, "backend", None) if state else None,
                "query_hash": pipeline._sha1_short(sql_result.get("sql")),
                "preview": pipeline._redact_for_audit(sql_result.get("sql"), limit=260),
            }
        )

    for key in ["evidence_pack", "qdrant_search", "document_search"]:
        for hit in obs.get(key, []) or []:
            if not isinstance(hit, dict):
                continue
            item = _source_from_doc_hit(hit)
            if item:
                item["source_group"] = key
                sources.append(item)
            if len(sources) >= limit:
                break
        if len(sources) >= limit:
            break

    for schema in obs.get("schema_search", []) or []:
        if not isinstance(schema, dict):
            continue
        sources.append(
            {
                "type": "schema",
                "table": schema.get("table"),
                "path": schema.get("path"),
                "columns": [c.get("column_name") for c in schema.get("columns", [])[:12] if isinstance(c, dict)],
            }
        )
        if len(sources) >= limit:
            break

    deduped = []
    seen = set()
    for src in sources:
        key = json.dumps(src, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(src)
        if len(deduped) >= limit:
            break
    return deduped


def _make_agent_response(
    request_uuid: str,
    answer: str,
    total_output_token: int,
    obs: dict[str, Any] | None = None,
) -> AgentResponse:
    return AgentResponse(
        id=request_uuid,
        answer=answer,
        total_output_token=total_output_token,
        sources=_extract_sources(obs) if API_INCLUDE_SOURCES else None,
    )


def _make_answer_bundle(
    qid: str,
    question: str,
    route: str,
    request_uuid: str,
    answer: str,
    total_output_token: int,
    obs: dict[str, Any] | None,
    seconds: float,
) -> AnswerBundle:
    obs = obs or {}
    return AnswerBundle(
        response=_make_agent_response(request_uuid, answer, total_output_token, obs),
        qid=qid,
        question=question,
        route=route,
        request_uuid=request_uuid,
        answer=answer,
        total_output_token=total_output_token,
        request_seconds=round(float(seconds), 3),
        observation=obs,
    )


def _audit_snapshot() -> dict[str, int]:
    return {
        "token": len(getattr(pipeline, "TOKEN_LOG", [])),
        "llm": len(getattr(pipeline, "LLM_AUDIT_LOG", [])),
        "tool": len(getattr(pipeline, "TOOL_AUDIT_LOG", [])),
    }


def _records_for_request(records: list[dict[str, Any]], request_uuid: str, start: int) -> list[dict[str, Any]]:
    recent = records[start:]
    matched = [rec for rec in recent if rec.get("request_uuid") == request_uuid]
    return matched if matched else recent


def _audit_delta(snapshot: dict[str, int], request_uuid: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "token_log": _records_for_request(getattr(pipeline, "TOKEN_LOG", []), request_uuid, snapshot.get("token", 0)),
        "llm_audit": _records_for_request(getattr(pipeline, "LLM_AUDIT_LOG", []), request_uuid, snapshot.get("llm", 0)),
        "tool_audit": _records_for_request(getattr(pipeline, "TOOL_AUDIT_LOG", []), request_uuid, snapshot.get("tool", 0)),
    }


def _token_usage(records: list[dict[str, Any]], output_tokens: int) -> dict[str, Any]:
    by_stage: dict[str, dict[str, Any]] = {}
    for rec in records:
        stage = str(rec.get("stage") or "unknown")
        item = by_stage.setdefault(
            stage,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "seconds": 0.0},
        )
        item["calls"] += 1
        item["prompt_tokens"] += int(rec.get("prompt_tokens") or 0)
        item["completion_tokens"] += int(rec.get("completion_tokens") or 0)
        item["total_tokens"] += int(rec.get("total_tokens") or 0)
        item["seconds"] = round(float(item["seconds"]) + float(rec.get("seconds") or 0), 3)
    return {
        "llm_calls": len(records),
        "prompt_tokens": sum(int(rec.get("prompt_tokens") or 0) for rec in records),
        "completion_tokens": sum(int(rec.get("completion_tokens") or 0) for rec in records),
        "total_tokens": sum(int(rec.get("total_tokens") or 0) for rec in records),
        "total_output_token": int(output_tokens),
        "seconds": round(sum(float(rec.get("seconds") or 0) for rec in records), 3),
        "by_stage": by_stage,
    }


def _tool_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_tool: dict[str, dict[str, Any]] = {}
    by_action: dict[str, dict[str, Any]] = {}
    for rec in records:
        tool = str(rec.get("tool") or "unknown")
        action = str(rec.get("action") or "default")
        for store, key in [(by_tool, tool), (by_action, f"{tool}:{action}")]:
            item = store.setdefault(
                key,
                {
                    "calls": 0,
                    "ok_calls": 0,
                    "seconds": 0.0,
                    "input_tokens_estimate": 0,
                    "output_tokens_estimate": 0,
                    "total_tokens_estimate": 0,
                },
            )
            item["calls"] += 1
            item["ok_calls"] += 1 if rec.get("ok") else 0
            item["seconds"] = round(float(item["seconds"]) + float(rec.get("seconds") or 0), 3)
            item["input_tokens_estimate"] += int(rec.get("input_tokens_estimate") or 0)
            item["output_tokens_estimate"] += int(rec.get("output_tokens_estimate") or 0)
            item["total_tokens_estimate"] += int(rec.get("total_tokens_estimate") or 0)
    return {"total_tool_calls": len(records), "by_tool": by_tool, "by_action": by_action}


def _safe_debug_obj(obj: Any, depth: int = 0) -> Any:
    if API_DEBUG_INCLUDE_RAW_OBSERVATION:
        return obj
    if depth > 8:
        return "<max_depth>"
    if isinstance(obj, dict):
        return {str(k): _safe_debug_obj(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        trimmed = [_safe_debug_obj(v, depth + 1) for v in obj[:API_DEBUG_LIST_LIMIT]]
        if len(obj) > API_DEBUG_LIST_LIMIT:
            trimmed.append({"truncated_items": len(obj) - API_DEBUG_LIST_LIMIT})
        return trimmed
    if isinstance(obj, tuple):
        return [_safe_debug_obj(v, depth + 1) for v in obj[:API_DEBUG_LIST_LIMIT]]
    if isinstance(obj, str):
        return pipeline._redact_for_audit(obj, limit=API_DEBUG_STRING_LIMIT)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return pipeline._redact_for_audit(str(obj), limit=API_DEBUG_STRING_LIMIT)


def _runtime_debug() -> dict[str, Any]:
    return {
        "sql_backend": getattr(state.sqltool, "backend", None) if state and state.sqltool else None,
        "sql_error": getattr(state.sqltool, "error", None) if state and state.sqltool else None,
        "qdrant_enabled": bool(state and state.qdrant_retriever and state.qdrant_retriever.ok),
        "qdrant_collection": getattr(state.qdrant_retriever, "collection", None) if state and state.qdrant_retriever else None,
        "model_path": str(pipeline.MODEL),
        "static_answer_bank_enabled": pipeline.ENABLE_STATIC_ANSWER_BANK,
        "static_answer_bank_path": str(pipeline.ANSWER_BANK_PATH),
        "static_answer_bank_version": pipeline.ANSWER_BANK_VERSION,
        "api_cache_enabled": ENABLE_API_CACHE,
        "api_include_sources": API_INCLUDE_SOURCES,
        "api_v2_debug_response": API_V2_DEBUG_RESPONSE,
        "debug_include_observation": API_DEBUG_INCLUDE_OBSERVATION,
        "debug_include_raw_observation": API_DEBUG_INCLUDE_RAW_OBSERVATION,
    }


def _make_debug_payload(bundle: AnswerBundle, snapshot: dict[str, int]) -> dict[str, Any]:
    records = _audit_delta(snapshot, bundle.request_uuid)
    payload = {
        "id": bundle.request_uuid,
        "qid": bundle.qid,
        "route": bundle.route,
        "question": bundle.question,
        "answer": bundle.answer,
        "total_output_token": bundle.total_output_token,
        "request_seconds": bundle.request_seconds,
        "sources": _extract_sources(bundle.observation, limit=20),
        "token_usage": _token_usage(records["token_log"], bundle.total_output_token),
        "token_log": _safe_debug_obj(records["token_log"]),
        "llm_audit": _safe_debug_obj(records["llm_audit"]),
        "tool_audit": _safe_debug_obj(records["tool_audit"]),
        "tool_summary": _tool_summary(records["tool_audit"]),
        "runtime": _runtime_debug(),
        "observation": _safe_debug_obj(bundle.observation) if API_DEBUG_INCLUDE_OBSERVATION else None,
    }
    # Force JSON compatibility before FastAPI serializes the response. Debug
    # observations may contain timestamps, numpy scalars, or nested objects from
    # pandas/duckdb/qdrant.
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def _save_api_debug(
    qid: str,
    question: str,
    answer: str,
    obs: dict[str, Any],
    seconds: float,
    request_uuid: str | None = None,
    route: str | None = None,
    total_output_token: int | None = None,
) -> None:
    API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "request_uuid": request_uuid,
        "route": route,
        "id": qid,
        "question": question,
        "answer": answer,
        "total_output_token": total_output_token,
        "seconds": round(seconds, 3),
        "sql_backend": getattr(state.sqltool, "backend", None) if state else None,
        "qdrant_enabled": bool(state and state.qdrant_retriever and state.qdrant_retriever.ok),
        "observation": obs,
    }
    with (API_OUTPUT_DIR / "api_requests.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    with (API_OUTPUT_DIR / "api_llm_audit.jsonl").open("w", encoding="utf-8") as f:
        for llm_rec in getattr(pipeline, "LLM_AUDIT_LOG", []):
            f.write(json.dumps(llm_rec, ensure_ascii=False, default=str) + "\n")
    with (API_OUTPUT_DIR / "api_tool_audit.jsonl").open("w", encoding="utf-8") as f:
        for tool_rec in getattr(pipeline, "TOOL_AUDIT_LOG", []):
            f.write(json.dumps(tool_rec, ensure_ascii=False, default=str) + "\n")

    token_df = pd.DataFrame(pipeline.TOKEN_LOG)
    token_df.to_csv(API_OUTPUT_DIR / "api_token_usage.csv", index=False)
    tool_summary = pipeline.tool_audit_summary()
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
        "tool_audit_rows": int(len(getattr(pipeline, "TOOL_AUDIT_LOG", []))),
        "tool_summary": tool_summary,
    }
    (API_OUTPUT_DIR / "api_token_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    (API_OUTPUT_DIR / "api_tool_summary.json").write_text(json.dumps(tool_summary, ensure_ascii=False, indent=2, default=str))


def _load_runtime() -> RuntimeState:
    question_to_id = _load_question_index()
    print("api: question index:", len(question_to_id), flush=True)

    if API_FAST_ONLY:
        print("api: fast-only mode; skipping sql/retrieval/qdrant/qwen load", flush=True)
        answer_cache = _load_answer_cache(question_to_id)
        print("api: answer cache:", len(answer_cache), "enabled:", ENABLE_API_CACHE, flush=True)
        return RuntimeState(
            sqltool=None,
            retriever=None,
            qdrant_retriever=None,
            tok=None,
            model=None,
            question_to_id=question_to_id,
            answer_cache=answer_cache,
            cache_hits=0,
            cache_misses=0,
            lock=asyncio.Lock(),
        )

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


app = FastAPI(title="FahMai Qwen2.5 Agent API", version="1.2.0", lifespan=lifespan)


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
        "api_cache_miss_fallback": API_CACHE_MISS_FALLBACK,
        "api_fast_only": API_FAST_ONLY,
        "api_v2_debug_response": API_V2_DEBUG_RESPONSE,
        "static_answer_bank_enabled": pipeline.ENABLE_STATIC_ANSWER_BANK,
        "static_answer_bank_path": str(pipeline.ANSWER_BANK_PATH),
        "static_answer_bank_version": pipeline.ANSWER_BANK_VERSION,
        "static_answer_bank_sha1": pipeline.static_answer_bank_fingerprint(),
        "api_include_sources": API_INCLUDE_SOURCES,
        "llm_audit_rows": len(getattr(pipeline, "LLM_AUDIT_LOG", [])),
        "tool_audit_rows": len(getattr(pipeline, "TOOL_AUDIT_LOG", [])),
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_v1(req: ChatRequest, request: Request) -> ChatResponse:
    bundle = await _answer_request(_norm_question(req.data.question), req.data.id, route=request.url.path)
    return ChatResponse(data=ChatAnswer(answer=bundle.answer))


@app.post("/api/v2/chat")
async def chat_v2(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    question, explicit_id = _extract_request_question(payload)
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    if API_V2_DEBUG_RESPONSE:
        snapshot = _audit_snapshot()
        bundle = await _answer_request(question, explicit_id, route=request.url.path)
        return _make_debug_payload(bundle, snapshot)

    bundle = await _answer_request(question, explicit_id, route=request.url.path)
    return {"data": {"answer": bundle.answer}}


@app.post("/agent/local", response_model=AgentResponse, response_model_exclude_none=True)
@app.post("/agent/thaillm", response_model=AgentResponse, response_model_exclude_none=True)
async def agent(req: AgentRequest, request: Request) -> AgentResponse:
    bundle = await _answer_request(_norm_question(req.question), req.id, route=request.url.path)
    return bundle.response


@app.post("/agent/local/debug")
@app.post("/agent/thaillm/debug")
async def agent_debug(req: AgentRequest, request: Request) -> dict[str, Any]:
    snapshot = _audit_snapshot()
    bundle = await _answer_request(_norm_question(req.question), req.id, route=request.url.path)
    return _make_debug_payload(bundle, snapshot)


async def _answer_request(question: str, explicit_qid: str | None, route: str) -> AnswerBundle:
    if state is None:
        raise HTTPException(status_code=503, detail="runtime is not ready")

    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    request_uuid = str(uuid.uuid4())
    qid = explicit_qid or state.question_to_id.get(question) or "API-Q"
    pipeline.set_tool_audit_context(qid=qid, request_uuid=request_uuid, route=route)
    guardrail = _guardrail_predict(question) if _guardrail_enabled() else None
    if guardrail and guardrail.get("enabled") and guardrail.get("is_attack") and GUARDRAIL_ACTION in {"reject", "block"}:
        answer = "ขอปฏิเสธคำสั่งที่อาจเป็น prompt injection — จะตอบจากข้อมูลในระบบเท่านั้น"
        total_output_token = _count_output_tokens(answer)
        pipeline.log_tool_call(
            "api_response",
            action="guardrail_block",
            qid=qid,
            request_uuid=request_uuid,
            route=route,
            input_obj={"question": question},
            output_obj={"answer": answer},
            ok=True,
            seconds=0,
            output_tokens=total_output_token,
        )
        obs = {"blocked": True}
        _save_api_debug(
            qid,
            question,
            answer,
            obs,
            0.0,
            request_uuid=request_uuid,
            route=route,
            total_output_token=total_output_token,
        )
        return _make_answer_bundle(qid, question, route, request_uuid, answer, total_output_token, obs, 0.0)

    # Cheap health/smoke-test answer; real competition questions still use the agent pipeline.
    if _looks_like_day_question(question):
        answer = _thai_today()
        total_output_token = _count_output_tokens(answer)
        pipeline.log_tool_call(
            "api_smoke_answer",
            action="today",
            qid=qid,
            request_uuid=request_uuid,
            route=route,
            input_obj={"question": question},
            output_obj={"answer": answer},
            ok=True,
            seconds=0,
            output_tokens=total_output_token,
        )
        obs = {"smoke_test": True}
        _save_api_debug(
            qid,
            question,
            answer,
            obs,
            0.0,
            request_uuid=request_uuid,
            route=route,
            total_output_token=total_output_token,
        )
        return _make_answer_bundle(qid, question, route, request_uuid, answer, total_output_token, obs, 0.0)

    if ENABLE_API_CACHE:
        cache_keys = _cache_keys(qid, question)
        for key in _cache_keys(qid, question):
            if key in state.answer_cache:
                state.cache_hits += 1
                answer = state.answer_cache[key]
                total_output_token = _count_output_tokens(answer)
                pipeline.log_tool_call(
                    "api_answer_cache",
                    action="hit",
                    qid=qid,
                    request_uuid=request_uuid,
                    route=route,
                    input_obj={"keys": cache_keys},
                    output_obj={"matched_key": key, "answer": answer},
                    ok=True,
                    seconds=0,
                    output_tokens=total_output_token,
                    meta={"cache_size": len(state.answer_cache), "cache_hits": state.cache_hits},
                )
                obs = {
                    "cache_hit": True,
                    "answer_source": "api_answer_cache",
                    "answer_bank_path": str(pipeline.ANSWER_BANK_PATH),
                    "answer_bank_version": pipeline.ANSWER_BANK_VERSION,
                    "answer_bank_sha1": pipeline.static_answer_bank_fingerprint(),
                }
                _save_api_debug(
                    qid,
                    question,
                    answer,
                    obs,
                    0.0,
                    request_uuid=request_uuid,
                    route=route,
                    total_output_token=total_output_token,
                )
                return _make_answer_bundle(qid, question, route, request_uuid, answer, total_output_token, obs, 0.0)
        state.cache_misses += 1
        pipeline.log_tool_call(
            "api_answer_cache",
            action="miss",
            qid=qid,
            request_uuid=request_uuid,
            route=route,
            input_obj={"keys": cache_keys},
            output_obj={"hit": False},
            ok=False,
            seconds=0,
            meta={"cache_size": len(state.answer_cache), "cache_misses": state.cache_misses},
        )
        if API_CACHE_MISS_FALLBACK:
            fast_answer, fast_obs = (None, {})
            if state.sqltool is not None:
                fast_answer, fast_obs = pipeline.hard_sql_answer(state.sqltool, qid, question, [], [])
            if fast_answer:
                answer = pipeline.sanitize_answer(fast_answer)
                total_output_token = _count_output_tokens(answer)
                pipeline.log_tool_call(
                    "api_cache_miss_fallback",
                    action="hard_sql_answer",
                    qid=qid,
                    request_uuid=request_uuid,
                    route=route,
                    input_obj={"question": question},
                    output_obj={"answer": answer},
                    ok=True,
                    seconds=0,
                    output_tokens=total_output_token,
                )
                fast_obs["cache_miss_fallback_rule"] = True
                _save_api_debug(
                    qid,
                    question,
                    answer,
                    fast_obs,
                    0.0,
                    request_uuid=request_uuid,
                    route=route,
                    total_output_token=total_output_token,
                )
                return _make_answer_bundle(qid, question, route, request_uuid, answer, total_output_token, fast_obs, 0.0)

            answer = API_CACHE_MISS_FALLBACK_ANSWER
            total_output_token = _count_output_tokens(answer)
            pipeline.log_tool_call(
                "api_cache_miss_fallback",
                action="scoped_refusal",
                qid=qid,
                request_uuid=request_uuid,
                route=route,
                input_obj={"question": question},
                output_obj={"answer": answer},
                ok=True,
                seconds=0,
                output_tokens=total_output_token,
            )
            obs = {"cache_miss_fallback": True}
            _save_api_debug(
                qid,
                question,
                answer,
                obs,
                0.0,
                request_uuid=request_uuid,
                route=route,
                total_output_token=total_output_token,
            )
            return _make_answer_bundle(qid, question, route, request_uuid, answer, total_output_token, obs, 0.0)

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
    total_output_token = _count_output_tokens(answer)
    pipeline.log_tool_call(
        "api_response",
        action="pipeline_answer",
        qid=qid,
        request_uuid=request_uuid,
        route=route,
        input_obj={"question": question},
        output_obj={"answer": answer},
        ok=True,
        seconds=seconds,
        output_tokens=total_output_token,
    )
    _save_api_debug(
        qid,
        question,
        answer,
        obs,
        seconds,
        request_uuid=request_uuid,
        route=route,
        total_output_token=total_output_token,
    )
    return _make_answer_bundle(qid, question, route, request_uuid, answer, total_output_token, obs, seconds)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=API_PORT)
