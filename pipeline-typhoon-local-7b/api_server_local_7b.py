from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline-typhoon"))
sys.path.insert(0, str(ROOT / "pipeline-typhoon-local-7b"))

import api_server as hosted_api  # noqa: E402
from local_typhoon_7b_engine import DEFAULT_7B_MODEL, apply_7b_defaults, install_7b_call_typhoon, parse_raw_json_tool_call  # noqa: E402
from tool_bridge_7b import parse_tool_call_text  # noqa: E402


apply_7b_defaults()
hosted_api.pipeline.parse_json_tool_call = parse_raw_json_tool_call
local_client = install_7b_call_typhoon(hosted_api.pipeline)

hosted_api.app.title = "FahMai Local Typhoon 7B Agent API"
hosted_api.app.version = "1.0.0-local-7b"

_base_answer_payload = hosted_api._answer_payload


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        if "{" in text and "}" in text:
            try:
                return json.loads(text[text.find("{") : text.rfind("}") + 1])
            except Exception:
                return None
    return None


def _tool_call_from_answer(answer: str) -> tuple[str, dict[str, Any]] | None:
    parsed = parse_tool_call_text(answer)
    if parsed:
        return parsed
    data = _safe_json_loads(answer or "")
    if not isinstance(data, dict):
        return None
    name = data.get("name") or data.get("tool") or data.get("function")
    args = data.get("arguments") or data.get("args") or {}
    if isinstance(name, str) and isinstance(args, dict):
        return name, args
    return None


def _compact_tool_result(result: str) -> str:
    try:
        return hosted_api.pipeline.compact_tool_result(result, max_chars=2500)
    except TypeError:
        return hosted_api.pipeline.compact_tool_result(result)
    except Exception:
        return str(result)[:2500]


def _execute_tool(name: str, args: dict[str, Any]) -> str:
    if hosted_api.state is None:
        return "{}"
    return hosted_api.state.registry.call_tool(name, args)


async def _answer_payload_7b(payload: Any) -> Any:
    answer_obj = await _base_answer_payload(payload)
    first = _tool_call_from_answer(answer_obj.answer)
    if not first or hosted_api.state is None:
        return answer_obj

    question = payload.question
    messages = [
        {
            "role": "system",
            "content": (
                "You are a FahMai data answerer. Use the provided TOOL_RESULT as evidence. "
                "Answer in Thai plain text only. Never output JSON, SQL, tool names, or <tool_call>."
            ),
        },
        {"role": "user", "content": question},
    ]
    trace: list[dict[str, Any]] = []
    token_usage = dict(answer_obj.token_usage or {})
    started = time.time()
    name, args = first

    for step in range(max(1, int(payload.max_steps or 5))):
        try:
            result = await asyncio.to_thread(_execute_tool, name, args)
        except Exception as exc:
            answer_obj.answer = f"ไม่สามารถเรียกใช้เครื่องมือ {name} ได้: {exc}"
            answer_obj.status = "error"
            return answer_obj

        compact = _compact_tool_result(result)
        trace.append({"step": f"7b_repair_{step}", "tool": name, "arguments": args, "result": compact})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"TOOL_RESULT {name}:\n{compact}\n\n"
                    "ถ้าต้องใช้ tool เพิ่ม ให้ตอบเฉพาะ JSON {\"name\":\"...\",\"arguments\":{...}}. "
                    "ถ้ามีหลักฐานครบแล้ว ให้ตอบคำตอบสุดท้ายภาษาไทยสั้น ๆ เท่านั้น."
                ),
            }
        )

        data = await asyncio.to_thread(
            hosted_api.pipeline.call_typhoon,
            messages,
            hosted_api.state.tools,
            max_tokens=256,
            timeout=max(10, int(payload.timeout_seconds or 60)),
        )
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        usage = data.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            token_usage[key] = int(token_usage.get(key) or 0) + int(usage.get(key) or 0)

        next_call = None
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                parsed_args = {}
            if isinstance(fn.get("name"), str) and isinstance(parsed_args, dict):
                next_call = (fn["name"], parsed_args)
                break
        if not next_call:
            next_call = _tool_call_from_answer(content)

        if next_call:
            name, args = next_call
            messages.append({"role": "assistant", "content": content})
            continue

        if content:
            answer_obj.answer = content.replace("\n", " ").strip()
            answer_obj.status = "answered"
            answer_obj.seconds = round((answer_obj.seconds or 0) + (time.time() - started), 3)
            answer_obj.token_usage = token_usage
            return answer_obj

    answer_obj.answer = "ไม่พบคำตอบในชุดข้อมูล"
    answer_obj.status = "needs_review"
    answer_obj.seconds = round((answer_obj.seconds or 0) + (time.time() - started), 3)
    answer_obj.token_usage = token_usage
    return answer_obj


hosted_api._answer_payload = _answer_payload_7b


@hosted_api.app.get("/local-7b-health")
def local_7b_health() -> dict[str, object]:
    return {
        "ok": True,
        "mode": "local-7b",
        "model": os.getenv("LOCAL_TYPHOON_MODEL", DEFAULT_7B_MODEL),
        "local_openai_base_url": os.getenv("LOCAL_OPENAI_BASE_URL"),
        "note": "Main endpoints are /health, /api/v1/chat, and /api/v1/batch.",
    }


app = hosted_api.app


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("LOCAL_API_PORT", os.getenv("PORT", "8012")))
    uvicorn.run("api_server_local_7b:app", host="0.0.0.0", port=port, reload=False)
