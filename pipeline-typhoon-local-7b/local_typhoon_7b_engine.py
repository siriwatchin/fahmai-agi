from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline-typhoon-local"))

import local_typhoon_engine as base_engine  # noqa: E402
from tool_bridge_7b import normalize_assistant_message, parse_tool_call_text  # noqa: E402


DEFAULT_7B_MODEL = "typhoon-ai/typhoon2-qwen2.5-7b-instruct"
DEFAULT_BACKUP_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def apply_7b_defaults() -> None:
    os.environ.setdefault("LOCAL_TYPHOON_MODEL", DEFAULT_7B_MODEL)
    os.environ.setdefault("LOCAL_API_PORT", "8012")
    os.environ.setdefault("LOCAL_TORCH_DTYPE", "bfloat16")
    os.environ.setdefault("LOCAL_DEVICE_MAP", "auto")
    os.environ.setdefault("LOCAL_MAX_NEW_TOKENS", "256")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_raw_json_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    return parse_tool_call_text(content)


def normalize_7b_tool_call_content(content: str) -> str:
    parsed = parse_raw_json_tool_call(content)
    if not parsed:
        return content
    name, args = parsed
    payload = {"name": name, "arguments": args}
    return f"<tool_call> {json.dumps(payload, ensure_ascii=False)} </tool_call>"


def install_7b_call_typhoon(pipeline_module: Any, client: Any | None = None) -> Any:
    apply_7b_defaults()
    local_client = client or base_engine.build_local_client()

    def call_7b_typhoon(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.0,
        max_tokens: int = 700,
        timeout: int = 120,
    ) -> dict[str, Any]:
        data = local_client.chat_completion(
            messages,
            tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        try:
            msg = data["choices"][0]["message"]
            normalize_assistant_message(msg)
        except Exception:
            pass
        return data

    pipeline_module.call_typhoon = call_7b_typhoon
    pipeline_module.parse_json_tool_call = parse_raw_json_tool_call
    return local_client
