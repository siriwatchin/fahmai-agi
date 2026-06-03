from __future__ import annotations

import json
import re
from typing import Any


TOOL_NAME_RE = re.compile(r"^(postgres_|domain_|qdrant_)[A-Za-z0-9_]+$")


def _balanced_json_prefix(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _json_from_text(text: str) -> Any:
    clean = (text or "").strip()
    if not clean:
        return None
    if clean.startswith("```"):
        clean = clean.strip("`")
        clean = clean.split("\n", 1)[-1].strip()
    candidates = [clean]
    balanced = _balanced_json_prefix(clean)
    if balanced:
        candidates.insert(0, balanced)
    if "{" in clean and "}" in clean:
        candidates.append(clean[clean.find("{") : clean.rfind("}") + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _repair_args_from_text(text: str) -> dict[str, Any]:
    arg_match = re.search(r'"arguments"\s*:\s*(\{.*)', text, flags=re.DOTALL)
    if not arg_match:
        return {}
    raw = arg_match.group(1)
    balanced = _balanced_json_prefix(raw)
    if balanced:
        parsed = _json_from_text(balanced)
        return parsed if isinstance(parsed, dict) else {}

    args: dict[str, Any] = {}
    for key, value in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*"([^"]*)"', raw):
        args[key] = value
    for key, value in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*(\d+)', raw):
        args[key] = int(value)
    return args


def _repair_tool_call_from_text(text: str) -> tuple[str, dict[str, Any]] | None:
    name_match = re.search(r'"(?:name|tool|function)"\s*:\s*"([^"]+)"', text)
    if not name_match:
        return None
    name = name_match.group(1)
    args = _repair_args_from_text(text)
    if TOOL_NAME_RE.match(name):
        return name, args
    return None


def parse_tool_call_text(content: str) -> tuple[str, dict[str, Any]] | None:
    data = _json_from_text(content)
    if isinstance(data, str):
        data = _json_from_text(data)
    if not isinstance(data, dict):
        return _repair_tool_call_from_text(content)

    name = data.get("name") or data.get("tool") or data.get("function")
    args = data.get("arguments") or data.get("args") or {}
    if isinstance(args, str):
        parsed_args = _json_from_text(args)
        args = parsed_args if isinstance(parsed_args, dict) else {}

    if isinstance(name, str) and isinstance(args, dict) and TOOL_NAME_RE.match(name):
        return name, args
    return _repair_tool_call_from_text(content)


def as_openai_tool_calls(content: str) -> list[dict[str, Any]]:
    parsed = parse_tool_call_text(content)
    if not parsed:
        return []
    name, args = parsed
    return [
        {
            "id": f"call_7b_{name}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }
    ]


def normalize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content") or ""
    tool_calls = as_openai_tool_calls(content)
    if tool_calls:
        message["content"] = ""
        message["tool_calls"] = tool_calls
    return message


def looks_like_tool_leak(answer: str) -> bool:
    return parse_tool_call_text(answer) is not None
