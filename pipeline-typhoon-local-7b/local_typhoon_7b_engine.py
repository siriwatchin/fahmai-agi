from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline-typhoon-local"))

try:
    import local_typhoon_engine as base_engine  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - optional local runtime module
    base_engine = None
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


class OpenAICompatibleLocalClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.0,
        max_tokens: int = 700,
        timeout: int = 120,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


def build_local_client() -> Any:
    if base_engine is not None:
        return base_engine.build_local_client()
    base_url = os.getenv("LOCAL_OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "Missing local_typhoon_engine.py and LOCAL_OPENAI_BASE_URL. "
            "Set LOCAL_OPENAI_BASE_URL to an OpenAI-compatible local model server, "
            "for example http://127.0.0.1:8000/v1."
        )
    return OpenAICompatibleLocalClient(base_url, os.getenv("LOCAL_TYPHOON_MODEL", DEFAULT_7B_MODEL))


def _message_content(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(msg.get("content") or "") for msg in messages or [])


def _is_product_value_question(messages: list[dict[str, Any]]) -> bool:
    text = _message_content(messages)
    return bool(re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", text)) and any(
        marker in text.lower() for marker in ["msrp", "ราคา", "เท่าไหร่", "warranty"]
    )


def _needs_parameter_guidance(messages: list[dict[str, Any]]) -> bool:
    text = _message_content(messages).lower()
    markers = [
        "policy",
        "นโยบาย",
        "refund",
        "return",
        "threshold",
        "points",
        "top sku",
        "stockout",
        "shipping",
        "partner brand",
        "loyalty",
        "ceo",
        "vendor",
        "customer",
        "branch",
        "employee",
        "sku",
        "สินค้า",
    ]
    return any(marker in text for marker in markers)


def _with_7b_tool_fewshot(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _needs_parameter_guidance(messages):
        return messages
    examples = [
        "GENERAL PARAMETER RULES:",
        "- Do not pass the full Thai question as a tool parameter unless the tool explicitly wants `question` or `query`.",
        "- Extract IDs, dates, years, policy variables, and entity types first.",
        "- Prefer exact domain tools or read-only SQL before document retrieval for structured facts.",
    ]
    if _is_product_value_question(messages):
        examples.extend([
            "",
            "SKU/MSRP example:",
            "Question: MSRP ของสินค้ารหัส NT-LT-001 เป็นเท่าไหร่ครับ",
            "First tool JSON:",
            "{\"name\":\"postgres_execute_readonly_sql\",\"arguments\":{\"sql\":\"SELECT sku_id, msrp_thb FROM public.\\\"DIM_PRODUCT\\\" WHERE upper(trim(sku_id::text)) = upper('NT-LT-001') LIMIT 1\",\"limit\":1}}",
        ])
    examples.extend([
        "",
        "Policy examples:",
        "Return window as of 2024-12-15 -> {\"name\":\"domain_policy_resolver\",\"arguments\":{\"policy_variable\":\"return_window_days\",\"as_of_date\":\"2024-12-15\"}}",
        "Refund threshold as of 2025-04-01 -> {\"name\":\"domain_policy_resolver\",\"arguments\":{\"policy_variable\":\"refund_threshold_thb\",\"as_of_date\":\"2025-04-01\"}}",
        "Point rate -> {\"name\":\"domain_policy_resolver\",\"arguments\":{\"policy_variable\":\"point_earning_rate_per_thb\",\"as_of_date\":\"2025-04-01\"}}",
        "",
        "Year/ranking examples:",
        "Top SKU by units in 2024 -> {\"name\":\"domain_top_sku_by_units\",\"arguments\":{\"year\":2024}}",
        "Most stockout SKU in 2025 -> {\"name\":\"domain_stockout_top_sku\",\"arguments\":{\"year\":2025}}",
        "",
        "Entity/domain examples:",
        "Current CEO -> {\"name\":\"domain_current_ceo\",\"arguments\":{\"as_of_date\":\"2025-06-01\"}}",
        "Shipping vendor share -> {\"name\":\"domain_shipping_vendor_share\",\"arguments\":{}}",
        "Partner brand vendors -> {\"name\":\"domain_partner_brand_vendors\",\"arguments\":{}}",
        "Customer loyalty tier counts -> {\"name\":\"domain_customer_loyalty_counts\",\"arguments\":{}}",
        "",
        "Only use domain_evidence_pack/qdrant/domain_hybrid_search for document, memo, OCR, chat, report, or source-text questions.",
        "After TOOL_RESULT rows are available, answer from those rows only in Thai plain text.",
    ])
    guidance = "7B TOOL PARAMETER ROUTING FEW-SHOT:\n" + "\n".join(examples)
    if messages and messages[0].get("role") == "system":
        out = [dict(messages[0])]
        out[0]["content"] = f"{guidance}\n\n{out[0].get('content') or ''}"
        out.extend(messages[1:])
        return out
    return [{"role": "system", "content": guidance}, *messages]


def install_7b_call_typhoon(pipeline_module: Any, client: Any | None = None) -> Any:
    apply_7b_defaults()
    local_client = client or build_local_client()

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
            _with_7b_tool_fewshot(messages),
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
