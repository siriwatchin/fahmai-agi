from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


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


class LocalChatPayload(BaseModel):
    question: str = Field(..., min_length=1)
    id: str | None = None
    use_answer_bank: bool | None = None
    max_steps: int | None = Field(default=None, ge=1, le=12)
    user_role: str | None = None
    timeout_seconds: int | None = Field(default=None, ge=1, le=600)


class LocalChatRequest(BaseModel):
    data: LocalChatPayload


class LocalChatAnswer(BaseModel):
    id: str | None = None
    answer: str
    status: str | None = None
    confidence: str | None = None
    refs: list[dict[str, Any]] = Field(default_factory=list)
    security: dict[str, Any] = Field(default_factory=dict)
    route: dict[str, Any] = Field(default_factory=dict)
    run_log: list[dict[str, Any]] | None = None
    seconds: float | None = None
    token_usage: dict[str, int] | None = None
    answer_bank: bool | None = None


class LocalChatResponse(BaseModel):
    data: LocalChatAnswer


class LocalBatchRequest(BaseModel):
    data: list[LocalChatPayload] = Field(..., min_length=1, max_length=100)


class LocalBatchResponse(BaseModel):
    data: list[LocalChatAnswer]


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


def _looks_like_refusal(answer: str) -> bool:
    return str(answer or "").strip().startswith("ไม่พบ")


def _looks_like_non_answer(answer: str) -> bool:
    text = str(answer or "").strip().lower()
    if not text:
        return True
    markers = [
        "ขออภัย",
        "โปรดระบุ",
        "กรุณาระบุ",
        "ไม่สามารถช่วย",
        "i need more",
        "please provide",
        "could you provide",
    ]
    return any(marker in text for marker in markers)


def _table_refs_from_sql(sql: str) -> list[dict[str, Any]]:
    tables = sorted(set(re.findall(r'\b(?:FACT|DIM|T2)_[A-Za-z0-9_]+\b|dim_[A-Za-z0-9_]+\b', sql, flags=re.IGNORECASE)))
    if not tables:
        return [{"type": "postgres", "source": "custom_sql", "sql": sql}]
    return [{"type": "postgres", "source": table, "sql": sql} for table in tables]


def _refs_from_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(ref: dict[str, Any]) -> None:
        key = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            refs.append(ref)

    for item in trace:
        tool = item.get("tool")
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        if tool:
            sql = str(args.get("sql") or "").strip()
            if sql:
                for ref in _table_refs_from_sql(sql):
                    ref["tool"] = tool
                    add(ref)
            else:
                add({"type": "tool", "source": str(tool), "arguments": args})

        result = item.get("result")
        if isinstance(result, str) and result.strip():
            parsed = _safe_json_loads(result)
            if isinstance(parsed, dict):
                result_sql = str(parsed.get("sql") or "").strip()
                if result_sql:
                    for ref in _table_refs_from_sql(result_sql):
                        ref["tool"] = str(tool or "")
                        add(ref)
                for row in parsed.get("rows") or []:
                    if isinstance(row, dict):
                        for key in ("table", "source", "file", "path"):
                            value = row.get(key)
                            if value:
                                add({"type": "evidence", "source": str(value), "tool": str(tool or "")})
                                break
    return refs


def _repair_trace_summary(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"step": item.get("step"), "tool": item.get("tool"), "arguments": item.get("arguments")}
        for item in trace
        if item.get("tool")
    ]


def _to_local_answer(answer_obj: Any, *, default_route: dict[str, Any] | None = None) -> LocalChatAnswer:
    answer = str(getattr(answer_obj, "answer", "") or "")
    refs = list(getattr(answer_obj, "refs", None) or [])
    status = getattr(answer_obj, "status", None)
    confidence = getattr(answer_obj, "confidence", None)
    route = dict(getattr(answer_obj, "route", None) or default_route or {})
    security = dict(getattr(answer_obj, "security", None) or {})
    if answer and not _looks_like_refusal(answer) and not _looks_like_non_answer(answer) and not refs:
        refs = [{"type": "pipeline", "source": route.get("intent_type") or "base_pipeline"}]
    if answer and not _looks_like_refusal(answer) and not _looks_like_non_answer(answer):
        status = status or "answered"
        confidence = confidence or ("medium" if refs else "low")
    elif not status:
        status = "needs_review"
    return LocalChatAnswer(
        id=getattr(answer_obj, "id", None),
        answer=answer,
        status=status,
        confidence=confidence,
        refs=refs,
        security=security,
        route=route,
        run_log=getattr(answer_obj, "run_log", None),
        seconds=getattr(answer_obj, "seconds", None),
        token_usage=getattr(answer_obj, "token_usage", None),
        answer_bank=getattr(answer_obj, "answer_bank", None),
    )


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


def _read_tool_json(name: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(_execute_tool(name, args))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _sku_from_question(question: str) -> str | None:
    match = re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", str(question or ""))
    return match.group(0) if match else None


def _is_msrp_question(question: str) -> bool:
    q = str(question or "").lower()
    return any(word in q for word in ["msrp", "ราคา", "เท่าไหร่"])


def _extract_years(question: str) -> list[int]:
    text = str(question or "")
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    thai_years = [int(y) - 543 for y in re.findall(r"\b(25[6-7]\d)\b", text)]
    out: list[int] = []
    for year in years + thai_years:
        if 2020 <= year <= 2030 and year not in out:
            out.append(year)
    return out


def _extract_iso_date(question: str) -> str | None:
    match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", str(question or ""))
    if match:
        return match.group(0)
    return None


def _policy_args_from_question(question: str) -> dict[str, str] | None:
    q = str(question or "").lower()
    as_of = _extract_iso_date(question)
    if "return_window" in q or "คืนสินค้า" in q or "return window" in q:
        return {"policy_variable": "return_window_days", "as_of_date": as_of or "2024-12-15"}
    if "refund_threshold" in q or "threshold" in q:
        return {"policy_variable": "refund_threshold_thb", "as_of_date": as_of or "2025-04-01"}
    if "point_earning_rate" in q or "points" in q or "สะสม" in q:
        return {"policy_variable": "point_earning_rate_per_thb", "as_of_date": as_of or "2025-04-01"}
    if "signing" in q or "authority" in q or "อนุมัติ" in q:
        return {"policy_variable": "refund_signing_authority_ladder", "as_of_date": as_of or "2025-04-01"}
    return None


def _domain_query_candidates(question: str) -> list[tuple[str, dict[str, Any]]]:
    q = str(question or "").lower()
    candidates: list[tuple[str, dict[str, Any]]] = []

    policy_args = _policy_args_from_question(question)
    if policy_args:
        candidates.append(("domain_policy_resolver", policy_args))

    years = _extract_years(question)
    if "top" in q and ("sku" in q or "สินค้า" in q) and ("unit" in q or "ขาย" in q):
        for year in years or [2024, 2025]:
            candidates.append(("domain_top_sku_by_units", {"year": year}))
    if "stockout" in q or "สต็อก" in q or "ขาด stock" in q:
        candidates.append(("domain_stockout_top_sku", {"year": (years or [2025])[-1]}))
    if "ceo" in q or "ผู้บริหาร" in q:
        candidates.append(("domain_current_ceo", {"as_of_date": _extract_iso_date(question) or "2025-06-01"}))
    if "shipping" in q or "ขนส่ง" in q:
        candidates.append(("domain_shipping_vendor_share", {}))
    if "partner brand" in q:
        candidates.append(("domain_partner_brand_vendors", {}))
    if "loyalty" in q and ("count" in q or "จำนวน" in q or "tier" in q):
        candidates.append(("domain_customer_loyalty_counts", {}))

    entity_type = None
    if "vendor" in q:
        entity_type = "vendor"
    elif "customer" in q or "ลูกค้า" in q:
        entity_type = "customer"
    elif "employee" in q or "พนักงาน" in q:
        entity_type = "employee"
    elif "branch" in q or "สาขา" in q:
        entity_type = "branch"
    elif "sku" in q or "สินค้า" in q:
        entity_type = "sku"
    if entity_type:
        candidates.append(("domain_entity_resolver", {"query": question, "entity_type": entity_type, "limit": 10}))

    return candidates


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _product_msrp_sql_candidates(sku: str) -> list[str]:
    sku_lit = _sql_literal(sku)
    sku_key_lit = _sql_literal(re.sub(r"[^A-Z0-9]", "", sku.upper()))
    return [
        (
            'SELECT sku_id, msrp_thb '
            'FROM public."DIM_PRODUCT" '
            f"WHERE sku_id = {sku_lit} "
            "LIMIT 1"
        ),
        (
            "SELECT sku_id, msrp_thb "
            "FROM DIM_PRODUCT "
            f"WHERE sku_id = {sku_lit} "
            "LIMIT 1"
        ),
        (
            'SELECT sku_id, msrp_thb '
            'FROM public."DIM_PRODUCT" '
            f"WHERE upper(trim(sku_id::text)) = upper({sku_lit}) "
            "LIMIT 1"
        ),
        (
            'SELECT sku_id, msrp_thb '
            'FROM public."DIM_PRODUCT" '
            "WHERE regexp_replace(upper(sku_id::text), '[^A-Z0-9]', '', 'g') = "
            f"{sku_key_lit} "
            "LIMIT 1"
        ),
        (
            "SELECT sku_id, msrp_thb "
            "FROM DIM_PRODUCT "
            f"WHERE upper(trim(sku_id::text)) = upper({sku_lit}) "
            "LIMIT 1"
        ),
        (
            "SELECT sku_id, msrp_thb "
            "FROM DIM_PRODUCT "
            "WHERE regexp_replace(upper(sku_id::text), '[^A-Z0-9]', '', 'g') = "
            f"{sku_key_lit} "
            "LIMIT 1"
        ),
    ]


def _is_vendor_payment_month_mismatch_question(question: str) -> bool:
    q = str(question or "").lower()
    required = ["fact_vendor_payment", "posting_date", "business_event_date"]
    return all(token in q for token in required) and any(token in q for token in ["เดือน", "month", "ไม่ตรง", "mismatch"])


def _is_shipping_vendor_share_question(question: str) -> bool:
    q = str(question or "").lower()
    return ("fact_shipping" in q or "ขนส่ง" in q or "shipping" in q) and "vendor" in q and any(
        token in q for token in ["ส่วนแบ่ง", "เปอร์เซ็นต์", "percent", "share", "ทั้งหมด"]
    )


def _vendor_payment_month_mismatch_sql_candidates() -> list[str]:
    return [
        """
        SELECT COUNT(*) AS mismatch_count
        FROM public."FACT_VENDOR_PAYMENT"
        WHERE to_char(NULLIF(posting_date, '')::date, 'YYYY-MM')
           <> to_char(NULLIF(business_event_date, '')::date, 'YYYY-MM')
        """.strip(),
        """
        SELECT COUNT(*) AS mismatch_count
        FROM FACT_VENDOR_PAYMENT
        WHERE to_char(NULLIF(posting_date, '')::date, 'YYYY-MM')
           <> to_char(NULLIF(business_event_date, '')::date, 'YYYY-MM')
        """.strip(),
        """
        SELECT COUNT(*) AS mismatch_count
        FROM public."FACT_VENDOR_PAYMENT"
        WHERE date_trunc('month', posting_date::date)
           <> date_trunc('month', business_event_date::date)
        """.strip(),
    ]


def _preflight_evidence_trace(question: str) -> list[dict[str, Any]]:
    if hosted_api.state is None:
        return []
    candidates: list[tuple[str, dict[str, Any]]] = []
    sku = _sku_from_question(question)
    if sku and _is_msrp_question(question):
        candidates.extend(
            ("postgres_execute_readonly_sql", {"sql": sql, "limit": 1})
            for sql in _product_msrp_sql_candidates(sku)
        )
    if _is_vendor_payment_month_mismatch_question(question):
        candidates.extend(
            ("postgres_execute_readonly_sql", {"sql": sql, "limit": 1})
            for sql in _vendor_payment_month_mismatch_sql_candidates()
        )
    if _is_shipping_vendor_share_question(question):
        candidates.append(("domain_shipping_vendor_share", {}))
    candidates.extend(_domain_query_candidates(question))
    candidates.extend([
        ("domain_evidence_pack", {"question": question, "top_k": 5}),
        ("postgres_search_schema", {"query": question, "schema": "public"}),
    ])
    trace: list[dict[str, Any]] = []
    for name, args in candidates:
        try:
            result = _execute_tool(name, args)
        except Exception as exc:
            trace.append({"step": "7b_preflight", "tool": name, "arguments": args, "result": json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)})
            continue
        trace.append({"step": "7b_preflight", "tool": name, "arguments": args, "result": _compact_tool_result(result)})
        parsed = _safe_json_loads(result)
        if not isinstance(parsed, dict) or parsed.get("ok") is not False:
            break
    return trace


def _apply_trace_validation(answer_obj: Any, trace: list[dict[str, Any]], route: dict[str, Any] | None = None) -> Any:
    route_meta = route or {"intent_type": "local_7b_repair"}
    local_answer = _to_local_answer(answer_obj, default_route=route_meta)
    local_answer.refs = _refs_from_trace(trace) or local_answer.refs
    local_answer.route = {**route_meta, "repair_trace": _repair_trace_summary(trace)}
    if local_answer.answer and not _looks_like_refusal(local_answer.answer):
        if _looks_like_non_answer(local_answer.answer):
            local_answer.status = "needs_review"
            local_answer.confidence = "low"
        else:
            local_answer.status = "answered"
            local_answer.confidence = "high" if local_answer.refs else "medium"
    else:
        local_answer.status = local_answer.status or "needs_review"
        local_answer.confidence = local_answer.confidence or "low"
    return local_answer


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return str(value)


def _deterministic_easy_answer(question: str) -> tuple[str | None, dict[str, Any]]:
    sku = _sku_from_question(question)
    if sku and _is_msrp_question(question):
        last_error: dict[str, Any] = {}
        for sql in _product_msrp_sql_candidates(sku):
            data = _read_tool_json("postgres_execute_readonly_sql", {"sql": sql, "limit": 1})
            rows = data.get("rows") or []
            if data.get("ok") and rows:
                row = rows[0]
                answer = f"MSRP ของสินค้ารหัส {row.get('sku_id', sku)} คือ {_money(row.get('msrp_thb'))} บาท"
                validation = {
                    "status": "answered",
                    "confidence": "high",
                    "refs": [{"type": "postgres", "source": "DIM_PRODUCT", "sql": sql}],
                    "security": {},
                    "route": {"intent_type": "deterministic_7b_easy", "tool": "postgres_execute_readonly_sql"},
                }
                return answer, {"validation": validation, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            last_error = data
        if last_error:
            return None, {"last_error": last_error}
    if _is_vendor_payment_month_mismatch_question(question):
        last_error: dict[str, Any] = {}
        for sql in _vendor_payment_month_mismatch_sql_candidates():
            data = _read_tool_json("postgres_execute_readonly_sql", {"sql": sql, "limit": 1})
            rows = data.get("rows") or []
            if data.get("ok") and rows:
                count = rows[0].get("mismatch_count")
                answer = f"มี {count} รายการ"
                validation = {
                    "status": "answered",
                    "confidence": "high",
                    "refs": [{"type": "postgres", "source": "FACT_VENDOR_PAYMENT", "sql": sql}],
                    "security": {},
                    "route": {"intent_type": "deterministic_7b_vendor_payment_month_mismatch", "tool": "postgres_execute_readonly_sql"},
                }
                return answer, {"validation": validation, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            last_error = data
        if last_error:
            return None, {"last_error": last_error}
    if _is_shipping_vendor_share_question(question):
        data = _read_tool_json("domain_shipping_vendor_share", {})
        rows = data.get("rows") or []
        if data.get("ok") and rows:
            parts = []
            for row in rows:
                vendor_name = row.get("vendor_name") or row.get("vendor_id") or "unknown"
                vendor_id = row.get("vendor_id")
                share = row.get("share_pct")
                count = row.get("shipment_count")
                label = f"{vendor_name} ({vendor_id})" if vendor_id and vendor_id != vendor_name else str(vendor_name)
                if share is not None:
                    parts.append(f"{label} {share}%")
                elif count is not None:
                    parts.append(f"{label} {count} รายการ")
            if parts:
                answer = "; ".join(parts)
                validation = {
                    "status": "answered",
                    "confidence": "high",
                    "refs": [{"type": "tool", "source": "domain_shipping_vendor_share", "arguments": {}}],
                    "security": {},
                    "route": {"intent_type": "deterministic_7b_shipping_vendor_share", "tool": "domain_shipping_vendor_share"},
                }
                return answer, {"validation": validation, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        return None, {"last_error": data}
    return None, {}


async def _answer_payload_7b(payload: Any) -> Any:
    if hosted_api.state is not None:
        det_answer, det_trace = await asyncio.to_thread(_deterministic_easy_answer, payload.question)
        if det_answer:
            answer_obj = LocalChatAnswer(
                id=payload.id or "API-Q",
                answer=det_answer,
                status=det_trace["validation"].get("status"),
                confidence=det_trace["validation"].get("confidence"),
                refs=det_trace["validation"].get("refs", []),
                security=det_trace["validation"].get("security", {}),
                route=det_trace["validation"].get("route", {}),
                run_log=None,
                seconds=0.0,
                token_usage=det_trace.get("usage", {}),
                answer_bank=payload.use_answer_bank,
            )
            return answer_obj

    preflight_trace = await asyncio.to_thread(_preflight_evidence_trace, payload.question)
    answer_obj = await _base_answer_payload(payload)
    first = _tool_call_from_answer(answer_obj.answer)
    if not first or hosted_api.state is None:
        return _apply_trace_validation(answer_obj, preflight_trace, {"intent_type": "base_pipeline_preflight"})

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
    trace: list[dict[str, Any]] = list(preflight_trace)
    token_usage = dict(answer_obj.token_usage or {})
    started = time.time()
    name, args = first

    for step in range(max(1, int(payload.max_steps or 5))):
        try:
            result = await asyncio.to_thread(_execute_tool, name, args)
        except Exception as exc:
            local_answer = _to_local_answer(answer_obj, default_route={"intent_type": "local_7b_repair"})
            local_answer.answer = f"ไม่สามารถเรียกใช้เครื่องมือ {name} ได้: {exc}"
            local_answer.status = "error"
            local_answer.refs = _refs_from_trace(trace)
            local_answer.route = {
                "intent_type": "local_7b_repair",
                "repair_error_tool": name,
                "repair_trace": _repair_trace_summary(trace),
            }
            return local_answer

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
            return _apply_trace_validation(answer_obj, trace)

    answer_obj.answer = "ไม่พบคำตอบในชุดข้อมูล"
    answer_obj.status = "needs_review"
    answer_obj.seconds = round((answer_obj.seconds or 0) + (time.time() - started), 3)
    answer_obj.token_usage = token_usage
    return _apply_trace_validation(answer_obj, trace)


hosted_api._answer_payload = _answer_payload_7b


app = hosted_api.app


def _remove_route(path: str, methods: set[str]) -> None:
    app.router.routes = [
        route
        for route in app.router.routes
        if not (getattr(route, "path", None) == path and methods.issubset(set(getattr(route, "methods", set()))))
    ]


_remove_route("/api/v1/chat", {"POST"})
_remove_route("/api/v1/batch", {"POST"})


@app.post("/api/v1/chat", response_model=LocalChatResponse)
async def chat_7b(req: LocalChatRequest) -> LocalChatResponse:
    return LocalChatResponse(data=await _answer_payload_7b(req.data))


@app.post("/api/v1/batch", response_model=LocalBatchResponse)
async def batch_7b(req: LocalBatchRequest) -> LocalBatchResponse:
    answers = []
    for payload in req.data:
        answers.append(await _answer_payload_7b(payload))
    return LocalBatchResponse(data=answers)


@hosted_api.app.get("/local-7b-health")
def local_7b_health() -> dict[str, object]:
    return {
        "ok": True,
        "mode": "local-7b",
        "model": os.getenv("LOCAL_TYPHOON_MODEL", DEFAULT_7B_MODEL),
        "local_openai_base_url": os.getenv("LOCAL_OPENAI_BASE_URL"),
        "note": "Main endpoints are /health, /api/v1/chat, and /api/v1/batch.",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("LOCAL_API_PORT", os.getenv("PORT", "8012")))
    uvicorn.run("api_server_local_7b:app", host="0.0.0.0", port=port, reload=False)
