from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def default_run_dir() -> Path:
    return ROOT / "outputs" / "runs" / f"local_7b_api_{time.strftime('%Y%m%d-%H%M%S')}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def looks_like_tool_leak(answer: str) -> bool:
    text = str(answer or "")
    return bool(re.search(r'"name"\s*:\s*"(postgres_|domain_|qdrant_)', text) and '"arguments"' in text)


def refusal_from_question(question: str) -> str:
    q = str(question or "")
    if any(token in q for token in ["รหัสสินค้า", "sku", "SKU", "สินค้า"]):
        return "ไม่พบ <รหัสสินค้า> ในชุดข้อมูล"
    return "ไม่พบคำตอบในชุดข้อมูล"


def finalize_answer(answer: str, question: str) -> str:
    text = str(answer or "").replace("\n", " ").strip()
    if not text:
        return refusal_from_question(question)
    if looks_like_tool_leak(text) or "<tool_call" in text.lower():
        return refusal_from_question(question)
    return text


def finalize_submission_answer(item: dict[str, Any], question: str) -> tuple[str, bool, str]:
    answer = finalize_answer(item.get("answer", ""), question)
    status = item.get("status")
    refs = item.get("refs") or []
    if answer.startswith("ไม่พบ"):
        return answer, False, "refusal"
    if status in {"error", "forbidden", "timeout", "needs_review"}:
        return refusal_from_question(question), False, f"bad_status:{status}"
    if not refs:
        return refusal_from_question(question), False, "no_refs"
    return answer, True, "accepted"


def count_output_tokens(answer: str) -> int:
    return max(0, len(re.findall(r"\S+", str(answer or ""))))


def write_submission(path: Path, sample_ids: list[str], answers: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        for qid in sample_ids:
            writer.writerow({"id": qid, "response": answers.get(qid, "")})


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def write_audit_files(run_dir: Path, audit_rows: list[dict[str, Any]]) -> None:
    llm_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    for rec in audit_rows:
        usage = rec.get("token_usage") or {}
        token_rows.append(
            {
                "request_uuid": rec.get("request_uuid"),
                "id": rec.get("id"),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "seconds": rec.get("seconds"),
            }
        )
        llm_rows.append(
            {
                "ts": rec.get("ts"),
                "request_uuid": rec.get("request_uuid"),
                "id": rec.get("id"),
                "model": rec.get("model"),
                "answer_preview": str(rec.get("answer") or "")[:500],
                "usage": usage,
                "seconds": rec.get("seconds"),
                "ok": rec.get("ok"),
            }
        )
        for ref in rec.get("refs") or []:
            tool_rows.append(
                {
                    "ts": rec.get("ts"),
                    "request_uuid": rec.get("request_uuid"),
                    "id": rec.get("id"),
                    "tool": "reference",
                    "output_obj": ref,
                    "ok": True,
                }
            )

    with (run_dir / "api_llm_audit.jsonl").open("w", encoding="utf-8") as f:
        for row in llm_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    with (run_dir / "api_tool_audit.jsonl").open("w", encoding="utf-8") as f:
        for row in tool_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    with (run_dir / "api_token_usage.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["request_uuid", "id", "prompt_tokens", "completion_tokens", "total_tokens", "seconds"])
        writer.writeheader()
        writer.writerows(token_rows)

    total_prompt = sum(r["prompt_tokens"] for r in token_rows)
    total_completion = sum(r["completion_tokens"] for r in token_rows)
    total_tokens = sum(r["total_tokens"] for r in token_rows)
    total_seconds = sum(float(r["seconds"] or 0) for r in token_rows)
    tool_summary = {
        "tool_audit_rows": len(tool_rows),
        "tools": {},
    }
    summary = {
        "num_llm_calls": len(token_rows),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "seconds": total_seconds,
        "qdrant_enabled": any(bool(r.get("qdrant_loaded")) for r in audit_rows),
        "guardrail_enabled": False,
        "tool_audit_rows": len(tool_rows),
        "tool_summary": tool_summary,
    }
    (run_dir / "api_token_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (run_dir / "api_tool_summary.json").write_text(json.dumps(tool_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_report(path: Path, *, api_url: str, audit_rows: list[dict[str, Any]], total_seconds: float, output: Path) -> None:
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status_counts: dict[str, int] = {}
    answered = 0
    leaks = 0
    rows = []

    for rec in audit_rows:
        status = rec.get("status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        answer = rec.get("answer") or ""
        if looks_like_tool_leak(answer):
            leaks += 1
        if answer and not answer.startswith("ไม่พบ") and not looks_like_tool_leak(answer) and status not in {"error", "forbidden", "timeout"}:
            answered += 1
        usage = rec.get("token_usage") or {}
        for key in total_tokens:
            total_tokens[key] += int(usage.get(key) or 0)
        refs = rec.get("refs") or []
        ref_text = ", ".join(f"{r.get('type')}:{r.get('source')}" for r in refs[:8] if isinstance(r, dict)) or "-"
        rows.append(f"- {rec.get('id')}: status={status}, seconds={rec.get('seconds')}, refs={ref_text}, answer={answer[:180]}")

    lines = [
        "Typhoon 7B API Client Run Report",
        f"API URL: {api_url}",
        f"Questions: {len(audit_rows)}",
        f"Answered: {answered}/{len(audit_rows)}",
        f"Tool-call leaks: {leaks}",
        f"Wall-clock seconds: {total_seconds:.1f}",
        "",
        "Token usage:",
        f"- prompt_tokens: {total_tokens['prompt_tokens']:,}",
        f"- completion_tokens: {total_tokens['completion_tokens']:,}",
        f"- total_tokens: {total_tokens['total_tokens']:,}",
        "",
        "Status counts:",
        *[f"- {k}: {v}" for k, v in sorted(status_counts.items())],
        "",
        "Files:",
        f"- submission: {output}",
        f"- api_requests: {path.parent / 'api_requests.jsonl'}",
        f"- api_llm_audit: {path.parent / 'api_llm_audit.jsonl'}",
        f"- api_tool_audit: {path.parent / 'api_tool_audit.jsonl'}",
        f"- api_token_usage: {path.parent / 'api_token_usage.csv'}",
        "",
        "Details:",
        *rows,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Send questions.csv to the local Typhoon 7B API server.")
    ap.add_argument("--api-url", default="http://127.0.0.1:8012/api/v1/chat")
    ap.add_argument("--health-url", default="http://127.0.0.1:8012/health")
    ap.add_argument("--questions", type=Path, default=ROOT / "questions.csv")
    ap.add_argument("--sample", type=Path, default=ROOT / "sample_submission.csv")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--use-answer-bank", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--user-role", default="audit")
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--timeout-seconds", type=int, default=60)
    ap.add_argument("--request-timeout", type=int, default=180)
    args = ap.parse_args()

    run_dir = args.run_dir or default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "submission.csv"
    debug_path = run_dir / "api_client_debug.json"
    report_path = run_dir / "api_client_report.md"
    request_log_path = run_dir / "api_requests.jsonl"
    meta_path = run_dir / "run_meta.json"

    questions = read_csv(args.questions)
    sample_ids = [r["id"] for r in read_csv(args.sample)]
    question_by_id = {r["id"]: r for r in questions}
    target_ids = sample_ids[args.offset :]
    if args.limit:
        target_ids = target_ids[: args.limit]

    meta = {
        "mode": "local-7b-api-client",
        "api_url": args.api_url,
        "health_url": args.health_url,
        "questions": str(args.questions),
        "sample": str(args.sample),
        "run_dir": str(run_dir),
        "limit": args.limit,
        "offset": args.offset,
        "use_answer_bank": args.use_answer_bank,
        "user_role": args.user_role,
        "max_steps": args.max_steps,
        "timeout_seconds": args.timeout_seconds,
        "request_timeout": args.request_timeout,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("health check...", flush=True)
    print(json.dumps(get_json(args.health_url), ensure_ascii=False), flush=True)

    answers: dict[str, str] = {}
    debug: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    start = time.time()

    for idx, qid in enumerate(target_ids, 1):
        row = question_by_id.get(qid)
        request_uuid = str(uuid.uuid4())
        if not row:
            answers[qid] = ""
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "request_uuid": request_uuid,
                "route": "/api/v1/chat",
                "id": qid,
                "question": "",
                "answer": "",
                "status": "missing_question",
                "ok": False,
                "error": "missing question in questions.csv",
            }
            debug[qid] = rec
            audit_rows.append(rec)
            append_jsonl(request_log_path, rec)
            continue

        payload = {
            "data": {
                "id": qid,
                "question": row["question"],
                "use_answer_bank": args.use_answer_bank,
                "user_role": args.user_role,
                "timeout_seconds": args.timeout_seconds,
                "max_steps": args.max_steps,
            }
        }
        print(f"[{idx}/{len(target_ids)}] {qid} sending...", flush=True)
        t0 = time.time()
        try:
            data = post_json(args.api_url, payload, timeout=args.request_timeout)
            item = data.get("data") if isinstance(data.get("data"), dict) else data
            raw_answer = item.get("answer", "")
            answer, answer_ok, answer_decision = finalize_submission_answer(item, row["question"])
            answers[qid] = answer
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "request_uuid": request_uuid,
                "route": "/api/v1/chat",
                "id": qid,
                "question": row["question"],
                "answer": answer,
                "raw_answer": raw_answer,
                "answer_decision": answer_decision,
                "total_output_token": count_output_tokens(answer),
                "seconds": round(time.time() - t0, 3),
                "api_seconds": item.get("seconds"),
                "status": item.get("status"),
                "confidence": item.get("confidence"),
                "refs": item.get("refs") or [],
                "security": item.get("security") or {},
                "route_meta": item.get("route") or {},
                "token_usage": item.get("token_usage") or {},
                "answer_bank": item.get("answer_bank"),
                "run_log": item.get("run_log"),
                "qdrant_loaded": (item.get("route") or {}).get("qdrant_loaded"),
                "tool_call_leak": looks_like_tool_leak(str(raw_answer or "")),
                "ok": answer_ok,
                "model": "local-7b",
            }
            debug[qid] = rec
            audit_rows.append(rec)
            append_jsonl(request_log_path, rec)
            print(f"  -> {answer[:180]}", flush=True)
            print(f"  status={item.get('status')} seconds={item.get('seconds')} refs={len(item.get('refs') or [])}", flush=True)
        except Exception as exc:
            answers[qid] = "ไม่พบคำตอบในชุดข้อมูล"
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "request_uuid": request_uuid,
                "route": "/api/v1/chat",
                "id": qid,
                "question": row["question"],
                "answer": answers[qid],
                "total_output_token": count_output_tokens(answers[qid]),
                "seconds": round(time.time() - t0, 3),
                "status": "error",
                "ok": False,
                "error": repr(exc),
                "model": "local-7b",
            }
            debug[qid] = rec
            audit_rows.append(rec)
            append_jsonl(request_log_path, rec)
            print(f"  ERROR: {exc}", flush=True)

        write_submission(output, target_ids, answers)
        debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        write_audit_files(run_dir, audit_rows)

    total_seconds = time.time() - start
    write_report(report_path, api_url=args.api_url, audit_rows=audit_rows, total_seconds=total_seconds, output=output)
    print(
        json.dumps(
            {
                "ok": True,
                "run_dir": str(run_dir),
                "submission": str(output),
                "debug": str(debug_path),
                "report": str(report_path),
                "api_requests": str(request_log_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
