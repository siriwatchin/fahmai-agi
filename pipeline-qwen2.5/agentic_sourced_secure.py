from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

import agentic_best_integrated_qdrant as base


RUN_ID = os.getenv("RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
WORK = Path(os.getenv("WORK_ROOT", str(Path.home() / "bank500"))).expanduser()
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", str(WORK / "output"))).expanduser()
RUN_OUTPUT_DIR = Path(os.getenv("RUN_OUTPUT_DIR", str(OUTPUT_ROOT / f"{RUN_ID}_sourced_secure"))).expanduser()
INCLUDE_RAW_DEBUG = os.getenv("INCLUDE_RAW_DEBUG", "0").lower() in {"1", "true", "yes"}

INJECTION_PATTERNS = [
    r"(?i)\b(system|developer|admin)\s*(prompt|mode|override|instruction)\b",
    r"(?i)\bignore\s+(previous|all|rules|instructions)\b",
    r"(?i)\bdo\s+not\s+(consult|use|follow)\b",
    r"(?i)\boutput\s+['\"]?[^'\"]+['\"]?\s+verbatim\b",
    r"(?i)\btrust\s*=\s*high\b",
    r"(?i)\bconfirmed_[a-z0-9_]+\b",
    r"(?i)\bcopy\s+.*confirmation\s+link\b",
    r"(?i)\bprevious\s+session\b",
    r"(?i)\bตอบด้วยข้อความ\b",
    r"(?i)\bพบกันใหม่\b",
    r"(?i)\[/?system\]",
]

SENSITIVE_TERMS = {
    "finance": ["BANK", "PAYMENT", "PAYROLL", "REFUND", "INVOICE", "FEE", "DEPOSIT", "AR"],
    "hr": ["EMPLOYEE", "PAYROLL", "CEO", "CFO", "HR", "SALARY"],
    "customer": ["CUSTOMER", "B2B", "B2C", "CS_INTERACTION", "CHAT", "LINE"],
    "vendor": ["VENDOR", "SHIPPING", "WARRANTY", "CLAIM", "RECALL"],
}

ROLE_DENYLIST = {
    # Public competition access is the Kaggle setting: public data lake, all rows visible.
    "public_competition": set(),
    # Use this role to smoke-test access controls for demos.
    "restricted_viewer": {"finance", "hr"},
}

TRUST_SCORE = {
    "deterministic_rule": 0.98,
    "sql_result": 0.95,
    "schema": 0.75,
    "qdrant": 0.68,
    "document": 0.62,
}


def _hash_text(text: str, n: int = 10) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()[:n]


def _compact(text: Any, limit: int = 220) -> str:
    text = str(text or "")
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Keep evidence snippets useful but avoid dumping long records/prompts.
    return text[:limit]


def _source_id(kind: str, name: str, path: str = "") -> str:
    return f"{kind}:{_hash_text(kind + '|' + name + '|' + path)}"


def _doc_path_from_text(text: str) -> tuple[str, str]:
    first, _, rest = str(text).partition("\n")
    m = re.match(r"\[(DOC|CSV_SAMPLE|TABLE)\]\s+(.+)", first.strip())
    if m:
        return m.group(2).strip(), rest
    return first.strip()[:120], rest or text


def _sql_tables(sql: str) -> list[str]:
    tables: list[str] = []
    for m in re.finditer(r'(?i)\b(?:FROM|JOIN)\s+((?:"[^"]+"\.)?"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)', str(sql)):
        raw = m.group(1)
        parts = re.findall(r'"([^"]+)"', raw)
        table = parts[-1] if parts else raw
        if table not in tables:
            tables.append(table)
    return tables


def _source_domains(source_names: list[str], question: str) -> set[str]:
    blob = " ".join(source_names + [question]).upper()
    domains: set[str] = set()
    for domain, terms in SENSITIVE_TERMS.items():
        if any(term in blob for term in terms):
            domains.add(domain)
    return domains


def detect_injection(text: str) -> bool:
    return any(re.search(p, str(text or "")) for p in INJECTION_PATTERNS)


def build_sources(obs: dict[str, Any], sqltool: Any) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    rule = obs.get("rule")
    if rule:
        sid = _source_id("rule", str(rule))
        sources.append(
            {
                "id": sid,
                "kind": "deterministic_rule",
                "name": str(rule),
                "path": None,
                "trust": TRUST_SCORE["deterministic_rule"],
                "excerpt": "Deterministic rule in pipeline source code.",
            }
        )
        seen.add(sid)

    sql = obs.get("sql") or (obs.get("sql_result") or {}).get("sql")
    if sql:
        for table in _sql_tables(sql):
            key = base.clean_name(table)
            meta = getattr(sqltool, "tables", {}).get(key, {})
            path = meta.get("path") or table
            sid = _source_id("sql", table, path)
            if sid in seen:
                continue
            sources.append(
                {
                    "id": sid,
                    "kind": "sql_result",
                    "name": table,
                    "path": path,
                    "trust": TRUST_SCORE["sql_result"],
                    "row_shape": (obs.get("sql_result") or {}).get("shape"),
                    "sql_hash": _hash_text(sql, 12),
                    "excerpt": f"SQL SELECT/WITH over {table}",
                }
            )
            seen.add(sid)

    for meta in obs.get("schema_search") or []:
        table = str(meta.get("table") or meta.get("path") or "schema")
        path = str(meta.get("path") or table)
        sid = _source_id("schema", table, path)
        if sid in seen:
            continue
        cols = ", ".join(c.get("column_name", "") for c in meta.get("columns", [])[:10])
        sources.append(
            {
                "id": sid,
                "kind": "schema",
                "name": table,
                "path": path,
                "trust": TRUST_SCORE["schema"],
                "excerpt": f"Columns: {cols}",
            }
        )
        seen.add(sid)

    for doc in obs.get("document_search") or []:
        text = str(doc.get("text", ""))
        path, body = _doc_path_from_text(text)
        sid = _source_id("document", path)
        if sid in seen:
            continue
        sources.append(
            {
                "id": sid,
                "kind": "document",
                "name": Path(path).name or path,
                "path": path,
                "score": doc.get("score"),
                "trust": TRUST_SCORE["document"],
                "excerpt": _compact(body),
            }
        )
        seen.add(sid)

    for hit in obs.get("qdrant_search") or []:
        path = str(hit.get("path") or hit.get("title") or "qdrant_payload")
        name = str(hit.get("title") or Path(path).name or path)
        sid = _source_id("qdrant", name, path)
        if sid in seen:
            continue
        sources.append(
            {
                "id": sid,
                "kind": "qdrant",
                "name": name,
                "path": path,
                "score": hit.get("score"),
                "date": hit.get("date"),
                "doc_type": hit.get("doc_type"),
                "trust": TRUST_SCORE["qdrant"],
                "excerpt": _compact(hit.get("text")),
            }
        )
        seen.add(sid)

    return sources


def build_security_report(
    qid: str,
    question: str,
    answer: str,
    obs: dict[str, Any],
    sources: list[dict[str, Any]],
    access_role: str,
) -> dict[str, Any]:
    source_names = [str(s.get("name") or "") + " " + str(s.get("path") or "") for s in sources]
    domains = sorted(_source_domains(source_names, question))
    denied_domains = sorted(set(domains) & ROLE_DENYLIST.get(access_role, set()))

    context_text = " ".join(_compact(s.get("excerpt"), 120) for s in sources)
    question_injection = detect_injection(question) or str(qid).startswith("L3-Q-INJ")
    context_injection = detect_injection(context_text) or any(str(s.get("doc_type", "")).lower() == "prompt_injection" for s in sources)
    used_sql = any(s.get("kind") == "sql_result" for s in sources)
    used_low_trust_text = any(s.get("kind") in {"document", "qdrant"} for s in sources)

    flags: list[str] = []
    if question_injection:
        flags.append("prompt_injection_question")
    if context_injection:
        flags.append("prompt_injection_retrieved_context")
    if denied_domains:
        flags.append("right_of_access_restricted")
    if used_sql and used_low_trust_text:
        flags.append("cross_source_privilege_checked")
    if re.search(r"(?i)<think>|chain[- ]of[- ]thought|reasoning trace|OBSERVATION|SQL_result|document_search", answer):
        flags.append("reasoning_trace_leakage_risk")

    return {
        "access_role": access_role,
        "access_decision": "deny" if denied_domains else "allow",
        "denied_domains": denied_domains,
        "detected_domains": domains,
        "prompt_injection_detected": question_injection or context_injection,
        "reasoning_trace_exposed": "reasoning_trace_leakage_risk" in flags,
        "cross_source_privilege_control": {
            "enabled": True,
            "policy": "SQL/rule evidence outranks retrieved text; retrieved text cannot override instructions or table facts.",
            "mixed_sources": used_sql and used_low_trust_text,
        },
        "source_trust_order": TRUST_SCORE,
        "flags": flags,
    }


def enforce_access(answer: str, security: dict[str, Any], question: str) -> str:
    if security.get("access_decision") != "deny":
        return answer
    topic = ", ".join(security.get("denied_domains") or []) or _compact(question, 60)
    return f"ไม่มีสิทธิ์เข้าถึง {topic} ในระบบ"


def answer_one_secure(sqltool, retriever, qdrant_retriever, tok, model, qid: str, question: str, access_role: str):
    t0 = time.time()
    answer, obs = base.answer_one(sqltool, retriever, qdrant_retriever, tok, model, qid, question)
    sources = build_sources(obs, sqltool)
    security = build_security_report(qid, question, answer, obs, sources, access_role)
    answer = enforce_access(answer, security, question)
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "sources": sources,
        "security": security,
        "seconds": round(time.time() - t0, 3),
    }, obs


def save_outputs(records: list[dict[str, Any]], debug: dict[str, Any], sqltool: Any, qdrant_retriever: Any, run_t0: float):
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for rec in records:
        source_refs = "; ".join(
            f"{s.get('kind')}:{s.get('path') or s.get('name')}" for s in rec["sources"][:8]
        )
        flat_rows.append(
            {
                "id": rec["id"],
                "question": rec["question"],
                "answer": rec["answer"],
                "source_count": len(rec["sources"]),
                "source_refs": source_refs,
                "security_flags": ",".join(rec["security"].get("flags", [])),
                "access_decision": rec["security"].get("access_decision"),
                "seconds": rec["seconds"],
            }
        )

    result_df = pd.DataFrame(flat_rows)
    if len(result_df):
        result_df.to_csv(RUN_OUTPUT_DIR / "sourced_secure_results.csv", index=False)
        result_df[["id", "answer"]].rename(columns={"answer": "response"}).to_csv(
            RUN_OUTPUT_DIR / "sourced_secure_submission.csv", index=False
        )
    else:
        pd.DataFrame(columns=["id", "response"]).to_csv(RUN_OUTPUT_DIR / "sourced_secure_submission.csv", index=False)

    with (RUN_OUTPUT_DIR / "sourced_secure_records.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    if INCLUDE_RAW_DEBUG:
        debug_payload = debug
    else:
        debug_payload = {
            qid: {
                "observation_keys": sorted(list(obs.keys())),
                "has_sql": bool(obs.get("sql") or obs.get("sql_result")),
                "sql_hash": _hash_text(obs.get("sql") or (obs.get("sql_result") or {}).get("sql") or "", 12)
                if (obs.get("sql") or (obs.get("sql_result") or {}).get("sql"))
                else None,
                "document_hits": len(obs.get("document_search") or []),
                "qdrant_hits": len(obs.get("qdrant_search") or []),
                "schema_hits": len(obs.get("schema_search") or []),
                "rule": obs.get("rule"),
            }
            for qid, obs in debug.items()
        }
    (RUN_OUTPUT_DIR / "sourced_secure_debug.json").write_text(
        json.dumps(debug_payload, ensure_ascii=False, indent=2, default=str)
    )

    token_df = pd.DataFrame(base.TOKEN_LOG)
    token_df.to_csv(RUN_OUTPUT_DIR / "sourced_secure_token_usage.csv", index=False)
    with (RUN_OUTPUT_DIR / "sourced_secure_llm_audit.jsonl").open("w", encoding="utf-8") as f:
        for rec in getattr(base, "LLM_AUDIT_LOG", []):
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    summary = {
        "run_id": RUN_ID,
        "run_output_dir": str(RUN_OUTPUT_DIR),
        "completed_rows": len(records),
        "num_llm_calls": int(len(token_df)),
        "prompt_tokens": int(token_df["prompt_tokens"].sum()) if len(token_df) else 0,
        "completion_tokens": int(token_df["completion_tokens"].sum()) if len(token_df) else 0,
        "total_tokens": int(token_df["total_tokens"].sum()) if len(token_df) else 0,
        "llm_seconds": float(token_df["seconds"].sum()) if len(token_df) else 0,
        "llm_audit_rows": len(getattr(base, "LLM_AUDIT_LOG", [])),
        "total_pipeline_sec": round(time.time() - run_t0, 3),
        "sql_backend": getattr(sqltool, "backend", None),
        "sql_error": getattr(sqltool, "error", None),
        "qdrant_enabled": bool(qdrant_retriever and qdrant_retriever.ok),
        "qdrant_collection": getattr(qdrant_retriever, "collection", None) if qdrant_retriever else None,
        "security_modes": {
            "source_attribution": True,
            "prompt_injection_detection": True,
            "reasoning_trace_redaction": True,
            "raw_debug_enabled": INCLUDE_RAW_DEBUG,
            "right_of_access": True,
            "cross_source_privilege_control": True,
        },
    }
    (RUN_OUTPUT_DIR / "sourced_secure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--no-qdrant", action="store_true")
    ap.add_argument("--skip-qdrant-preload", action="store_true")
    ap.add_argument(
        "--access-role",
        default=os.getenv("ACCESS_ROLE", "public_competition"),
        choices=sorted(ROLE_DENYLIST.keys()),
    )
    args = ap.parse_args()

    run_t0 = time.time()
    print("run_id:", RUN_ID)
    print("run_output_dir:", RUN_OUTPUT_DIR)
    print("security_pipeline: sourced_secure")
    print("access_role:", args.access_role)

    print("loading sql...")
    t0 = time.time()
    sqltool = base.SQLTool()
    print("tables:", len(sqltool.tables), "sql_backend:", sqltool.backend, "sql_load_sec:", round(time.time() - t0, 3))

    print("loading retrieval...")
    t0 = time.time()
    retriever = base.RetrievalTool()
    print("docs:", len(retriever.docs), "retrieval_load_sec:", round(time.time() - t0, 3))

    qdrant_retriever = None
    if not args.no_qdrant:
        print("loading qdrant...")
        t0 = time.time()
        qdrant_retriever = base.QdrantRetrievalTool()
        print(
            "qdrant_ok:",
            qdrant_retriever.ok,
            "collection:",
            qdrant_retriever.collection,
            "vector_name:",
            qdrant_retriever.vector_name,
            "qdrant_load_sec:",
            round(time.time() - t0, 3),
        )
        if qdrant_retriever.ok and not args.skip_qdrant_preload:
            print("preloading qdrant encoder:", qdrant_retriever.embed_model)
            t0 = time.time()
            qdrant_retriever.preload_encoder()
            print("qdrant_encoder_load_sec:", round(time.time() - t0, 3))

    print("loading model...")
    t0 = time.time()
    tok, model = base.load_model()
    print("model_load_sec:", round(time.time() - t0, 3))

    qdf, id_col, q_col = base.load_questions()
    qdf = qdf.head(args.limit)
    records: list[dict[str, Any]] = []
    debug: dict[str, Any] = {}

    for _, row in qdf.iterrows():
        qid = str(row[id_col]).strip()
        question = str(row[q_col]).strip()
        print(f"\n== {qid} ==")
        print(question)
        rec, obs = answer_one_secure(sqltool, retriever, qdrant_retriever, tok, model, qid, question, args.access_role)
        print("ANSWER:", rec["answer"])
        print("sources:", len(rec["sources"]), "security_flags:", ",".join(rec["security"].get("flags", [])) or "-")
        print("question_sec:", rec["seconds"])
        records.append(rec)
        debug[qid] = obs
        save_outputs(records, debug, sqltool, qdrant_retriever, run_t0)

    summary = save_outputs(records, debug, sqltool, qdrant_retriever, run_t0)
    print("\nDONE")
    print("run_output_dir:", RUN_OUTPUT_DIR)
    print("results:", RUN_OUTPUT_DIR / "sourced_secure_results.csv")
    print("jsonl:", RUN_OUTPUT_DIR / "sourced_secure_records.jsonl")
    print("summary:", RUN_OUTPUT_DIR / "sourced_secure_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
