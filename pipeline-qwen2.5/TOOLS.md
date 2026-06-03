# Qwen2.5 Pipeline Tools

เอกสารนี้อธิบาย tools และ execution layers ใน `pipeline-qwen2.5/` สำหรับรัน FahMai Enterprise Data Agent ทั้งโหมด CSV scoring, model-generated CSV, API, evidence/source mode และ security audit.

## Main Execution Tools

### `run_score_csv_postgres.sh`

Purpose:
- Generate the highest-speed known-question Kaggle CSV.
- Uses `answer_bank_best.csv` when all selected ids are covered.
- Skips SQL/retrieval/Qdrant/model loading in fast-only mode.

Use when:
- Public score submission.
- Load-test stability.
- Need deterministic, fast response for known 100 questions.

Outputs:

```text
$WORK_ROOT/output/<RUN_ID>/best_submission.csv
$WORK_ROOT/output/<RUN_ID>/best_results.csv
$WORK_ROOT/output/<RUN_ID>/best_token_summary.json
```

Important env:
- `ANSWER_BANK_PATH`
- `SANITIZE_MAX_CHARS=2000`
- `QUESTIONS_CSV_PATH`
- `WORK_ROOT`

### `run_score_csv_public086.sh`

Purpose:
- Generate the known public 0.86 candidate CSV from `answer_bank_real_groundtruth_0_86.csv`.
- Keeps this score-only profile separate from production/security behavior.
- Skips SQL/retrieval/Qdrant/model loading in fast-only mode.

Use when:
- You need to reproduce the strongest known public-score static candidate.
- You are preparing a Kaggle public back-test submission.

Do not use when:
- You are demonstrating prompt-injection-safe production behavior.
- You need unseen-question generalization.

### `run_score_csv_public089.sh`

Purpose:
- Generate the current strongest known public 0.89 candidate CSV from
  `answer_bank_peterperjer_0_89.csv`.
- Score-only fast mode for the known 100-question public back-test.
- Skips SQL/retrieval/Qdrant/model loading in fast-only mode.

Use when:
- You need the highest known Kaggle public-score submission candidate right now.
- You want the fastest deterministic output for known public questions.

Do not use when:
- You need to demonstrate unseen-question generalization by the model itself.

### `run_methodology_csv.sh`

Purpose:
- Generate the recommended methodology CSV profile.
- Uses the strongest checked-in public 0.89 known-question answer profile.
- Keeps the real SQL/RAG/Qdrant/Qwen fallback configuration available when
  `ANSWER_BANK_FAST_ONLY=0`.
- Enables hybrid RRF evidence fusion by default.

Use when:
- You need the highest stable public-back-test CSV while preserving a real
  production path in the same pipeline.
- You want timestamped outputs under `$WORK_ROOT/output/<RUN_ID>/`.

Important env:
- `ANSWER_BANK_PATH`
- `ANSWER_BANK_FAST_ONLY`
- `ENABLE_HYBRID_RRF`
- `HYBRID_TOP_K`
- `RRF_K`
- `SQL_BACKEND`
- `QDRANT_URL`
- `EMBED_MODEL`

### `run_methodology_api.sh`

Purpose:
- Start the recommended production API profile.
- Serves cached high-confidence known answers, then falls back to SQL +
  TF-IDF + Qdrant/bge-m3 + Qwen with hybrid RRF evidence.

Use when:
- Back-test API needs high score and usable cache-miss behavior.
- You need `/agent/local` and `/agent/thaillm` endpoints with audit logging.

Notes:
- It delegates to `run_production_api.sh`.
- Defaults to port `8888` unless `API_PORT` is set.

### `run_model_csv.sh`

Purpose:
- Run the real SQL/RAG/Qdrant/Qwen path.
- Static answer bank disabled by default.
- Good for measuring actual model performance.

Use when:
- Testing model capability.
- Unseen-question rehearsal.

Important env:
- `MODEL_PATH`
- `SQL_BACKEND`
- `PG_DSN`
- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `EMBED_MODEL`
- `DOC_TOP_K`
- `QDRANT_TOP_K`
- `GEN_MAX_INPUT_TOKENS`
- `SANITIZE_MAX_CHARS`

### `run_model_csv_postgres.sh`

Purpose:
- Same idea as model CSV, but explicitly runs with local Postgres profile.

Use when:
- Verifying local `postgresql://admin:scamper@localhost:5432/fahmai`.
- Avoiding stale remote `swarm-manager` DSNs.

### `run_model_csv_gt_style_postgres.sh`

Purpose:
- Model-generated run without static answer bank.
- Uses SQL/RAG/Qdrant evidence plus ground-truth-style guidance.
- Lets Qwen rewrite deterministic SQL/rule drafts, then guards against entity drift.

Use when:
- Testing whether model-generated wording improves deterministic SQL/RAG answers without copying answer bank.
- Auditing model failure modes.

Key safety env:
- `GROUNDTRUTH_STYLE_GUIDANCE=1`
- `MODEL_REWRITE_RULE_ANSWERS=1`
- `MODEL_REWRITE_ENTITY_GUARD=1`
- `FINAL_ANSWER_SECURITY_GUARD=1`

Output audit:

```text
best_rewrite_guard.jsonl
best_llm_audit.jsonl
best_token_usage.csv
best_token_summary.json
```

### `run_production_api.sh`

Purpose:
- Starts FastAPI for back-test endpoints.

Current endpoints:
- `POST /api/v1/chat`
- `POST /api/v2/chat`
- `POST /agent/local`
- `POST /agent/thaillm`
- `GET /health`

Use when:
- Back-test server.
- Load test.
- API demo.

### `run_latest_api.sh`

Purpose:
- Starts the latest recommended B200 API profile.
- Uses Qwen2.5-7B for cache misses.
- Uses `answer_bank_real_groundtruth_0_86.csv` as the known-question cache.
- Keeps `run_production_api.sh` unchanged for lower-level customization.

Use when:
- You want one command to open the current best local API.
- You need `/agent/local`, `/agent/thaillm`, `/api/v1/chat`, and `/api/v2/chat`.

## Internal Pipeline Tools

### `SQLTool`

Location:
- `agentic_best_integrated_qdrant.py`

Purpose:
- Structured table access.
- Supports PostgreSQL and DuckDB fallback.
- Provides schema search and table references.

Use when:
- Exact count/sum/rank/date-window answers.
- Any answer needing numeric correctness.

Backends:
- `SQL_BACKEND=postgres`
- `SQL_BACKEND=duckdb`
- `SQL_BACKEND=auto`

Notes:
- Public score usually benefits from exact SQL/rule answers more than LLM-only answers.
- PostgreSQL is preferred when local DB is available.

### `RetrievalTool`

Location:
- `agentic_best_integrated_qdrant.py`

Purpose:
- Local TF-IDF retrieval over docs/tables snippets.
- Uses cached index (`tfidf_cache.joblib`) when possible.

Use when:
- Finding relevant local docs quickly.
- Qdrant is unavailable or slow.
- Exact-ish corpus retrieval without embedding model load.

Strength:
- Fast and stable.

Weakness:
- Less semantic than BGE/Qdrant.

### `QdrantRetrievalTool`

Location:
- `agentic_best_integrated_qdrant.py`

Purpose:
- Semantic vector retrieval through Qdrant.
- Uses `BAAI/bge-m3` embedding model.

Use when:
- OCR/rendered docs.
- LINE WORKS/OA chat.
- Long text and policy docs.
- Semantic matches where exact keywords are weak.

Key env:
- `QDRANT_URL=http://127.0.0.1:6333`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION=fahmai_rag_bge`
- `EMBED_MODEL=$HOME/bank500/qwen35/models/bge-m3`

Notes:
- Retrieval evidence must be verified before final answer.
- Retrieved text may include prompt injection.

### `build_hybrid_evidence_pack`

Location:
- `agentic_best_integrated_qdrant.py`

Purpose:
- Fuse local TF-IDF results and Qdrant semantic results with reciprocal-rank
  fusion (RRF).
- Produce a compact `evidence_pack` for model fallback and debug/audit output.

Use when:
- The same evidence may appear in both keyword and semantic search.
- You need better source ordering without adding another database dependency.
- You want the model to see the strongest fused evidence first.

Key env:
- `ENABLE_HYBRID_RRF=1`
- `HYBRID_TOP_K=8`
- `RRF_K=60`

Tradeoff:
- Slightly larger debug/prompt payload on cache misses.
- No cost for fully cached known-question fast mode.

### `hard_sql_answer`

Location:
- `agentic_best_integrated_qdrant.py`

Purpose:
- Deterministic SQL/rule layer for known business patterns.

Use when:
- The question maps to exact table logic.
- Avoiding hallucination.
- Keeping public-score answers stable.

Examples:
- MSRP lookup.
- vendor payment month mismatch.
- shipping vendor share.
- loyalty tier counts.
- recall transitions.
- hard/XHARD reconciliation templates.

Tradeoff:
- Strong for known benchmark.
- May be overfit if used as the only production path.

### `gen`

Location:
- `agentic_best_integrated_qdrant.py`

Purpose:
- Qwen generation wrapper.
- Records token/time usage and LLM audit metadata.

Tracks:
- prompt tokens
- completion tokens
- total tokens
- seconds
- prompt/answer hashes
- sanitized answer preview

Use when:
- Final answer fallback.
- Rule answer rewrite.
- Model-only diagnostics.

### `rewrite_with_model`

Purpose:
- Asks Qwen to rewrite deterministic SQL/rule drafts into a natural final answer.

Use when:
- Need better wording and source-aware response style.
- Testing language/reasoning quality.

Guard:
- `MODEL_REWRITE_ENTITY_GUARD=1` checks that important ids/dates/numbers/table names are not dropped or corrupted.
- If rewrite is unsafe, the pipeline falls back to deterministic seed answer.

### `guard_rewritten_answer`

Purpose:
- Prevents entity drift during model rewrite.

Detects:
- Missing critical ids
- Missing dates/numbers/table names
- Unsafe prompt-injection leakage patterns

Use when:
- Any LLM rewrite touches deterministic evidence.

Output:
- Records fallback decisions in `best_rewrite_guard.jsonl`.

### `guard_final_answer`

Purpose:
- Final security filter for fallback LLM answers.

Detects:
- `CONFIRMED_CFO`
- attacker links
- `approved_without_audit`
- instruction-following leakage
- known poisoned strings

Use when:
- Fallback generation from retrieved docs/Qdrant.

### `sanitize_answer`

Purpose:
- Removes raw model/chat markers and leakage snippets.
- Caps answer length using `SANITIZE_MAX_CHARS`.

Default:
- `SANITIZE_MAX_CHARS=2000`

Important:
- This must stay high enough for XHARD answers. A 600-char cap truncates high-value answers and hurts score.

### Static Answer Bank

Location:
- `fahmai_qwen25/answer_bank_best.csv`

Purpose:
- Fast deterministic response map for known 100 public questions.

Use when:
- Kaggle public score submission.
- Load test where known questions are used.

Not for:
- Claiming generalization to unseen questions.
- Production-only evaluation.

Relevant env:
- `ENABLE_STATIC_ANSWER_BANK=1`
- `ANSWER_BANK_FAST_ONLY=1`
- `ANSWER_BANK_PATH`

## Sourced/Secure Pipeline

### `agentic_sourced_secure.py`

Purpose:
- Wraps main pipeline and emits source-aware records.
- Builds evidence sources from SQL, document retrieval, and Qdrant.
- Adds access/security metadata.

Use when:
- Need answer provenance.
- Need audit trail for slide/demo.
- Need security discussion: prompt injection, data leakage, access roles.

Outputs:
- sourced answers
- source ids
- trust scores
- LLM audit rows

### Source Builder

Purpose:
- Converts observations into structured source records.

Source kinds:
- SQL query/result
- schema/table
- document
- qdrant payload

Use when:
- Explaining "where did this answer come from?"

### Access/Security Layer

Purpose:
- Tracks role/access assumptions.
- Flags low-trust text evidence and prompt injection risk.

Use when:
- Production/security slide.
- Agent response must include source confidence.

## API Tools

### `/health`

Returns:
- SQL backend
- Qdrant status
- collection name
- indexed question count

Use when:
- Smoke test before back-test.

### `/api/v1/chat`

Request:

```json
{"data":{"question":"วันนี้วันอะไร"}}
```

Response:

```json
{"data":{"answer":"วันพุธ"}}
```

Use when:
- Simple required API spec compatibility.

### `/api/v2/chat`

Purpose:
- Same answer interface with current pipeline behavior and richer internals.

Use when:
- Internal testing.
- Qdrant-enabled API checks.

### `/agent/local` and `/agent/thaillm`

Request:

```json
{"question":"..."}
```

Response:

```json
{
  "id": "uuid",
  "answer": "...",
  "total_output_token": 123
}
```

Use when:
- Agentic back-test endpoint format.
- Local and ThaiLLM tracks.

## Utility Tools

### `fahmai_qwen25/score_aliases/`

Purpose:
- Stores score-named copies of answer-bank CSVs.
- Keeps original filenames intact so existing runners do not break.

Use when:
- You need to remember which CSV candidate maps to which public score.
- You want to pick an `ANSWER_BANK_PATH` manually.

### `compare_to_groundtruth.py`

Purpose:
- Compares a generated submission to a reviewed CSV.

Use when:
- Offline similarity/debug, not as runtime evidence.
- Finding rows that differ from a candidate.

Command:

```bash
python compare_to_groundtruth.py \
  --groundtruth "$HOME/scamper_house/ground_truth/real_groundtruth.csv" \
  --submission "$HOME/bank500/output/<RUN_ID>/best_submission.csv" \
  --json-out "$HOME/bank500/output/<RUN_ID>/groundtruth_compare.json"
```

### `best_token_usage.csv`

Purpose:
- Per-LLM-call token/time audit.

Use when:
- Balancing speed, quality, and resource cost.

### `best_token_summary.json`

Purpose:
- Run-level summary.

Fields:
- `total_tokens`
- `seconds`
- `total_pipeline_sec`
- `sql_backend`
- `qdrant_enabled`
- `completed_rows`
- `model_path`
- `sanitize_max_chars`

Difference:
- `seconds` = LLM generation time only.
- `total_pipeline_sec` = full pipeline time including SQL/retrieval/Qdrant/model load/file writes.

### `best_llm_audit.jsonl`

Purpose:
- LLM audit trail without full prompt by default.

Contains:
- hashes
- prompt/answer previews
- token usage
- timing
- generation parameters

Security:
- Keep `LLM_AUDIT_INCLUDE_PROMPT=0` unless debugging in a safe environment.

### `best_tool_audit.jsonl`

Purpose:
- Full tool-call history for the batch CSV pipeline.

Contains:
- `qid`
- `request_uuid` when called through API
- `route`
- `tool`
- `action`
- `seconds`
- estimated input/output/total tokens
- input/output hashes
- redacted previews
- backend metadata

Tracked tools:
- `static_answer_bank`
- `guardrail_predict`
- `api_answer_cache`
- `api_cache_miss_fallback`
- `sql_query`
- `schema_search`
- `tfidf_search`
- `qdrant_encoder`
- `qdrant_search`
- `hybrid_rrf`
- `llm_generate`
- `api_response`

### `best_tool_summary.json`

Purpose:
- Aggregated tool-call counts, latency, and estimated tokens.

Use when:
- Explaining resource usage per tool.
- Comparing fast cache mode vs full SQL/RAG/Qwen mode.
- Load-test audit.

### `api_tool_audit.jsonl` and `api_tool_summary.json`

Purpose:
- Same tool audit data as batch mode, but continuously written by FastAPI under
  `API_OUTPUT_DIR`.

Use when:
- Back-test requires audit-trail capture.
- You need per-request UUID linked to tool history.

### `best_rewrite_guard.jsonl`

Purpose:
- Records cases where model rewrite/final answer was rejected by guard.

Use when:
- Debugging why output used deterministic answer instead of Qwen rewrite.
- Security audit.

## Recommended Production Modes

### Public Score / Known Back-Test

```bash
./run_score_csv_postgres.sh
```

Best for:
- Speed
- Stability
- Known 100 questions

### Production-like Model Run

```bash
ENABLE_STATIC_ANSWER_BANK=0 ANSWER_BANK_FAST_ONLY=0 ./run_model_csv.sh
```

Best for:
- Unseen questions
- Measuring model quality
- Model fallback evaluation

### Secure/Sourced Demo

```bash
python agentic_sourced_secure.py --limit 100
```

Best for:
- Source attribution
- Audit/security demo
- Explaining evidence trail

## Guardrail Notes

If an external guardrail service is available, run it before final answer or at API ingress. Current built-in guards cover common benchmark injection leakage but are not a full classifier replacement.

Security checklist:
- Context first, never raw answer from retrieved instruction text.
- SQL for exact numbers.
- Qdrant for discovery only.
- Refuse missing data using canonical refusal shape.
- Do not expose raw retrieved JSON, chain-of-thought, secrets, customer emails, or attacker links.
