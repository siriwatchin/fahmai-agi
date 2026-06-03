# FahMai Qwen2.5 Agentic Pipeline

Local/remote hybrid pipeline for FahMai Enterprise Data Agentic Showdown.

Core tools:

- PostgreSQL read-only SQL tool for structured tables.
- Qdrant vector search tool for documents, logs, and table snippets.
- Local Qwen2.5 inference through `transformers`.
- Deterministic SQL-first answer path with RAG fallback.
- Token/time accounting for every LLM call.

Secrets are loaded from environment variables. Do not commit `.env`.

## Setup

```bash
cd pipeline-qwen2.5
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with real credentials and paths.

## Environment

```bash
export PG_DSN="postgresql://USER:PASSWORD@HOST:PORT/DBNAME"
export QDRANT_URL="http://HOST:6333"
export QDRANT_API_KEY="..."
export QDRANT_COLLECTION="fahmai_public"
export EMBED_MODEL="BAAI/bge-m3"
export QWEN_MODEL_PATH="$HOME/scamper_house/qwen35/models/Qwen2.5-7B-Instruct"
export DATA_DIR="$HOME/scamper_house/fah-mai-the-finale-enterprise-data-agentic-showdown"
export QUESTIONS_CSV="$HOME/scamper_house/questions.csv"
```

## Ingest To Qdrant

```bash
python -m fahmai_qwen25.ingest_qdrant \
  --data-dir "$DATA_DIR" \
  --recreate
```

## Run Pipeline

```bash
python run_pipeline.py --limit 100
```

Outputs:

- `outputs/qwen25_results.csv`
- `outputs/qwen25_submission.csv`
- `outputs/qwen25_debug.json`
- `outputs/qwen25_token_usage.csv`
- `outputs/qwen25_token_summary.json`

The integrated B200 runner also writes every run to a timestamped folder:

```text
$WORK_ROOT/output/<RUN_ID>/
  best_results.csv
  best_submission.csv
  best_debug.json
  best_token_usage.csv
  best_token_summary.json
  best_llm_audit.jsonl
```

Set `RUN_ID`, `OUTPUT_ROOT`, or `RUN_OUTPUT_DIR` to customize the run folder.

## Run Source + Security Pipeline

`agentic_sourced_secure.py` is a separate wrapper around the current best
pipeline. It keeps the same SQL-first / RAG fallback answer logic, then adds:

- structured source attribution for SQL tables, rules, schema hits, TF-IDF docs, and Qdrant hits
- prompt-injection detection on both the question and retrieved context
- reasoning-trace leakage checks; public outputs do not expose chain-of-thought or raw prompt traces
- role-based access-control hooks through `ACCESS_ROLE`
- cross-source privilege metadata so lower-trust retrieved text cannot override SQL/rule evidence

Run on B200:

```bash
cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate

export MODEL_PATH="$HOME/bank500/qwen35/models/Qwen2.5-7B-Instruct"
export FAHMAI_SRC_ROOT="$HOME/scamper_house"
export WORK_ROOT="$HOME/bank500"
export SQL_BACKEND="duckdb"

export QDRANT_URL="http://127.0.0.1:6333"
export QDRANT_API_KEY="..."
export QDRANT_COLLECTION="fahmai_rag_bge"
export EMBED_MODEL="$HOME/bank500/qwen35/models/bge-m3"

export MODEL_LOAD_STRATEGY="cuda_direct"
export DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP="1"
export GEN_DO_SAMPLE="0"
export DOC_TOP_K="8"
export QDRANT_TOP_K="8"
export GEN_MAX_INPUT_TOKENS="7000"
export TORCH_NUM_THREADS="1"

python agentic_sourced_secure.py --limit 100 --skip-qdrant-preload
```

Outputs:

```text
$WORK_ROOT/output/<RUN_ID>_sourced_secure/
  sourced_secure_results.csv
  sourced_secure_submission.csv
  sourced_secure_records.jsonl
  sourced_secure_debug.json
  sourced_secure_token_usage.csv
  sourced_secure_summary.json
  sourced_secure_llm_audit.jsonl
```

`sourced_secure_results.csv` is for quick review. `sourced_secure_records.jsonl`
contains per-answer `sources` and `security` objects for downstream audit.
`sourced_secure_debug.json` is redacted by default. To write raw observations for
local debugging only, set `INCLUDE_RAW_DEBUG=1`.

Access roles:

```bash
python agentic_sourced_secure.py --limit 10 --access-role public_competition
python agentic_sourced_secure.py --limit 10 --access-role restricted_viewer
```

`public_competition` matches the Kaggle public data-lake setting. `restricted_viewer`
is a smoke-test role that denies finance/HR domains and returns an access refusal.

## Notes

- The pipeline prefers deterministic SQL when the question has clear table/field intent.
- Qdrant is used for document snippets, logs, memos, refusal evidence, and schema-ish text.
- `BAAI/bge-m3` is the required vector embedding/search model only. It is separate from the Qwen generation model.
- The LLM is used as a final synthesizer, not as the primary calculator.
- Qwen runs with a FahMai system prompt that enforces context-first tool use. When `OBSERVATIONS` are already supplied by the pipeline, it switches to final-answer mode and returns a concise Thai answer instead of tool-call JSON.

## Run FastAPI Chat Server

This wraps `agentic_best_integrated_qdrant.py`, which is the current B200 runner with SQL-first rules, Qdrant retrieval, and Qwen final answer generation.

```bash
cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate

source ~/.fahmai_db_env 2>/dev/null || true

# Use duckdb to start fast from the local data lake.
export SQL_BACKEND="duckdb"
export ALLOW_SQL_FALLBACK="1"

# If local Postgres is available on B200, switch to:
# export SQL_BACKEND="postgres"
# export ALLOW_SQL_FALLBACK="0"
# export PG_DSN="postgresql://admin:scamper@localhost:5432/fahmai"
# export PG_SCHEMA="public"

export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY="..."
export QDRANT_COLLECTION="fahmai_rag_bge"
export EMBED_MODEL="$HOME/bank500/qwen35/models/bge-m3"
export API_OUTPUT_DIR="$HOME/bank500"
export API_PORT="8888"
export ENABLE_API_CACHE="1"
export API_PRELOAD_ANSWERS="1"
export API_CACHE_MISS_FALLBACK="1"

# Optional input guardrail. Keep audit_only for Kaggle-style injection answers;
# use reject/block for production API safety.
export GUARDRAIL_URL="http://127.0.0.1:8000"
export GUARDRAIL_ACTION="audit_only"
export GUARDRAIL_THRESHOLD="0.75"

pip install -U fastapi "uvicorn[standard]"

uvicorn api_server:app --host 0.0.0.0 --port "$API_PORT"
```

Smoke test:

```bash
curl -s http://127.0.0.1:8888/health

curl -s -X POST http://127.0.0.1:8888/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"data":{"question":"วันนี้วันอะไร"}}'

curl -s -X POST http://127.0.0.1:8888/api/v2/chat \
  -H "Content-Type: application/json" \
  -d '{"data":{"question":"วันนี้วันอะไร"}}'
```

Agentic back-test endpoints:

```bash
curl -s -X POST http://127.0.0.1:8888/agent/local \
  -H "Content-Type: application/json" \
  -d '{"question":"วันนี้วันอะไร"}'

curl -s -X POST http://127.0.0.1:8888/agent/thaillm \
  -H "Content-Type: application/json" \
  -d '{"question":"MSRP ของสินค้ารหัส NT-LT-001 (NovaTech laptop) เป็นเท่าไหร่ครับ"}'
```

Agentic response format:

```json
{
  "id": "b8b9b5f0-9f69-4ef5-89f8-b85ac0086da9",
  "answer": "วันพุธ",
  "total_output_token": 3
}
```

For load-test mode, keep `ENABLE_API_CACHE=1`, `API_PRELOAD_ANSWERS=1`, and
`API_CACHE_MISS_FALLBACK=1`. Known competition questions are answered from the
precomputed cache. Cache misses first try a deterministic SQL/rule answer, then
return a scoped refusal instead of blocking on long Qwen generation.

`id` is a per-request UUID. `total_output_token` is counted from the final answer
with the active Qwen tokenizer, including cached/rule-based answers.

For load tests, pre-run the 100-question batch once, then keep `ENABLE_API_CACHE=1`.
The API preloads the newest `$WORK_ROOT/output/<RUN_ID>/best_results.csv` so repeated
questions return from memory instead of hitting Qwen/GPU.

API audit outputs are written under `API_OUTPUT_DIR`:

```text
api_requests.jsonl
api_token_usage.csv
api_token_summary.json
api_llm_audit.jsonl
```

Guardrail behavior:

- `GUARDRAIL_URL` unset: guardrail disabled.
- `GUARDRAIL_ACTION=audit_only`: log guardrail result but still let the FahMai agent answer. This is best for the competition because prompt-injection questions often need a defensive answer, not a hard block.
- `GUARDRAIL_ACTION=reject` or `block`: return a refusal immediately when guardrail says `is_attack=true`. This is best for production API safety.
- `GUARDRAIL_FAIL_CLOSED=1`: reject when the guardrail API is unreachable. Default is fail-open.

API contract:

```json
{
  "data": {
    "question": "วันนี้วันอะไร"
  }
}
```

Response:

```json
{
  "data": {
    "answer": "วันอังคาร"
  }
}
```
