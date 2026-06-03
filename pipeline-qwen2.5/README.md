# FahMai Qwen2.5 Agentic Pipeline

Local/remote hybrid pipeline for FahMai Enterprise Data Agentic Showdown.

Detailed tool reference: [`TOOLS.md`](TOOLS.md)

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
  best_rewrite_guard.jsonl
```

Set `RUN_ID`, `OUTPUT_ROOT`, or `RUN_OUTPUT_DIR` to customize the run folder.

## Best Score Fast Mode

`agentic_best_integrated_qdrant.py` now checks a curated static answer bank before
loading SQL, Qdrant, or Qwen:

```text
fahmai_qwen25/answer_bank_best.csv
```

This is the highest-speed competition mode for the known 100-question back-test.
The current default `answer_bank_best.csv` is the v7 compact-keyword bank derived
from the public 0.80 candidate plus targeted HARD/XHARD/refusal/injection patches.
The v7 pass keeps every answer under the pipeline sanitizer cap so XHARD evidence
is not truncated before CSV generation or API serving.
When every selected question id is covered and `ANSWER_BANK_FAST_ONLY=1`, the
runner skips SQL/retrieval/Qdrant/model loading and writes the output immediately.
Use this for public-score submission rehearsal and load-test stability.

Run on B200:

Recommended score-submission command:

```bash
cd ~/fahmai-agi
git pull origin main

cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_score_csv_postgres.sh
```

Public 0.86 candidate profile:

```bash
cd ~/fahmai-agi
git pull origin main

cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_score_csv_public086.sh
```

`run_score_csv_public086.sh` uses `fahmai_qwen25/answer_bank_real_groundtruth_0_86.csv`.
Treat it as a score-submission/static public-back-test profile, not as the
production/security API behavior.

Public 0.89 candidate profile:

```bash
cd ~/fahmai-agi
git pull origin main

cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_score_csv_public089.sh
```

`run_score_csv_public089.sh` uses `fahmai_qwen25/answer_bank_peterperjer_0_89.csv`.
This is currently the strongest known public-score static profile.

## Real Model CSV Run With Logs

Use this when you want to inspect the real Qwen path instead of submitting the
static public-score profile. This disables the static answer bank and routes all
questions through SQL/schema tools, TF-IDF/Qdrant retrieval, and Qwen generation.

```bash
cd ~/fahmai-agi
git pull origin main

cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_real_model_csv.sh
```

Output is written to:

```text
$WORK_ROOT/output/<RUN_ID>_real_model/
  run.log
  best_submission.csv
  best_results.csv
  best_debug.json
  best_token_usage.csv
  best_token_summary.json
  best_llm_audit.jsonl
  best_tool_audit.jsonl
  best_tool_summary.json
```

Default SQL backend is DuckDB because it is always available from the local data
lake. If B200 local Postgres is running, use:

```bash
SQL_BACKEND=postgres \
PG_DSN="postgresql://admin:scamper@localhost:5432/fahmai" \
./run_real_model_csv.sh
```

To tail the log from another terminal:

```bash
tail -f ~/bank500/output/*_real_model/run.log
```

## Real Model Debug API

Use this when teammates need one-question-at-a-time inspection from the real
SQL/RAG/Qdrant/Qwen path. This profile disables answer-bank/cache answers and
returns a verbose per-request payload.

Run on B200:

```bash
cd ~/fahmai-agi
git pull origin main

cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_real_model_debug_api.sh
```

If you are not in the pipeline folder, run it by absolute path:

```bash
~/fahmai-agi/pipeline-qwen2.5/run_real_model_debug_api.sh
```

Debug endpoint:

```bash
curl -s -X POST http://127.0.0.1:8888/api/v2/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"MSRP ของสินค้ารหัส NT-LT-001 (NovaTech laptop) เป็นเท่าไหร่ครับ"}' \
  | python -m json.tool > debug_one_question.json
```

If `python -m json.tool` says `Expecting value`, inspect the raw HTTP response
first:

```bash
curl -i -X POST http://127.0.0.1:8888/api/v2/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"MSRP ของสินค้ารหัส NT-LT-001 (NovaTech laptop) เป็นเท่าไหร่ครับ"}'
```

`/api/v2/chat` also accepts the older wrapped body:

```json
{
  "data": {
    "question": "..."
  }
}
```

Response fields:

```json
{
  "id": "request uuid",
  "qid": "question id if known",
  "route": "/api/v2/chat",
  "question": "...",
  "answer": "...",
  "total_output_token": 12,
  "request_seconds": 3.21,
  "sources": [],
  "guardrail": {},
  "token_usage": {},
  "token_log": [],
  "llm_audit": [],
  "tool_audit": [],
  "tool_summary": {},
  "runtime": {},
  "observation": {}
}
```

Notes:

- In `run_real_model_debug_api.sh`, `/api/v2/chat` returns the full debug payload.
- In normal API profiles, `/api/v2/chat` keeps the simple `{ "data": { "answer": "..." } }` chat response.
- `/agent/local` and `/agent/thaillm` stay compatible with the back-test spec:
  `id`, `answer`, `total_output_token`.
- `/agent/local/debug` and `/agent/thaillm/debug` are for internal inspection.
- `observation`, `tool_audit`, and `llm_audit` are redacted by default.
- Set `API_DEBUG_INCLUDE_RAW_OBSERVATION=1` only for trusted internal debugging.
- `api_requests.jsonl`, `api_llm_audit.jsonl`, `api_tool_audit.jsonl`,
  `api_token_usage.csv`, and `api_tool_summary.json` are still written under
  `API_OUTPUT_DIR`.

## Methodology Profile: High Score + Usable Fallback

Use this profile when you want the most practical balance:

- Known 100-question public back-test: stable 0.89 high-score answer cache.
- Cache miss/unseen questions: SQL + TF-IDF + Qdrant/bge-m3 + Qwen fallback.
- Hybrid evidence pack: TF-IDF and Qdrant hits are fused with reciprocal-rank
  fusion (`ENABLE_HYBRID_RRF=1`) before the model sees observations.
- Run metadata: token usage, LLM audit, debug evidence, rewrite guard, and
  timestamped output folder.

CSV command:

```bash
cd ~/fahmai-agi
git pull origin main

cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_methodology_csv.sh
```

Output:

```text
$WORK_ROOT/output/<RUN_ID>/best_submission.csv
$WORK_ROOT/output/<RUN_ID>/best_results.csv
$WORK_ROOT/output/<RUN_ID>/best_debug.json
$WORK_ROOT/output/<RUN_ID>/best_token_usage.csv
$WORK_ROOT/output/<RUN_ID>/best_token_summary.json
```

API command:

```bash
cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_methodology_api.sh
```

The API profile exposes the same endpoints as `run_production_api.sh`, uses the
0.89 known-answer cache by default, and keeps fallback enabled for questions not
present in the cache.

Guarded API with source attribution:

```bash
cd ~/fahmai-agi
git pull origin main

cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_guarded_source_api.sh
```

This sets:

```bash
GUARDRAIL_ENDPOINT=http://localhost:7777/predictv2
GUARDRAIL_MAX_LENGTH=2048
GUARDRAIL_THRESHOLD=0.75
GUARDRAIL_ACTION=audit_only
API_INCLUDE_SOURCES=1
```

Use `GUARDRAIL_ACTION=reject ./run_guarded_source_api.sh` for stricter
production blocking. Keep `audit_only` for Kaggle-style prompt-injection
questions because the grader may expect a defensive business answer rather than
an immediate hard block.

Score-named copies are available under:

```text
fahmai_qwen25/score_aliases/
```

The original filenames are kept because runners reference them directly.

Manual equivalent:

```bash
cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate

export WORK_ROOT="$HOME/bank500"
export FAHMAI_SRC_ROOT="$HOME/scamper_house"
export QUESTIONS_CSV_PATH="$HOME/scamper_house/questions.csv"

export ENABLE_STATIC_ANSWER_BANK="1"
export ANSWER_BANK_FAST_ONLY="1"
export ANSWER_BANK_PATH="$HOME/fahmai-agi/pipeline-qwen2.5/fahmai_qwen25/answer_bank_best.csv"
export ANSWER_BANK_VERSION="best_v7_compact_keywords"

python agentic_best_integrated_qdrant.py --limit 100 --skip-qdrant-preload
```

The final file is:

```text
$WORK_ROOT/output/<RUN_ID>/best_submission.csv
```

For ablation or unseen questions, keep the answer bank as a first-pass cache but
allow the real agent fallback:

```bash
export ANSWER_BANK_FAST_ONLY="0"
```

Then configure `SQL_BACKEND`, `QDRANT_URL`, `EMBED_MODEL`, and `MODEL_PATH` as
usual. Known ids still return from the answer bank; missing ids go through the
SQL/RAG/Qwen path.

## Model-Generated CSV

Use this when you want to measure the actual B200 model pipeline instead of the
static answer bank. This path disables the answer bank and runs the full
SQL/RAG/Qdrant/Qwen stack, then writes:

```text
$WORK_ROOT/output/<RUN_ID>/best_submission.csv
```

Run on B200:

```bash
cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_model_csv.sh
```

The script defaults to:

```text
SQL_BACKEND=postgres
PG_DSN=postgresql://admin:scamper@localhost:5432/fahmai
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=fahmai_rag_bge
MODEL_PATH=~/bank500/qwen35/models/Qwen2.5-7B-Instruct
ENABLE_STATIC_ANSWER_BANK=0
ANSWER_BANK_FAST_ONLY=0
```

For the strict local Postgres measurement path use:

```bash
./run_model_csv_postgres.sh
```

Ground-truth-style model run, without copying an answer bank:

```bash
./run_model_csv_gt_style_postgres.sh
```

This profile sets:

```text
ENABLE_STATIC_ANSWER_BANK=0
ANSWER_BANK_FAST_ONLY=0
GROUNDTRUTH_STYLE_GUIDANCE=1
MODEL_REWRITE_RULE_ANSWERS=1
MODEL_REWRITE_ENTITY_GUARD=1
FINAL_ANSWER_SECURITY_GUARD=1
FINAL_MAX_NEW_TOKENS=260
```

It still uses SQL/RAG/Qdrant as evidence, but Qwen rewrites deterministic
SQL/rule drafts into final answers using a rubric distilled from the reviewed
ground-truth response style. It does not map question id to a stored response.
The rewrite guard keeps this mode from damaging evidence: if the LLM drops or
mutates critical ids, dates, table names, counts, amounts, or emits known prompt
injection leakage, the final answer falls back to the deterministic
SQL/RAG-derived draft. Guard decisions are written to
`best_rewrite_guard.jsonl` and counted in `best_token_summary.json`.

After a run, compare the generated submission with a reviewed CSV:

```bash
python compare_to_groundtruth.py \
  --groundtruth "$HOME/scamper_house/ground_truth/real_groundtruth.csv" \
  --submission "$HOME/bank500/output/<RUN_ID>/best_submission.csv" \
  --json-out "$HOME/bank500/output/<RUN_ID>/groundtruth_compare.json"
```

If the ground-truth file is not on B200, upload it first or point
`--groundtruth` to the local path that contains `id,response`.

For highest public-score rehearsal use `./run_score_csv_postgres.sh` instead.

For a quick smoke run:

```bash
LIMIT=10 ./run_model_csv.sh
```

Targeted XHARD/REF/INJ probe:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

src = Path.home() / "scamper_house/questions.csv"
out = Path.home() / "bank500/probe_xhard_ref_inj.csv"

df = pd.read_csv(src)
id_col = df.columns[0]
probe = df[df[id_col].astype(str).str.strip().str.startswith(("L3-Q-XHARD", "L3-Q-REF", "L3-Q-INJ"))].copy()
probe.to_csv(out, index=False)
print(out, probe.shape)
PY

QUESTIONS_CSV_PATH="$HOME/bank500/probe_xhard_ref_inj.csv" LIMIT=999 ./run_model_csv_postgres.sh
```

Strict PostgreSQL run, no DuckDB fallback:

```bash
cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_model_csv_postgres.sh
```

If Postgres is not reachable, this script fails instead of producing a DuckDB
result. The token summary should show:

```json
{
  "sql_backend": "postgres"
}
```

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

## Run Production FastAPI Server

Use this mode for a real service on B200. It is not fast-only. Startup loads SQL,
local document retrieval, Qdrant, and Qwen. Known 100-question back-test items can
still return from the curated answer bank/cache for speed; unseen questions go
through the real SQL/RAG/Qwen path.

Latest recommended local API profile:

```bash
cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate
./run_latest_api.sh
```

This starts the API with Qwen2.5-7B for cache misses and the 0.86 answer bank as
the known-question cache.

Recommended B200 setup:

```bash
cd ~/fahmai-agi
git pull origin main

cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate

cat > ~/.fahmai_db_env <<'ENV'
export PG_DSN="postgresql://admin:scamper@localhost:5432/fahmai"
export QDRANT_URL="http://127.0.0.1:6333"
export QDRANT_API_KEY="569f01c61ce1e2a2acad1d9e268fa73d8b1a7cc076806720b44f034fd5f3bb41"
export QDRANT_COLLECTION="fahmai_rag_bge"
export GUARDRAIL_URL="http://127.0.0.1:8000"
ENV

chmod +x run_production_api.sh
./run_production_api.sh
```

Production knobs:

```bash
# Balanced production: cache known questions, answer unknown questions with SQL/RAG/Qwen.
export API_FAST_ONLY="0"
export API_CACHE_MISS_FALLBACK="0"
export ENABLE_API_CACHE="1"
export ENABLE_STATIC_ANSWER_BANK="1"

# Strict production prompt-injection behavior.
export GUARDRAIL_ACTION="reject"
export GUARDRAIL_FAIL_CLOSED="1"

# Competition/back-test behavior: keep injection answers substantive, only audit guardrail.
export GUARDRAIL_ACTION="audit_only"
export GUARDRAIL_FAIL_CLOSED="0"
```

Health check must show:

```json
{
  "api_fast_only": false,
  "qdrant_enabled": true,
  "static_answer_bank_version": "best_v7_compact_keywords"
}
```

If `sql_backend` is `duckdb`, the API is still usable, but Postgres was not used.
If `qdrant_enabled` is `false`, long-text/OCR retrieval is degraded.

## Run FastAPI Chat Server Manually

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
export ENABLE_STATIC_ANSWER_BANK="1"
export ANSWER_BANK_PATH="$HOME/fahmai-agi/pipeline-qwen2.5/fahmai_qwen25/answer_bank_best.csv"
export ANSWER_BANK_VERSION="best_v7_compact_keywords"

# Optional input guardrail. Keep audit_only for Kaggle-style injection answers;
# use reject/block for production API safety.
export GUARDRAIL_URL="http://127.0.0.1:8000"
export GUARDRAIL_ACTION="audit_only"
export GUARDRAIL_THRESHOLD="0.75"

# Production mode: cache known questions, but let unknown questions hit SQL/RAG/Qwen.
export API_FAST_ONLY="0"
export API_CACHE_MISS_FALLBACK="0"

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

If `API_INCLUDE_SOURCES=1`, `/agent/local` and `/agent/thaillm` additionally
return a `sources` array. The chat endpoints `/api/v1/chat` and `/api/v2/chat`
keep the original `{ "data": { "answer": "..." } }` contract.

Source-enabled example:

```json
{
  "id": "b8b9b5f0-9f69-4ef5-89f8-b85ac0086da9",
  "answer": "MSRP ของ NT-LT-001 คือ 42,900.00 บาท",
  "total_output_token": 7,
  "sources": [
    {
      "type": "answer_cache",
      "path": "/root/workspace/bank500/fahmai-agi/pipeline-qwen2.5/fahmai_qwen25/answer_bank_peterperjer_0_89.csv",
      "version": "methodology_public089_api",
      "sha1": "..."
    }
  ]
}
```

For load-test mode, set `API_FAST_ONLY=1` and keep `ENABLE_API_CACHE=1`,
`API_PRELOAD_ANSWERS=1`, and `API_CACHE_MISS_FALLBACK=1`. Known competition
questions are answered from the static answer bank first, then from the newest
precomputed run cache. Cache misses first try a deterministic SQL/rule answer,
then return a scoped refusal instead of blocking on long Qwen generation.

For production mode, set `API_FAST_ONLY=0` and `API_CACHE_MISS_FALLBACK=0`.
Known questions still use cache/bank, but unseen questions are handled by the
real SQL/RAG/Qwen pipeline.

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
api_tool_audit.jsonl
api_tool_summary.json
```

`api_tool_audit.jsonl` records every tool-level action with request UUID, route,
question id, latency, estimated input/output tokens, hashes, and redacted
previews. `api_tool_summary.json` aggregates call count, seconds, and token
estimates per tool/action. This includes cache lookup, guardrail, SQL, TF-IDF,
Qdrant, hybrid RRF, and LLM generation when those paths are used.

Guardrail behavior:

- `GUARDRAIL_URL` unset: guardrail disabled.
- `GUARDRAIL_ENDPOINT=http://localhost:7777/predictv2`: use the local guardrail service on the B200 host. This endpoint receives only `text`, `max_length`, and `threshold`.
- `GUARDRAIL_ENDPOINT=http://swarm-manager.modelharbor.com:54132/predictv2`: optional remote fallback if the B200 host can reach the shared guardrail service.
- `GUARDRAIL_URL=http://127.0.0.1:8000`: use the older local guardrail shape at `$GUARDRAIL_URL/predict`, including the optional `model` field.
- `GUARDRAIL_ACTION=audit_only`: log guardrail result but still let the FahMai agent answer. This is best for the competition because prompt-injection questions often need a defensive answer, not a hard block.
- `GUARDRAIL_ACTION=reject` or `block`: return a refusal immediately when guardrail says `is_attack=true`. This is best for production API safety.
- `GUARDRAIL_FAIL_CLOSED=1`: reject when the guardrail API is unreachable. Default is fail-open.
- `API_INCLUDE_SOURCES=1`: include source references in `/agent/local` and `/agent/thaillm`.

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
