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

# On B200, Postgres host currently times out. Use duckdb to start fast.
# Switch to auto/postgres only when PG_DSN is reachable from B200.
export SQL_BACKEND="duckdb"
export ALLOW_SQL_FALLBACK="1"

export QDRANT_URL="http://localhost:6333"
export QDRANT_COLLECTION="fahmai_rag_bge"
export EMBED_MODEL="$HOME/bank500/qwen35/models/bge-m3"
export API_OUTPUT_DIR="$HOME/bank500"

pip install -U fastapi "uvicorn[standard]"

uvicorn api_server:app --host 0.0.0.0 --port 5555
```

Smoke test:

```bash
curl -s http://127.0.0.1:5555/health

curl -s -X POST http://127.0.0.1:5555/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"data":{"question":"วันนี้วันอะไร"}}'
```

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
