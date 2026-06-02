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
export EMBED_MODEL="intfloat/multilingual-e5-base"
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
- The LLM is used as a final synthesizer, not as the primary calculator.

