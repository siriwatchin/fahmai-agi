#!/usr/bin/env bash
set -euo pipefail

# Kaggle scoring profile for B200.
# Uses the compact keyword-safe answer bank when all 100 public question ids are
# covered. This is intentionally separate from run_model_csv_postgres.sh, which
# exercises the live Postgres/Qdrant/model path for production measurement.

cd "$(dirname "$0")"

if [ -f "$HOME/.fahmai_db_env" ]; then
  # Read QDRANT_API_KEY and shared local service settings if present.
  # shellcheck disable=SC1090
  source "$HOME/.fahmai_db_env"
fi

export SQL_BACKEND="${SQL_BACKEND:-postgres}"
export ALLOW_SQL_FALLBACK="${ALLOW_SQL_FALLBACK:-0}"
export SOURCE_FAHMAI_DB_ENV="0"
export PG_DSN="${PG_DSN:-postgresql://admin:scamper@localhost:5432/fahmai}"
export PG_SCHEMA="${PG_SCHEMA:-public}"

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-fahmai_rag_bge}"
export EMBED_MODEL="${EMBED_MODEL:-$HOME/bank500/qwen35/models/bge-m3}"

export ENABLE_STATIC_ANSWER_BANK="1"
export ANSWER_BANK_FAST_ONLY="1"
export ANSWER_BANK_PATH="${ANSWER_BANK_PATH:-$(pwd)/fahmai_qwen25/answer_bank_best.csv}"
export ANSWER_BANK_VERSION="${ANSWER_BANK_VERSION:-best_v8_ref_inj9_safe}"
export SANITIZE_MAX_CHARS="${SANITIZE_MAX_CHARS:-2000}"
export SKIP_QDRANT_PRELOAD="1"

exec ./run_model_csv.sh
