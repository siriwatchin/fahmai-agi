#!/usr/bin/env bash
set -euo pipefail

# Strict PostgreSQL batch model-generation profile for B200.
# Fails loudly if Postgres is unavailable instead of silently falling back to
# DuckDB. Use this to measure the production SQL path.

cd "$(dirname "$0")"

if [ -f "$HOME/.fahmai_db_env" ]; then
  # Read shared keys such as QDRANT_API_KEY, then override SQL below so stale
  # remote PG_DSN values cannot leak into the strict local Postgres run.
  # shellcheck disable=SC1090
  source "$HOME/.fahmai_db_env"
fi

export SQL_BACKEND="postgres"
export ALLOW_SQL_FALLBACK="0"
export SOURCE_FAHMAI_DB_ENV="0"
export PG_DSN="postgresql://admin:scamper@localhost:5432/fahmai"
export PG_SCHEMA="${PG_SCHEMA:-public}"
export QDRANT_URL="http://127.0.0.1:6333"
export EMBED_MODEL="${EMBED_MODEL:-$HOME/bank500/qwen35/models/bge-m3}"
export NO_QDRANT="0"
export SKIP_QDRANT_PRELOAD="${SKIP_QDRANT_PRELOAD:-0}"

exec ./run_model_csv.sh
