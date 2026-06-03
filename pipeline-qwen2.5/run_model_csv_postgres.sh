#!/usr/bin/env bash
set -euo pipefail

# Strict PostgreSQL batch model-generation profile for B200.
# Fails loudly if Postgres is unavailable instead of silently falling back to
# DuckDB. Use this to measure the production SQL path.

cd "$(dirname "$0")"

export SQL_BACKEND="postgres"
export ALLOW_SQL_FALLBACK="0"
export SOURCE_FAHMAI_DB_ENV="0"
export PG_DSN="${PG_DSN:-postgresql://admin:scamper@localhost:5432/fahmai}"
export PG_SCHEMA="${PG_SCHEMA:-public}"
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export EMBED_MODEL="${EMBED_MODEL:-$HOME/bank500/qwen35/models/bge-m3}"

exec ./run_model_csv.sh
