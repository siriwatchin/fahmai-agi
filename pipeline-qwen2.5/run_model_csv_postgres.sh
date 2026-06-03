#!/usr/bin/env bash
set -euo pipefail

# Strict PostgreSQL batch model-generation profile for B200.
# Fails loudly if Postgres is unavailable instead of silently falling back to
# DuckDB. Use this to measure the production SQL path.

cd "$(dirname "$0")"

export SQL_BACKEND="postgres"
export ALLOW_SQL_FALLBACK="0"
export PG_DSN="${PG_DSN:-postgresql://admin:scamper@localhost:5432/fahmai}"
export PG_SCHEMA="${PG_SCHEMA:-public}"

exec ./run_model_csv.sh
