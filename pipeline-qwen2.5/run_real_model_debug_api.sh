#!/usr/bin/env bash
set -euo pipefail

# Real-model debug API profile for teammate testing.
# This disables static answer-bank/cache answers and exposes verbose debug
# endpoints with sources, per-request token logs, LLM audit, tool audit, and
# redacted observations.
#
# Main debug endpoints:
#   POST /api/v2/chat
#   POST /agent/local/debug
#   POST /agent/thaillm/debug
#
# Back-test-compatible endpoints still exist:
#   POST /agent/local
#   POST /agent/thaillm

cd "$(dirname "$0")"

if [ "${SOURCE_FAHMAI_DB_ENV:-1}" = "1" ] && [ -f "$HOME/.fahmai_db_env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.fahmai_db_env"
fi

export WORK_ROOT="${WORK_ROOT:-$HOME/bank500}"
export FAHMAI_SRC_ROOT="${FAHMAI_SRC_ROOT:-$HOME/scamper_house}"
export MODEL_PATH="${MODEL_PATH:-$WORK_ROOT/qwen35/models/Qwen2.5-7B-Instruct}"

export API_PORT="${API_PORT:-8888}"
export API_OUTPUT_DIR="${API_OUTPUT_DIR:-$WORK_ROOT}"

# Use local PostgreSQL by default; keep fallback enabled for B200 sessions where
# the local database is temporarily unavailable.
export SQL_BACKEND="${SQL_BACKEND:-postgres}"
export ALLOW_SQL_FALLBACK="${ALLOW_SQL_FALLBACK:-1}"
export PG_DSN="${PG_DSN:-postgresql://admin:scamper@localhost:5432/fahmai}"
export PG_SCHEMA="${PG_SCHEMA:-public}"

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-fahmai_rag_bge}"
export EMBED_MODEL="${EMBED_MODEL:-$WORK_ROOT/qwen35/models/bge-m3}"
export SKIP_QDRANT_PRELOAD="${SKIP_QDRANT_PRELOAD:-1}"
export NO_QDRANT="${NO_QDRANT:-0}"

export ENABLE_STATIC_ANSWER_BANK="0"
export ENABLE_API_CACHE="0"
export API_PRELOAD_ANSWERS="0"
export API_CACHE_MISS_FALLBACK="0"
export API_FAST_ONLY="0"

export API_INCLUDE_SOURCES="1"
export API_V2_DEBUG_RESPONSE="1"
export API_DEBUG_INCLUDE_OBSERVATION="${API_DEBUG_INCLUDE_OBSERVATION:-1}"
export API_DEBUG_INCLUDE_RAW_OBSERVATION="${API_DEBUG_INCLUDE_RAW_OBSERVATION:-0}"
export API_DEBUG_STRING_LIMIT="${API_DEBUG_STRING_LIMIT:-2000}"
export API_DEBUG_LIST_LIMIT="${API_DEBUG_LIST_LIMIT:-80}"

unset GUARDRAIL_ENDPOINT
unset GUARDRAIL_URL

export MODEL_LOAD_STRATEGY="${MODEL_LOAD_STRATEGY:-cuda_direct}"
export DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP="${DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-0}"
export GEN_MAX_INPUT_TOKENS="${GEN_MAX_INPUT_TOKENS:-7000}"
export FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-220}"
export DOC_TOP_K="${DOC_TOP_K:-8}"
export QDRANT_TOP_K="${QDRANT_TOP_K:-8}"
export SANITIZE_MAX_CHARS="${SANITIZE_MAX_CHARS:-2000}"

echo "Starting FahMai real-model debug API"
echo "  port: $API_PORT"
echo "  sql_backend: $SQL_BACKEND"
echo "  pg_dsn_set: $([ -n "${PG_DSN:-}" ] && echo yes || echo no)"
echo "  qdrant_url: ${QDRANT_URL:-disabled}"
echo "  qdrant_collection: ${QDRANT_COLLECTION:-}"
echo "  model_path: $MODEL_PATH"
echo "  static_answer_bank: disabled"
echo "  api_cache: disabled"
echo "  debug_sources: $API_INCLUDE_SOURCES"
echo "  api_v2_debug_response: $API_V2_DEBUG_RESPONSE"
echo "  debug_observation: $API_DEBUG_INCLUDE_OBSERVATION"

exec uvicorn api_server:app --host 0.0.0.0 --port "$API_PORT"
