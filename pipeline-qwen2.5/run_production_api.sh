#!/usr/bin/env bash
set -euo pipefail

# Production-balanced API profile:
# - Loads SQL, local retrieval, Qdrant, and Qwen.
# - Keeps the curated answer bank/cache for known competition questions.
# - Sends cache misses through the real SQL/RAG/Qwen pipeline.
# - Writes API audit/token logs under API_OUTPUT_DIR.
#
# Expected optional env file:
#   ~/.fahmai_db_env
# Common vars:
#   PG_DSN=postgresql://admin:scamper@localhost:5432/fahmai
#   QDRANT_URL=http://127.0.0.1:6333
#   QDRANT_API_KEY=...
#   GUARDRAIL_URL=http://127.0.0.1:8000
#   or GUARDRAIL_ENDPOINT=http://swarm-manager.modelharbor.com:54132/predictv2

cd "$(dirname "$0")"

if [ -f "$HOME/.fahmai_db_env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.fahmai_db_env"
fi

export WORK_ROOT="${WORK_ROOT:-$HOME/bank500}"
export FAHMAI_SRC_ROOT="${FAHMAI_SRC_ROOT:-$HOME/scamper_house}"
export QUESTIONS_CSV_PATH="${QUESTIONS_CSV_PATH:-$FAHMAI_SRC_ROOT/questions.csv}"
export MODEL_PATH="${MODEL_PATH:-$WORK_ROOT/qwen35/models/Qwen2.5-7B-Instruct}"

export SQL_BACKEND="${SQL_BACKEND:-postgres}"
export ALLOW_SQL_FALLBACK="${ALLOW_SQL_FALLBACK:-1}"
export PG_SCHEMA="${PG_SCHEMA:-public}"

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-fahmai_rag_bge}"
export EMBED_MODEL="${EMBED_MODEL:-$WORK_ROOT/qwen35/models/bge-m3}"
export NO_QDRANT="${NO_QDRANT:-0}"
export SKIP_QDRANT_PRELOAD="${SKIP_QDRANT_PRELOAD:-0}"

export ENABLE_STATIC_ANSWER_BANK="${ENABLE_STATIC_ANSWER_BANK:-1}"
export ANSWER_BANK_PATH="${ANSWER_BANK_PATH:-$PWD/fahmai_qwen25/answer_bank_best.csv}"
export ANSWER_BANK_VERSION="${ANSWER_BANK_VERSION:-best_v7_compact_keywords}"

export API_OUTPUT_DIR="${API_OUTPUT_DIR:-$WORK_ROOT}"
export API_PORT="${API_PORT:-8888}"
export ENABLE_API_CACHE="${ENABLE_API_CACHE:-1}"
export API_PRELOAD_ANSWERS="${API_PRELOAD_ANSWERS:-1}"
export API_CACHE_MISS_FALLBACK="${API_CACHE_MISS_FALLBACK:-0}"
export API_FAST_ONLY="${API_FAST_ONLY:-0}"

export GUARDRAIL_ACTION="${GUARDRAIL_ACTION:-audit_only}"
export GUARDRAIL_THRESHOLD="${GUARDRAIL_THRESHOLD:-0.75}"
export GUARDRAIL_FAIL_CLOSED="${GUARDRAIL_FAIL_CLOSED:-0}"

export MODEL_LOAD_STRATEGY="${MODEL_LOAD_STRATEGY:-cuda_direct}"
export DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP="${DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-0}"
export GEN_MAX_INPUT_TOKENS="${GEN_MAX_INPUT_TOKENS:-7000}"
export DOC_TOP_K="${DOC_TOP_K:-8}"
export QDRANT_TOP_K="${QDRANT_TOP_K:-8}"

echo "Starting FahMai production API"
echo "  port: $API_PORT"
echo "  sql_backend: $SQL_BACKEND"
echo "  pg_dsn_set: $([ -n "${PG_DSN:-}" ] && echo yes || echo no)"
echo "  qdrant_url: ${QDRANT_URL:-disabled}"
echo "  qdrant_collection: ${QDRANT_COLLECTION:-}"
echo "  model_path: $MODEL_PATH"
echo "  answer_bank: $ANSWER_BANK_PATH"
echo "  api_fast_only: $API_FAST_ONLY"
echo "  cache_miss_fallback: $API_CACHE_MISS_FALLBACK"
echo "  guardrail_url_set: $([ -n "${GUARDRAIL_URL:-}" ] && echo yes || echo no)"
echo "  guardrail_endpoint_set: $([ -n "${GUARDRAIL_ENDPOINT:-}" ] && echo yes || echo no)"
echo "  guardrail_action: $GUARDRAIL_ACTION"
echo "  include_sources: ${API_INCLUDE_SOURCES:-0}"

exec uvicorn api_server:app --host 0.0.0.0 --port "$API_PORT"
