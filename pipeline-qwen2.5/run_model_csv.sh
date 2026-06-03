#!/usr/bin/env bash
set -euo pipefail

# Batch model-generation profile for B200.
# This intentionally disables the static answer bank so the CSV is generated
# through the real SQL/RAG/Qdrant/Qwen pipeline.

cd "$(dirname "$0")"

export SOURCE_FAHMAI_DB_ENV="${SOURCE_FAHMAI_DB_ENV:-1}"

if [ "$SOURCE_FAHMAI_DB_ENV" = "1" ] && [ -f "$HOME/.fahmai_db_env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.fahmai_db_env"
fi

export WORK_ROOT="${WORK_ROOT:-$HOME/bank500}"
export FAHMAI_SRC_ROOT="${FAHMAI_SRC_ROOT:-$HOME/scamper_house}"
export QUESTIONS_CSV_PATH="${QUESTIONS_CSV_PATH:-$FAHMAI_SRC_ROOT/questions.csv}"
export MODEL_PATH="${MODEL_PATH:-$WORK_ROOT/qwen35/models/Qwen2.5-7B-Instruct}"

export SQL_BACKEND="${SQL_BACKEND:-postgres}"
export ALLOW_SQL_FALLBACK="${ALLOW_SQL_FALLBACK:-1}"
export PG_DSN="${PG_DSN:-postgresql://admin:scamper@localhost:5432/fahmai}"
export PG_SCHEMA="${PG_SCHEMA:-public}"

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-fahmai_rag_bge}"
export EMBED_MODEL="${EMBED_MODEL:-$WORK_ROOT/qwen35/models/bge-m3}"
export NO_QDRANT="${NO_QDRANT:-0}"
export SKIP_QDRANT_PRELOAD="${SKIP_QDRANT_PRELOAD:-0}"

export ENABLE_STATIC_ANSWER_BANK=0
export ANSWER_BANK_FAST_ONLY=0
export ANSWER_BANK_VERSION="${ANSWER_BANK_VERSION:-model_qwen25_7b_rag}"

export MODEL_LOAD_STRATEGY="${MODEL_LOAD_STRATEGY:-cuda_direct}"
export DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP="${DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

export GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-0}"
export GEN_MAX_INPUT_TOKENS="${GEN_MAX_INPUT_TOKENS:-7000}"
export GEN_REPETITION_PENALTY="${GEN_REPETITION_PENALTY:-1.05}"
export DOC_TOP_K="${DOC_TOP_K:-8}"
export QDRANT_TOP_K="${QDRANT_TOP_K:-8}"
export SCHEMA_TOP_K="${SCHEMA_TOP_K:-10}"

export LLM_AUDIT_INCLUDE_PROMPT="${LLM_AUDIT_INCLUDE_PROMPT:-0}"

LIMIT="${LIMIT:-100}"

echo "Starting FahMai model CSV run"
echo "  limit: $LIMIT"
echo "  output_root: $WORK_ROOT/output/<RUN_ID>"
echo "  sql_backend: $SQL_BACKEND"
echo "  pg_dsn_set: $([ -n "${PG_DSN:-}" ] && echo yes || echo no)"
echo "  qdrant_url: ${QDRANT_URL:-disabled}"
echo "  qdrant_api_key_set: $([ -n "${QDRANT_API_KEY:-}" ] && echo yes || echo no)"
echo "  qdrant_collection: ${QDRANT_COLLECTION:-}"
echo "  embed_model: $EMBED_MODEL"
echo "  model_path: $MODEL_PATH"
echo "  static_answer_bank: disabled"
echo "  gen_do_sample: $GEN_DO_SAMPLE"
echo "  doc_top_k: $DOC_TOP_K"
echo "  qdrant_top_k: $QDRANT_TOP_K"
echo "  gen_max_input_tokens: $GEN_MAX_INPUT_TOKENS"

args=(--limit "$LIMIT")
if [ "$NO_QDRANT" = "1" ]; then
  args+=(--no-qdrant)
fi
if [ "$SKIP_QDRANT_PRELOAD" = "1" ]; then
  args+=(--skip-qdrant-preload)
fi

python agentic_best_integrated_qdrant.py "${args[@]}"
