#!/usr/bin/env bash
set -euo pipefail

# Ground-truth-style model-generation profile for B200.
# This does NOT use the static answer bank. It uses SQL/RAG/Qdrant evidence,
# then asks Qwen to rewrite deterministic SQL/rule drafts in the same response
# style family as the reviewed ground-truth CSV.

cd "$(dirname "$0")"

if [ -f "$HOME/.fahmai_db_env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.fahmai_db_env"
fi

export SQL_BACKEND="postgres"
export ALLOW_SQL_FALLBACK="${ALLOW_SQL_FALLBACK:-0}"
export SOURCE_FAHMAI_DB_ENV="0"
# Force local Postgres. ~/.fahmai_db_env may contain stale remote swarm-manager
# DSNs, which time out from the B200 runtime.
export PG_DSN="postgresql://admin:scamper@localhost:5432/fahmai"
export PG_SCHEMA="${PG_SCHEMA:-public}"

export WORK_ROOT="${WORK_ROOT:-$HOME/bank500}"
export FAHMAI_SRC_ROOT="${FAHMAI_SRC_ROOT:-$HOME/scamper_house}"
export QUESTIONS_CSV_PATH="${QUESTIONS_CSV_PATH:-$FAHMAI_SRC_ROOT/questions.csv}"
export MODEL_PATH="${MODEL_PATH:-$WORK_ROOT/qwen35/models/Qwen2.5-7B-Instruct}"

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-fahmai_rag_bge}"
export EMBED_MODEL="${EMBED_MODEL:-$WORK_ROOT/qwen35/models/bge-m3}"
export NO_QDRANT="${NO_QDRANT:-0}"
export SKIP_QDRANT_PRELOAD="${SKIP_QDRANT_PRELOAD:-1}"

export ENABLE_STATIC_ANSWER_BANK="0"
export ANSWER_BANK_FAST_ONLY="0"
export GROUNDTRUTH_STYLE_GUIDANCE="1"
export MODEL_REWRITE_RULE_ANSWERS="1"
export MODEL_REWRITE_ENTITY_GUARD="${MODEL_REWRITE_ENTITY_GUARD:-1}"
export FINAL_ANSWER_SECURITY_GUARD="${FINAL_ANSWER_SECURITY_GUARD:-1}"
export ANSWER_BANK_VERSION="model_qwen25_gt_style_no_bank"

export MODEL_LOAD_STRATEGY="${MODEL_LOAD_STRATEGY:-cuda_direct}"
export DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP="${DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

export GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-0}"
export GEN_MAX_INPUT_TOKENS="${GEN_MAX_INPUT_TOKENS:-8500}"
export FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-260}"
export GEN_REPETITION_PENALTY="${GEN_REPETITION_PENALTY:-1.03}"
export SANITIZE_MAX_CHARS="${SANITIZE_MAX_CHARS:-2000}"
export DOC_TOP_K="${DOC_TOP_K:-10}"
export QDRANT_TOP_K="${QDRANT_TOP_K:-10}"
export SCHEMA_TOP_K="${SCHEMA_TOP_K:-12}"
export LLM_AUDIT_INCLUDE_PROMPT="${LLM_AUDIT_INCLUDE_PROMPT:-0}"

LIMIT="${LIMIT:-100}"

echo "Starting FahMai groundtruth-style model CSV run"
echo "  limit: $LIMIT"
echo "  sql_backend: $SQL_BACKEND"
echo "  pg_dsn_set: $([ -n "${PG_DSN:-}" ] && echo yes || echo no)"
echo "  pg_host: localhost:5432/fahmai"
echo "  qdrant_url: ${QDRANT_URL:-disabled}"
echo "  qdrant_collection: ${QDRANT_COLLECTION:-}"
echo "  model_path: $MODEL_PATH"
echo "  static_answer_bank: disabled"
echo "  groundtruth_style_guidance: $GROUNDTRUTH_STYLE_GUIDANCE"
echo "  model_rewrite_rule_answers: $MODEL_REWRITE_RULE_ANSWERS"
echo "  model_rewrite_entity_guard: $MODEL_REWRITE_ENTITY_GUARD"
echo "  final_answer_security_guard: $FINAL_ANSWER_SECURITY_GUARD"
echo "  final_max_new_tokens: $FINAL_MAX_NEW_TOKENS"
echo "  sanitize_max_chars: $SANITIZE_MAX_CHARS"

args=(--limit "$LIMIT")
if [ "$NO_QDRANT" = "1" ]; then
  args+=(--no-qdrant)
fi
if [ "$SKIP_QDRANT_PRELOAD" = "1" ]; then
  args+=(--skip-qdrant-preload)
fi

python agentic_best_integrated_qdrant.py "${args[@]}"
