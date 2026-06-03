#!/usr/bin/env bash
set -euo pipefail

# Real model CSV profile for inspection/debugging.
# This disables static answer banks and forces the pipeline through
# SQL/schema tools, TF-IDF/Qdrant retrieval, and Qwen generation.
#
# Output directory:
#   $WORK_ROOT/output/<RUN_ID>_real_model/
# Files:
#   run.log
#   best_submission.csv
#   best_results.csv
#   best_debug.json
#   best_token_usage.csv
#   best_token_summary.json
#   best_llm_audit.jsonl
#   best_tool_audit.jsonl
#   best_tool_summary.json

cd "$(dirname "$0")"

if [ "${SOURCE_FAHMAI_DB_ENV:-1}" = "1" ] && [ -f "$HOME/.fahmai_db_env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.fahmai_db_env"
fi

export WORK_ROOT="${WORK_ROOT:-$HOME/bank500}"
export FAHMAI_SRC_ROOT="${FAHMAI_SRC_ROOT:-$HOME/scamper_house}"
export QUESTIONS_CSV_PATH="${QUESTIONS_CSV_PATH:-$FAHMAI_SRC_ROOT/questions.csv}"
export MODEL_PATH="${MODEL_PATH:-$WORK_ROOT/qwen35/models/Qwen2.5-7B-Instruct}"

# DuckDB is the safest default on B200 because local Postgres may not always be
# mounted. Override with SQL_BACKEND=postgres when localhost:5432/fahmai is up.
export SQL_BACKEND="${SQL_BACKEND:-duckdb}"
export ALLOW_SQL_FALLBACK="${ALLOW_SQL_FALLBACK:-1}"
export PG_DSN="${PG_DSN:-postgresql://admin:scamper@localhost:5432/fahmai}"
export PG_SCHEMA="${PG_SCHEMA:-public}"

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-fahmai_rag_bge}"
export EMBED_MODEL="${EMBED_MODEL:-$WORK_ROOT/qwen35/models/bge-m3}"
export NO_QDRANT="${NO_QDRANT:-0}"
export SKIP_QDRANT_PRELOAD="${SKIP_QDRANT_PRELOAD:-1}"

export ENABLE_STATIC_ANSWER_BANK="0"
export ANSWER_BANK_FAST_ONLY="0"
export GROUNDTRUTH_STYLE_GUIDANCE="${GROUNDTRUTH_STYLE_GUIDANCE:-1}"
export MODEL_REWRITE_RULE_ANSWERS="${MODEL_REWRITE_RULE_ANSWERS:-1}"
export MODEL_REWRITE_ENTITY_GUARD="${MODEL_REWRITE_ENTITY_GUARD:-1}"
export FINAL_ANSWER_SECURITY_GUARD="${FINAL_ANSWER_SECURITY_GUARD:-1}"

export MODEL_LOAD_STRATEGY="${MODEL_LOAD_STRATEGY:-cuda_direct}"
export DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP="${DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

export GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-0}"
export GEN_MAX_INPUT_TOKENS="${GEN_MAX_INPUT_TOKENS:-7000}"
export GEN_REPETITION_PENALTY="${GEN_REPETITION_PENALTY:-1.05}"
export FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-220}"
export DOC_TOP_K="${DOC_TOP_K:-8}"
export QDRANT_TOP_K="${QDRANT_TOP_K:-8}"
export SCHEMA_TOP_K="${SCHEMA_TOP_K:-10}"
export SANITIZE_MAX_CHARS="${SANITIZE_MAX_CHARS:-2000}"

export LLM_AUDIT_INCLUDE_PROMPT="${LLM_AUDIT_INCLUDE_PROMPT:-0}"

export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_real_model}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/output}"
export RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_ID}"
mkdir -p "$RUN_OUTPUT_DIR"

LIMIT="${LIMIT:-100}"
LOG_FILE="$RUN_OUTPUT_DIR/run.log"

echo "Starting FahMai real-model CSV run"
echo "  limit: $LIMIT"
echo "  run_output_dir: $RUN_OUTPUT_DIR"
echo "  log_file: $LOG_FILE"
echo "  sql_backend: $SQL_BACKEND"
echo "  pg_dsn_set: $([ -n "${PG_DSN:-}" ] && echo yes || echo no)"
echo "  qdrant_url: ${QDRANT_URL:-disabled}"
echo "  qdrant_collection: ${QDRANT_COLLECTION:-}"
echo "  embed_model: $EMBED_MODEL"
echo "  model_path: $MODEL_PATH"
echo "  static_answer_bank: disabled"
echo "  groundtruth_style_guidance: $GROUNDTRUTH_STYLE_GUIDANCE"
echo "  model_rewrite_rule_answers: $MODEL_REWRITE_RULE_ANSWERS"
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

python agentic_best_integrated_qdrant.py "${args[@]}" 2>&1 | tee "$LOG_FILE"
