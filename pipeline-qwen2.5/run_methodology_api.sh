#!/usr/bin/env bash
set -euo pipefail

# Methodology production API profile.
# Serves cached high-confidence known answers, then falls back to
# SQL + TF-IDF + Qdrant/bge-m3 + Qwen with hybrid RRF evidence.

cd "$(dirname "$0")"

export WORK_ROOT="${WORK_ROOT:-$HOME/bank500}"
export FAHMAI_SRC_ROOT="${FAHMAI_SRC_ROOT:-$HOME/scamper_house}"
export MODEL_PATH="${MODEL_PATH:-$WORK_ROOT/qwen35/models/Qwen2.5-7B-Instruct}"

export API_PORT="${API_PORT:-8888}"
export SQL_BACKEND="${SQL_BACKEND:-postgres}"
export ALLOW_SQL_FALLBACK="${ALLOW_SQL_FALLBACK:-1}"
export PG_DSN="${PG_DSN:-postgresql://admin:scamper@localhost:5432/fahmai}"
export PG_SCHEMA="${PG_SCHEMA:-public}"
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-fahmai_rag_bge}"
export EMBED_MODEL="${EMBED_MODEL:-$WORK_ROOT/qwen35/models/bge-m3}"

export ENABLE_STATIC_ANSWER_BANK="${ENABLE_STATIC_ANSWER_BANK:-1}"
export ANSWER_BANK_PATH="${ANSWER_BANK_PATH:-$PWD/fahmai_qwen25/answer_bank_peterperjer_0_89.csv}"
export ANSWER_BANK_VERSION="${ANSWER_BANK_VERSION:-methodology_public089_api}"
export ENABLE_API_CACHE="${ENABLE_API_CACHE:-1}"
export API_PRELOAD_ANSWERS="${API_PRELOAD_ANSWERS:-1}"

export API_FAST_ONLY="${API_FAST_ONLY:-0}"
export API_CACHE_MISS_FALLBACK="${API_CACHE_MISS_FALLBACK:-0}"

export ENABLE_HYBRID_RRF="${ENABLE_HYBRID_RRF:-1}"
export HYBRID_TOP_K="${HYBRID_TOP_K:-8}"
export RRF_K="${RRF_K:-60}"

export MODEL_LOAD_STRATEGY="${MODEL_LOAD_STRATEGY:-cuda_direct}"
export DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP="${DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-0}"
export GEN_MAX_INPUT_TOKENS="${GEN_MAX_INPUT_TOKENS:-7000}"
export FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-220}"
export DOC_TOP_K="${DOC_TOP_K:-8}"
export QDRANT_TOP_K="${QDRANT_TOP_K:-8}"

exec ./run_production_api.sh
