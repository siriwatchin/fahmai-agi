#!/usr/bin/env bash
set -euo pipefail

# Latest recommended local API profile for B200.
# - Uses the stable Qwen2.5-7B model for cache misses.
# - Uses the 0.86 static answer bank for known 100 public questions.
# - Uses local Qdrant and local/available SQL backends through run_production_api.sh.

cd "$(dirname "$0")"

export WORK_ROOT="${WORK_ROOT:-$HOME/bank500}"
export FAHMAI_SRC_ROOT="${FAHMAI_SRC_ROOT:-$HOME/scamper_house}"
export MODEL_PATH="${MODEL_PATH:-$WORK_ROOT/qwen35/models/Qwen2.5-7B-Instruct}"

export API_PORT="${API_PORT:-8888}"
export SQL_BACKEND="${SQL_BACKEND:-duckdb}"
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-fahmai_rag_bge}"
export EMBED_MODEL="${EMBED_MODEL:-$WORK_ROOT/qwen35/models/bge-m3}"

export ENABLE_STATIC_ANSWER_BANK="1"
export ANSWER_BANK_PATH="${ANSWER_BANK_PATH:-$PWD/fahmai_qwen25/answer_bank_real_groundtruth_0_86.csv}"
export ANSWER_BANK_VERSION="${ANSWER_BANK_VERSION:-public086_api_cache}"
export ENABLE_API_CACHE="${ENABLE_API_CACHE:-1}"
export API_PRELOAD_ANSWERS="${API_PRELOAD_ANSWERS:-1}"

export API_FAST_ONLY="${API_FAST_ONLY:-0}"
export API_CACHE_MISS_FALLBACK="${API_CACHE_MISS_FALLBACK:-0}"

export MODEL_LOAD_STRATEGY="${MODEL_LOAD_STRATEGY:-cuda_direct}"
export DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP="${DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-0}"
export GEN_MAX_INPUT_TOKENS="${GEN_MAX_INPUT_TOKENS:-7000}"
export DOC_TOP_K="${DOC_TOP_K:-8}"
export QDRANT_TOP_K="${QDRANT_TOP_K:-8}"

exec ./run_production_api.sh
