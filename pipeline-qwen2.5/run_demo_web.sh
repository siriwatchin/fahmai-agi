#!/usr/bin/env bash
set -euo pipefail

# Standalone demo web service.
# It does not load Qwen, PostgreSQL, or Qdrant directly. It serves the browser UI
# and proxies browser requests to the already-running model API.

cd "$(dirname "$0")"

export MODEL_API_URL="${MODEL_API_URL:-http://127.0.0.1:8888}"
export DEMO_WEB_PORT="${DEMO_WEB_PORT:-8890}"
export DEMO_PROXY_TIMEOUT_SEC="${DEMO_PROXY_TIMEOUT_SEC:-120}"

echo "Starting FahMai demo web service"
echo "  web_port: $DEMO_WEB_PORT"
echo "  model_api_url: $MODEL_API_URL"
echo
echo "Open one of these:"
echo "  local:  http://127.0.0.1:$DEMO_WEB_PORT/demo"
echo "  b200:   https://b200.thescamperss6.com/user/bank500/proxy/$DEMO_WEB_PORT/demo"
echo

exec uvicorn demo_web_server:app --host 0.0.0.0 --port "$DEMO_WEB_PORT"

