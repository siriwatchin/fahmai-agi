#!/usr/bin/env bash
set -euo pipefail

# API profile for the agentic back-test contract with:
# - guardrail disabled by default; set GUARDRAIL_ENDPOINT or GUARDRAIL_URL
#   explicitly if you want to enable an external classifier
# - optional source attribution in /agent/local and /agent/thaillm responses
# - competition-valid prompt-injection handling is done by the agent/refusal
#   logic unless an external guardrail is explicitly configured.

cd "$(dirname "$0")"

export GUARDRAIL_ENDPOINT="${GUARDRAIL_ENDPOINT:-}"
export GUARDRAIL_URL="${GUARDRAIL_URL:-}"
export GUARDRAIL_THRESHOLD="${GUARDRAIL_THRESHOLD:-0.75}"
export GUARDRAIL_MAX_LENGTH="${GUARDRAIL_MAX_LENGTH:-2048}"
export GUARDRAIL_TIMEOUT_SEC="${GUARDRAIL_TIMEOUT_SEC:-2.0}"
export GUARDRAIL_ACTION="${GUARDRAIL_ACTION:-audit_only}"
export GUARDRAIL_FAIL_CLOSED="${GUARDRAIL_FAIL_CLOSED:-0}"

export API_INCLUDE_SOURCES="${API_INCLUDE_SOURCES:-1}"

exec ./run_methodology_api.sh
