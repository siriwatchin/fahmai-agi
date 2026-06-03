#!/usr/bin/env bash
set -euo pipefail

# API profile for the agentic back-test contract with:
# - input guardrail through the shared /predictv2 endpoint
# - optional source attribution in /agent/local and /agent/thaillm responses
# - audit-only guardrail by default, so injection questions can still receive
#   competition-valid defensive answers instead of being hard-blocked.

cd "$(dirname "$0")"

export GUARDRAIL_ENDPOINT="${GUARDRAIL_ENDPOINT:-http://swarm-manager.modelharbor.com:54132/predictv2}"
export GUARDRAIL_THRESHOLD="${GUARDRAIL_THRESHOLD:-0.75}"
export GUARDRAIL_MAX_LENGTH="${GUARDRAIL_MAX_LENGTH:-2048}"
export GUARDRAIL_TIMEOUT_SEC="${GUARDRAIL_TIMEOUT_SEC:-2.0}"
export GUARDRAIL_ACTION="${GUARDRAIL_ACTION:-audit_only}"
export GUARDRAIL_FAIL_CLOSED="${GUARDRAIL_FAIL_CLOSED:-0}"

export API_INCLUDE_SOURCES="${API_INCLUDE_SOURCES:-1}"

exec ./run_methodology_api.sh
