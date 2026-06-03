#!/usr/bin/env bash
set -euo pipefail

# API profile for the agentic back-test contract with:
# - external guardrail disabled; scripts unset GUARDRAIL_ENDPOINT/GUARDRAIL_URL
# - optional source attribution in /agent/local and /agent/thaillm responses
# - competition-valid prompt-injection handling is done by the agent/refusal
#   logic.

cd "$(dirname "$0")"

unset GUARDRAIL_ENDPOINT
unset GUARDRAIL_URL

export API_INCLUDE_SOURCES="${API_INCLUDE_SOURCES:-1}"

exec ./run_methodology_api.sh
