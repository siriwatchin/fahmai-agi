#!/usr/bin/env bash
set -euo pipefail

# Public-score static profile.
# This intentionally uses the checked-in 0.86 public candidate answer bank.
# Do not use this profile as the production/security API behavior; it is a
# score-submission profile for the known 100-question public back-test only.

cd "$(dirname "$0")"

export ANSWER_BANK_PATH="${ANSWER_BANK_PATH:-$(pwd)/fahmai_qwen25/answer_bank_real_groundtruth_0_86.csv}"
export ANSWER_BANK_VERSION="${ANSWER_BANK_VERSION:-public086_static_profile}"
export ENABLE_STATIC_ANSWER_BANK="1"
export ANSWER_BANK_FAST_ONLY="1"
export SANITIZE_MAX_CHARS="${SANITIZE_MAX_CHARS:-2000}"

exec ./run_score_csv_postgres.sh
