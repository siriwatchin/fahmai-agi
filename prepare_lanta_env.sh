#!/bin/bash
set -euo pipefail

module purge || true
module load Mamba/23.11.0-0 || true

PROJECT_ROOT="${PROJECT_ROOT:-/project/zz992000-zdevb}"
cd "$PROJECT_ROOT/fahmai-agi/pipeline-qwen2.5"

mkdir -p "$PROJECT_ROOT/venvs"

if [ ! -d "$PROJECT_ROOT/venvs/qwen25" ]; then
  BASE_PYTHON="/lustrefs/disk/modules/easybuild/software/Mamba/23.11.0-0/envs/pytorch-2.2.2/bin/python"
  if [ -x "$BASE_PYTHON" ]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$PROJECT_ROOT/venvs/qwen25"
  else
    python3 -m venv "$PROJECT_ROOT/venvs/qwen25"
  fi
fi

source "$PROJECT_ROOT/venvs/qwen25/bin/activate"
python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt

python - <<'PY'
import importlib
mods = ["torch", "transformers", "accelerate", "duckdb", "pandas", "qdrant_client", "sentence_transformers"]
for mod in mods:
    m = importlib.import_module(mod)
    print(mod, getattr(m, "__version__", "ok"))
PY
