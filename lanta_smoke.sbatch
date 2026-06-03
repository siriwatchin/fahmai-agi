#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -J fahmai-qwen-smoke
#SBATCH -o lanta_smoke.%j.out
#SBATCH -e lanta_smoke.%j.err

set -euo pipefail

module purge || true
module load Mamba/23.11.0-0 || true
module load cuda/12.6 || true

cd /project/zz992000-zdevb/fahmai-agi/pipeline-qwen2.5
source ./lanta.env

mkdir -p "$WORK_DIR"

if [ -n "${LANTA_PYTHON:-}" ] && [ -x "$LANTA_PYTHON" ]; then
  PYTHON="$LANTA_PYTHON"
elif [ -d "$PROJECT_ROOT/venvs/qwen25" ]; then
  source "$PROJECT_ROOT/venvs/qwen25/bin/activate"
  PYTHON=python
elif command -v conda >/dev/null 2>&1; then
  conda activate qwen25
  PYTHON=python
else
  echo "No Python environment found. Create $PROJECT_ROOT/venvs/qwen25 or conda env qwen25 first." >&2
  exit 2
fi

"$PYTHON" - <<'PY'
import os
from pathlib import Path
checks = {
    "DATA_DIR": os.environ["DATA_DIR"],
    "QUESTIONS_CSV": os.environ["QUESTIONS_CSV"],
    "QWEN_MODEL_PATH": os.environ["QWEN_MODEL_PATH"],
    "EMBED_MODEL": os.environ["EMBED_MODEL"],
}
for name, value in checks.items():
    print(f"{name}={value} exists={Path(value).exists()}")
PY

"$PYTHON" agentic_best_integrated_qdrant.py --limit 1 --skip-qdrant-preload
