#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 05:00:00
#SBATCH -J fahmai-qwen-run
#SBATCH -o lanta_run.%j.out
#SBATCH -e lanta_run.%j.err

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

"$PYTHON" agentic_best_integrated_qdrant.py --limit 100
