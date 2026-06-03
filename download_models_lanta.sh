#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/zz992000-zdevb}"
SRC_DIR="${SRC_DIR:-$PROJECT_ROOT/scamper_house}"
MODEL_ROOT="$SRC_DIR/qwen35/models"
QWEN_DIR="$MODEL_ROOT/Qwen2.5-7B-Instruct"
BGE_DIR="$MODEL_ROOT/bge-m3"
PYTHON="${LANTA_PYTHON:-$PROJECT_ROOT/envs/ml-base/bin/python}"

mkdir -p "$MODEL_ROOT"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ ! -x "$PYTHON" ]; then
  module load Mamba/23.11.0-0 || true
  PYTHON=python
fi

"$PYTHON" -m pip install -U "huggingface-hub>=0.34,<1" hf_transfer

"$PYTHON" - <<PY
from huggingface_hub import snapshot_download

targets = [
    ("Qwen/Qwen2.5-7B-Instruct", r"$QWEN_DIR"),
    ("BAAI/bge-m3", r"$BGE_DIR"),
]

for repo_id, local_dir in targets:
    print(f"Downloading {repo_id} -> {local_dir}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
PY

echo "Qwen: $QWEN_DIR"
echo "BGE : $BGE_DIR"
