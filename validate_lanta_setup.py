from pathlib import Path
import os

from transformers import AutoConfig, AutoTokenizer


for name in ["DATA_DIR", "QUESTIONS_CSV", "QWEN_MODEL_PATH", "EMBED_MODEL"]:
    value = os.environ.get(name, "")
    path = Path(value)
    print(f"{name}={path} exists={path.exists()}")

config = AutoConfig.from_pretrained(os.environ["QWEN_MODEL_PATH"])
print(f"model_type={config.model_type}")

tokenizer = AutoTokenizer.from_pretrained(os.environ["QWEN_MODEL_PATH"])
print(f"tokenizer_vocab={len(tokenizer)}")
