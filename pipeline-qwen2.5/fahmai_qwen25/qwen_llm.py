from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .formatting import clean_answer


class QwenLocalLLM:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
        )
        self.token_log: list[dict[str, Any]] = []

    def generate(self, prompt: str, qid: str = "", stage: str = "final", max_new_tokens: int = 180) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=7000).to(device)
        input_len = int(inputs["input_ids"].shape[-1])
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        sec = time.time() - t0
        completion = int(out.shape[-1]) - input_len
        raw = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        self.token_log.append(
            {
                "qid": qid,
                "stage": stage,
                "prompt_tokens": input_len,
                "completion_tokens": completion,
                "total_tokens": input_len + completion,
                "seconds": round(sec, 3),
                "prompt_chars": len(prompt),
                "answer_chars": len(raw),
            }
        )
        return clean_answer(raw)

