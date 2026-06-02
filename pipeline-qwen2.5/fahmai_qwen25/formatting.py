from __future__ import annotations

import re
from typing import Any


def money(x: Any) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def clean_answer(s: str, limit: int = 700) -> str:
    s = str(s).strip()
    for marker in ["\nuser\n", "\nassistant\n", "\nOBSERVATION", "\nSQL_RESULT", "\nQUESTION:"]:
        if marker in s:
            s = s.split(marker)[0].strip()
    s = re.sub(r"(?i)^assistant\s*", "", s).strip()
    return s[:limit]


def refuse_missing(topic: str) -> str:
    return f"ไม่พบ {topic} ในชุดข้อมูล"

