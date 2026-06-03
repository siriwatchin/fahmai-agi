#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ENTITY_RE = re.compile(
    r"""
    (?:EMP-L3-\d+|CUST-L3-[A-Z0-9-]+|SKU-[A-Z0-9-]+|[A-Z]{2,}-[A-Z0-9-]+|
       V-\d{3}|VP-\d+-\d+|BT-\d+-\d+|RFD-[A-Z0-9-]+|TXN-\d+-\d+|
       \d{4}-\d{2}-\d{2}|\d[\d,]*(?:\.\d+)?)
    """,
    re.X,
)


def load_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "id" not in rows[0]:
        raise ValueError(f"{path} must contain an id column")
    value_col = "response" if "response" in rows[0] else "answer"
    return {str(r["id"]).strip(): str(r.get(value_col, "")).strip() for r in rows}


def toks(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_ก-๙.-]{2,}", s.lower()))


def entities(s: str) -> set[str]:
    return {x.strip("`.,;:()[]{} ") for x in ENTITY_RE.findall(s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    gt = load_csv(Path(args.groundtruth).expanduser())
    pred = load_csv(Path(args.submission).expanduser())
    common = sorted(set(gt) & set(pred))
    missing = sorted(set(gt) - set(pred))
    extra = sorted(set(pred) - set(gt))

    details = []
    exact = 0
    for qid in common:
        g, p = gt[qid], pred[qid]
        if g == p:
            exact += 1
        gt_t, pr_t = toks(g), toks(p)
        overlap = len(gt_t & pr_t) / max(1, len(gt_t))
        gt_e, pr_e = entities(g), entities(p)
        entity_recall = len(gt_e & pr_e) / max(1, len(gt_e))
        details.append(
            {
                "id": qid,
                "exact": g == p,
                "token_recall": round(overlap, 4),
                "entity_recall": round(entity_recall, 4),
                "missing_entities": sorted(gt_e - pr_e)[:20],
                "pred_extra_entities": sorted(pr_e - gt_e)[:20],
                "gt_len": len(g),
                "pred_len": len(p),
            }
        )

    details.sort(key=lambda x: (x["exact"], x["entity_recall"], x["token_recall"]))
    summary = {
        "groundtruth": str(Path(args.groundtruth).expanduser()),
        "submission": str(Path(args.submission).expanduser()),
        "common": len(common),
        "missing": missing,
        "extra": extra,
        "exact_matches": exact,
        "exact_rate": round(exact / max(1, len(common)), 4),
        "avg_token_recall": round(sum(d["token_recall"] for d in details) / max(1, len(details)), 4),
        "avg_entity_recall": round(sum(d["entity_recall"] for d in details) / max(1, len(details)), 4),
        "worst": details[: args.top],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).expanduser().write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
