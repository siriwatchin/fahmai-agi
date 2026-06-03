from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline-typhoon"))
sys.path.insert(0, str(ROOT / "pipeline-typhoon-local-7b"))

import run_typhoon_database_tools as pipeline  # noqa: E402
from local_typhoon_7b_engine import DEFAULT_7B_MODEL, apply_7b_defaults, install_7b_call_typhoon, parse_raw_json_tool_call  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run FahMai pipeline with local Typhoon/Qwen 7B.")
    ap.add_argument("--questions", type=Path, default=ROOT / "questions.csv")
    ap.add_argument("--sample", type=Path, default=ROOT / "sample_submission.csv")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--timeout-seconds", type=int, default=60)
    ap.add_argument("--user-role", default="audit")
    ap.add_argument("--no-answer-bank", action="store_true")
    ap.add_argument("--no-qdrant", action="store_true")
    ap.add_argument("--qdrant-mode", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--schema-cache", type=Path, default=ROOT / "outputs" / "schema_cache.json")
    ap.add_argument("--refresh-schema-cache", action="store_true")
    ap.add_argument("--local-model", default=os.getenv("LOCAL_TYPHOON_MODEL", DEFAULT_7B_MODEL))
    args = ap.parse_args()

    apply_7b_defaults()
    os.environ["LOCAL_TYPHOON_MODEL"] = args.local_model
    pipeline.parse_json_tool_call = parse_raw_json_tool_call
    install_7b_call_typhoon(pipeline)

    run_dir = args.run_dir or pipeline.default_run_dir("local_7b_batch")
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "submission.csv"
    debug_path = run_dir / "typhoon_local_7b_debug.json"
    report_path = run_dir / "typhoon_local_7b_report.md"
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "mode": "local-7b",
                "model": args.local_model,
                "run_dir": str(run_dir),
                "questions": str(args.questions),
                "sample": str(args.sample),
                "output": str(output),
                "debug": str(debug_path),
                "report": str(report_path),
                "qdrant_mode": "never" if args.no_qdrant else args.qdrant_mode,
                "answer_bank": not args.no_answer_bank,
                "user_role": args.user_role,
                "timeout_seconds": args.timeout_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    sys.argv = [
        "run_typhoon_database_tools.py",
        "--questions",
        str(args.questions),
        "--sample",
        str(args.sample),
        "--run-dir",
        str(run_dir),
        "--output",
        str(output),
        "--debug",
        str(debug_path),
        "--report",
        str(report_path),
        "--max-steps",
        str(args.max_steps),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--user-role",
        args.user_role,
        "--schema-cache",
        str(args.schema_cache),
    ]
    if args.limit:
        sys.argv += ["--limit", str(args.limit)]
    if args.no_answer_bank:
        sys.argv.append("--no-answer-bank")
    if args.no_qdrant:
        sys.argv.append("--no-qdrant")
    else:
        sys.argv += ["--qdrant-mode", args.qdrant_mode]
    if args.refresh_schema_cache:
        sys.argv.append("--refresh-schema-cache")

    pipeline.main()


if __name__ == "__main__":
    main()

