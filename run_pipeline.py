from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from fahmai_qwen25.agent import answer_one
from fahmai_qwen25.config import Settings
from fahmai_qwen25.postgres_tool import PostgresTool
from fahmai_qwen25.qwen_llm import QwenLocalLLM
from fahmai_qwen25.vector_tool import QdrantVectorTool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    settings = Settings.from_env()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    qdf = pd.read_csv(settings.questions_csv)
    if args.limit:
        qdf = qdf.head(args.limit)

    pg = PostgresTool(settings.pg_dsn)
    vec = QdrantVectorTool(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        embed_model=settings.embed_model,
    )
    llm = QwenLocalLLM(settings.qwen_model_path)

    rows = []
    debug = {}
    t0 = time.time()

    for _, row in qdf.iterrows():
        qid = str(row.iloc[0]).strip()
        q = str(row.iloc[1])
        print(f"\n== {qid} ==\n{q}")
        qt = time.time()
        ans, obs = answer_one(pg, vec, llm, qid, q, settings.max_context_chars)
        print("ANSWER:", ans)
        print("question_sec:", round(time.time() - qt, 3))
        rows.append({"id": qid, "response": ans})
        debug[qid] = obs

    out = pd.DataFrame(rows)
    out.to_csv(settings.output_dir / "qwen25_results.csv", index=False)
    out.to_csv(settings.output_dir / "qwen25_submission.csv", index=False)

    (settings.output_dir / "qwen25_debug.json").write_text(json.dumps(debug, ensure_ascii=False, indent=2, default=str))

    token_df = pd.DataFrame(llm.token_log)
    token_df.to_csv(settings.output_dir / "qwen25_token_usage.csv", index=False)
    summary = {
        "num_llm_calls": int(len(token_df)),
        "prompt_tokens": int(token_df["prompt_tokens"].sum()) if len(token_df) else 0,
        "completion_tokens": int(token_df["completion_tokens"].sum()) if len(token_df) else 0,
        "total_tokens": int(token_df["total_tokens"].sum()) if len(token_df) else 0,
        "seconds": round(float(token_df["seconds"].sum()), 3) if len(token_df) else 0,
        "total_pipeline_sec": round(time.time() - t0, 3),
    }
    (settings.output_dir / "qwen25_token_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nDONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

