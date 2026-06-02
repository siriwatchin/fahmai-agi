# FahMai Database Tools For Agentic AI

Reusable database/vector tools for agents that need to answer FahMai analytics questions.

This folder is intentionally separate from model pipelines. Any agent can import `database_tools.py`, get OpenAI-style tool schemas, dispatch tool calls, and use PostgreSQL plus Qdrant safely.

Secrets are read from environment variables. Do not commit `.env`.

## Install

```bash
cd database-tools
pip install -r requirements.txt
cp .env.example .env
```

## Environment

```bash
export PG_DSN="postgresql://USER:PASSWORD@HOST:PORT/DBNAME"
export QDRANT_URL="http://HOST:6333"
export QDRANT_API_KEY=""
export QDRANT_COLLECTION="fahmai_public"
export EMBED_MODEL="intfloat/multilingual-e5-base"
```

## Tool Categories

PostgreSQL structured tools:

- `postgres_healthcheck`
- `postgres_list_schemas`
- `postgres_list_tables`
- `postgres_describe_table`
- `postgres_search_schema`
- `postgres_sample_rows`
- `postgres_count_rows`
- `postgres_distinct_values`
- `postgres_aggregate`
- `postgres_group_by`
- `postgres_top_k`
- `postgres_time_series`
- `postgres_execute_readonly_sql`

Qdrant vector tools:

- `qdrant_healthcheck`
- `qdrant_list_collections`
- `qdrant_search`
- `qdrant_recreate_collection`
- `qdrant_upsert_texts`

Agent helpers:

- `get_openai_tool_schemas()`
- `call_tool(name, arguments)`
- `build_default_registry()`

Hackathon domain tools live in `domain_tools.py`.

These are FahMai-specific tools for common competition patterns:

- `domain_profile_table`: table profile, columns, row count, date ranges, numeric stats.
- `domain_date_range`: min/max date of a fact/dim table.
- `domain_file_catalog_search`: find corpus files by exact path/name keyword.
- `domain_text_exact_search`: exact string search in local corpus files.
- `domain_hybrid_search`: Qdrant semantic search plus local exact search.
- `domain_evidence_pack`: one-shot schema + file + retrieval + injection evidence bundle.
- `domain_policy_resolver`: resolve active `DIM_POLICY_VERSION` row at a date.
- `domain_entity_resolver`: map text/name/id to product/vendor/customer/employee/branch.
- `domain_prompt_injection_detector`: detect override/admin/copy-verbatim injection patterns.
- `domain_refusal_checker`: verify refusal has verb + topic + data-scope marker.
- `domain_answer_verifier`: catch bad refusal shape, Chinese leakage, injection failures.
- `domain_top_sku_by_units`: top SKU by units for a year.
- `domain_top_sku_by_revenue`: top SKU by line revenue.
- `domain_shipping_vendor_share`: shipping vendor count/share.
- `domain_customer_loyalty_counts`: customer count by loyalty tier.
- `domain_partner_brand_vendors`: partner-brand vendors.
- `domain_stockout_top_sku`: top stockout SKU by year.
- `domain_current_ceo`: active CEO at an as-of date.
- `domain_duplicate_vendor_invoice`: duplicate vendor invoice rows.
- `domain_recall_window`: recall state transitions for a SKU.
- `domain_return_refund_reconciliation`: return/refund mismatch summary.
- `domain_pos_log_schema_summary`: local POS TSV schema variants and BKK-CTW sanity figures.
- `domain_validate_submission`: validate final Kaggle submission shape.

## Example

```python
from database_tools import build_default_registry

registry = build_default_registry()

print(registry.call_tool("postgres_search_schema", {"query": "FACT_SALES net_total_thb"}))

print(registry.call_tool("postgres_execute_readonly_sql", {
    "sql": '''
        SELECT branch_code, COUNT(*) AS n, SUM(net_total_thb) AS revenue
        FROM FACT_SALES
        GROUP BY branch_code
        ORDER BY revenue DESC
        LIMIT 5
    '''
}))

print(registry.call_tool("qdrant_search", {
    "query": "refund signing authority ladder current version",
    "top_k": 5
}))
```

Domain tool example:

```python
from domain_tools import build_domain_registry

registry = build_domain_registry(include_qdrant=True)

print(registry.call_tool("domain_policy_resolver", {
    "policy_variable": "point_earning_rate_per_thb",
    "as_of_date": "2025-03-31"
}))

print(registry.call_tool("domain_entity_resolver", {
    "query": "NovaTech laptop",
    "entity_type": "product"
}))

print(registry.call_tool("domain_evidence_pack", {
    "question": "ใครเป็น CEO ของ FahMai ในเดือนพฤษภาคม 2025"
}))
```

## Shared Vector Database

Qdrant is model-agnostic. Every LLM pipeline can use the same collection as long as it sends search queries through the same embedding model used at ingest time.

Recommended shared setup:

- Collection name: `fahmai_public`
- Embedding model: `intfloat/multilingual-e5-base`
- Payload fields: `text`, `path`, `source`, `chunk`

Ingest once:

```bash
python qdrant_ingest.py --data-dir "$DATA_DIR" --recreate
```

Then any model can query:

```python
registry.call_tool("qdrant_search", {
    "query": "Powercell X3 recall early warning warranty claim",
    "top_k": 8
})
```

Do not create separate vector collections per model unless the embedding model or corpus changes. Separate per-model collections usually waste time and make retrieval inconsistent.

## Safety

- SQL execution is read-only only: `SELECT` and `WITH`.
- Mutation keywords are blocked: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, etc.
- Identifier-based helper tools validate table/column names.
- Free-form SQL should be used only by trusted agent code after planning/validation.
- Credentials are never hardcoded in this package.

## Tool Selection Guide

Use Postgres generic tools for:

- exact counts
- sums/averages/max
- table joins
- time series
- row-level reconciliation

Use Qdrant/vector tools for:

- policy/memo/chat/report retrieval
- semantic search
- prompt-injection evidence
- refusal confirmation

Use domain tools for:

- recurring FahMai business questions
- temporal policy lookup
- entity resolution
- answer/refusal verification
- submission QA
