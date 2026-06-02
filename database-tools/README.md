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
