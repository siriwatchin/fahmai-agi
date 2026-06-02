from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

import psycopg
from psycopg.rows import dict_row

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


READ_ONLY_DENY = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|GRANT|REVOKE|COPY|CALL|DO|VACUUM|ANALYZE)\b",
    re.IGNORECASE,
)
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_env() -> None:
    if load_dotenv:
        load_dotenv()


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _ident(name: str) -> str:
    if not IDENT_RE.match(str(name)):
        raise ValueError(f"Unsafe identifier: {name}")
    return f'"{name}"'


def _table_ref(table: str, schema: str | None = None) -> str:
    if schema:
        return f"{_ident(schema)}.{_ident(table)}"
    return _ident(table)


def _where_equals(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    if not filters:
        return "", []
    parts = []
    params = []
    for col, val in filters.items():
        parts.append(f"{_ident(col)} = %s")
        params.append(val)
    return " WHERE " + " AND ".join(parts), params


def _check_readonly(sql: str) -> str:
    sql_clean = sql.strip().rstrip(";")
    if READ_ONLY_DENY.search(sql_clean):
        raise ValueError("Only read-only SELECT/WITH queries are allowed")
    if not re.match(r"^(SELECT|WITH)\b", sql_clean, re.IGNORECASE):
        raise ValueError("SQL must start with SELECT or WITH")
    return sql_clean


@dataclass(frozen=True)
class DbToolConfig:
    pg_dsn: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    embed_model: str

    @classmethod
    def from_env(cls) -> "DbToolConfig":
        _load_env()
        return cls(
            pg_dsn=os.getenv("PG_DSN", ""),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "fahmai_public"),
            embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        )


class PostgresDatabaseTools:
    def __init__(self, dsn: str):
        if not dsn:
            raise ValueError("PG_DSN is required")
        self.dsn = dsn

    def _query(self, sql: str, params: list[Any] | tuple[Any, ...] = (), limit: int | None = None) -> dict[str, Any]:
        try:
            sql_clean = _check_readonly(sql)
            if limit is not None:
                sql_clean = f"SELECT * FROM ({sql_clean}) AS tool_query LIMIT {int(limit)}"
            with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '45s'")
                    cur.execute(sql_clean, params)
                    rows = cur.fetchall()
            return {"ok": True, "rows": rows, "shape": [len(rows), len(rows[0]) if rows else 0], "sql": sql_clean}
        except Exception as e:
            return {"ok": False, "error": str(e), "sql": sql}

    def healthcheck(self) -> str:
        return _json(self._query("SELECT current_database() AS database, current_user AS user, now() AS server_time"))

    def list_schemas(self) -> str:
        sql = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
        ORDER BY schema_name
        """
        return _json(self._query(sql))

    def list_tables(self, schema: str = "public") -> str:
        sql = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = %s
        ORDER BY table_name
        """
        return _json(self._query(sql, [schema], limit=1000))

    def describe_table(self, table: str, schema: str = "public") -> str:
        sql = """
        SELECT column_name, data_type, is_nullable, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """
        return _json(self._query(sql, [schema, table], limit=500))

    def search_schema(self, query: str, schema: str | None = None, limit: int = 20) -> str:
        sql = """
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name, ordinal_position
        """
        res = self._query(sql, limit=10000)
        if not res.get("ok"):
            return _json(res)
        tokens = [t.upper() for t in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", query)]
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in res["rows"]:
            if schema and row["table_schema"] != schema:
                continue
            key = (row["table_schema"], row["table_name"])
            grouped.setdefault(key, []).append({"column": row["column_name"], "type": row["data_type"]})
        hits = []
        q_upper = query.upper()
        for (sch, tbl), cols in grouped.items():
            blob = f"{sch} {tbl} " + " ".join(c["column"] for c in cols)
            blob_u = blob.upper()
            score = 0
            if tbl.upper() in q_upper:
                score += 1000
            for tok in tokens:
                if tok in blob_u:
                    score += len(tok) * 3
            if score:
                hits.append({"schema": sch, "table": tbl, "score": score, "columns": cols})
        hits.sort(key=lambda x: x["score"], reverse=True)
        return _json({"ok": True, "rows": hits[:limit]})

    def sample_rows(self, table: str, schema: str = "public", limit: int = 5) -> str:
        sql = f"SELECT * FROM {_table_ref(table, schema)}"
        return _json(self._query(sql, limit=limit))

    def count_rows(self, table: str, schema: str = "public", filters: dict[str, Any] | None = None) -> str:
        where, params = _where_equals(filters)
        sql = f"SELECT COUNT(*) AS n FROM {_table_ref(table, schema)}{where}"
        return _json(self._query(sql, params))

    def distinct_values(self, table: str, column: str, schema: str = "public", limit: int = 100) -> str:
        sql = f"""
        SELECT {_ident(column)} AS value, COUNT(*) AS n
        FROM {_table_ref(table, schema)}
        GROUP BY {_ident(column)}
        ORDER BY n DESC, value
        """
        return _json(self._query(sql, limit=limit))

    def aggregate(
        self,
        table: str,
        column: str,
        op: str = "sum",
        schema: str = "public",
        filters: dict[str, Any] | None = None,
    ) -> str:
        allowed = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX", "count": "COUNT"}
        fn = allowed.get(op.lower())
        if not fn:
            return _json({"ok": False, "error": f"Unsupported aggregate op: {op}"})
        where, params = _where_equals(filters)
        expr = "*" if fn == "COUNT" and column == "*" else _ident(column)
        sql = f"SELECT {fn}({expr}) AS value FROM {_table_ref(table, schema)}{where}"
        return _json(self._query(sql, params))

    def group_by(
        self,
        table: str,
        group_columns: list[str],
        metric_column: str = "*",
        op: str = "count",
        schema: str = "public",
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> str:
        allowed = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX", "count": "COUNT"}
        fn = allowed.get(op.lower())
        if not fn:
            return _json({"ok": False, "error": f"Unsupported aggregate op: {op}"})
        groups = ", ".join(_ident(c) for c in group_columns)
        metric = "*" if fn == "COUNT" and metric_column == "*" else _ident(metric_column)
        where, params = _where_equals(filters)
        sql = f"""
        SELECT {groups}, {fn}({metric}) AS value
        FROM {_table_ref(table, schema)}
        {where}
        GROUP BY {groups}
        ORDER BY value DESC
        """
        return _json(self._query(sql, params, limit=limit))

    def top_k(
        self,
        table: str,
        order_by: str,
        schema: str = "public",
        descending: bool = True,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> str:
        where, params = _where_equals(filters)
        direction = "DESC" if descending else "ASC"
        sql = f"SELECT * FROM {_table_ref(table, schema)}{where} ORDER BY {_ident(order_by)} {direction}"
        return _json(self._query(sql, params, limit=limit))

    def time_series(
        self,
        table: str,
        date_column: str,
        metric_column: str,
        grain: str = "month",
        op: str = "sum",
        schema: str = "public",
        filters: dict[str, Any] | None = None,
        limit: int = 500,
    ) -> str:
        allowed_grains = {"day", "week", "month", "quarter", "year"}
        allowed_ops = {"sum": "SUM", "avg": "AVG", "count": "COUNT", "min": "MIN", "max": "MAX"}
        if grain not in allowed_grains:
            return _json({"ok": False, "error": f"Unsupported grain: {grain}"})
        fn = allowed_ops.get(op.lower())
        if not fn:
            return _json({"ok": False, "error": f"Unsupported op: {op}"})
        metric = "*" if fn == "COUNT" and metric_column == "*" else _ident(metric_column)
        where, params = _where_equals(filters)
        sql = f"""
        SELECT date_trunc('{grain}', {_ident(date_column)}::date) AS period,
               {fn}({metric}) AS value
        FROM {_table_ref(table, schema)}
        {where}
        GROUP BY period
        ORDER BY period
        """
        return _json(self._query(sql, params, limit=limit))

    def execute_readonly_sql(self, sql: str, limit: int = 200) -> str:
        return _json(self._query(sql, limit=limit))


class QdrantDatabaseTools:
    def __init__(self, url: str, api_key: str | None, collection: str, embed_model: str):
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key)
        self.encoder = SentenceTransformer(embed_model)

    def healthcheck(self) -> str:
        try:
            cols = self.client.get_collections()
            return _json({"ok": True, "collections": [c.name for c in cols.collections]})
        except Exception as e:
            return _json({"ok": False, "error": str(e)})

    def list_collections(self) -> str:
        return self.healthcheck()

    def recreate_collection(self, collection: str | None = None) -> str:
        try:
            from qdrant_client.models import Distance, VectorParams

            dim = int(self.encoder.get_sentence_embedding_dimension())
            self.client.recreate_collection(
                collection_name=collection or self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            return _json({"ok": True, "collection": collection or self.collection, "dimension": dim})
        except Exception as e:
            return _json({"ok": False, "error": str(e)})

    def upsert_texts(self, records: list[dict[str, Any]], collection: str | None = None, start_id: int = 0) -> str:
        try:
            from qdrant_client.models import PointStruct

            points = []
            texts = [str(r.get("text", "")) for r in records]
            vectors = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
            for offset, (record, vector) in enumerate(zip(records, vectors)):
                payload = dict(record)
                points.append(PointStruct(id=start_id + offset, vector=vector, payload=payload))
            self.client.upsert(collection_name=collection or self.collection, points=points)
            return _json({"ok": True, "upserted": len(points), "collection": collection or self.collection})
        except Exception as e:
            return _json({"ok": False, "error": str(e)})

    def search(self, query: str, top_k: int = 8, collection: str | None = None) -> str:
        try:
            vector = self.encoder.encode([query], normalize_embeddings=True)[0].tolist()
            hits = self.client.search(
                collection_name=collection or self.collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
            rows = []
            for h in hits:
                payload = dict(h.payload or {})
                rows.append({"score": float(h.score), "payload": payload, "text": payload.get("text", "")})
            return _json({"ok": True, "rows": rows})
        except Exception as e:
            return _json({"ok": False, "error": str(e)})


class DatabaseToolRegistry:
    def __init__(self, pg: PostgresDatabaseTools, qdrant: QdrantDatabaseTools | None = None):
        self.pg = pg
        self.qdrant = qdrant
        self.tools: dict[str, Callable[..., str]] = {
            "postgres_healthcheck": lambda: self.pg.healthcheck(),
            "postgres_list_schemas": lambda: self.pg.list_schemas(),
            "postgres_list_tables": self.pg.list_tables,
            "postgres_describe_table": self.pg.describe_table,
            "postgres_search_schema": self.pg.search_schema,
            "postgres_sample_rows": self.pg.sample_rows,
            "postgres_count_rows": self.pg.count_rows,
            "postgres_distinct_values": self.pg.distinct_values,
            "postgres_aggregate": self.pg.aggregate,
            "postgres_group_by": self.pg.group_by,
            "postgres_top_k": self.pg.top_k,
            "postgres_time_series": self.pg.time_series,
            "postgres_execute_readonly_sql": self.pg.execute_readonly_sql,
        }
        if qdrant is not None:
            self.tools.update(
                {
                    "qdrant_healthcheck": lambda: qdrant.healthcheck(),
                    "qdrant_list_collections": lambda: qdrant.list_collections(),
                    "qdrant_recreate_collection": qdrant.recreate_collection,
                    "qdrant_upsert_texts": qdrant.upsert_texts,
                    "qdrant_search": qdrant.search,
                }
            )

    def call_tool(self, name: str, arguments: dict[str, Any] | str | None = None) -> str:
        if name not in self.tools:
            return _json({"ok": False, "error": f"Unknown tool: {name}"})
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        try:
            return self.tools[name](**arguments)
        except Exception as e:
            return _json({"ok": False, "error": str(e), "tool": name})

    def get_openai_tool_schemas(self) -> list[dict[str, Any]]:
        return get_openai_tool_schemas(include_qdrant=self.qdrant is not None)


def build_default_registry(include_qdrant: bool = True) -> DatabaseToolRegistry:
    cfg = DbToolConfig.from_env()
    pg = PostgresDatabaseTools(cfg.pg_dsn)
    qdrant = None
    if include_qdrant:
        qdrant = QdrantDatabaseTools(cfg.qdrant_url, cfg.qdrant_api_key, cfg.qdrant_collection, cfg.embed_model)
    return DatabaseToolRegistry(pg, qdrant)


def get_openai_tool_schemas(include_qdrant: bool = True) -> list[dict[str, Any]]:
    def tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required or [],
                },
            },
        }

    schemas = [
        tool("postgres_healthcheck", "Check PostgreSQL connection.", {}),
        tool("postgres_list_schemas", "List database schemas.", {}),
        tool("postgres_list_tables", "List tables in a schema.", {"schema": {"type": "string", "default": "public"}}),
        tool(
            "postgres_describe_table",
            "Describe table columns and types.",
            {"table": {"type": "string"}, "schema": {"type": "string", "default": "public"}},
            ["table"],
        ),
        tool(
            "postgres_search_schema",
            "Search table/column schema by natural language or keywords.",
            {"query": {"type": "string"}, "schema": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
            ["query"],
        ),
        tool(
            "postgres_sample_rows",
            "Fetch sample rows from a table.",
            {"table": {"type": "string"}, "schema": {"type": "string", "default": "public"}, "limit": {"type": "integer", "default": 5}},
            ["table"],
        ),
        tool(
            "postgres_count_rows",
            "Count table rows with optional equality filters.",
            {"table": {"type": "string"}, "schema": {"type": "string", "default": "public"}, "filters": {"type": "object"}},
            ["table"],
        ),
        tool(
            "postgres_distinct_values",
            "List distinct values and counts for a column.",
            {"table": {"type": "string"}, "column": {"type": "string"}, "schema": {"type": "string", "default": "public"}, "limit": {"type": "integer", "default": 100}},
            ["table", "column"],
        ),
        tool(
            "postgres_aggregate",
            "Aggregate a numeric column with sum/avg/min/max/count.",
            {"table": {"type": "string"}, "column": {"type": "string"}, "op": {"type": "string", "enum": ["sum", "avg", "min", "max", "count"]}, "schema": {"type": "string", "default": "public"}, "filters": {"type": "object"}},
            ["table", "column"],
        ),
        tool(
            "postgres_group_by",
            "Group by one or more columns and aggregate a metric.",
            {"table": {"type": "string"}, "group_columns": {"type": "array", "items": {"type": "string"}}, "metric_column": {"type": "string", "default": "*"}, "op": {"type": "string", "enum": ["sum", "avg", "min", "max", "count"]}, "schema": {"type": "string", "default": "public"}, "filters": {"type": "object"}, "limit": {"type": "integer", "default": 100}},
            ["table", "group_columns"],
        ),
        tool(
            "postgres_top_k",
            "Fetch top-k rows ordered by a column.",
            {"table": {"type": "string"}, "order_by": {"type": "string"}, "schema": {"type": "string", "default": "public"}, "descending": {"type": "boolean", "default": True}, "filters": {"type": "object"}, "limit": {"type": "integer", "default": 10}},
            ["table", "order_by"],
        ),
        tool(
            "postgres_time_series",
            "Build day/week/month/quarter/year aggregate time series.",
            {"table": {"type": "string"}, "date_column": {"type": "string"}, "metric_column": {"type": "string"}, "grain": {"type": "string", "enum": ["day", "week", "month", "quarter", "year"]}, "op": {"type": "string", "enum": ["sum", "avg", "count", "min", "max"]}, "schema": {"type": "string", "default": "public"}, "filters": {"type": "object"}, "limit": {"type": "integer", "default": 500}},
            ["table", "date_column", "metric_column"],
        ),
        tool(
            "postgres_execute_readonly_sql",
            "Execute trusted read-only SELECT/WITH SQL. Use after schema validation.",
            {"sql": {"type": "string"}, "limit": {"type": "integer", "default": 200}},
            ["sql"],
        ),
    ]
    if include_qdrant:
        schemas.extend(
            [
                tool("qdrant_healthcheck", "Check Qdrant connection and collections.", {}),
                tool("qdrant_list_collections", "List Qdrant collections.", {}),
                tool(
                    "qdrant_recreate_collection",
                    "Delete and recreate a Qdrant collection with the configured embedding dimension.",
                    {"collection": {"type": "string"}},
                ),
                tool(
                    "qdrant_upsert_texts",
                    "Upsert text records into Qdrant. Each record should include text plus optional path/source/chunk metadata.",
                    {"records": {"type": "array", "items": {"type": "object"}}, "collection": {"type": "string"}, "start_id": {"type": "integer", "default": 0}},
                    ["records"],
                ),
                tool(
                    "qdrant_search",
                    "Semantic vector search over embedded corpus.",
                    {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 8}, "collection": {"type": "string"}},
                    ["query"],
                ),
            ]
        )
    return schemas


if __name__ == "__main__":
    registry = build_default_registry(include_qdrant=False)
    print(registry.call_tool("postgres_healthcheck"))
