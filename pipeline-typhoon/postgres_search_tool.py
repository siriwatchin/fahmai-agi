#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FahMai PostgreSQL tool kit for small/medium agent models.

This module is intentionally lightweight. The main runner uses
database-tools/, but this file remains useful for notebook experiments and
model pipelines that want a compact set of granular PostgreSQL tools.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PostgresSearchTool")


IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
READ_ONLY_DENY = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|GRANT|REVOKE|COPY|CALL|DO|VACUUM|ANALYZE)\b",
    re.IGNORECASE,
)


def _json(data: Any) -> str:
    return json.dumps(data, default=str, ensure_ascii=False)


class FahMaiDBAgentKit:
    def __init__(self):
        if load_dotenv:
            load_dotenv()

        self.ALLOWED_TABLES = {
            "DIM_BANK_ACCOUNT",
            "DIM_BRANCH",
            "DIM_CUSTOMER",
            "DIM_DATE",
            "DIM_DEPARTMENT",
            "DIM_EMPLOYEE",
            "DIM_POLICY_VERSION",
            "DIM_POSITION_LEVEL",
            "DIM_PRODUCT",
            "DIM_PROMO_CAMPAIGN",
            "DIM_VENDOR",
            "DIM_VENDOR_CONTRACT_VERSION",
            "FACT_BANK_TRANSACTION",
            "FACT_CS_INTERACTION",
            "FACT_INVENTORY_MONTHLY_SNAPSHOT",
            "FACT_INVENTORY_MOVEMENT",
            "FACT_LOYALTY_LEDGER",
            "FACT_PROMO_REDEMPTION",
            "FACT_REFUND_PAID",
            "FACT_RETURN",
            "FACT_SALES",
            "FACT_SALES_LINE_ITEM",
            "FACT_SHIPPING",
            "FACT_VENDOR_PAYMENT",
            "FACT_WARRANTY_CLAIM",
            "T2_DOC_INVENTORY",
            "dim_product_recall_history",
            "dim_signing_authority_ladder",
        }
        self.BLOCKED_TABLES = {"FACT_PAYROLL"}
        self.CUTOVER_DATE_STR = "2025-04-01"

    def _get_connection(self):
        pg_dsn = os.getenv("PG_DSN")
        if pg_dsn:
            return psycopg2.connect(pg_dsn)
        return psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
        )

    def _ident(self, identifier: str) -> str:
        if not IDENT_RE.match(str(identifier)):
            raise ValueError(f"Unsafe identifier: {identifier}")
        return f'"{identifier}"'

    def _validate_table_access(self, target_table: str) -> str:
        clean_table = str(target_table).replace(".csv", "").strip()
        upper = clean_table.upper()
        if upper in self.BLOCKED_TABLES:
            raise PermissionError("Access denied for sensitive table.")
        if clean_table not in self.ALLOWED_TABLES and upper not in self.ALLOWED_TABLES:
            raise PermissionError(f"Access denied or unknown table: {target_table}")
        if upper in self.ALLOWED_TABLES:
            return upper
        return clean_table

    def _table_ref(self, target_table: str, schema: str = "public") -> str:
        table = self._validate_table_access(target_table)
        return f"{self._ident(schema)}.{self._ident(table)}"

    def _column_ref(self, column: str) -> str:
        return self._ident(str(column).strip())

    def _handle_sales_schema_routing(self, column_name: str, filter_date: Optional[str]) -> str:
        clean_col = column_name.lower()
        if clean_col in {"discount_amt", "discount_total_thb", "discount"}:
            return "discount_total_thb" if filter_date and filter_date >= self.CUTOVER_DATE_STR else "discount_total_thb"
        return column_name

    def _check_readonly_sql(self, sql: str) -> str:
        sql_clean = sql.strip().rstrip(";")
        if READ_ONLY_DENY.search(sql_clean):
            raise ValueError("Only read-only SELECT/WITH queries are allowed")
        if not re.match(r"^(SELECT|WITH)\b", sql_clean, re.IGNORECASE):
            raise ValueError("SQL must start with SELECT or WITH")
        return self._quote_known_tables(sql_clean)

    def _quote_known_tables(self, sql: str) -> str:
        for table in sorted(self.ALLOWED_TABLES, key=len, reverse=True):
            if "*" in table or "/" in table:
                continue
            sql = re.sub(
                rf'(?<!")\bpublic\.{re.escape(table)}\b(?!")',
                f'public."{table}"',
                sql,
                flags=re.IGNORECASE,
            )
            sql = re.sub(
                rf'(?<![.\w"])\b{re.escape(table)}\b(?!")',
                f'"{table}"',
                sql,
                flags=re.IGNORECASE,
            )
        return sql

    def _execute(self, sql: str, params: list[Any] | tuple[Any, ...] = (), limit: int | None = None) -> dict[str, Any]:
        try:
            sql_clean = self._check_readonly_sql(sql)
            if limit is not None:
                sql_clean = f"SELECT * FROM ({sql_clean}) AS tool_query LIMIT {int(limit)}"
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SET statement_timeout = '45s'")
                    cursor.execute(sql_clean, params)
                    rows = cursor.fetchall()
            return {"status": "success", "ok": True, "data": rows, "rows": rows, "shape": [len(rows), len(rows[0]) if rows else 0], "sql": sql_clean}
        except Exception as e:
            return {"status": "error", "ok": False, "message": str(e), "error": str(e), "sql": sql}

    def healthcheck(self) -> str:
        return _json(self._execute("SELECT current_database() AS database, current_user AS user, now() AS server_time"))

    def list_tables(self, schema: str = "public") -> str:
        sql = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = %s
        ORDER BY table_name
        """
        return _json(self._execute(sql, [schema], limit=1000))

    def describe_table(self, target_table: str, schema: str = "public") -> str:
        table = self._validate_table_access(target_table)
        sql = """
        SELECT column_name, data_type, is_nullable, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """
        return _json(self._execute(sql, [schema, table], limit=500))

    def search_schema(self, query: str, schema: str = "public", limit: int = 20) -> str:
        res = self._execute(
            """
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position
            """,
            [schema],
            limit=10000,
        )
        if not res.get("ok"):
            return _json(res)
        tokens = [t.upper() for t in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", query)]
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in res["rows"]:
            table = row["table_name"]
            if table not in self.ALLOWED_TABLES and table.upper() not in self.ALLOWED_TABLES:
                continue
            grouped.setdefault(table, []).append({"column": row["column_name"], "type": row["data_type"]})

        hits = []
        q_upper = query.upper()
        for table, cols in grouped.items():
            blob = f"{table} " + " ".join(c["column"] for c in cols)
            score = 1000 if table.upper() in q_upper else 0
            score += sum(len(tok) * 3 for tok in tokens if tok in blob.upper())
            if score:
                hits.append({"schema": schema, "table": table, "score": score, "columns": cols})
        hits.sort(key=lambda x: x["score"], reverse=True)
        return _json({"status": "success", "ok": True, "rows": hits[:limit], "data": hits[:limit]})

    def count_rows(self, target_table: str, filter_column: Optional[str] = None, filter_value: Optional[str] = None) -> str:
        table = self._table_ref(target_table)
        if filter_column and filter_value is not None:
            sql = f"SELECT COUNT(*) AS total_rows FROM {table} WHERE {self._column_ref(filter_column)} = %s"
            return _json(self._execute(sql, [filter_value], limit=1))
        return _json(self._execute(f"SELECT COUNT(*) AS total_rows FROM {table}", limit=1))

    def sum_column(
        self,
        target_table: str,
        target_column: str,
        filter_column: Optional[str] = None,
        filter_value: Optional[str] = None,
        filter_date: Optional[str] = None,
    ) -> str:
        table_name = self._validate_table_access(target_table)
        column = self._handle_sales_schema_routing(target_column, filter_date) if table_name == "FACT_SALES" else target_column
        table = self._table_ref(table_name)
        metric = self._column_ref(column)
        if filter_column and filter_value is not None:
            sql = f"SELECT SUM({metric}::numeric) AS sum_value FROM {table} WHERE {self._column_ref(filter_column)} = %s"
            return _json(self._execute(sql, [filter_value], limit=1))
        return _json(self._execute(f"SELECT SUM({metric}::numeric) AS sum_value FROM {table}", limit=1))

    def find_max(self, target_table: str, target_column: str, filter_column: Optional[str] = None, filter_value: Optional[str] = None) -> str:
        table = self._table_ref(target_table)
        metric = self._column_ref(target_column)
        if filter_column and filter_value is not None:
            sql = f"SELECT * FROM {table} WHERE {self._column_ref(filter_column)} = %s ORDER BY {metric}::numeric DESC LIMIT 1"
            return _json(self._execute(sql, [filter_value], limit=1))
        return _json(self._execute(f"SELECT * FROM {table} ORDER BY {metric}::numeric DESC LIMIT 1", limit=1))

    def group_by_count(self, target_table: str, group_column: str, limit: int = 100) -> str:
        table = self._table_ref(target_table)
        group_col = self._column_ref(group_column)
        sql = f"""
        SELECT {group_col} AS value, COUNT(*) AS row_count
        FROM {table}
        GROUP BY {group_col}
        ORDER BY row_count DESC, value
        """
        return _json(self._execute(sql, limit=limit))

    def fetch_sample(self, target_table: str, limit: int = 5) -> str:
        return _json(self._execute(f"SELECT * FROM {self._table_ref(target_table)}", limit=limit))

    def execute_readonly_sql(self, sql: str, limit: int = 200) -> str:
        return _json(self._execute(sql, limit=limit))

    def get_openai_tool_schema(self) -> list[dict[str, Any]]:
        def tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": properties, "required": required or []},
                },
            }

        return [
            tool("healthcheck", "Check PostgreSQL connection.", {}),
            tool("list_tables", "List tables in a schema.", {"schema": {"type": "string", "default": "public"}}),
            tool("describe_table", "Describe columns for an allowed table.", {"target_table": {"type": "string"}, "schema": {"type": "string", "default": "public"}}, ["target_table"]),
            tool("search_schema", "Search allowed tables/columns by keywords.", {"query": {"type": "string"}, "schema": {"type": "string", "default": "public"}, "limit": {"type": "integer", "default": 20}}, ["query"]),
            tool("count_rows", "Count rows in an allowed table, optionally filtered by equality.", {"target_table": {"type": "string"}, "filter_column": {"type": "string"}, "filter_value": {"type": "string"}}, ["target_table"]),
            tool("sum_column", "Sum a numeric column, optionally filtered by equality.", {"target_table": {"type": "string"}, "target_column": {"type": "string"}, "filter_column": {"type": "string"}, "filter_value": {"type": "string"}, "filter_date": {"type": "string"}}, ["target_table", "target_column"]),
            tool("find_max", "Return row with maximum numeric value in a column.", {"target_table": {"type": "string"}, "target_column": {"type": "string"}, "filter_column": {"type": "string"}, "filter_value": {"type": "string"}}, ["target_table", "target_column"]),
            tool("group_by_count", "Count rows grouped by one column.", {"target_table": {"type": "string"}, "group_column": {"type": "string"}, "limit": {"type": "integer", "default": 100}}, ["target_table", "group_column"]),
            tool("fetch_sample", "Fetch sample rows from an allowed table.", {"target_table": {"type": "string"}, "limit": {"type": "integer", "default": 5}}, ["target_table"]),
            tool("execute_readonly_sql", "Execute trusted read-only SELECT/WITH SQL. Use after schema validation.", {"sql": {"type": "string"}, "limit": {"type": "integer", "default": 200}}, ["sql"]),
        ]


if __name__ == "__main__":
    kit = FahMaiDBAgentKit()
    print(kit.healthcheck())
