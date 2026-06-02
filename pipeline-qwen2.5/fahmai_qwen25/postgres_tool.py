from __future__ import annotations

import re
from typing import Any

import psycopg
from psycopg.rows import dict_row


READ_ONLY_DENY = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|GRANT|REVOKE|COPY|CALL|DO)\b",
    re.IGNORECASE,
)


class PostgresTool:
    def __init__(self, dsn: str):
        if not dsn:
            raise ValueError("PG_DSN is required for PostgresTool")
        self.dsn = dsn

    def query(self, sql: str, params: tuple[Any, ...] = (), limit: int = 200) -> dict[str, Any]:
        if READ_ONLY_DENY.search(sql):
            return {"ok": False, "error": "Only read-only SELECT/WITH queries are allowed", "sql": sql}
        sql_clean = sql.strip().rstrip(";")
        if not re.match(r"^(SELECT|WITH)\b", sql_clean, re.IGNORECASE):
            return {"ok": False, "error": "Query must start with SELECT or WITH", "sql": sql}
        wrapped = f"SELECT * FROM ({sql_clean}) AS q LIMIT {int(limit)}"
        try:
            with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '30s'")
                    cur.execute(wrapped, params)
                    rows = cur.fetchall()
            return {"ok": True, "rows": rows, "shape": [len(rows), len(rows[0]) if rows else 0], "sql": sql_clean}
        except Exception as e:
            return {"ok": False, "error": str(e), "sql": sql_clean}

    def schema_search(self, query: str, limit: int = 20) -> dict[str, Any]:
        tokens = [t.upper() for t in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", query)]
        sql = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_name, ordinal_position
        """
        res = self.query(sql, limit=5000)
        if not res.get("ok"):
            return res
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in res["rows"]:
            grouped.setdefault(str(row["table_name"]).upper(), []).append(
                {"column": row["column_name"], "type": row["data_type"]}
            )
        hits = []
        for table, cols in grouped.items():
            blob = table + " " + " ".join(c["column"].upper() for c in cols)
            score = sum(len(t) for t in tokens if t in blob)
            if table in query.upper():
                score += 1000
            if score:
                hits.append({"table": table, "score": score, "columns": cols})
        hits.sort(key=lambda x: x["score"], reverse=True)
        return {"ok": True, "rows": hits[:limit]}

