from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CORE_PREFIXES = ("FACT_", "DIM_", "T2_")


def _tool_json(registry: Any, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return json.loads(registry.call_tool(name, args or {}))


def load_or_build_schema_cache(registry: Any, cache_path: Path, refresh: bool = False) -> dict[str, Any]:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    tables_res = _tool_json(registry, "postgres_list_tables", {"schema": "public"})
    if not tables_res.get("ok"):
        return {"ok": False, "error": tables_res.get("error", "failed to list tables"), "tables": {}}

    tables: dict[str, Any] = {}
    for row in tables_res.get("rows", []):
        schema = row.get("table_schema", "public")
        table = row.get("table_name", "")
        desc = _tool_json(registry, "postgres_describe_table", {"schema": schema, "table": table})
        tables[f"{schema}.{table}"] = {
            "schema": schema,
            "table": table,
            "table_type": row.get("table_type"),
            "columns": desc.get("rows", []) if desc.get("ok") else [],
        }

    cache = {"ok": True, "tables": tables}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return cache


def build_schema_summary(registry: Any, cache_path: Path, refresh: bool = False, whitelist: set[str] | None = None) -> str:
    cache = load_or_build_schema_cache(registry, cache_path, refresh=refresh)
    if not cache.get("ok"):
        return f"Schema summary unavailable: {cache.get('error', 'unknown error')}"

    tables = cache.get("tables", {})
    table_names = [meta.get("table", "") for meta in tables.values()]
    important = [t for t in table_names if t.startswith(CORE_PREFIXES)]

    lines = [
        "PostgreSQL public tables are case-sensitive. Use quoted uppercase table names in SQL.",
    ]
    if whitelist:
        lines.append("Data layer whitelist:")
        lines.append(", ".join(sorted(whitelist)))
    lines.append("Available core tables:")
    lines.append(", ".join(sorted(important)))

    compact_columns = []
    for key, meta in sorted(tables.items()):
        table = meta.get("table", "")
        if not table.startswith(CORE_PREFIXES):
            continue
        cols = [c.get("column_name") for c in meta.get("columns", []) if c.get("column_name")]
        compact_columns.append(f"- {table}: {', '.join(cols[:28])}")
    if compact_columns:
        lines.append("Core table columns:")
        lines.extend(compact_columns)

    return "\n".join(lines)
