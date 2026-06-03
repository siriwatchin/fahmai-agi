import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

try:
    from context_prompt import CONTEXT_PACKS
except Exception:
    CONTEXT_PACKS = {
        "sales_context": ["FACT_SALES", "FACT_SALES_LINE_ITEM", "DIM_PRODUCT", "DIM_BRANCH", "DIM_CUSTOMER"],
        "customer_cs_context": ["DIM_CUSTOMER", "FACT_CS_INTERACTION", "docs/chat_line_oa"],
        "policy_context": ["DIM_POLICY_VERSION", "dim_signing_authority_ladder", "docs/memo", "docs/email"],
        "vendor_shipping_context": ["DIM_VENDOR", "FACT_SHIPPING", "FACT_VENDOR_PAYMENT", "DIM_VENDOR_CONTRACT_VERSION"],
        "finance_bank_context": ["FACT_BANK_TRANSACTION", "DIM_BANK_ACCOUNT", "FACT_REFUND_PAID", "FACT_VENDOR_PAYMENT"],
        "inventory_context": ["FACT_INVENTORY_MOVEMENT", "FACT_INVENTORY_MONTHLY_SNAPSHOT", "DIM_PRODUCT", "T2_DOC_INVENTORY"],
        "employee_context": ["DIM_EMPLOYEE", "DIM_DEPARTMENT", "DIM_POSITION_LEVEL", "FACT_PAYROLL"],
        "document_render_context": ["renders/*", "T2_DOC_INVENTORY"],
        "report_context": ["reports/*"],
    }


REPO_ROOT = Path(__file__).resolve().parents[1]
DATABASE_TOOLS_DIR = REPO_ROOT / "database-tools"
DATA_DIR = REPO_ROOT / "data"
TABLES_DIR = DATA_DIR / "tables"
CODEX_PYTHON_SITE_PACKAGES = (
    Path.home()
    / ".cache"
    / "codex-runtimes"
    / "codex-primary-runtime"
    / "dependencies"
    / "python"
    / "Lib"
    / "site-packages"
)
if CODEX_PYTHON_SITE_PACKAGES.exists() and str(CODEX_PYTHON_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(CODEX_PYTHON_SITE_PACKAGES))
if str(DATABASE_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DATABASE_TOOLS_DIR))
os.environ.setdefault("DATA_DIR", str(DATA_DIR))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv(REPO_ROOT / ".env")

ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


DATABASE_TOOL_NAMES = [
    "postgres_describe_table",
    "postgres_sample_rows",
    "postgres_count_rows",
    "postgres_distinct_values",
    "postgres_aggregate",
    "postgres_group_by",
    "postgres_top_k",
    "postgres_time_series",
    "postgres_execute_readonly_sql",
    "domain_file_catalog_search",
    "domain_text_exact_search",
    "domain_hybrid_search",
    "domain_policy_resolver",
    "domain_entity_resolver",
]

DATABASE_TOOL_DESCRIPTIONS = {
    "postgres_describe_table": 'inspect columns and types. Arguments: {"table": "FACT_SALES", "schema": "public"}. Required: table. Optional: schema.',
    "postgres_sample_rows": 'sample rows. Arguments: {"table": "FACT_SALES", "schema": "public", "limit": 5}. Required: table. Optional: schema, limit.',
    "postgres_count_rows": 'count rows with optional equality filters. Arguments: {"table": "DIM_CUSTOMER", "schema": "public", "filters": {"is_b2b": true}}. Required: table. Optional: schema, filters.',
    "postgres_distinct_values": 'list values and counts for a column. Arguments: {"table": "DIM_CUSTOMER", "column": "loyalty_tier", "schema": "public", "limit": 100}. Required: table, column. Optional: schema, limit.',
    "postgres_aggregate": 'aggregate one column. Arguments: {"table": "FACT_SALES", "column": "net_total_thb", "op": "sum", "schema": "public", "filters": {"payment_status": "paid"}}. Required: table, column. Optional: op ("sum", "avg", "min", "max", "count"), schema, filters.',
    "postgres_group_by": 'group and aggregate. Arguments: {"table": "FACT_SALES", "group_columns": ["branch_code"], "metric_column": "net_total_thb", "op": "sum", "schema": "public", "filters": {}, "limit": 100}. Required: table, group_columns. Optional: metric_column, op ("sum", "avg", "min", "max", "count"), schema, filters, limit. For row counts use {"metric_column": "*", "op": "count"}; do not invent count columns such as interaction_count.',
    "postgres_top_k": 'return top rows ordered by a column. Arguments: {"table": "FACT_SALES", "order_by": "net_total_thb", "schema": "public", "descending": true, "filters": {}, "limit": 10}. Required: table, order_by. Optional: schema, descending, filters, limit.',
    "postgres_time_series": 'aggregate by time grain. Arguments: {"table": "FACT_SALES", "date_column": "business_event_date", "metric_column": "net_total_thb", "grain": "month", "op": "sum", "schema": "public", "filters": {}, "limit": 500}. Required: table, date_column, metric_column. Optional: grain ("day", "week", "month", "quarter", "year"), op, schema, filters, limit. Use only for explicit trends or time-series questions.',
    "postgres_execute_readonly_sql": 'execute trusted read-only SELECT/WITH SQL. Arguments: {"sql": "SELECT COUNT(*) AS n FROM public.\\"FACT_SALES\\"", "limit": 200}. Required: sql. Optional: limit.',
    "domain_file_catalog_search": 'find matching files under the data catalog. Arguments: {"query": "chat_line_oa NT-LT-001", "limit": 30}. Required: query. Optional: limit.',
    "domain_text_exact_search": 'exact keyword search in local text files. Arguments: {"query": "NT-LT-001", "limit": 20}. Required: query. Optional: limit.',
    "domain_hybrid_search": 'hybrid local retrieval over text corpus. Arguments: {"query": "refund policy", "top_k": 10}. Required: query. Optional: top_k.',
    "domain_policy_resolver": 'resolve an active policy variable for a date. Arguments: {"policy_variable": "point_earning_rate_per_thb", "as_of_date": "2025-04-01", "schema": "public"}. Required: policy_variable, as_of_date. Optional: schema.',
    "domain_entity_resolver": 'resolve product/vendor/customer/employee/branch IDs. Arguments: {"query": "NT-LT-001", "entity_type": "product", "limit": 10, "schema": "public"}. Required: query. Optional: entity_type ("any", "sku", "product", "vendor", "customer", "employee", "branch"), limit, schema.',
}

ALWAYS_ALLOWED_DATABASE_TOOLS = [
    "postgres_describe_table",
    "postgres_sample_rows",
    "postgres_count_rows",
    "postgres_distinct_values",
    "postgres_aggregate",
    "postgres_group_by",
    "postgres_top_k",
    "postgres_execute_readonly_sql",
    "domain_file_catalog_search",
    "domain_text_exact_search",
    "domain_hybrid_search",
    "domain_policy_resolver",
    "domain_entity_resolver",
]

POSTGRES_REQUIRED_TOOLS = {
    "postgres_describe_table",
    "postgres_sample_rows",
    "postgres_count_rows",
    "postgres_distinct_values",
    "postgres_aggregate",
    "postgres_group_by",
    "postgres_top_k",
    "postgres_time_series",
    "postgres_execute_readonly_sql",
    "domain_policy_resolver",
    "domain_entity_resolver",
}

TIME_SERIES_KEYWORDS = [
    "trend",
    "time series",
    "timeseries",
    "timeline",
    "monthly",
    "daily",
    "weekly",
    "yearly",
    "by month",
    "by day",
    "by week",
    "by year",
    "รายวัน",
    "รายเดือน",
    "รายสัปดาห์",
    "รายปี",
    "แนวโน้ม",
    "เทรนด์",
    "เปรียบเทียบเดือน",
    "เปรียบเทียบปี",
]

DATABASE_TOOL_PROMPT = """
Shared database tools from database-tools:

PostgreSQL tools:
- postgres_describe_table: inspect columns and types. Arguments: {"table": "FACT_SALES", "schema": "public"}.
- postgres_sample_rows: sample rows. Arguments: {"table": "FACT_SALES", "schema": "public", "limit": 5}.
- postgres_count_rows: count rows. Arguments: {"table": "DIM_CUSTOMER", "schema": "public", "filters": {"is_b2b": true}}.
- postgres_distinct_values: list values and counts. Arguments: {"table": "DIM_CUSTOMER", "column": "loyalty_tier", "schema": "public", "limit": 100}.
- postgres_aggregate: aggregate one column. Arguments: {"table": "FACT_SALES", "column": "net_total_thb", "op": "sum", "schema": "public", "filters": {}}.
- postgres_group_by: group and aggregate. Arguments: {"table": "FACT_SALES", "group_columns": ["branch_code"], "metric_column": "net_total_thb", "op": "sum", "schema": "public", "filters": {}, "limit": 100}.
- postgres_top_k: top rows ordered by a column. Arguments: {"table": "FACT_SALES", "order_by": "net_total_thb", "schema": "public", "descending": true, "filters": {}, "limit": 10}.
- postgres_execute_readonly_sql: execute trusted SELECT/WITH SQL only. Arguments: {"sql": "SELECT COUNT(*) AS n FROM public.\"FACT_SALES\"", "limit": 200}.

FahMai domain tools:
- domain_file_catalog_search: Arguments: {"query": "chat_line_oa NT-LT-001", "limit": 30}.
- domain_text_exact_search: Arguments: {"query": "NT-LT-001", "limit": 20}.
- domain_hybrid_search: Arguments: {"query": "refund policy", "top_k": 10}.
- domain_policy_resolver: Arguments: {"policy_variable": "point_earning_rate_per_thb", "as_of_date": "2025-04-01", "schema": "public"}.
- domain_entity_resolver: Arguments: {"query": "NT-LT-001", "entity_type": "product", "limit": 10, "schema": "public"}.

Use get_*_context tools to choose tables and sources first. Prefer PostgreSQL tools for structured data and retrieval tools for documents. Use only read-only SQL. Do not repeat the same tool call with the same arguments.

Tool call output format:
- Return only one raw JSON object when calling a tool.
- The first character must be { and the last character must be }.
- Do not use markdown, code fences, ```json, or explanatory text around tool JSON.
- Do not repeat any tool call with the same arguments. Specialized domain_* tools are one-shot helpers; if one is insufficient, switch to PostgreSQL or retrieval tools.
"""


def select_database_tool_names(question):
    q = str(question or "").lower()
    names = list(ALWAYS_ALLOWED_DATABASE_TOOLS)
    if any(keyword in q for keyword in TIME_SERIES_KEYWORDS):
        names.insert(names.index("postgres_execute_readonly_sql"), "postgres_time_series")
    return names


def build_database_tool_prompt(question=None):
    names = select_database_tool_names(question)
    lines = [
        "Allowed database/retrieval tools for this question:",
        *[f"- {name}: {DATABASE_TOOL_DESCRIPTIONS[name]}" for name in names],
        "",
        "Do not call tools that are not listed above for this question.",
        "Use get_*_context tools first. Then use only the allowed tools listed above.",
        'For direct PostgreSQL tool arguments, pass "table" as the exact table name from context, e.g. "FACT_SALES", and pass "schema": "public" separately.',
        'Use real column names from context. For dates, prefer "business_event_date" unless the context shows a more specific date column.',
        'For range filters use operator objects, e.g. {"business_event_date": {"gte": "2025-01-01", "lte": "2025-12-31"}}.',
        "CRITICAL TOOL OUTPUT FORMAT:",
        "- Return only one raw JSON object when calling a tool.",
        "- Tool-call output MUST start with { and MUST end with }.",
        "- The first character of the whole response must be {.",
        "- The last character of the whole response must be }.",
        "- Do not use markdown.",
        "- Do not use code fences.",
        "- Do not wrap JSON in ```json or ```.",
        "- Do not write any text before or after the JSON object.",
        "FINAL REMINDER: when calling a tool, your entire response must start with { and end with }.",
    ]
    return "\n".join(lines)


def _json(data):
    return json.dumps(data, ensure_ascii=False, default=str)


_CONTEXT_RETRIEVED = False
_LAST_DATABASE_TOOL_CALL = None
_DATABASE_TOOL_CALL_COUNTS = {}
_ONE_SHOT_DATABASE_TOOLS = set()


def _tool_call_signature(tool_name, arguments):
    return (
        tool_name,
        json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str),
    )


def _normalize_identifier(value):
    return value


def _split_table_reference(value):
    text = str(value or "").strip()
    match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\."([^"]+)"$', text)
    if match:
        return match.group(2), match.group(1)
    match = re.match(r'^"([^"]+)"\."([^"]+)"$', text)
    if match:
        return match.group(2), match.group(1)
    match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$', text)
    if match:
        return match.group(2), match.group(1)
    match = re.match(r'^"([^"]+)"$', text)
    if match:
        return match.group(1), None
    return value, None


def _safe_schema(schema):
    schema = str(schema or "public")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", schema):
        raise ValueError(f"Unsafe schema: {schema}")
    return schema


def _normalize_database_arguments(tool_name, arguments):
    normalized = dict(arguments or {})
    if tool_name.startswith("postgres_") and tool_name != "postgres_execute_readonly_sql":
        if "table" in normalized:
            table, schema = _split_table_reference(normalized["table"])
            normalized["table"] = table
            if schema and not normalized.get("schema"):
                normalized["schema"] = schema
        for key in ("table", "schema", "column", "date_column", "metric_column", "order_by"):
            if key in normalized:
                normalized[key] = _normalize_identifier(normalized[key])
        if "group_columns" in normalized and isinstance(normalized["group_columns"], list):
            normalized["group_columns"] = [_normalize_identifier(col) for col in normalized["group_columns"]]
        if "filters" in normalized and isinstance(normalized["filters"], dict):
            normalized["filters"] = {_normalize_identifier(k): v for k, v in normalized["filters"].items()}
    return normalized


def _reset_database_tool_loop_guard():
    global _LAST_DATABASE_TOOL_CALL, _DATABASE_TOOL_CALL_COUNTS
    _LAST_DATABASE_TOOL_CALL = None
    _DATABASE_TOOL_CALL_COUNTS = {}


def reset_tool_state():
    global _CONTEXT_RETRIEVED
    _CONTEXT_RETRIEVED = False
    _reset_database_tool_loop_guard()


def _mark_context_retrieved():
    global _CONTEXT_RETRIEVED
    _CONTEXT_RETRIEVED = True
    _reset_database_tool_loop_guard()


@lru_cache(maxsize=2)
def _database_registry(include_qdrant=True):
    from domain_tools import build_domain_registry

    return build_domain_registry(include_qdrant=include_qdrant)


@lru_cache(maxsize=1)
def _local_domain_tools():
    from domain_tools import FahMaiDomainTools

    return FahMaiDomainTools(pg=None, qdrant=None, data_dir=DATA_DIR)


@lru_cache(maxsize=1)
def _postgres_tools():
    from database_tools import DbToolConfig, PostgresDatabaseTools

    cfg = DbToolConfig.from_env()
    return PostgresDatabaseTools(cfg.pg_dsn)


@lru_cache(maxsize=1)
def _qdrant_tools():
    from database_tools import DbToolConfig, QdrantDatabaseTools

    cfg = DbToolConfig.from_env()
    return QdrantDatabaseTools(cfg.qdrant_url, cfg.qdrant_api_key, cfg.qdrant_collection, cfg.embed_model)


def _get_table_columns(table, schema="public"):
    csv_path = TABLES_DIR / f"{table}.csv"
    if csv_path.exists():
        try:
            with open(csv_path, encoding="utf-8") as f:
                header = f.readline().strip()
            if header:
                return set(header.split(","))
        except Exception:
            pass

    if not os.getenv("PG_DSN"):
        return None

    try:
        res = json.loads(_postgres_tools().describe_table(table, schema))
    except Exception:
        return None
    if not res.get("ok"):
        return None
    return {row.get("column_name") for row in res.get("rows", [])}


COLUMN_ALIASES = {
    "transaction_date": ["business_event_date", "posting_date", "date_iso"],
    "txn_date": ["business_event_date", "posting_date"],
    "sales_date": ["business_event_date", "posting_date"],
    "order_date": ["business_event_date", "posting_date"],
    "date": ["business_event_date", "date_iso", "posting_date"],
    "month": ["business_event_date", "month_end_date", "date_iso"],
    "received_date": ["payment_received_date", "business_event_date", "posting_date"],
    "paid_date": ["payment_received_date", "business_event_date", "posting_date"],
    "payment_date": ["payment_received_date", "business_event_date", "posting_date"],
    "vendor": ["vendor_id"],
    "vendor_name": ["vendor_id"],
    "vendor_name_en": ["vendor_id"],
    "vendor_name_th": ["vendor_id"],
    "product": ["sku_id"],
    "product_name": ["sku_id"],
    "sku": ["sku_id"],
    "branch": ["branch_code"],
    "branch_name": ["branch_code"],
    "employee": ["employee_id"],
    "employee_name": ["employee_id"],
    "customer": ["customer_id"],
    "customer_name": ["customer_id"],
    "is_b2b": ["customer_type"],
    "customer_type": ["is_b2b"],
    "units_sold": ["quantity"],
    "unit_sold": ["quantity"],
    "units": ["quantity"],
    "points_earned": ["points_delta"],
    "earned_points": ["points_delta"],
    "redemption_count": ["redemption_id"],
    "redemptions": ["redemption_id"],
    "campaign": ["campaign_id", "promo_campaign_id"],
    "campaign_code": ["campaign_id", "promo_campaign_id"],
    "promo": ["campaign_id", "promo_campaign_id"],
    "promo_code": ["campaign_id", "promo_campaign_id"],
    "promo_campaign": ["campaign_id", "promo_campaign_id"],
}


COUNT_METRIC_ALIASES = {
    "count",
    "n",
    "row_count",
    "record_count",
    "interaction_count",
    "transaction_count",
    "redemption_count",
    "redemptions",
}


DIMENSION_GROUP_JOINS = {
    ("FACT_SALES", "branch_type"): {
        "dimension_table": "DIM_BRANCH",
        "join_key": "branch_code",
        "dimension_column": "branch_type",
    },
    ("FACT_SALES", "branch_name"): {
        "dimension_table": "DIM_BRANCH",
        "join_key": "branch_code",
        "dimension_column": "name_th",
    },
}


def _is_count_metric(value):
    value_l = str(value or "").lower()
    return value_l.endswith("_count") or value_l in COUNT_METRIC_ALIASES


def _repair_count_metric_arguments(tool_name, arguments):
    repaired = dict(arguments)
    count_ops = {"sum", "avg", "min", "max"}

    if tool_name == "postgres_aggregate":
        op = str(repaired.get("op", "sum")).lower()
        if op in count_ops and _is_count_metric(repaired.get("column")):
            repaired["op"] = "count"
            repaired["column"] = "*"

    if tool_name == "postgres_group_by":
        op = str(repaired.get("op", "count")).lower()
        if op in count_ops and _is_count_metric(repaired.get("metric_column")):
            repaired["op"] = "count"
            repaired["metric_column"] = "*"

    return repaired


def _repair_column_name(column, columns):
    if column in {None, "*"} or not columns:
        return column
    if column in columns:
        return column
    for candidate in COLUMN_ALIASES.get(str(column).lower(), []):
        if candidate in columns:
            return candidate
    return column


def _repair_postgres_columns(tool_name, arguments):
    if tool_name == "postgres_execute_readonly_sql":
        return arguments
    table = arguments.get("table")
    schema = arguments.get("schema", "public")
    if not table:
        return arguments
    columns = _get_table_columns(table, schema)
    if not columns:
        return arguments

    repaired = dict(arguments)
    for key in ("column", "date_column", "metric_column", "order_by"):
        if key in repaired:
            repaired[key] = _repair_column_name(repaired[key], columns)
    if isinstance(repaired.get("group_columns"), list):
        repaired["group_columns"] = [_repair_column_name(col, columns) for col in repaired["group_columns"]]
    if isinstance(repaired.get("filters"), dict):
        fixed_filters = {}
        for key, value in repaired["filters"].items():
            fixed_key = _repair_column_name(key, columns)
            if str(key).lower() == "is_b2b" and fixed_key == "customer_type":
                if value is True or str(value).lower() == "true":
                    fixed_filters[fixed_key] = "B2B"
                elif value is False or str(value).lower() == "false":
                    fixed_filters[fixed_key] = {"ne": "B2B"}
                else:
                    fixed_filters[fixed_key] = value
            elif str(key).lower() == "customer_type" and fixed_key == "is_b2b":
                if str(value).upper() == "B2B":
                    fixed_filters[fixed_key] = True
                elif isinstance(value, dict) and str(value.get("ne", "")).upper() == "B2B":
                    fixed_filters[fixed_key] = False
                else:
                    fixed_filters[fixed_key] = value
            else:
                fixed_filters[fixed_key] = value
        repaired["filters"] = fixed_filters
    return repaired


def _ident(name):
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(name)):
        raise ValueError(f"Unsafe identifier: {name}")
    return f'"{name}"'


def _table_ref(table, schema="public"):
    return f'{_ident(schema)}.{_ident(table)}'


def _where_from_filters(filters, table_alias=None):
    if not filters:
        return "", []
    parts = []
    params = []
    op_map = {
        ">": ">",
        ">=": ">=",
        "gt": ">",
        "gte": ">=",
        "min": ">=",
        "from": ">=",
        "<": "<",
        "<=": "<=",
        "lt": "<",
        "lte": "<=",
        "max": "<=",
        "to": "<=",
        "!=": "<>",
        "<>": "<>",
        "ne": "<>",
        "neq": "<>",
    }
    for column, value in filters.items():
        col = f"{table_alias}.{_ident(column)}" if table_alias else _ident(column)
        if isinstance(value, dict):
            if "op" in value and "value" in value:
                op_l = str(value.get("op")).lower()
                op_value = value.get("value")
                if op_l in op_map:
                    parts.append(f"{col} {op_map[op_l]} %s")
                    params.append(op_value)
                elif op_l in {"eq", "="}:
                    parts.append(f"{col} = %s")
                    params.append(op_value)
                elif op_l == "between" and isinstance(op_value, (list, tuple)) and len(op_value) == 2:
                    parts.append(f"{col} BETWEEN %s AND %s")
                    params.extend(op_value)
                elif op_l == "in" and isinstance(op_value, (list, tuple)) and op_value:
                    parts.append(f"{col} IN ({', '.join(['%s'] * len(op_value))})")
                    params.extend(op_value)
                elif op_l in {"contains", "ilike"}:
                    parts.append(f"{col} ILIKE %s")
                    params.append(f"%{op_value}%")
                else:
                    raise ValueError(f"Unsupported filter operator for {column}: {value.get('op')}")
                continue

            for op, op_value in value.items():
                op_l = str(op).lower()
                if op_l in op_map:
                    parts.append(f"{col} {op_map[op_l]} %s")
                    params.append(op_value)
                elif op_l in {"eq", "="}:
                    parts.append(f"{col} = %s")
                    params.append(op_value)
                elif op_l == "between" and isinstance(op_value, (list, tuple)) and len(op_value) == 2:
                    parts.append(f"{col} BETWEEN %s AND %s")
                    params.extend(op_value)
                elif op_l == "in" and isinstance(op_value, (list, tuple)) and op_value:
                    parts.append(f"{col} IN ({', '.join(['%s'] * len(op_value))})")
                    params.extend(op_value)
                elif op_l in {"contains", "ilike"}:
                    parts.append(f"{col} ILIKE %s")
                    params.append(f"%{op_value}%")
                else:
                    raise ValueError(f"Unsupported filter operator for {column}: {op}")
        elif isinstance(value, (list, tuple)) and value:
            parts.append(f"{col} IN ({', '.join(['%s'] * len(value))})")
            params.extend(value)
        else:
            parts.append(f"{col} = %s")
            params.append(value)
    return " WHERE " + " AND ".join(parts), params


def _call_enhanced_postgres_tool(tool_name, arguments):
    pg = _postgres_tools()
    table = arguments.get("table")
    schema = arguments.get("schema", "public")
    filters = arguments.get("filters")

    if tool_name == "postgres_count_rows" and filters:
        where, params = _where_from_filters(filters)
        return _json(pg._query(f"SELECT COUNT(*) AS n FROM {_table_ref(table, schema)}{where}", params))

    if tool_name == "postgres_aggregate" and filters:
        op = str(arguments.get("op", "sum")).lower()
        allowed = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX", "count": "COUNT"}
        fn = allowed.get(op)
        if not fn:
            return _json({"ok": False, "error": f"Unsupported aggregate op: {op}"})
        column = arguments.get("column", "*")
        expr = "*" if fn == "COUNT" and column == "*" else _ident(column)
        where, params = _where_from_filters(filters)
        return _json(pg._query(f"SELECT {fn}({expr}) AS value FROM {_table_ref(table, schema)}{where}", params))

    if tool_name == "postgres_group_by":
        group_columns = arguments.get("group_columns") or []
        metric_column = arguments.get("metric_column", "*")
        op = str(arguments.get("op", "count")).lower()
        limit = int(arguments.get("limit", 100))
        allowed = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX", "count": "COUNT"}
        fn = allowed.get(op)
        if not fn:
            return _json({"ok": False, "error": f"Unsupported aggregate op: {op}"})

        if len(group_columns) == 1:
            join = DIMENSION_GROUP_JOINS.get((str(table), str(group_columns[0])))
            if join:
                fact_ref = _table_ref(table, schema)
                dim_ref = _table_ref(join["dimension_table"], schema)
                group_expr = f'd.{_ident(join["dimension_column"])}'
                metric = "*" if fn == "COUNT" and metric_column == "*" else f'f.{_ident(metric_column)}'
                where, params = _where_from_filters(filters, table_alias="f")
                sql = f"""
                SELECT {group_expr} AS {group_columns[0]}, {fn}({metric}) AS value
                FROM {fact_ref} f
                JOIN {dim_ref} d ON f.{_ident(join["join_key"])} = d.{_ident(join["join_key"])}
                {where}
                GROUP BY {group_expr}
                ORDER BY value DESC
                """
                return _json(pg._query(sql, params, limit=limit))

        groups = ", ".join(_ident(c) for c in group_columns)
        metric = "*" if fn == "COUNT" and metric_column == "*" else _ident(metric_column)
        where, params = _where_from_filters(filters)
        sql = f"""
        SELECT {groups}, {fn}({metric}) AS value
        FROM {_table_ref(table, schema)}
        {where}
        GROUP BY {groups}
        ORDER BY value DESC
        """
        return _json(pg._query(sql, params, limit=limit))

    if tool_name == "postgres_top_k" and filters:
        order_by = arguments.get("order_by")
        descending = bool(arguments.get("descending", True))
        limit = int(arguments.get("limit", 10))
        direction = "DESC" if descending else "ASC"
        where, params = _where_from_filters(filters)
        return _json(pg._query(f"SELECT * FROM {_table_ref(table, schema)}{where} ORDER BY {_ident(order_by)} {direction}", params, limit=limit))

    if tool_name == "postgres_time_series":
        date_column = arguments.get("date_column")
        metric_column = arguments.get("metric_column")
        grain = str(arguments.get("grain", "month"))
        op = str(arguments.get("op", "sum")).lower()
        limit = int(arguments.get("limit", 500))
        allowed_grains = {"day", "week", "month", "quarter", "year"}
        allowed_ops = {"sum": "SUM", "avg": "AVG", "count": "COUNT", "min": "MIN", "max": "MAX"}
        if grain not in allowed_grains:
            return _json({"ok": False, "error": f"Unsupported grain: {grain}"})
        fn = allowed_ops.get(op)
        if not fn:
            return _json({"ok": False, "error": f"Unsupported op: {op}"})
        metric = "*" if fn == "COUNT" and metric_column == "*" else _ident(metric_column)
        where, params = _where_from_filters(filters)
        sql = f"""
        SELECT date_trunc('{grain}', {_ident(date_column)}::date) AS period,
               {fn}({metric}) AS value
        FROM {_table_ref(table, schema)}
        {where}
        GROUP BY period
        ORDER BY period
        """
        return _json(pg._query(sql, params, limit=limit))

    return None


def _repair_group_by_arguments(arguments):
    if not os.getenv("PG_DSN"):
        return arguments

    table = arguments.get("table")
    schema = arguments.get("schema", "public")
    metric = arguments.get("metric_column", "*")
    op = str(arguments.get("op", "count")).lower()
    group_columns = arguments.get("group_columns") or []
    metric_l = str(metric).lower()

    if metric_l == "payment_received_date" and any(str(col).lower() == "customer_id" for col in group_columns):
        repaired = dict(arguments)
        repaired["table"] = "FACT_SALES"
        return repaired

    if table == "FACT_SALES" and any(str(col).lower() == "sku_id" for col in group_columns):
        if metric_l in {
            "units_sold",
            "unit_sold",
            "units",
            "quantity",
            "line_total_thb",
            "line_discount_thb",
            "unit_price_thb",
            "*",
        }:
            repaired = dict(arguments)
            repaired["table"] = "FACT_SALES_LINE_ITEM"
            return repaired

    if not table or metric == "*":
        return arguments

    columns = _get_table_columns(table, schema)
    if not columns or metric in columns:
        return arguments

    metric_l = str(metric).lower()
    if op in {"sum", "avg", "min", "max"} and _is_count_metric(metric_l):
        repaired = dict(arguments)
        repaired["op"] = "count"
        repaired["metric_column"] = "*"
        return repaired

    return arguments


def call_database_tool(tool_name, arguments=None, include_qdrant=None):
    global _LAST_DATABASE_TOOL_CALL, _DATABASE_TOOL_CALL_COUNTS

    if tool_name not in DATABASE_TOOL_NAMES:
        return _json({"ok": False, "error": f"Unsupported database tool: {tool_name}"})

    if not _CONTEXT_RETRIEVED:
        return _json(
            {
                "ok": False,
                "tool": tool_name,
                "error": "context_required",
                "message": "You must call the relevant get_*_context tool before any PostgreSQL, Qdrant, or domain database tool. Stop this tool path and call a context tool first.",
            }
        )

    arguments = _normalize_database_arguments(tool_name, arguments or {})
    arguments = _repair_count_metric_arguments(tool_name, arguments)
    arguments = _repair_postgres_columns(tool_name, arguments)
    signature = _tool_call_signature(tool_name, arguments)
    if signature == _LAST_DATABASE_TOOL_CALL:
        return _json(
            {
                "ok": False,
                "tool": tool_name,
                "error": "duplicate_tool_call",
                "message": "This exact database tool call was already made after the current context retrieval. Do not retry it. Answer from the previous result or switch to a different, more specific tool.",
            }
        )

    call_count = _DATABASE_TOOL_CALL_COUNTS.get(tool_name, 0)
    if call_count and tool_name in _ONE_SHOT_DATABASE_TOOLS:
        return _json(
            {
                "ok": False,
                "tool": tool_name,
                "error": "tool_already_used",
                "message": "This discovery/domain tool was already used after the current context retrieval. Do not call it again for this question. Use the previous result, switch to a more specific PostgreSQL/retrieval tool, or answer that evidence is insufficient.",
            }
        )

    _LAST_DATABASE_TOOL_CALL = signature
    _DATABASE_TOOL_CALL_COUNTS[tool_name] = call_count + 1

    if tool_name in {"domain_file_catalog_search", "domain_text_exact_search", "domain_hybrid_search"}:
        local_tools = _local_domain_tools()
        try:
            if tool_name == "domain_file_catalog_search":
                return local_tools.file_catalog_search(**arguments)
            if tool_name == "domain_text_exact_search":
                return local_tools.text_exact_search(**arguments)
            return local_tools.hybrid_search(**arguments)
        except Exception as e:
            return _json({"ok": False, "tool": tool_name, "error": f"{type(e).__name__}: {e}"})

    if tool_name in POSTGRES_REQUIRED_TOOLS and not os.getenv("PG_DSN"):
        return _json(
            {
                "ok": False,
                "tool": tool_name,
                "error": "pg_dsn_required",
                "message": "PG_DSN is not set. Configure the PostgreSQL connection string before using structured database tools.",
            }
        )

    if tool_name == "postgres_group_by":
        arguments = _repair_group_by_arguments(arguments)
        arguments = _repair_postgres_columns(tool_name, arguments)
        if (
            arguments.get("table") == "FACT_LOYALTY_LEDGER"
            and arguments.get("metric_column") == "points_delta"
            and str((arguments.get("op") or "")).lower() == "sum"
        ):
            filters = dict(arguments.get("filters") or {})
            filters.setdefault("points_delta", {"gt": 0})
            arguments = {**arguments, "filters": filters}

    enhanced_result = _call_enhanced_postgres_tool(tool_name, arguments)
    if enhanced_result is not None:
        return enhanced_result

    if tool_name == "domain_policy_resolver":
        try:
            pg = _postgres_tools()
            schema = _safe_schema(arguments.get("schema", "public"))
            policy_variable = str(arguments.get("policy_variable", ""))
            as_of_date = str(arguments.get("as_of_date", ""))
            sql = f"""
            SELECT policy_version_id, policy_class, policy_variable, scope_filter,
                   value_numeric, value_text, policy_value_table_ref, effective_date, end_date
            FROM "{schema}"."DIM_POLICY_VERSION"
            WHERE policy_variable = %s
              AND effective_date::date <= %s::date
              AND (NULLIF(end_date, '') IS NULL OR NULLIF(end_date, '')::date > %s::date)
            ORDER BY effective_date::date DESC
            LIMIT 1
            """
            return _json(pg._query(sql, [policy_variable, as_of_date, as_of_date], limit=1))
        except Exception as e:
            return _json({"ok": False, "tool": tool_name, "error": f"{type(e).__name__}: {e}"})

    if tool_name == "domain_entity_resolver":
        try:
            pg = _postgres_tools()
            schema = _safe_schema(arguments.get("schema", "public"))
            query = str(arguments.get("query", ""))
            entity_type = str(arguments.get("entity_type", "any")).lower()
            limit = int(arguments.get("limit", 10))
            q = f"%{query}%"
            entity_queries = []

            if entity_type in {"any", "sku", "product"}:
                entity_queries.append(
                    (
                        "product",
                        f"""
                        SELECT sku_id AS id, brand_family, category, subcategory, msrp_thb
                        FROM "{schema}"."DIM_PRODUCT"
                        WHERE sku_id ILIKE %s OR brand_family ILIKE %s OR category ILIKE %s OR subcategory ILIKE %s
                        LIMIT %s
                        """,
                        [q, q, q, q, limit],
                    )
                )
            if entity_type in {"any", "vendor"}:
                entity_queries.append(
                    (
                        "vendor",
                        f"""
                        SELECT vendor_id AS id, name_th, name_en, category, role
                        FROM "{schema}"."DIM_VENDOR"
                        WHERE vendor_id ILIKE %s OR name_th ILIKE %s OR name_en ILIKE %s OR category ILIKE %s
                        LIMIT %s
                        """,
                        [q, q, q, q, limit],
                    )
                )
            if entity_type in {"any", "customer"}:
                entity_queries.append(
                    (
                        "customer",
                        f"""
                        SELECT customer_id AS id, first_name_th, last_name_th, first_name_en, last_name_en,
                               province, region, customer_type, loyalty_tier
                        FROM "{schema}"."DIM_CUSTOMER"
                        WHERE customer_id ILIKE %s OR first_name_th ILIKE %s OR last_name_th ILIKE %s
                           OR first_name_en ILIKE %s OR last_name_en ILIKE %s OR email ILIKE %s
                        LIMIT %s
                        """,
                        [q, q, q, q, q, q, limit],
                    )
                )
            if entity_type in {"any", "employee"}:
                entity_queries.append(
                    (
                        "employee",
                        f"""
                        SELECT employee_id AS id, first_name_th, last_name_th, first_name_en, last_name_en,
                               dept_code, position_title, position_level, canon_role_label
                        FROM "{schema}"."DIM_EMPLOYEE"
                        WHERE employee_id ILIKE %s OR first_name_th ILIKE %s OR last_name_th ILIKE %s
                           OR first_name_en ILIKE %s OR last_name_en ILIKE %s OR position_title ILIKE %s
                           OR canon_role_label ILIKE %s
                        LIMIT %s
                        """,
                        [q, q, q, q, q, q, q, limit],
                    )
                )
            if entity_type in {"any", "branch"}:
                entity_queries.append(
                    (
                        "branch",
                        f"""
                        SELECT branch_code AS id, name_th, name_en, branch_type
                        FROM "{schema}"."DIM_BRANCH"
                        WHERE branch_code ILIKE %s OR name_th ILIKE %s OR name_en ILIKE %s OR branch_type ILIKE %s
                        LIMIT %s
                        """,
                        [q, q, q, q, limit],
                    )
                )

            rows = []
            for kind, sql, params in entity_queries:
                res = pg._query(sql, params, limit=limit)
                if not res.get("ok"):
                    return _json({"ok": False, "tool": tool_name, "error": res.get("error"), "sql": res.get("sql")})
                for row in res.get("rows", []):
                    rows.append({"entity_type": kind, **row})
                    if len(rows) >= limit:
                        break
                if len(rows) >= limit:
                    break
            return _json({"ok": True, "rows": rows})
        except Exception as e:
            return _json({"ok": False, "tool": tool_name, "error": f"{type(e).__name__}: {e}"})

    if include_qdrant is None:
        include_qdrant = os.getenv("PATHUMMA_INCLUDE_QDRANT", "0").lower() not in {"0", "false", "no"}

    try:
        registry = _database_registry(include_qdrant=include_qdrant)
        return registry.call_tool(tool_name, arguments)
    except Exception as e:
        return _json({"ok": False, "tool": tool_name, "error": f"{type(e).__name__}: {e}"})


def _make_database_tool(tool_name):
    def _tool(**arguments):
        return call_database_tool(tool_name, arguments)

    _tool.__name__ = tool_name
    _tool.__doc__ = f"Call shared database-tools tool `{tool_name}`."
    return _tool


def get_database_tool_functions():
    return {name: _make_database_tool(name) for name in DATABASE_TOOL_NAMES}


def _read_csv_header(table_name):
    csv_path = TABLES_DIR / f"{table_name}.csv"
    if not csv_path.exists():
        return None

    try:
        with csv_path.open("r", encoding="utf-8-sig") as f:
            header = f.readline().strip()
    except UnicodeDecodeError:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip()

    if not header:
        return None
    return [column.strip() for column in header.split(",") if column.strip()]


def _context_source_prompt(source):
    if source.endswith("/*"):
        return f"- {source}: file collection under data/{source}; use domain_file_catalog_search, domain_text_exact_search, or domain_hybrid_search for discovery and evidence."

    if "/" in source:
        return f"- {source}: document source under data/{source}; use domain_file_catalog_search, domain_text_exact_search, or domain_hybrid_search with this source."

    columns = _read_csv_header(source)
    if columns:
        return f"- {source}: table. Columns: {', '.join(columns)}"

    return f"- {source}: table/source listed in this context pack. Inspect with query tools before answering."


def _build_context_prompt(context_name):
    _mark_context_retrieved()

    sources = CONTEXT_PACKS.get(context_name)
    if not sources:
        return _json({"ok": False, "error": f"Unknown context: {context_name}", "available_contexts": sorted(CONTEXT_PACKS)})

    source_lines = "\n".join(_context_source_prompt(source) for source in sources)
    return f"""Context pack: {context_name}

Allowed sources:
{source_lines}

Instructions:
- Use only the tables, document sources, and file patterns listed above for this context.
- For structured facts, call only the PostgreSQL tools listed in the runtime allowed-tool section after this context call.
- For long text, chats, memos, emails, reports, renders, or document evidence, call domain_file_catalog_search, domain_text_exact_search, or domain_hybrid_search after this context call.
- Never repeat the same tool call with the same arguments. If a tool result is not enough, switch to a more specific PostgreSQL or retrieval tool.
- Do not invent table names, column names, joins, file paths, or business rules beyond this returned context.
- In raw SQL, quote uppercase table names exactly as listed, e.g. public."FACT_SALES"; direct table-tool arguments should also use the exact table name from this context.
- If a needed table, column, source, or join is missing from this context, say what is missing or call another relevant context tool first.
"""



def get_sales_context (**kwargs) :
    # print("using sales_context")
    return _build_context_prompt("sales_context")

def get_customer_cs_context (**kwargs) :
    # print("using customer_cs_context")
    return _build_context_prompt("customer_cs_context")

def get_policy_context (**kwargs) :
    #print("using policy_context")
    return _build_context_prompt("policy_context")

def get_vendor_shipping_context (**kwargs) :
    #print("using vendor_shipping_context")
    return _build_context_prompt("vendor_shipping_context")

def get_finance_bank_context (**kwargs) :
    #print("using finance_bank_context")
    return _build_context_prompt("finance_bank_context")

def get_inventory_context (**kwargs) :
    #print("using inventory_context")
    return _build_context_prompt("inventory_context")

def get_employee_context (**kwargs) :
    #print("using employee_context")
    return _build_context_prompt("employee_context")

def get_document_render_context (**kwargs) :
    #print("using document_render_context")
    return _build_context_prompt("document_render_context")

def get_report_context (**kwargs) :
    #print("using report_context")
    return _build_context_prompt("report_context")

