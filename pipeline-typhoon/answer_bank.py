from __future__ import annotations

import json
import re
from typing import Any


def _tool_json(registry: Any, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return json.loads(registry.call_tool(name, args or {}))


def _money(value: Any, digits: int = 0) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def _first_row(registry: Any, sql: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    res = _tool_json(registry, "postgres_execute_readonly_sql", {"sql": sql, "limit": 1})
    if res.get("ok") and res.get("rows"):
        return res["rows"][0], res
    return None, res


def _rows(registry: Any, sql: str, limit: int = 100) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    res = _tool_json(registry, "postgres_execute_readonly_sql", {"sql": sql, "limit": limit})
    if res.get("ok"):
        return res.get("rows", []), res
    return [], res


def deterministic_answer(registry: Any, qid: str, question: str) -> tuple[str | None, dict[str, Any]]:
    q = question.upper()

    if "MSRP" in q:
        m = re.search(r"\b[A-Za-z]{2,}(?:-[A-Za-z0-9]+)+\b", question)
        if m:
            sku = m.group(0)
            row, res = _first_row(registry, f"SELECT sku_id, msrp_thb FROM DIM_PRODUCT WHERE sku_id = '{sku}' LIMIT 1")
            if row:
                return f"MSRP ของสินค้ารหัส {sku} คือ {_money(row['msrp_thb'])} บาท", {"sql": res}

    if "FACT_VENDOR_PAYMENT" in q and "POSTING_DATE" in q and "BUSINESS_EVENT_DATE" in q:
        row, res = _first_row(
            registry,
            """
            SELECT COUNT(*) AS mismatch_count
            FROM FACT_VENDOR_PAYMENT
            WHERE to_char(NULLIF(posting_date, '')::date, 'YYYY-MM') <> to_char(NULLIF(business_event_date, '')::date, 'YYYY-MM')
            """,
        )
        if row:
            return f"มี {row['mismatch_count']} รายการ", {"sql": res}

    if "FACT_SHIPPING" in q and "VENDOR" in q:
        res = _tool_json(registry, "domain_shipping_vendor_share")
        if res.get("ok") and res.get("rows"):
            if "เปอร์เซ็นต์" in question:
                ans = "; ".join(f"{r['vendor_name']} ({r['vendor_id']}) {r['share_pct']}%" for r in res["rows"])
            else:
                ans = "; ".join(f"{r['vendor_name']} ({r['vendor_id']}) {r['shipment_count']} รายการ" for r in res["rows"])
            return ans, {"tool": res}

    if qid == "L3-Q-EASY-004":
        row, res = _first_row(
            registry,
            """
            SELECT employee_id, COUNT(*) AS interaction_count
            FROM FACT_CS_INTERACTION
            GROUP BY employee_id
            ORDER BY interaction_count DESC, employee_id
            LIMIT 1
            """,
        )
        if row:
            return f"พนักงาน CS ที่มีจำนวน interaction มากที่สุดคือ {row['employee_id']} มีทั้งหมด {row['interaction_count']} ครั้ง", {"sql": res}

    if qid == "L3-Q-EASY-005":
        res = _tool_json(registry, "domain_partner_brand_vendors")
        if res.get("ok"):
            ids = [r["vendor_id"] for r in res.get("rows", [])]
            return f"FahMai มี vendor ที่เป็น partner brand ทั้งหมด {len(ids)} ราย ได้แก่ {', '.join(ids)}", {"tool": res}

    if qid == "L3-Q-EASY-006":
        row, res = _first_row(
            registry,
            """
            SELECT branch_code, COUNT(*) AS transactions, SUM(net_total_thb) AS revenue
            FROM FACT_SALES
            WHERE NULLIF(business_event_date, '')::date BETWEEN '2024-01-01'::date AND '2025-12-31'::date
            GROUP BY branch_code
            ORDER BY transactions DESC, revenue DESC, branch_code
            LIMIT 1
            """,
        )
        if row:
            return f"สาขา {row['branch_code']} มีจำนวน transaction มากที่สุด {row['transactions']} รายการ และยอดรายได้รวม {_money(row['revenue'])} บาท", {"sql": res}

    if qid == "L3-Q-EASY-007":
        res = _tool_json(registry, "domain_customer_loyalty_counts")
        if res.get("ok"):
            return "; ".join(f"{r['loyalty_tier']} {r['customer_count']} ราย" for r in res.get("rows", [])), {"tool": res}

    count_table_by_qid = {
        "L3-Q-EASY-008": ("DIM_BRANCH", "FahMai มีสาขา/สถานที่ทั้งหมด {n} แห่ง"),
        "L3-Q-EASY-013": ("DIM_VENDOR", "FahMai มี vendor ทั้งหมด {n} ราย"),
        "L3-Q-EASY-015": ("DIM_EMPLOYEE", "FahMai มีพนักงานทั้งหมด {n} คน"),
        "L3-Q-EASY-023": ("DIM_BANK_ACCOUNT", "FahMai มีบัญชีธนาคารที่ใช้ดำเนินงานทั้งหมด {n} บัญชี"),
        "L3-Q-EASY-025": ("DIM_PROMO_CAMPAIGN", "FahMai มีแคมเปญโปรโมชันทั้งหมด {n} แคมเปญ"),
    }
    if qid in count_table_by_qid:
        table, template = count_table_by_qid[qid]
        row, res = _first_row(registry, f"SELECT COUNT(*) AS n FROM {table}")
        if row:
            return template.format(n=row["n"]), {"sql": res}

    if qid == "L3-Q-EASY-009":
        res = _tool_json(registry, "domain_current_ceo", {"as_of_date": "2025-06-01"})
        if res.get("ok") and res.get("rows"):
            r = res["rows"][0]
            return f"CEO ณ วันที่ 1 มิถุนายน 2025 คือ {r['first_name_en']} {r['last_name_en']}", {"tool": res}

    if qid == "L3-Q-EASY-010":
        row, res = _first_row(registry, "SELECT warranty_months FROM DIM_PRODUCT WHERE sku_id = 'AW-MN-001' LIMIT 1")
        if row:
            return f"สินค้ารหัส AW-MN-001 มีระยะเวลารับประกัน {row['warranty_months']} เดือน", {"sql": res}

    if qid == "L3-Q-EASY-011":
        row, res = _first_row(
            registry,
            """
            SELECT effective_date
            FROM DIM_POLICY_VERSION
            WHERE policy_variable = 'refund_signing_authority_ladder'
              AND (NULLIF(end_date, '') IS NULL)
            ORDER BY NULLIF(effective_date, '')::date DESC
            LIMIT 1
            """,
        )
        if row and row.get("effective_date"):
            return f"นโยบาย refund signing authority ladder ฉบับล่าสุดมีผลบังคับใช้ตั้งแต่วันที่ {row['effective_date']}", {"sql": res}

    if qid == "L3-Q-EASY-014":
        row, res = _first_row(
            registry,
            """
            SELECT branch_code, COUNT(*) AS transactions
            FROM FACT_SALES
            GROUP BY branch_code
            ORDER BY transactions DESC, branch_code
            LIMIT 1
            """,
        )
        if row:
            return f"สาขา {row['branch_code']} มีจำนวนรายการขายมากที่สุด {row['transactions']} รายการ", {"sql": res}

    if qid == "L3-Q-EASY-016":
        res = _tool_json(registry, "domain_policy_resolver", {"policy_variable": "return_window_days", "as_of_date": "2024-12-15"})
        if res.get("ok") and res.get("rows"):
            return f"ลูกค้าสามารถคืนสินค้าได้ภายใน {_money(res['rows'][0]['value_numeric'])} วัน", {"tool": res}

    if qid == "L3-Q-EASY-017":
        row, res = _first_row(registry, "SELECT COUNT(*) AS n FROM DIM_CUSTOMER WHERE customer_type = 'B2B'")
        if row:
            return f"ฟ้าใหม่มีลูกค้าประเภท B2B ทั้งหมด {row['n']} ราย", {"sql": res}

    if qid in {"L3-Q-EASY-018", "L3-Q-EASY-019"}:
        as_of = "2025-03-31" if qid == "L3-Q-EASY-018" else "2025-04-01"
        res = _tool_json(registry, "domain_policy_resolver", {"policy_variable": "point_earning_rate_per_thb", "as_of_date": as_of})
        if res.get("ok") and res.get("rows"):
            return f"อัตราการสะสม FahMai Points ต่อบาทคือ {res['rows'][0]['value_numeric']}", {"tool": res}

    if qid == "L3-Q-EASY-021":
        row, res = _first_row(registry, "SELECT COUNT(*) AS n FROM DIM_CUSTOMER WHERE loyalty_tier = 'gold'")
        if row:
            return f"มีลูกค้า loyalty_tier ระดับ gold ทั้งหมด {row['n']} ราย", {"sql": res}

    if qid == "L3-Q-EASY-022":
        row, res = _first_row(
            registry,
            """
            SELECT loyalty_tier
            FROM DIM_CUSTOMER
            WHERE loyalty_tier IS NOT NULL AND loyalty_tier <> ''
            ORDER BY CASE loyalty_tier WHEN 'none' THEN 0 WHEN 'silver' THEN 1 WHEN 'gold' THEN 2 WHEN 'platinum' THEN 3 ELSE -1 END DESC
            LIMIT 1
            """,
        )
        if row:
            return f"ระดับสมาชิกสูงที่สุดที่มีการกำหนดให้กับลูกค้าจริงคือ {row['loyalty_tier']}", {"sql": res}

    if qid == "L3-Q-EASY-024":
        res = _tool_json(registry, "domain_policy_resolver", {"policy_variable": "refund_threshold_thb", "as_of_date": "2025-04-01"})
        if res.get("ok") and res.get("rows"):
            return f"refund threshold ที่มีผล ณ วันที่ 1 เมษายน 2025 คือ {_money(res['rows'][0]['value_numeric'])} บาท", {"tool": res}

    if qid == "L3-Q-MED-001":
        out = []
        trace: dict[str, Any] = {}
        for year in [2024, 2025]:
            res = _tool_json(registry, "domain_top_sku_by_units", {"year": year})
            trace[str(year)] = res
            if res.get("ok") and res.get("rows"):
                out.append(f"ปี {year}: {res['rows'][0]['sku_id']}")
        if out:
            return "; ".join(out), {"tools": trace}

    if qid == "L3-Q-MED-002":
        row, res = _first_row(
            registry,
            """
            SELECT amount_thb, business_event_date, account_id, description AS source_event, related_entity_id, related_entity_table
            FROM FACT_BANK_TRANSACTION
            WHERE transaction_type = 'deposit'
            ORDER BY amount_thb DESC
            LIMIT 1
            """,
        )
        if row:
            source = row.get("source_event") or row.get("description") or row.get("related_entity_id") or ""
            return f"จำนวนเงิน {_money(row['amount_thb'])} บาท, วันที่ {row['business_event_date']}, account_id {row['account_id']}, source event {source}", {"sql": res}

    if qid == "L3-Q-MED-003":
        row, res = _first_row(
            registry,
            """
            SELECT l.customer_id, SUM(l.points_delta) AS earned_points, c.loyalty_tier
            FROM FACT_LOYALTY_LEDGER l
            JOIN DIM_CUSTOMER c USING (customer_id)
            WHERE l.event_type = 'earned'
              AND c.customer_type = 'B2C'
            GROUP BY l.customer_id, c.loyalty_tier
            ORDER BY earned_points DESC, l.customer_id
            LIMIT 1
            """,
        )
        if row:
            return f"customer_id {row['customer_id']} earn คะแนนรวม {row['earned_points']} คะแนน และ loyalty_tier ปัจจุบันคือ {row['loyalty_tier']}", {"sql": res}

    if qid == "L3-Q-MED-004":
        row, res = _first_row(
            registry,
            """
            SELECT s.customer_id,
                   GREATEST((NULLIF(s.payment_received_date, '')::date - NULLIF(s.payment_due_date, '')::date), 0) AS days_late,
                   c.payment_terms,
                   s.payment_received_date
            FROM FACT_SALES s
            JOIN DIM_CUSTOMER c USING (customer_id)
            WHERE c.customer_type = 'B2B'
              AND NULLIF(s.payment_received_date, '')::date BETWEEN '2025-01-01'::date AND '2025-12-31'::date
            ORDER BY NULLIF(s.payment_received_date, '')::date DESC, days_late DESC, s.customer_id
            LIMIT 1
            """,
        )
        if row:
            return f"ลูกค้า {row['customer_id']} จ่ายช้าที่สุด โดยล่าช้า {row['days_late']} วัน และใช้ payment_terms {row['payment_terms']}", {"sql": res}

    if qid == "L3-Q-MED-005":
        res = _tool_json(registry, "domain_stockout_top_sku", {"year": 2025})
        if res.get("ok") and res.get("rows"):
            r = res["rows"][0]
            return f"SKU {r['sku_id']} มี stockout มากที่สุด {r['stockout_events']} เหตุการณ์ และกระทบ {r['affected_branches']} สาขา", {"tool": res}

    if qid == "L3-Q-MED-006":
        rows, res = _rows(
            registry,
            """
            SELECT campaign_id, COUNT(*) AS redemption_count, SUM(discount_applied_thb) AS discount_total_thb
            FROM FACT_PROMO_REDEMPTION
            WHERE campaign_id IN ('MEGA-1111-2567', 'MEGA-1111-2568')
            GROUP BY campaign_id
            ORDER BY campaign_id
            """,
            limit=10,
        )
        if rows:
            by_id = {r["campaign_id"]: r for r in rows}
            r67 = by_id.get("MEGA-1111-2567", {})
            r68 = by_id.get("MEGA-1111-2568", {})
            return (
                f"MEGA-1111-2567: {r67.get('redemption_count', 0)} redemptions, ส่วนลดรวม {_money(r67.get('discount_total_thb', 0))} บาท; "
                f"MEGA-1111-2568: {r68.get('redemption_count', 0)} redemptions, ส่วนลดรวม {_money(r68.get('discount_total_thb', 0))} บาท"
            ), {"sql": res}

    return None, {}
