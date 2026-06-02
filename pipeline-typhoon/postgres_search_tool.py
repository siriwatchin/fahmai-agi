#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FahMai Enterprise Data-Agent Database Connectivity Library (Granular Function Version)
Authoritative Tool-Calling Provider for Relational PostgreSQL Operations
Updated: Production Database Endpoint Integration
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Tuple, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

# ตั้งค่าระบบตรวจจับและบันทึกเหตุการณ์ระบบ
logging.basicConfig(level=logging.INFO, format="%(asctime)s – %(levelname)s – %(message)s")
logger = logging.getLogger("PostgresSearchTool")


class FahMaiDBAgentKit:
    def __init__(self):
        """
        เริ่มต้นโมดูลไลบรารีและโหลดการตั้งค่าความปลอดภัยของคลังข้อมูล
        """
        # รายชื่อตารางธุรกรรมหลักที่อนุญาตให้ Agent เข้าถึงได้ (Data Layer Whitelist)
        self.ALLOWED_TABLES = {
            "FACT_VENDOR_PAYMENT", "FACT_SHIPPING", "FACT_CS_INTERACTION", 
            "FACT_SALES", "FACT_SALES_LINE_ITEM", "FACT_RETURN", "FACT_PROMO_REDEMPTION",
            "DIM_PRODUCT", "DIM_VENDOR", "DIM_POLICY_VERSION", "DIM_STORE", "DIM_CUSTOMER",
            "FACT_INVENTORY_MOVEMENT", "FACT_INVENTORY_MONTHLY_SNAPSHOT", "FACT_WARRANTY_CLAIM",
            "FACT_LOYALTY_LEDGER", "DIM_PROMO_CAMPAIGN", "DIM_BRANCH", "DIM_DEPARTMENT"
        }
        
        # กำหนดวันที่ตัดเปลี่ยนระบบขาย (Schema Cutover Date: 1 เมษายน 2568)
        self.CUTOVER_DATE_STR = "2025-04-01"

    def _get_connection(self):
        """
        สร้างการเชื่อมต่อเครือข่ายไปยัง PostgreSQL Server ตัวจริงตามสิทธิ์เข้าใช้งานที่ระบุ
        """
        return psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"), 
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432")
        )

    def _sanitize_identifier(self, identifier: str) -> str:
        """
        ทำความสะอาดชื่อตารางหรือชื่อคอลัมน์เพื่อป้องกัน SQL Injection ระดับ Identifier
        """
        if not identifier:
            return ""
        return re.sub(r"[^a-zA-Z0-9_]", "", identifier.strip())

    def _handle_sales_schema_routing(self, column_name: str, filter_date: Optional[str]) -> str:
        """
        ตรรกะจัดการ Schema Cutover อัตโนมัติสำหรับตาราง FACT_SALES
        ช่วยแปลงชื่อคอลัมน์ส่วนลดตามกรอบเวลาข้อมูลจริง (v1 vs v2)
        """
        clean_col = column_name.lower()
        if clean_col in ["discount_amt", "discount_total_thb", "discount"]:
            if filter_date and filter_date >= self.CUTOVER_DATE_STR:
                logger.info("Schema Cutover Match: Routing to v2 'discount_total_thb'")
                return "discount_total_thb"
            else:
                logger.info("Schema Cutover Match: Routing to v1 'discount_amt'")
                return "discount_amt"
        return column_name

    def _validate_table_access(self, target_table: str) -> str:
        """
        ระบบตรวจสอบความปลอดภัยของชื่อตาราง ป้องกันการละเมิดสิทธิ์เข้าถึง (Data Leak Guard)
        """
        clean_table = target_table.replace(".csv", "").strip().upper()
        if clean_table not in self.ALLOWED_TABLES or "EMPLOYEE" in clean_table or "PAYROLL" in clean_table:
            raise PermissionError("Access Denied. Unauthorized data layer access.")
        return self._sanitize_identifier(clean_table)

    def count_rows(self, target_table: str, filter_column: Optional[str] = None, filter_value: Optional[str] = None) -> str:
        """
        [Tool] นับจำนวนแถวในตารางตามเงื่อนไขที่กำหนด
        """
        try:
            safe_table = self._validate_table_access(target_table)
            safe_filter_col = self._sanitize_identifier(filter_column) if filter_column else None
            
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if safe_filter_col and filter_value:
                query = f"SELECT COUNT(*) as total_rows FROM {safe_table} WHERE {safe_filter_col} = %s"
                params = [filter_value]
            else:
                query = f"SELECT COUNT(*) as total_rows FROM {safe_table}"
                params = []
                
            cursor.execute(query, params)
            db_result = cursor.fetchall()
            cursor.close()
            conn.close()
            return json.dumps({"status": "success", "data": db_result}, default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def sum_column(self, target_table: str, target_column: str, filter_column: Optional[str] = None, filter_value: Optional[str] = None, filter_date: Optional[str] = None) -> str:
        """
        [Tool] คำนวณผลรวมของคอลัมน์ที่เป็นตัวเลข พร้อมรองรับตรรกะ Schema Cutover ยอดขายอัตโนมัติ
        """
        try:
            safe_table = self._validate_table_access(target_table)
            safe_target_col = self._sanitize_identifier(target_column)
            safe_filter_col = self._sanitize_identifier(filter_column) if filter_column else None
            
            # ตรวจสอบตรรกะการเปลี่ยนโครงสร้างข้อมูลยอดขายอัตโนมัติ (v1 vs v2)
            if safe_table == "FACT_SALES":
                safe_target_col = self._handle_sales_schema_routing(safe_target_col, filter_date)
                
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if safe_filter_col and filter_value:
                query = f"SELECT SUM({safe_target_col}::numeric) as sum_value FROM {safe_table} WHERE {safe_filter_col} = %s"
                params = [filter_value]
            else:
                query = f"SELECT SUM({safe_target_col}::numeric) as sum_value FROM {safe_table}"
                params = []
                
            cursor.execute(query, params)
            db_result = cursor.fetchall()
            cursor.close()
            conn.close()
            return json.dumps({"status": "success", "data": db_result}, default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def find_max(self, target_table: str, target_column: str, filter_column: Optional[str] = None, filter_value: Optional[str] = None) -> str:
        """
        [Tool] ค้นหาแถวข้อมูลที่มีค่าสูงสุดในคอลัมน์ที่กำหนด
        """
        try:
            safe_table = self._validate_table_access(target_table)
            safe_target_col = self._sanitize_identifier(target_column)
            safe_filter_col = self._sanitize_identifier(filter_column) if filter_column else None
            
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if safe_filter_col and filter_value:
                query = f"SELECT * FROM {safe_table} WHERE {safe_filter_col} = %s ORDER BY {safe_target_col}::numeric DESC LIMIT 1"
                params = [filter_value]
            else:
                query = f"SELECT * FROM {safe_table} ORDER BY {safe_target_col}::numeric DESC LIMIT 1"
                params = []
                
            cursor.execute(query, params)
            db_result = cursor.fetchall()
            cursor.close()
            conn.close()
            return json.dumps({"status": "success", "data": db_result}, default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def fetch_sample(self, target_table: str) -> str:
        """
        [Tool] ดึงตัวอย่างข้อมูลจำนวน 5 แถวแรกจากตารางเพื่อส่องดูโครงสร้างข้อมูลอภิพันธุ์
        """
        try:
            safe_table = self._validate_table_access(target_table)
            
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = f"SELECT * FROM {safe_table} LIMIT 5"
            cursor.execute(query)
            db_result = cursor.fetchall()
            cursor.close()
            conn.close()
            return json.dumps({"status": "success", "data": db_result}, default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def get_openai_tool_schema(self) -> List[Dict[str, Any]]:
        """
        ส่งออกโครงสร้างคำอธิบายฟังก์ชันแยกส่วน (Granular JSON Schema) 
        ทำให้โมเดลขนาดเล็ก เช่น Pathumma หรือ Qwen เรียกใช้งานได้ง่ายและถูกต้องแม่นยำสูงขึ้น
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "count_rows",
                    "description": "Count the total number of rows in an authorized core table based on an optional filtration condition.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_table": {"type": "string", "description": "The target table name (e.g., 'FACT_SALES')."},
                            "filter_column": {"type": "string", "description": "The optional table column name used for filtering rows."},
                            "filter_value": {"type": "string", "description": "The exact criteria string to filter by."}
                        },
                        "required": ["target_table"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sum_column",
                    "description": "Calculate the numeric summation of a target column. Includes automated bitemporal schema routing features for FACT_SALES table fields.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_table": {"type": "string", "description": "The target table name (e.g., 'FACT_SALES')."},
                            "target_column": {"type": "string", "description": "The numeric column name targeted for summation (e.g., 'net_total_thb', 'discount_total_thb')."},
                            "filter_column": {"type": "string", "description": "The optional column name used for filtering."},
                            "filter_value": {"type": "string", "description": "The criteria string to match the filter column."},
                            "filter_date": {"type": "string", "description": "Context date string in YYYY-MM-DD format to route between old and new schema versions automatically."}
                        },
                        "required": ["target_table", "target_column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_max",
                    "description": "Retrieve the record containing the maximum numeric value of the specified column within an authorized table.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_table": {"type": "string", "description": "The target table name (e.g., 'FACT_SHIPPING')."},
                            "target_column": {"type": "string", "description": "The numeric column name to evaluate (e.g., 'shipping_fee_thb')."},
                            "filter_column": {"type": "string", "description": "The optional column name used for filtering."},
                            "filter_value": {"type": "string", "description": "The exact textual criteria value."}
                        },
                        "required": ["target_table", "target_column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_sample",
                    "description": "Fetch a sample of 5 rows from the specified database table to inspect the available schema structure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_table": {"type": "string", "description": "The target table name (e.g., 'FACT_CS_INTERACTION')."}
                        },
                        "required": ["target_table"]
                    }
                }
            }
        ]