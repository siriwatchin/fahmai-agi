#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# นำเข้าคลาสไลบรารีจัดการฐานข้อมูล PostgreSQL จากไฟล์ postgres_search_tool.py
from postgres_search_tool import FahMaiDBAgentKit

# โหลดค่าคอนฟิกูเรชันสภาพแวดล้อมจากไฟล์ .env
load_dotenv()


def get_typhoon_api_key() -> str:
    """
    โหลด API key โดยรองรับทั้งชื่อมาตรฐานใหม่และชื่อเดิมในไฟล์ .env
    """
    api_key = (
        os.getenv("TYPHOON_API_KEY")
        or os.getenv("APIKEY")
        or os.getenv("TYPHOON_API_KEY_1")
        or os.getenv("TYPHOON_API_KEY_2")
        or os.getenv("TYPHOON_API_KEY_3")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Missing Typhoon API key. Set TYPHOON_API_KEY in .env "
            "(or legacy APIKEY for backward compatibility)."
        )
    return api_key


def get_typhoon_base_url() -> str:
    """
    OpenAI-compatible clients expect a base URL such as:
    https://api.opentyphoon.ai/v1

    Some .env files store the full chat completions endpoint, so normalize it.
    """
    url = os.getenv("TYPHOON_API_BASE_URL") or os.getenv("TYPHOON_API_URL") or "https://api.opentyphoon.ai/v1"
    url = url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url

# 1. เริ่มต้นใช้งานออบเจกต์คลังเครื่องมือฐานข้อมูลเพื่อดึง Granular Tool Schema
db_kit = FahMaiDBAgentKit()
tools_schema = db_kit.get_openai_tool_schema()

# 2. ตั้งค่า Client เชื่อมต่อไปยัง Cloud API Endpoint (รองรับสถาปัตยกรรม OpenAI-Compatible)
client = OpenAI(
    base_url=get_typhoon_base_url(),
    api_key=get_typhoon_api_key(),
)

# 3. กำหนดชื่อโมเดลเป้าหมายที่เปิดบริการใช้งาน
MODEL_NAME = os.getenv("TYPHOON_MODEL", "typhoon-v2.5-30b-a3b-instruct").strip()


def run_agent_pipeline(user_question: str):
    """
    ฟังก์ชันควบคุมท่อส่งข้อมูลประมวลผลกระบวนการ Tool Calling แบบแยกฟังก์ชัน (Granular Function)
    """
    print(f"คำถามจากผู้ใช้: {user_question}\n")
    
    messages = [
        {
            "role": "system", 
            "content": "You are an expert enterprise data analyst for FahMai. "
                       "You must select the most appropriate granular database tool to solve the query. "
                       "If querying FACT_SALES discounts, you must pass the date in YYYY-MM-DD format to 'filter_date' "
                       "to ensure correct bitemporal schema routing."
        },
        {"role": "user", "content": user_question}
    ]
    
    try:
        # ส่งชุดคำขอประมวลผลรอบแรก (First-Pass Inference) ไปยัง Cloud API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.0
        )
        
        response_message = response.choices[0].message
        
        # ตรวจสอบโครงสร้างคำสั่งเรียกใช้เครื่องมือ (Tool Calls)
        if response_message.tool_calls:
            print("สถานะ: ตรวจพบสัญญาณคำสั่ง Tool Calls จากระบบ API\n")
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"[Tool Call]: {function_name}")
                print(f"[Arguments]: {json.dumps(arguments, ensure_ascii=False, indent=2)}")
                
                # ทำการตรวจสอบและเรียกใช้งานฟังก์ชันภายในคลาสแบบพลวัต (Dynamic Mapping)
                if hasattr(db_kit, function_name):
                    target_function = getattr(db_kit, function_name)
                    
                    # ปฏิบัติตามคำสั่งคิวรีไปยังฐานข้อมูล PostgreSQL จริงผ่านการ Unpacking Keyword Arguments
                    tool_output = target_function(**arguments)
                else:
                    tool_output = json.dumps({"status": "error", "message": f"Function '{function_name}' not found."})
                
                print(f"[Database Output]: {tool_output}\n")
                
                # แนบผลลัพธ์เครื่องมือกลับเข้าสู่ประวัติข้อความ
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_output
                })
            
            # ส่งชุดประวัติกลับไปยัง Cloud API เพื่อประมวลผลสรุปคำตอบรอบสุดท้าย (Second-Pass Inference)
            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0
            )
            
            print(f"คำตอบสุดท้ายจาก Agent:\n{final_response.choices[0].message.content}")
            
        else:
            print(f"คำตอบโดยตรงจากโมเดล:\n{response_message.content}")
            
    except Exception as err:
        print(f"ระบบทำงานล้มเหลวเนื่องจากเกิดข้อผิดพลาด: {str(err)}")


if __name__ == "__main__":
    # ทดสอบส่งคำถามระบุเงื่อนไขเวลาเพื่อทดสอบระบบสลับเวอร์ชันคอลัมน์ส่วนลดอัตโนมัติ
    test_query = "ขอผลรวมของส่วนลดตาราง FACT_SALES สำหรับสาขา B-01 ในช่วงข้อมูลวันที่ 2025-08-22 ครับ"
    run_agent_pipeline(test_query)
