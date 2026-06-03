# Current Qwen2.5 Pipeline - สถานะล่าสุด

เอกสารนี้สรุป pipeline ปัจจุบันที่ใช้บน B200 สำหรับ FahMai Enterprise Data Agentic Showdown รวมถึง audit log, security mode, และ AI harness ที่มีใน repo ตอนนี้

## Runtime หลัก

ตำแหน่งบน B200:

```text
repo:        ~/fahmai-agi/pipeline-qwen2.5
workspace:   ~/bank500
data root:   ~/scamper_house
venv:        ~/venvs/qwen35
```

โมเดล generation หลัก:

```text
~/bank500/qwen35/models/Qwen2.5-7B-Instruct
```

โมเดล embedding สำหรับ vector search:

```text
~/bank500/qwen35/models/bge-m3
```

Qdrant local:

```text
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=fahmai_rag_bge
vector_name=dense
embedding_dim=1024
```

SQL backend ปัจจุบัน:

```text
SQL_BACKEND=duckdb
```

เหตุผล: Postgres host ภายนอกเคย timeout จาก B200 container ส่วน DuckDB โหลด public data lake จาก `~/scamper_house` ได้ครบและเร็วกว่าใน runtime นี้

## Pipeline หลักสำหรับคะแนน

ไฟล์:

```text
agentic_best_integrated_qdrant.py
```

แนวทาง:

```text
question
  -> deterministic rule / SQL fast path
  -> ถ้ายังตอบไม่ได้: TF-IDF document search + schema search
  -> hard SQL rules อีกครั้งพร้อม context
  -> Qdrant BGE-M3 vector retrieval
  -> Qwen2.5-7B final answer เฉพาะข้อที่ยังต้อง synthesize
  -> token/time/audit/debug outputs
```

หลักการสำคัญ:

- ใช้ SQL/rule ก่อนเสมอสำหรับคำถามตัวเลข ตาราง count, group by, join, date, policy
- ไม่เรียก TF-IDF/Qdrant ก่อนถ้า deterministic rule ตอบได้
- ใช้ Qwen เป็น final synthesizer ไม่ใช่เครื่องคิดเลขหลัก
- ใช้ `GEN_DO_SAMPLE=0` เพื่อคำตอบนิ่งและเหมาะกับ keyword grader
- ใช้ `MODEL_LOAD_STRATEGY=cuda_direct` บน B200 เมื่อโหลดผ่านแล้ว

Command ที่แนะนำ:

```bash
cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate

export MODEL_PATH="$HOME/bank500/qwen35/models/Qwen2.5-7B-Instruct"
export FAHMAI_SRC_ROOT="$HOME/scamper_house"
export WORK_ROOT="$HOME/bank500"
export SQL_BACKEND="duckdb"

export QDRANT_URL="http://127.0.0.1:6333"
export QDRANT_API_KEY="<qdrant-key>"
export QDRANT_COLLECTION="fahmai_rag_bge"
export EMBED_MODEL="$HOME/bank500/qwen35/models/bge-m3"

export MODEL_LOAD_STRATEGY="cuda_direct"
export DISABLE_TRANSFORMERS_ALLOCATOR_WARMUP="1"
export GEN_DO_SAMPLE="0"
export DOC_TOP_K="8"
export QDRANT_TOP_K="8"
export GEN_MAX_INPUT_TOKENS="7000"
export TORCH_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"

python agentic_best_integrated_qdrant.py --limit 100 --skip-qdrant-preload
```

Output ของแต่ละ run:

```text
~/bank500/output/<RUN_ID>/
  best_results.csv
  best_submission.csv
  best_debug.json
  best_token_usage.csv
  best_token_summary.json
  best_llm_audit.jsonl
```

## Source + Security Pipeline

ไฟล์:

```text
agentic_sourced_secure.py
```

ตัวนี้เป็น wrapper แยก ไม่แทน pipeline scoring หลัก ใช้เมื่อต้องการคำตอบพร้อม source และ security metadata

เพิ่มจาก pipeline หลัก:

- `sources`: ระบุที่มาจาก deterministic rule, SQL table, schema, TF-IDF document, Qdrant hit
- `security`: ตรวจ prompt injection, reasoning trace leakage, right of access, cross-source privilege
- debug ถูก redacted เป็นค่า default
- ถ้าต้องการ raw observation เฉพาะ local debugging ให้ตั้ง `INCLUDE_RAW_DEBUG=1`

Command:

```bash
python agentic_sourced_secure.py --limit 100 --skip-qdrant-preload
```

Output:

```text
~/bank500/output/<RUN_ID>_sourced_secure/
  sourced_secure_results.csv
  sourced_secure_submission.csv
  sourced_secure_records.jsonl
  sourced_secure_debug.json
  sourced_secure_token_usage.csv
  sourced_secure_summary.json
  sourced_secure_llm_audit.jsonl
```

ไฟล์ที่สำคัญ:

- `sourced_secure_records.jsonl`: ต่อ 1 บรรทัดต่อ 1 คำถาม มี `answer`, `sources`, `security`
- `sourced_secure_debug.json`: redacted debug ไม่ dump raw context ยาว
- `sourced_secure_llm_audit.jsonl`: audit เฉพาะ LLM calls

Access role:

```bash
python agentic_sourced_secure.py --limit 10 --access-role public_competition
python agentic_sourced_secure.py --limit 10 --access-role restricted_viewer
```

`public_competition` คือโหมด Kaggle data lake ปัจจุบัน ส่วน `restricted_viewer` เป็น smoke-test role ที่ deny finance/HR domain เพื่อทดสอบ right-of-access

## FastAPI Pipeline

ไฟล์:

```text
api_server.py
```

Endpoint:

```text
GET  /health
POST /api/v1/chat
POST /api/v2/chat
```

Request:

```json
{
  "data": {
    "question": "วันนี้วันอะไร"
  }
}
```

Response:

```json
{
  "data": {
    "answer": "วันพุธ"
  }
}
```

API ใช้ pipeline หลักเหมือน batch แต่มี answer cache สำหรับ load test:

- `ENABLE_API_CACHE=1`
- `API_PRELOAD_ANSWERS=1`
- โหลดคำตอบจาก run ล่าสุดใน `~/bank500/output/<RUN_ID>/best_results.csv`
- ถ้าคำถามซ้ำกับ 100 ข้อที่ precompute แล้ว จะตอบจาก memory ไม่แตะ GPU

Command:

```bash
cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate

export API_OUTPUT_DIR="$HOME/bank500"
export API_PORT="8888"
export ENABLE_API_CACHE="1"
export API_PRELOAD_ANSWERS="1"

uvicorn api_server:app --host 0.0.0.0 --port 8888
```

API outputs:

```text
~/bank500/api_requests.jsonl
~/bank500/api_token_usage.csv
~/bank500/api_token_summary.json
~/bank500/api_llm_audit.jsonl
```

## Guardrail API Integration

มี guardrail API แยกที่ตรวจ prompt-injection / attack text ได้ตาม spec:

```text
POST http://127.0.0.1:8000/predict
GET  http://127.0.0.1:8000/health
```

pipeline API หลักรองรับ guardrail แล้วผ่าน env:

```bash
export GUARDRAIL_URL="http://127.0.0.1:8000"
export GUARDRAIL_MODEL="model"
export GUARDRAIL_THRESHOLD="0.75"
export GUARDRAIL_ACTION="audit_only"
export GUARDRAIL_FAIL_CLOSED="0"
```

โหมดที่แนะนำ:

- `GUARDRAIL_ACTION=audit_only`: เหมาะกับ Kaggle/backtest เพราะข้อ injection ต้องตอบแบบต้านคำสั่งฝัง ไม่ใช่ hard reject เสมอไป
- `GUARDRAIL_ACTION=reject`: เหมาะกับ production API ถ้า guardrail บอก `is_attack=true` จะตอบ refusal ทันที ไม่ส่งต่อให้ Qwen
- `GUARDRAIL_FAIL_CLOSED=1`: ถ้า guardrail ล่มให้ reject ไปเลย เหมาะกับ production ที่เน้น safety

ตัวอย่าง run แบบ secure API:

```bash
# terminal 1: guardrail server
# รัน guardrail FastAPI ของทีมให้ฟังที่ port 8000
curl -s http://127.0.0.1:8000/health

# terminal 2: FahMai answer API
cd ~/fahmai-agi/pipeline-qwen2.5
source ~/venvs/qwen35/bin/activate

export GUARDRAIL_URL="http://127.0.0.1:8000"
export GUARDRAIL_ACTION="reject"
export GUARDRAIL_THRESHOLD="0.75"

uvicorn api_server:app --host 0.0.0.0 --port 8888
```

`GET /health` ของ FahMai API จะแสดง:

```text
guardrail_enabled
guardrail_url
guardrail_action
guardrail_fail_closed
```

## Audit Log ปัจจุบัน

มี 4 ระดับ

### 1. Token/time audit

ไฟล์:

```text
best_token_usage.csv
best_token_summary.json
api_token_usage.csv
api_token_summary.json
sourced_secure_token_usage.csv
sourced_secure_summary.json
```

เก็บ:

- `qid`
- `stage`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `seconds`
- generation params เช่น `do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`

### 2. LLM audit log

ไฟล์:

```text
best_llm_audit.jsonl
api_llm_audit.jsonl
sourced_secure_llm_audit.jsonl
```

เก็บต่อทุก LLM call:

- timestamp
- `qid`
- `stage`
- `model_path`
- `prompt_hash`
- `raw_answer_hash`
- `sanitized_answer_hash`
- token counts
- generation params
- redacted `prompt_preview`
- redacted `raw_answer_preview`
- redacted `sanitized_answer_preview`

ค่า default ไม่เก็บ prompt/raw answer เต็ม เพื่อกัน data leakage ผ่าน reasoning trace หรือ retrieved context

ถ้าต้องการ full raw prompt สำหรับ local debugging เท่านั้น:

```bash
export LLM_AUDIT_INCLUDE_PROMPT=1
```

คำเตือน: ห้าม commit/share output ที่เปิด `LLM_AUDIT_INCLUDE_PROMPT=1` เพราะอาจมี retrieved context, internal snippets, หรือ secret-like strings

### 3. Observation/debug audit

ไฟล์:

```text
best_debug.json
sourced_secure_debug.json
```

`best_debug.json` ของ pipeline หลักเก็บ observation เยอะกว่า เหมาะ debug accuracy ภายใน

`sourced_secure_debug.json` ถูก redacted default:

- keys ที่ใช้
- SQL hash
- จำนวน document/Qdrant/schema hits
- rule ที่ fire

### 4. API request audit

ไฟล์:

```text
api_requests.jsonl
```

เก็บ:

- timestamp
- id/question
- answer
- seconds
- sql backend
- qdrant enabled
- observation

## Security Controls

### Prompt Injection

มี 3 ชั้น:

1. `SYSTEM_PROMPT` สั่งให้ ignore embedded instruction และตอบจาก evidence เท่านั้น
2. deterministic injection rules สำหรับ `L3-Q-INJ-*` หลายข้อใน `hard_sql_answer`
3. `agentic_sourced_secure.py` ตรวจ pattern ทั้ง question และ retrieved context แล้วใส่ flag:

```text
prompt_injection_question
prompt_injection_retrieved_context
```

### Data Leakage Through Reasoning Trace

แนวทางปัจจุบัน:

- ไม่ output `<think>`
- `sanitize_answer()` ตัด marker เช่น `OBSERVATION`, `SQL_result`, `document_search`
- LLM audit default เก็บ hash/preview ไม่เก็บ prompt เต็ม
- sourced-secure debug default เป็น redacted

### Right Of Access

ปัจจุบัน competition mode คือ public data lake:

```text
ACCESS_ROLE=public_competition
```

ใน sourced-secure มี smoke-test role:

```text
ACCESS_ROLE=restricted_viewer
```

role นี้ deny finance/HR domain เพื่อทดสอบ access refusal

### Cross Source Privilege

หลักการ:

```text
SQL/rule > schema > Qdrant/TF-IDF retrieved text > LLM prior
```

retrieved text จากเอกสารหรือ Qdrant ไม่มีสิทธิ์ override:

- system prompt
- SQL result
- deterministic rules
- table facts

sourced-secure จะ flag:

```text
cross_source_privilege_checked
```

เมื่อคำตอบใช้ทั้ง SQL/rule และ low-trust retrieved text

## AI Harness ที่มี

### Batch harness

ไฟล์:

```text
agentic_best_integrated_qdrant.py
```

ทำหน้าที่:

- รัน 100 questions
- วัด per-question seconds
- วัด LLM token/time
- save submission/debug/audit
- ใช้เป็น offline accuracy/speed harness

### API/load-test harness

ไฟล์:

```text
api_server.py
```

ทำหน้าที่:

- serve `/api/v1/chat` และ `/api/v2/chat`
- preload answer cache จาก batch output
- วัด API request/token/audit
- เหมาะกับ load test 8 นาที ถ้าคำถามซ้ำกับชุดที่ precompute

### Source/security harness

ไฟล์:

```text
agentic_sourced_secure.py
```

ทำหน้าที่:

- บังคับ response มี source attribution
- สร้าง security report ต่อคำถาม
- ตรวจ prompt injection / trace leak / access / cross-source
- เหมาะกับ audit demo หรือ production-readiness review

### Database/domain tool harness

โฟลเดอร์:

```text
database-tools/
```

เครื่องมือสำคัญ:

- `domain_evidence_pack`: รวม schema + file + retrieval + injection evidence
- `domain_prompt_injection_detector`: ตรวจ pattern prompt injection
- `domain_refusal_checker`: ตรวจ refusal shape ให้เข้ากับ grader
- `domain_answer_verifier`: ตรวจคำตอบเบื้องต้น เช่น empty answer, bad refusal, prompt injection ที่ไม่ปฏิเสธ
- `domain_hybrid_search`: exact + vector search
- `domain_entity_resolver`: resolve SKU/vendor/customer/employee/branch

นี่คือ harness ฝั่ง tool/agent ที่เพื่อนหรือโมเดลอื่นใช้ซ้ำได้

## สถานะความแม่น

จุดแข็ง:

- EASY/MED ที่เป็น SQL/rule เร็วและค่อนข้างนิ่ง
- injection หลายข้อกันได้
- token ลดลงเพราะ SQL-first path ไม่เรียก LLM ถ้าไม่จำเป็น
- Qdrant local ใช้ได้จริงกับ BGE-M3

จุดอ่อน:

- HARD/XHARD ที่ต้อง reconcile หลาย source ยังต้องเพิ่ม deterministic SQL rules
- บางข้อ Qwen 7B สังเคราะห์จาก context ยาวแล้วหลุด/ตัดกลาง
- OCR/document-heavy ควรใช้ sourced-secure เพื่อตรวจว่า evidence มาจากไหนก่อนเชื่อคำตอบ
- right-of-access ตอนนี้เป็น smoke-test ไม่ใช่ RBAC production เต็ม

## สิ่งที่ควรทำต่อ

1. เพิ่ม deterministic rules ให้ HARD/XHARD ที่พลาดชัด เช่น `HARD-011`, `HARD-014`, `XHARD-010`, `XHARD-012`, `XHARD-014`
2. เพิ่ม `/api/v3/chat` ถ้าต้องการ API response ที่มี `sources` และ `security`
3. เพิ่ม evaluator เทียบ ground truth แบบ local เพื่อแยก pass/fail ต่อข้อ
4. เพิ่ม role policy จริงถ้า production ต้องมี user/tenant/access scope
