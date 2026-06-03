# FahMai Local Typhoon 7B

ชุดนี้เป็น local runner/API สำหรับโมเดลเล็กกว่า 30B โดยยังใช้ pipeline เดิม:

- endpoint เดิม: `/api/v1/chat`
- tools เดิม: PostgreSQL/domain tools/Qdrant ตาม config
- request body เดิม
- output/report/debug เหมือนชุด local เดิม

Default model:

```text
typhoon-ai/typhoon2-qwen2.5-7b-instruct
```

Backup model:

```text
Qwen/Qwen2.5-7B-Instruct
```

## Install / Download

```bash
cd /root/workspace/scamper_house/typhoon_testpipeline/fahmai-agi/fahmai-agi
pip install -r pipeline-typhoon-local-7b/requirements-local-7b.txt
huggingface-cli download typhoon-ai/typhoon2-qwen2.5-7b-instruct
```

ถ้าใช้ CLI ใหม่:

```bash
hf download typhoon-ai/typhoon2-qwen2.5-7b-instruct
```

## Start API

```bash
export LOCAL_TYPHOON_MODEL=typhoon-ai/typhoon2-qwen2.5-7b-instruct
export LOCAL_API_PORT=8012
export NO_QDRANT=1
export USE_ANSWER_BANK=1
export LOCAL_TORCH_DTYPE=bfloat16
export LOCAL_DEVICE_MAP=auto
export LOCAL_MAX_NEW_TOKENS=256
export LOCAL_7B_TEMPERATURE=0.1
export LOCAL_TOP_P=0.9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python pipeline-typhoon-local-7b/api_server_local_7b.py
```

## 7B Prompt Routing

The local 7B wrapper uses dynamic prompting:

- SQL/value/count questions receive a short SQL-first system prompt and only `postgres_execute_readonly_sql`.
- Schema questions receive schema tools only.
- Policy questions receive policy resolver + SQL.
- Document questions receive document retrieval tools.
- Security/prompt-injection questions receive injection detector + trusted SQL.

The prompt includes only a small table context relevant to the question, not the full database schema.

## Test One Question

```bash
curl --max-time 120 -X POST http://127.0.0.1:8012/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"data":{"id":"L3-Q-EASY-001","question":"MSRP ของสินค้ารหัส NT-LT-001 เป็นเท่าไหร่ครับ","use_answer_bank":false,"user_role":"audit","timeout_seconds":45,"max_steps":3}}'
```

## Run Questions Through API

```bash
python pipeline-typhoon-local-7b/run_api_questions_7b.py \
  --limit 10 \
  --user-role audit \
  --timeout-seconds 45 \
  --request-timeout 180 \
  --no-use-answer-bank \
  --run-dir outputs/runs/local_7b_test_10
```

รันทุกข้อ:

```bash
python pipeline-typhoon-local-7b/run_api_questions_7b.py \
  --user-role audit \
  --timeout-seconds 45 \
  --request-timeout 180 \
  --no-use-answer-bank \
  --run-dir outputs/runs/local_7b_full
```

## Direct Batch Without API

```bash
python pipeline-typhoon-local-7b/run_typhoon_local_7b_tools.py \
  --no-qdrant \
  --limit 10 \
  --user-role audit \
  --timeout-seconds 45 \
  --max-steps 3 \
  --no-answer-bank
```
