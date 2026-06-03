from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse


API_BASE_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:8888").rstrip("/")
DEMO_UI_PATH = Path(__file__).with_name("web_ui") / "index.html"
PROXY_TIMEOUT_SEC = float(os.getenv("DEMO_PROXY_TIMEOUT_SEC", "120"))


app = FastAPI(title="FahMai Demo Web UI", version="1.0.0")


def _read_demo_html() -> str:
    if not DEMO_UI_PATH.exists():
        raise HTTPException(status_code=404, detail=f"demo UI is missing: {DEMO_UI_PATH}")
    return DEMO_UI_PATH.read_text(encoding="utf-8")


def _proxy_json(method: str, path: str, body: Any | None = None) -> JSONResponse:
    url = f"{API_BASE_URL}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=PROXY_TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
            return JSONResponse(payload, status_code=resp.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"detail": raw or str(exc)}
        return JSONResponse(payload, status_code=exc.code)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"cannot reach model API at {API_BASE_URL}: {exc}",
        ) from exc


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="demo")


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo() -> HTMLResponse:
    return HTMLResponse(_read_demo_html())


@app.get("/health")
def health() -> JSONResponse:
    return _proxy_json("GET", "/health")


@app.post("/api/v2/chat")
async def chat_v2(request: Request) -> JSONResponse:
    return _proxy_json("POST", "/api/v2/chat", await request.json())


@app.post("/agent/local")
async def agent_local(request: Request) -> JSONResponse:
    return _proxy_json("POST", "/agent/local", await request.json())


@app.post("/agent/thaillm")
async def agent_thaillm(request: Request) -> JSONResponse:
    return _proxy_json("POST", "/agent/thaillm", await request.json())

