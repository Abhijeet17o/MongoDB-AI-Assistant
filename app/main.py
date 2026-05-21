from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import answer_question
from .db import get_schema_snapshot

load_dotenv()

app = FastAPI(title="AI Mongo Assistant")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str
    collection_hint: str | None = None


class ChatResponse(BaseModel):
    answer: str
    needs_clarification: bool
    choices: list[str] | None = None
    data: dict | None = None


@app.on_event("startup")
def warm_schema_cache() -> None:
    snapshot = get_schema_snapshot(force_refresh=True)
    if snapshot.get("error"):
        print(f"[startup] MongoDB connection error: {snapshot['error']}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/db")
def health_db() -> dict:
    return get_schema_snapshot(force_refresh=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    payload = answer_question(request.question, request.collection_hint)
    return ChatResponse(**payload)
