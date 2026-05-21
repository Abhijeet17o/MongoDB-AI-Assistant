from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    llm_api_key: str
    llm_model: str


def load_settings() -> Settings:
    return Settings(
        mongodb_uri=os.getenv("MONGODB_URI", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
    )


def require_settings() -> Settings:
    settings = load_settings()
    missing = []
    if not settings.mongodb_uri:
        missing.append("MONGODB_URI")
    if not settings.llm_api_key:
        missing.append("LLM_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    return settings
