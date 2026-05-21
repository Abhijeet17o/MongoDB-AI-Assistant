from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Literal, Optional, Tuple

from bson import ObjectId
import google.generativeai as genai
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from .config import require_settings
from .db import get_default_db, get_schema_snapshot


class QueryPlan(BaseModel):
    action: Literal["count", "sum", "top", "find", "clarify", "unknown"] = "unknown"
    collection: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    field: Optional[str] = None
    group_by: Optional[str] = None
    sort: Optional[Dict[str, int]] = None
    limit: Optional[int] = Field(default=None, ge=1, le=100)
    clarification_question: Optional[str] = None


def _render_prompt(schema_snapshot: Dict[str, Any], question: str, parser: PydanticOutputParser) -> str:
    system = (
        "You are a MongoDB query planner for an AI assistant. "
        "Use ONLY the collections listed in schema. If unsure, ask for clarification. "
        "Return a JSON object matching the provided schema.\n"
        "Schema snapshot: {schema_snapshot}\n"
        "Rules:\n"
        "- If user intent is unclear, set action=clarify and provide a short clarification_question.\n"
        "- Only use simple filters (equality or date ranges).\n"
        "- Keep limit <= 100.\n"
        "- If preferred_collection is provided, use it. Otherwise use primary_collection.\n"
        "- Use first_word as a fallback field when a field is needed.\n"
        "{format_instructions}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question}"),
        ]
    )
    rendered = prompt.format_messages(
        question=question,
        schema_snapshot=json.dumps(schema_snapshot, ensure_ascii=True),
        format_instructions=parser.get_format_instructions(),
    )
    return "\n\n".join(f"{msg.type.upper()}: {msg.content}" for msg in rendered)


def _render_refine_prompt(
    schema_snapshot: Dict[str, Any],
    question: str,
    parser: PydanticOutputParser,
    previous_plan: Dict[str, Any],
    issue: str,
) -> str:
    system = (
        "You are a MongoDB query planner for an AI assistant. "
        "The previous plan failed or returned no data. "
        "Adjust the plan to fix the issue.\n"
        "Schema snapshot: {schema_snapshot}\n"
        "Previous plan: {previous_plan}\n"
        "Issue: {issue}\n"
        "Rules:\n"
        "- Use ONLY the collections listed in schema.\n"
        "- Keep limit <= 100.\n"
        "- Return a JSON object matching the provided schema.\n"
        "{format_instructions}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question}"),
        ]
    )
    rendered = prompt.format_messages(
        question=question,
        schema_snapshot=json.dumps(schema_snapshot, ensure_ascii=True),
        previous_plan=json.dumps(previous_plan, ensure_ascii=True),
        issue=issue,
        format_instructions=parser.get_format_instructions(),
    )
    return "\n\n".join(f"{msg.type.upper()}: {msg.content}" for msg in rendered)


def _call_llm(prompt: str) -> str:
    settings = require_settings()
    genai.configure(api_key=settings.llm_api_key)
    model = genai.GenerativeModel(settings.llm_model)
    response = model.generate_content(prompt)
    return response.text or ""


def _guess_collection(question: str, collections: List[str]) -> Optional[str]:
    question_lower = question.lower()
    mapping = {
        "users": ["customer", "customers", "user", "users", "subscriber", "account"],
        "movies": ["movie", "movies", "film", "films", "title"],
        "comments": ["comment", "comments", "review", "reviews"],
        "sessions": ["session", "sessions", "login", "activity"],
        "theaters": ["theater", "theaters", "cinema", "venue"],
    }

    for collection, keywords in mapping.items():
        if collection in collections and any(keyword in question_lower for keyword in keywords):
            return collection
    return None


def _build_collection_choices(
    collections: List[str], preferred: Optional[str] = None
) -> List[str]:
    choices: List[str] = []
    if preferred:
        choices.append(preferred)
    for name in collections:
        if name not in choices:
            choices.append(name)
        if len(choices) >= 3:
            break
    return choices


def _needs_refinement(plan: QueryPlan, result: Dict[str, Any]) -> Tuple[bool, str]:
    if "error" in result:
        return True, result["error"]
    if plan.action in {"find", "top"}:
        items = result.get("items") or []
        if not items:
            return True, "No results"
    return False, ""


def _normalize_plan(plan: QueryPlan, schema_snapshot: Dict[str, Any]) -> QueryPlan:
    if plan.limit is None:
        plan.limit = 20
    plan.limit = max(1, min(plan.limit, 100))

    preferred_collection = schema_snapshot.get("preferred_collection")
    if not plan.collection:
        plan.collection = preferred_collection or schema_snapshot.get("primary_collection")

    collections = schema_snapshot.get("collections") or []
    if plan.collection and collections and plan.collection not in collections:
        return QueryPlan(
            action="clarify",
            clarification_question=(
                "Which collection should I use? Available: " + ", ".join(collections)
            ),
        )

    if plan.action in {"sum", "top"} and not plan.field:
        plan.field = schema_snapshot.get("primary_field") or schema_snapshot.get("first_word")

    if plan.action == "top" and not plan.group_by:
        plan.group_by = schema_snapshot.get("primary_field") or schema_snapshot.get("first_word")

    return plan


def _serialize_value(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    return value


def _pick_display_field(items: List[Dict[str, Any]]) -> Optional[str]:
    if not items:
        return None
    preferred = ["name", "full_name", "customer_name", "username", "email", "title"]
    keys = {key for item in items if isinstance(item, dict) for key in item.keys()}
    for key in preferred:
        if key in keys:
            return key
    for key in keys:
        if key.lower().endswith("name"):
            return key
    return None


def _redact_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    redacted: Dict[str, Any] = {}
    for key, value in doc.items():
        key_lower = key.lower()
        if any(token in key_lower for token in ["password", "pass", "secret", "token", "apikey", "api_key", "hash"]):
            continue
        redacted[key] = value
    return redacted


def _execute_plan(plan: QueryPlan) -> Dict[str, Any]:
    db = get_default_db()
    collection = db[plan.collection] if plan.collection else None
    if collection is None:
        return {"error": "No collection selected."}

    filter_doc = plan.filter or {}
    try:
        if plan.action == "count":
            count = collection.count_documents(filter_doc)
            return {"count": count}

        if plan.action == "sum":
            if not plan.field:
                return {"error": "Missing field for sum."}
            pipeline = [
                {"$match": filter_doc},
                {"$group": {"_id": None, "total": {"$sum": f"${plan.field}"}}},
            ]
            result = list(collection.aggregate(pipeline))
            total = result[0]["total"] if result else 0
            return {"total": total}

        if plan.action == "top":
            if not plan.field or not plan.group_by:
                return {"error": "Missing field or group_by for top query."}
            pipeline = [
                {"$match": filter_doc},
                {"$group": {"_id": f"${plan.group_by}", "total": {"$sum": f"${plan.field}"}}},
                {"$sort": {"total": -1}},
                {"$limit": plan.limit},
            ]
            result = list(collection.aggregate(pipeline))
            return {"items": result}

        if plan.action == "find":
            cursor = collection.find(filter_doc).limit(plan.limit)
            docs = [_serialize_value(doc) for doc in cursor]
            safe_docs = [_redact_doc(doc) for doc in docs if isinstance(doc, dict)]
            display_field = _pick_display_field(safe_docs)
            if display_field:
                values = [doc.get(display_field) for doc in safe_docs if doc.get(display_field)]
                values = [str(value) for value in values if str(value).strip()]
                return {
                    "items": [{"value": value} for value in values],
                    "display_field": display_field,
                    "count": len(safe_docs),
                }
            return {"items": safe_docs, "count": len(safe_docs)}

        return {"error": "Unsupported action."}
    except PyMongoError as exc:
        return {"error": str(exc)}


def _format_response(plan: QueryPlan, result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"I ran into an issue: {result['error']}"
    if plan.action == "count":
        return f"Count: {result['count']}"
    if plan.action == "sum":
        return f"Total: {result['total']}"
    if plan.action == "top":
        items = result.get("items") or []
        if not items:
            return "No results found."
        lines = []
        for item in items[:5]:
            label = item.get("_id")
            total = item.get("total")
            lines.append(f"{label}: {total}")
        return "Top results: " + "; ".join(lines)
    if plan.action == "find":
        items = result.get("items") or []
        if not items:
            return "No results found."
        count = result.get("count", len(items))
        field = result.get("display_field")
        if field:
            values = [item.get("value") for item in items if item.get("value")]
            preview = ", ".join(values[:10]) if values else ""
            suffix = f" {field} values: {preview}" if preview else ""
            return f"Found {count} documents.{suffix}"
        return f"Found {count} documents. See details below."
    return "I could not determine a valid response."


def answer_question(question: str, collection_hint: Optional[str] = None) -> Dict[str, Any]:
    schema_snapshot = get_schema_snapshot()
    if schema_snapshot.get("error"):
        return {
            "answer": "Database connection failed. Please check network access and MongoDB IP allowlist.",
            "needs_clarification": True,
            "data": {"error": schema_snapshot["error"]},
        }

    collections = schema_snapshot.get("collections") or []
    preferred = collection_hint or _guess_collection(question, collections)
    if preferred:
        schema_snapshot = {**schema_snapshot, "preferred_collection": preferred}

    try:
        parser = PydanticOutputParser(pydantic_object=QueryPlan)
        prompt = _render_prompt(schema_snapshot, question, parser)
        raw = _call_llm(prompt)
        plan = parser.parse(raw)
    except Exception:
        plan = QueryPlan(
            action="clarify",
            clarification_question="I could not parse that. Can you rephrase or be more specific?",
        )

    plan = _normalize_plan(plan, schema_snapshot)

    if plan.action in {"clarify", "unknown"}:
        return {
            "answer": plan.clarification_question or "Can you clarify your request?",
            "needs_clarification": True,
            "choices": _build_collection_choices(collections, preferred),
            "data": {"plan": plan.model_dump()},
        }

    result = _execute_plan(plan)
    needs_retry, issue = _needs_refinement(plan, result)
    if needs_retry:
        try:
            parser = PydanticOutputParser(pydantic_object=QueryPlan)
            prompt = _render_refine_prompt(
                schema_snapshot,
                question,
                parser,
                plan.model_dump(),
                issue,
            )
            raw = _call_llm(prompt)
            refined_plan = parser.parse(raw)
            refined_plan = _normalize_plan(refined_plan, schema_snapshot)
            refined_result = _execute_plan(refined_plan)
            if "error" not in refined_result:
                plan = refined_plan
                result = refined_result
        except Exception:
            pass
    return {
        "answer": _format_response(plan, result),
        "needs_clarification": False,
        "data": {"plan": plan.model_dump(), "result": result},
    }
