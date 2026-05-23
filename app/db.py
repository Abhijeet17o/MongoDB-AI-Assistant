from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, PyMongoError

from .config import require_settings

_client: MongoClient | None = None
_schema_cache: Dict[str, Any] | None = None
_FIELD_SAMPLE_SIZE = 20
_DOC_SAMPLE_SIZE = 2
_MAX_SAMPLE_DEPTH = 2
_MAX_LIST_ITEMS = 5
_MAX_DOC_FIELDS = 40
_MAX_STRING_LEN = 120


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


def _redact_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    redacted: Dict[str, Any] = {}
    for key, value in doc.items():
        key_lower = key.lower()
        if any(token in key_lower for token in ["password", "pass", "secret", "token", "apikey", "api_key", "hash"]):
            continue
        redacted[key] = value
    return redacted


def _prune_doc(value: Any, depth: int = 0) -> Any:
    if depth >= _MAX_SAMPLE_DEPTH:
        return "..."
    if isinstance(value, dict):
        pruned: Dict[str, Any] = {}
        for index, (key, entry) in enumerate(value.items()):
            if index >= _MAX_DOC_FIELDS:
                pruned["..."] = "..."
                break
            pruned[key] = _prune_doc(entry, depth + 1)
        return pruned
    if isinstance(value, list):
        trimmed = value[:_MAX_LIST_ITEMS]
        return [_prune_doc(item, depth + 1) for item in trimmed]
    if isinstance(value, str) and len(value) > _MAX_STRING_LEN:
        return value[:_MAX_STRING_LEN] + "..."
    return value


def _build_sample_docs(db, collection: str) -> List[Dict[str, Any]]:
    sample_docs: List[Dict[str, Any]] = []
    for doc in db[collection].find().limit(_DOC_SAMPLE_SIZE):
        safe_doc = _serialize_value(_redact_doc(doc))
        pruned = _prune_doc(safe_doc)
        if isinstance(pruned, dict):
            sample_docs.append(pruned)
    return sample_docs


def _load_schema_from_file() -> Optional[Dict[str, Any]]:
    try:
        schema_path = Path(__file__).resolve().parent.parent / "db_schema.json"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "collections" in data and "fields_by_collection" in data:
                return data
    except Exception as e:
        print(f"Error loading db_schema.json: {e}")
    return None


def _build_base_snapshot() -> Dict[str, Any]:
    file_schema = _load_schema_from_file()
    if file_schema:
        snapshot = {
            "default_db": file_schema.get("default_db", ""),
            "collections": file_schema.get("collections", []),
            "primary_collection": file_schema.get("primary_collection"),
            "primary_field": file_schema.get("primary_field"),
            "first_word": file_schema.get("first_word", ""),
            "fields_by_collection": file_schema.get("fields_by_collection", {}),
            "sample_docs_by_collection": file_schema.get("sample_docs_by_collection", {}),
            "error": None,
        }
        return snapshot

    snapshot: Dict[str, Any] = {
        "default_db": "",
        "collections": [],
        "primary_collection": None,
        "primary_field": None,
        "first_word": "",
        "fields_by_collection": {},
        "error": None,
    }

    try:
        db = get_default_db()
        snapshot["default_db"] = db.name
        # Use listCollections to avoid system collections (system.views can be restricted).
        collections: list[str] = []
        try:
            res = db.command(
                {"listCollections": 1, "nameOnly": True, "authorizedCollections": True}
            )
            batch = res.get("cursor", {}).get("firstBatch", [])
            collections = [item.get("name") for item in batch if item.get("name")]
        except PyMongoError:
            # Fallback to list_collection_names if listCollections is not allowed.
            collections = db.list_collection_names()

        collections = [name for name in collections if not name.startswith("system.")]
        collections.sort()
        snapshot["collections"] = collections
        fields_by_collection: Dict[str, list[str]] = {}
        for name in collections:
            fields: set[str] = set()
            # Sample multiple documents to avoid missing sparsely populated fields.
            for doc in db[name].find().limit(_FIELD_SAMPLE_SIZE):
                fields.update(doc.keys())
            fields_by_collection[name] = sorted(fields)
        snapshot["fields_by_collection"] = fields_by_collection
        if collections:
            primary = collections[0]
            snapshot["primary_collection"] = primary
            doc = db[primary].find_one()
            if doc:
                fields = list(doc.keys())
                if fields:
                    primary_field = fields[0]
                    snapshot["primary_field"] = primary_field
                    snapshot["first_word"] = (
                        str(primary_field).split()[0] if str(primary_field).strip() else ""
                    )
    except PyMongoError as exc:
        snapshot["error"] = str(exc)

    return snapshot


def get_client() -> MongoClient:
    global _client
    if _client is None:
        settings = require_settings()
        _client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)
    return _client


def get_default_db():
    client = get_client()
    try:
        return client.get_default_database()
    except ConfigurationError:
        # Fall back to the first non-system database when URI has no db name.
        databases = [
            name
            for name in client.list_database_names()
            if name not in {"admin", "local", "config"}
        ]
        if not databases:
            raise
        return client[databases[0]]


def get_schema_snapshot(
    force_refresh: bool = False,
    include_samples: bool = False,
    sample_collections: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    global _schema_cache
    if _schema_cache is None or force_refresh:
        _schema_cache = _build_base_snapshot()

    snapshot = _schema_cache
    if not include_samples:
        return snapshot

    snapshot_with_samples = dict(snapshot)
    target_collections = [name for name in (sample_collections or []) if name]
    if not target_collections and snapshot.get("primary_collection"):
        target_collections = [snapshot["primary_collection"]]

    sample_docs_by_collection: Dict[str, List[Dict[str, Any]]] = {}
    file_samples = snapshot.get("sample_docs_by_collection", {})
    
    # 1. First extract existing samples from the loaded file cache
    for name in target_collections:
        if name in file_samples:
            sample_docs_by_collection[name] = file_samples[name]

    # 2. For any remaining collections, dynamically query MongoDB if connected
    remaining_collections = [name for name in target_collections if name not in sample_docs_by_collection]
    if remaining_collections:
        try:
            db = get_default_db()
            for name in remaining_collections:
                if name in snapshot.get("collections", []):
                    sample_docs_by_collection[name] = _build_sample_docs(db, name)
        except Exception as exc:
            # Fall back gracefully to whatever was loaded, do not fail completely
            if not sample_docs_by_collection:
                snapshot_with_samples["error"] = str(exc)

    snapshot_with_samples["sample_docs_by_collection"] = sample_docs_by_collection
    return snapshot_with_samples
