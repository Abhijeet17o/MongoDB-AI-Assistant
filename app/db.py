from __future__ import annotations

from typing import Any, Dict

from pymongo import MongoClient
from pymongo.errors import ConfigurationError, PyMongoError

from .config import require_settings

_client: MongoClient | None = None
_schema_cache: Dict[str, Any] | None = None
_FIELD_SAMPLE_SIZE = 20


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


def get_schema_snapshot(force_refresh: bool = False) -> Dict[str, Any]:
    global _schema_cache
    if _schema_cache is not None and not force_refresh:
        return _schema_cache

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

    _schema_cache = snapshot
    return snapshot
