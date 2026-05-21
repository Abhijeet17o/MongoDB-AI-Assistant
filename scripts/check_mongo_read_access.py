from __future__ import annotations

import argparse
import sys
from typing import List

from pymongo import MongoClient
from pymongo.errors import ConfigurationError, PyMongoError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check read access on a MongoDB database without touching app code."
    )
    parser.add_argument("--uri", required=True, help="MongoDB connection string.")
    parser.add_argument("--timeout-ms", type=int, default=8000)
    args = parser.parse_args()

    client = MongoClient(args.uri, serverSelectionTimeoutMS=args.timeout_ms)

    try:
        client.admin.command("ping")
        print("Ping OK")
    except PyMongoError as exc:
        print("Ping FAILED:", exc)
        return 1

    try:
        db = client.get_default_database()
    except ConfigurationError:
        db_name = None
        for name in client.list_database_names():
            if name not in {"admin", "local", "config"}:
                db_name = name
                break
        if not db_name:
            print("No non-system database found.")
            return 1
        db = client[db_name]

    print("Using database:", db.name)

    collection_names: List[str] = []
    try:
        res = db.command({"listCollections": 1, "nameOnly": True, "authorizedCollections": True})
        batch = res.get("cursor", {}).get("firstBatch", [])
        collection_names = [item.get("name") for item in batch if item.get("name")]
        print("Collections (authorized):", ", ".join(collection_names) or "<none>")
    except PyMongoError as exc:
        print("listCollections FAILED:", exc)

    readable_collections: List[str] = []
    for name in collection_names:
        try:
            doc = db[name].find_one()
            if doc is not None:
                readable_collections.append(name)
        except PyMongoError:
            continue

    print("Readable collections:", ", ".join(readable_collections) or "<none>")

    if readable_collections:
        print("READ ACCESS: OK")
        return 0

    print("READ ACCESS: FAILED (no readable collections)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
