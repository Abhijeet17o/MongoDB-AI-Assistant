from __future__ import annotations

import argparse
import sys
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify MongoDB connection string.")
    parser.add_argument(
        "--uri",
        required=True,
        help="MongoDB connection string (mongodb+srv://...).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=8000,
        help="Server selection timeout in milliseconds.",
    )
    args = parser.parse_args()

    try:
        client = MongoClient(args.uri, serverSelectionTimeoutMS=args.timeout_ms)
        # Ping ensures the server is reachable and authenticated.
        client.admin.command("ping")
        dbs = client.list_database_names()
        print("Connection OK. Databases:", ", ".join(dbs))
        return 0
    except PyMongoError as exc:
        print("Connection FAILED:", exc)
        return 1
    except Exception as exc:  # Catch unexpected errors explicitly.
        print("Unexpected error:", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
