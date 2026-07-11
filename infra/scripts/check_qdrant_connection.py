"""Print a clear connection diagnostic without creating or changing any collection."""

from __future__ import annotations

import argparse

from qdrant_connection import QdrantConnectionError, get_collections, get_settings, load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check access to the Qdrant REST API.")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP request timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")

    load_env_file()
    base_url, api_key = get_settings()
    response = get_collections(timeout_seconds=args.timeout)
    result = response.get("result", {})
    collections = result.get("collections", []) if isinstance(result, dict) else []

    print("Qdrant connection OK")
    print(f"URL: {base_url}")
    print(f"Authentication: {'API key configured' if api_key else 'no API key'}")
    print(f"Collections: {len(collections)}")
    if collections:
        print("Collection names:")
        for item in collections:
            print(f"- {item.get('name', '<unknown>')}")


if __name__ == "__main__":
    try:
        main()
    except QdrantConnectionError as exc:
        raise SystemExit(f"Qdrant connection FAILED: {exc}") from exc
