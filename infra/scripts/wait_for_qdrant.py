"""Wait until Qdrant accepts REST API requests."""

from __future__ import annotations

import argparse
import time

from qdrant_connection import QdrantConnectionError, get_collections, get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait until Qdrant is reachable.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Maximum wait time in seconds.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between attempts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0 or args.interval <= 0:
        raise ValueError("--timeout and --interval must both be greater than 0.")

    base_url, _ = get_settings()
    deadline = time.monotonic() + args.timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            get_collections(timeout_seconds=min(args.interval, 5.0))
            print(f"Qdrant is ready: {base_url}")
            return
        except QdrantConnectionError as exc:
            last_error = exc
            time.sleep(args.interval)

    raise TimeoutError(
        f"Qdrant did not become ready within {args.timeout:.0f}s. Last error: {last_error}"
    )


if __name__ == "__main__":
    main()
