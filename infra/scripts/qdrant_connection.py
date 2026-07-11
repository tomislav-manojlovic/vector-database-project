"""Small dependency-free helpers for checking a local or Cloud Qdrant instance."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QdrantConnectionError(RuntimeError):
    """Raised when the Qdrant HTTP API cannot be reached successfully."""


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load a minimal KEY=VALUE .env file without overwriting shell variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_settings() -> tuple[str, str | None]:
    """Return the API base URL and optional API key from environment/.env."""
    load_env_file()
    port = os.getenv("QDRANT_HTTP_PORT", "6333")
    base_url = os.getenv("QDRANT_URL", f"http://localhost:{port}").rstrip("/")
    api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    return base_url, api_key


def get_collections(timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Call GET /collections and return Qdrant's JSON response."""
    base_url, api_key = get_settings()
    headers = {"Accept": "application/json"}
    if api_key:
        headers["api-key"] = api_key

    request = Request(f"{base_url}/collections", headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            if response.status != 200:
                raise QdrantConnectionError(
                    f"Qdrant returned HTTP {response.status} for {base_url}/collections."
                )
            return json.loads(body)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise QdrantConnectionError(
                "Qdrant rejected the request. Check QDRANT_API_KEY in .env."
            ) from exc
        raise QdrantConnectionError(
            f"Qdrant returned HTTP {exc.code} at {base_url}/collections."
        ) from exc
    except URLError as exc:
        raise QdrantConnectionError(
            f"Cannot reach Qdrant at {base_url}. Start Docker or check QDRANT_URL."
        ) from exc
    except json.JSONDecodeError as exc:
        raise QdrantConnectionError(
            f"Qdrant response from {base_url} was not valid JSON."
        ) from exc
