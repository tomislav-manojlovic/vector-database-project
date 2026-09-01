"""Lokalni UI server bez Flask/Node zavisnosti.

Pokreće se preko START_UI.bat i vezuje se isključivo za 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import traceback
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
REPORTS_DIR = ROOT_DIR / "reports"
DATA_DIR = ROOT_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qdrant_common import COLLECTION_NAME, VECTOR_SIZE, get_qdrant_client


DEMO_ID_MIN = 9_000_000
MAX_BODY_BYTES = 1_000_000


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if np.isnan(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def dataframe_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if limit is not None:
        frame = frame.head(limit)
    return clean_json(frame.to_dict(orient="records"))


def get_vector(record: Any) -> list[float]:
    vector = record.vector
    if isinstance(vector, list):
        return vector
    if isinstance(vector, dict) and vector:
        return next(iter(vector.values()))
    raise RuntimeError("Qdrant point nema čitljiv vektor.")


def point_to_dict(point: Any, score: float | None = None) -> dict[str, Any]:
    payload = point.payload or {}
    result = {
        "id": int(point.id),
        "label": str(payload.get("label", "unknown")),
        "image_path": str(payload.get("image_path", "")),
        "payload": clean_json(payload),
        "is_demo": payload.get("source") == "ui_demo",
    }
    if score is not None:
        result["score"] = float(score)
    return result


def qdrant_status() -> dict[str, Any]:
    try:
        client = get_qdrant_client()
        exists = client.collection_exists(COLLECTION_NAME)
        if not exists:
            return {
                "connected": True,
                "collection_exists": False,
                "collection": COLLECTION_NAME,
                "count": 0,
            }
        count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        info = client.get_collection(COLLECTION_NAME)
        vectors_config = info.config.params.vectors
        size = getattr(vectors_config, "size", VECTOR_SIZE)
        distance = getattr(vectors_config, "distance", "Cosine")
        return {
            "connected": True,
            "collection_exists": True,
            "collection": COLLECTION_NAME,
            "count": int(count),
            "vector_size": int(size),
            "distance": str(getattr(distance, "value", distance)),
            "dashboard_url": "http://localhost:6333/dashboard",
        }
    except Exception as exc:
        return {
            "connected": False,
            "collection_exists": False,
            "collection": COLLECTION_NAME,
            "error": str(exc),
        }


def demo_points() -> list[dict[str, Any]]:
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = get_qdrant_client()
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value="ui_demo"))
                ]
            ),
            limit=100,
            with_vectors=False,
            with_payload=True,
        )
        return [point_to_dict(point) for point in points]
    except Exception:
        return []


def overview() -> dict[str, Any]:
    metadata_path = DATA_DIR / "embeddings" / "embeddings_metadata.csv"
    embeddings_path = DATA_DIR / "embeddings" / "embeddings.npy"
    config = read_json_file(DATA_DIR / "embeddings" / "embedding_config.json") or {}
    dataset_count = 0
    class_counts: dict[str, int] = {}
    labeled_count = 0
    labeled_classes = 0
    unlabeled_count = 0
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        metadata = metadata[metadata["status"] == "ok"]
        dataset_count = len(metadata)
        class_counts = clean_json(metadata["label"].value_counts().sort_index().to_dict())
        labeled_mask = metadata["is_labeled"].astype(str).str.lower().isin({"true", "1"})
        labeled_metadata = metadata[labeled_mask]
        labeled_count = len(labeled_metadata)
        labeled_classes = int(labeled_metadata["label"].nunique())
        unlabeled_count = dataset_count - labeled_count

    embedding_shape = None
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path, mmap_mode="r")
        embedding_shape = list(embeddings.shape)

    error_summary = read_json_file(
        REPORTS_DIR / "error_analysis" / "summary.json"
    )
    cleaning_summary = read_json_file(
        REPORTS_DIR / "variant3_dataset_cleaning" / "summary.json"
    )
    cleaning_manifest = read_json_file(
        DATA_DIR / "cleaned" / "cleaning_manifest.json"
    )
    qdrant = qdrant_status()
    demos = demo_points() if qdrant.get("collection_exists") else []

    return {
        "project": "Qdrant Image Search",
        "dataset": {
            "name": "STL-10",
            "count": dataset_count,
            "classes": len(class_counts),
            "class_counts": class_counts,
            "labeled_count": labeled_count,
            "labeled_classes": labeled_classes,
            "unlabeled_count": unlabeled_count,
        },
        "embeddings": {
            "model": config.get("model_name", "openai/clip-vit-base-patch32"),
            "shape": embedding_shape,
            "dimension": config.get("embedding_dim", VECTOR_SIZE),
            "normalized": config.get("normalize", True),
        },
        "qdrant": qdrant,
        "demo_points": demos,
        "error_analysis": error_summary,
        "cleaning": cleaning_summary,
        "cleaned_dataset": cleaning_manifest,
        "files": {
            "metadata": metadata_path.exists(),
            "embeddings": embeddings_path.exists(),
            "error_report": error_summary is not None,
            "cleaning_report": cleaning_summary is not None,
            "cleaned_dataset": cleaning_manifest is not None,
        },
    }


def get_point(point_id: int, with_vector: bool = False) -> dict[str, Any]:
    client = get_qdrant_client()
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_vectors=with_vector,
        with_payload=True,
    )
    if not points:
        raise KeyError(f"Point ID={point_id} ne postoji.")
    result = point_to_dict(points[0])
    if with_vector:
        vector = get_vector(points[0])
        result["vector_dimension"] = len(vector)
        result["vector_preview"] = vector[:8]
    return result


def filter_points(label: str, limit: int) -> list[dict[str, Any]]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_qdrant_client()
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="label", match=MatchValue(value=label))]
        ),
        limit=limit,
        with_vectors=False,
        with_payload=True,
    )
    return [point_to_dict(point) for point in points]


def similar_points(
    point_id: int,
    top_k: int,
    label: str | None = None,
    exact: bool = False,
    hnsw_ef: int = 64,
) -> list[dict[str, Any]]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue, SearchParams

    client = get_qdrant_client()
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_vectors=True,
        with_payload=True,
    )
    if not points:
        raise KeyError(f"Point ID={point_id} ne postoji.")
    query_filter = None
    if label:
        query_filter = Filter(
            must=[FieldCondition(key="label", match=MatchValue(value=label))]
        )
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=get_vector(points[0]),
        query_filter=query_filter,
        limit=top_k + 3,
        with_payload=True,
        with_vectors=False,
        search_params=SearchParams(
            exact=exact,
            hnsw_ef=None if exact else hnsw_ef,
        ),
    )
    output = []
    for point in result.points:
        if int(point.id) == point_id:
            continue
        output.append(point_to_dict(point, point.score))
        if len(output) == top_k:
            break
    return output


def create_demo_point(source_id: int, new_id: int, label: str | None) -> dict[str, Any]:
    from qdrant_client.models import PointStruct

    if new_id < DEMO_ID_MIN:
        raise ValueError(f"Demo ID mora biti najmanje {DEMO_ID_MIN}.")
    client = get_qdrant_client()
    existing = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[new_id],
        with_vectors=False,
    )
    if existing:
        raise ValueError(f"Point ID={new_id} već postoji.")
    source = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[source_id],
        with_vectors=True,
        with_payload=True,
    )
    if not source:
        raise KeyError(f"Izvorni point ID={source_id} ne postoji.")
    source_payload = source[0].payload or {}
    payload = {
        "id": new_id,
        "image_path": str(source_payload.get("image_path", "")),
        "label": label or str(source_payload.get("label", "unknown")),
        "source": "ui_demo",
        "demo_source_id": source_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=new_id, vector=get_vector(source[0]), payload=payload)],
        wait=True,
    )
    return get_point(new_id, with_vector=True)


def require_demo_point(point_id: int) -> dict[str, Any]:
    point = get_point(point_id)
    if not point["is_demo"]:
        raise PermissionError(
            "UI dozvoljava izmene i brisanje samo privremenih ui_demo pointova."
        )
    return point


def update_demo_point(point_id: int, key: str, value: Any) -> dict[str, Any]:
    if not key or key in {"id", "source", "demo_source_id"}:
        raise ValueError("Ovo payload polje nije dozvoljeno za izmenu.")
    require_demo_point(point_id)
    client = get_qdrant_client()
    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={key: value},
        points=[point_id],
        wait=True,
    )
    return get_point(point_id)


def delete_demo_point(point_id: int) -> dict[str, Any]:
    from qdrant_client.models import PointIdsList

    require_demo_point(point_id)
    client = get_qdrant_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[point_id]),
        wait=True,
    )
    return {"deleted": True, "id": point_id}


def cleanup_demo_points() -> dict[str, Any]:
    from qdrant_client.models import PointIdsList

    points = demo_points()
    ids = [int(point["id"]) for point in points]
    if ids:
        get_qdrant_client().delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=ids),
            wait=True,
        )
    return {"deleted_count": len(ids), "ids": ids}


def parse_payload_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def error_data(limit: int = 50) -> dict[str, Any]:
    report_dir = REPORTS_DIR / "error_analysis"
    summary = read_json_file(report_dir / "summary.json")
    errors = dataframe_records(report_dir / "errors.csv", limit)
    matrix_path = report_dir / "confusion_matrix.csv"
    matrix = None
    if matrix_path.exists():
        frame = pd.read_csv(matrix_path)
        matrix = {
            "columns": list(frame.columns),
            "rows": clean_json(frame.to_dict(orient="records")),
        }
    return {"summary": summary, "errors": errors, "confusion_matrix": matrix}


def error_detail(point_id: int) -> dict[str, Any]:
    report_dir = REPORTS_DIR / "error_analysis"
    errors_path = report_dir / "errors.csv"
    neighbors_path = report_dir / "error_neighbors.csv"
    if not errors_path.exists() or not neighbors_path.exists():
        raise FileNotFoundError("Prvo pokreni analizu grešaka.")
    errors = pd.read_csv(errors_path)
    selected = errors[errors["id"] == point_id]
    if selected.empty:
        raise KeyError(f"ID={point_id} nije u listi grešaka.")
    neighbors = pd.read_csv(neighbors_path)
    selected_neighbors = neighbors[neighbors["query_id"] == point_id]
    return {
        "error": clean_json(selected.iloc[0].to_dict()),
        "neighbors": clean_json(selected_neighbors.to_dict(orient="records")),
    }


def cleaning_data() -> dict[str, Any]:
    report_dir = REPORTS_DIR / "variant3_dataset_cleaning"
    return {
        "summary": read_json_file(report_dir / "summary.json"),
        "groups": dataframe_records(report_dir / "group_summary.csv", 100),
        "pairs": dataframe_records(report_dir / "candidate_pairs.csv", 100),
        "manifest": read_json_file(DATA_DIR / "cleaned" / "cleaning_manifest.json"),
    }


def cleaning_group(group_id: int) -> dict[str, Any]:
    report_dir = REPORTS_DIR / "variant3_dataset_cleaning"
    members_path = report_dir / "group_members.csv"
    pairs_path = report_dir / "candidate_pairs.csv"
    if not members_path.exists() or not pairs_path.exists():
        raise FileNotFoundError("Prvo pokreni analizu čišćenja.")
    members = pd.read_csv(members_path)
    selected = members[members["group_id"] == group_id]
    if selected.empty:
        raise KeyError(f"Grupa {group_id} ne postoji.")
    ids = set(selected["point_id"].astype(int))
    pairs = pd.read_csv(pairs_path)
    selected_pairs = pairs[
        pairs["left_id"].isin(ids) & pairs["right_id"].isin(ids)
    ]
    return {
        "group_id": group_id,
        "members": clean_json(selected.to_dict(orient="records")),
        "pairs": clean_json(selected_pairs.to_dict(orient="records")),
    }


def run_project_command(command: list[str], timeout: int = 300) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, *command],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return {
        "success": completed.returncode == 0,
        "return_code": completed.returncode,
        "output": output.strip(),
        "command": " ".join(command),
    }


def image_for_id(point_id: int) -> tuple[bytes, str]:
    label = "image"
    image_path = ""
    try:
        point = get_point(point_id)
        label = point["label"]
        image_path = point["image_path"]
    except Exception:
        metadata_path = DATA_DIR / "embeddings" / "embeddings_metadata.csv"
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            selected = metadata[metadata["id"] == point_id]
            if not selected.empty:
                label = str(selected.iloc[0]["label"])
                image_path = str(selected.iloc[0]["image_path"])

    if image_path:
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = ROOT_DIR / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(ROOT_DIR.resolve())
            if resolved.is_file():
                content_type = mimetypes.guess_type(resolved.name)[0] or "image/jpeg"
                return resolved.read_bytes(), content_type
        except (OSError, ValueError):
            pass

    safe_label = "".join(character for character in label if character.isalnum() or character in " -_")[:24]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240" viewBox="0 0 320 240">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#e7eefb"/><stop offset="1" stop-color="#dce8f8"/></linearGradient></defs>
<rect width="320" height="240" rx="18" fill="url(#g)"/><circle cx="160" cy="96" r="34" fill="#9fb8dc"/><path d="M78 195l58-61 37 39 28-28 42 50z" fill="#6f91c2"/><text x="160" y="222" text-anchor="middle" font-family="Segoe UI,Arial" font-size="15" fill="#28466f">{safe_label} · ID {point_id}</text></svg>"""
    return svg.encode("utf-8"), "image/svg+xml"


class UIHandler(BaseHTTPRequestHandler):
    server_version = "QdrantProjectUI/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[UI] {self.address_string()} - {format % args}")

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(clean_json(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, exc: Exception, status: int = 400) -> None:
        self.send_json(
            {
                "error": str(exc),
                "type": exc.__class__.__name__,
            },
            status,
        )

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            raise ValueError("Zahtev je prevelik.")
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def serve_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                return self.serve_file(UI_DIR / "index.html", "text/html; charset=utf-8")
            if path.startswith("/assets/"):
                filename = path.removeprefix("/assets/")
                if filename not in {"app.js", "styles.css"}:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                return self.serve_file(UI_DIR / filename)
            if path.startswith("/data/images/"):
                relative = Path(path.lstrip("/"))
                candidate = (ROOT_DIR / relative).resolve()
                images_root = (DATA_DIR / "images").resolve()
                try:
                    candidate.relative_to(images_root)
                except ValueError:
                    return self.send_error(HTTPStatus.FORBIDDEN)
                return self.serve_file(candidate)
            if path == "/api/overview":
                return self.send_json(overview())
            if path == "/api/point":
                point_id = int(query.get("id", [""])[0])
                with_vector = query.get("with_vector", ["false"])[0].lower() == "true"
                return self.send_json(get_point(point_id, with_vector))
            if path == "/api/filter":
                label = query.get("label", [""])[0].strip()
                limit = min(max(int(query.get("limit", ["8"])[0]), 1), 50)
                return self.send_json({"results": filter_points(label, limit)})
            if path == "/api/similar":
                point_id = int(query.get("id", [""])[0])
                top_k = min(max(int(query.get("top_k", ["5"])[0]), 1), 20)
                label = query.get("label", [""])[0].strip() or None
                exact = query.get("exact", ["false"])[0].lower() in {
                    "1", "true", "yes"
                }
                hnsw_ef = int(query.get("hnsw_ef", ["64"])[0])
                return self.send_json(
                    {
                        "query": get_point(point_id),
                        "results": similar_points(
                            point_id,
                            top_k,
                            label,
                            exact=exact,
                            hnsw_ef=hnsw_ef,
                        ),
                    }
                )
            if path == "/api/errors":
                limit = min(max(int(query.get("limit", ["50"])[0]), 1), 200)
                return self.send_json(error_data(limit))
            if path == "/api/error-detail":
                return self.send_json(error_detail(int(query.get("id", [""])[0])))
            if path == "/api/cleaning":
                return self.send_json(cleaning_data())
            if path == "/api/cleaning-group":
                return self.send_json(
                    cleaning_group(int(query.get("id", [""])[0]))
                )
            if path == "/api/image":
                data, content_type = image_for_id(int(query.get("id", ["0"])[0]))
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=60")
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/reports/errors":
                return self.serve_file(
                    REPORTS_DIR / "error_analysis" / "report.html",
                    "text/html; charset=utf-8",
                )
            if path == "/reports/cleaning":
                return self.serve_file(
                    REPORTS_DIR / "variant3_dataset_cleaning" / "report.html",
                    "text/html; charset=utf-8",
                )
            self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self.send_error_json(exc, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_error_json(exc, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            payload = self.read_json()
            if path == "/api/crud/create-demo":
                result = create_demo_point(
                    source_id=int(payload.get("source_id", 1)),
                    new_id=int(payload.get("new_id", DEMO_ID_MIN + 1)),
                    label=str(payload.get("label", "")).strip() or None,
                )
                return self.send_json(result, HTTPStatus.CREATED)
            if path == "/api/crud/update":
                result = update_demo_point(
                    int(payload["id"]),
                    str(payload["key"]).strip(),
                    parse_payload_value(payload.get("value")),
                )
                return self.send_json(result)
            if path == "/api/crud/delete":
                return self.send_json(delete_demo_point(int(payload["id"])))
            if path == "/api/crud/cleanup":
                return self.send_json(cleanup_demo_points())
            if path == "/api/run/tests":
                return self.send_json(
                    run_project_command(
                        ["-m", "unittest", "tests.test_real_database", "-v"]
                    )
                )
            if path == "/api/run/error-analysis":
                cleanup = cleanup_demo_points()
                result = run_project_command(
                    ["src/07_error_analysis.py", "analyze", "--backend", "qdrant"],
                    timeout=300,
                )
                result["demo_cleanup"] = cleanup
                return self.send_json(result)
            if path == "/api/run/cleaning-analysis":
                cleanup = cleanup_demo_points()
                result = run_project_command(
                    ["src/08_dataset_cleaning.py", "analyze", "--backend", "qdrant"],
                    timeout=300,
                )
                result["demo_cleanup"] = cleanup
                return self.send_json(result)
            if path == "/api/run/build-cleaned":
                return self.send_json(
                    run_project_command(["src/08_dataset_cleaning.py", "build-clean-dataset"])
                )
            if path == "/api/run/verify-cleaned":
                return self.send_json(
                    run_project_command(["src/08_dataset_cleaning.py", "verify-cleaned"])
                )
            self.send_error(HTTPStatus.NOT_FOUND)
        except PermissionError as exc:
            self.send_error_json(exc, HTTPStatus.FORBIDDEN)
        except KeyError as exc:
            self.send_error_json(exc, HTTPStatus.NOT_FOUND)
        except subprocess.TimeoutExpired:
            self.send_error_json(RuntimeError("Komanda je prekoračila vreme čekanja."), 504)
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(exc, HTTPStatus.BAD_REQUEST)


def find_available_port(start: int) -> int:
    import socket

    for port in range(start, start + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Nije pronađen slobodan lokalni port.")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Lokalni Qdrant Project UI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    port = find_available_port(args.port)
    address = f"http://127.0.0.1:{port}"
    server = ThreadingHTTPServer(("127.0.0.1", port), UIHandler)

    print()
    print("Qdrant Image Search UI je pokrenut.")
    print(f"Adresa: {address}")
    print("Za zaustavljanje zatvori ovaj prozor ili pritisni Ctrl+C.")
    print()

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nUI je zaustavljen.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
