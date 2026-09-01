"""Analiza klasifikacionih grešaka pomoću CLIP embeddinga i Qdranta.

CLIP je feature extractor, dok weighted k-NN nad najbližim vektorima daje
predikciju klase. Svaka slika se klasifikuje leave-one-out postupkom: sama
slika se obavezno uklanja iz skupa suseda.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings.npy"
DEFAULT_METADATA_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings_metadata.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "reports" / "error_analysis"
ANALYSIS_IMAGES_PER_CLASS = 100

CANONICAL_STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

REQUIRED_METADATA_COLUMNS = {
    "id",
    "image_path",
    "label",
    "embedding_index",
    "status",
    "is_labeled",
}


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class Neighbor:
    point_id: int
    label: str
    image_path: str
    score: float


class NeighborBackend(Protocol):
    name: str

    def validate(self, expected_count: int) -> None:
        ...

    def search(self, query_id: int, query_vector: np.ndarray, limit: int) -> list[Neighbor]:
        ...


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Pronađen je embedding čija je L2 norma jednaka nuli.")
    return (vectors / norms).astype(np.float32)


def load_inputs(
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> tuple[np.ndarray, pd.DataFrame]:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Nedostaje embeddings fajl: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Nedostaje metadata fajl: {metadata_path}")

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)

    missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata fajlu nedostaju kolone: {sorted(missing)}")

    metadata = metadata[metadata["status"] == "ok"].copy()
    metadata["id"] = metadata["id"].astype(int)
    metadata["embedding_index"] = metadata["embedding_index"].astype(int)
    metadata["label"] = metadata["label"].astype(str)
    metadata["image_path"] = metadata["image_path"].astype(str)
    metadata = metadata.sort_values("embedding_index").reset_index(drop=True)

    if metadata.empty:
        raise ValueError("Nema metadata redova sa status='ok'.")
    if metadata["id"].duplicated().any():
        raise ValueError("Metadata sadrži duplirane ID-eve.")
    if metadata["embedding_index"].duplicated().any():
        raise ValueError("Metadata sadrži duplirane embedding_index vrednosti.")
    if embeddings.ndim != 2:
        raise ValueError("embeddings.npy mora biti 2D matrica oblika (N, D).")

    indices = metadata["embedding_index"].to_numpy(dtype=int)
    if indices.min() < 0 or indices.max() >= len(embeddings):
        raise ValueError("Neki embedding_index je izvan granica embeddings.npy fajla.")

    selected_embeddings = embeddings[indices].astype(np.float32)
    selected_embeddings = normalize_rows(selected_embeddings)

    if len(selected_embeddings) != len(metadata):
        raise ValueError("Broj embeddinga i broj validnih metadata redova se ne poklapaju.")

    return selected_embeddings, metadata


def select_analysis_sample(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    images_per_class: int = ANALYSIS_IMAGES_PER_CLASS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Izaberi mali, uravnotežen uzorak samo labeliranih STL-10 slika."""
    labeled_mask = metadata["is_labeled"].astype(str).str.lower().isin({"true", "1"})
    labeled = metadata[labeled_mask]
    if labeled.empty:
        raise ValueError("Nema labeliranih slika za analizu grešaka.")

    sampled = labeled.groupby("label", sort=False).head(images_per_class).copy()
    sampled_embeddings = embeddings[sampled.index.to_numpy(dtype=int)]
    return sampled_embeddings, sampled.reset_index(drop=True)


class LocalNeighborBackend:
    """Tačna cosine pretraga u NumPy-ju, korisna za offline proveru."""

    name = "local"

    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        self.embeddings = embeddings
        self.metadata = metadata.reset_index(drop=True)
        self.similarities = embeddings @ embeddings.T
        self.id_to_index = {
            int(point_id): index for index, point_id in enumerate(self.metadata["id"])
        }

    def validate(self, expected_count: int) -> None:
        if len(self.metadata) != expected_count:
            raise RuntimeError(
                f"Lokalni backend ima {len(self.metadata)} pointova, očekivano {expected_count}."
            )

    def search(self, query_id: int, query_vector: np.ndarray, limit: int) -> list[Neighbor]:
        if query_id not in self.id_to_index:
            raise KeyError(f"ID={query_id} ne postoji u lokalnim metapodacima.")

        query_index = self.id_to_index[query_id]
        scores = self.similarities[query_index].copy()
        scores[query_index] = -np.inf
        ordered_indices = np.argsort(-scores, kind="stable")[:limit]

        neighbors: list[Neighbor] = []
        for index in ordered_indices:
            row = self.metadata.iloc[int(index)]
            neighbors.append(
                Neighbor(
                    point_id=int(row["id"]),
                    label=str(row["label"]),
                    image_path=str(row["image_path"]),
                    score=float(scores[index]),
                )
            )
        return neighbors


class QdrantNeighborBackend:
    """Tačna cosine pretraga nad kolekcijom stl10_clip_images."""

    name = "qdrant"

    def __init__(
        self,
        client: Any | None = None,
        collection_name: str | None = None,
        batch_size: int = 100,
        labeled_only: bool = False,
    ):
        try:
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchValue,
                QueryRequest,
                SearchParams,
            )
        except ImportError as exc:
            raise ImportError(
                "Qdrant backend zahteva qdrant-client. Pokreni: "
                "python -m pip install -r requirements.txt"
            ) from exc

        if client is None:
            from qdrant_common import COLLECTION_NAME, get_qdrant_client

            client = get_qdrant_client()
            collection_name = collection_name or COLLECTION_NAME

        self.client = client
        self.collection_name = collection_name or "stl10_clip_images"
        self.search_params = SearchParams(exact=True)
        self.query_request_class = QueryRequest
        self.query_filter = (
            Filter(
                must=[
                    FieldCondition(
                        key="is_labeled",
                        match=MatchValue(value=True),
                    )
                ]
            )
            if labeled_only
            else None
        )
        self.batch_size = batch_size
        self._neighbor_cache: dict[int, list[Neighbor]] = {}

        if self.batch_size < 1:
            raise ValueError("Qdrant batch size mora biti najmanje 1.")

    def validate(self, expected_count: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            raise RuntimeError(
                f"Qdrant kolekcija '{self.collection_name}' ne postoji. "
                "Pokreni kreiranje i import pre analize."
            )

        count = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count
        if count != expected_count:
            raise RuntimeError(
                f"Qdrant ima {count} pointova, a lokalni validni metadata imaju "
                f"{expected_count}. Očekuje se da brojevi budu jednaki."
            )

    def _convert_result(self, query_id: int, points: list[Any], limit: int) -> list[Neighbor]:
        neighbors: list[Neighbor] = []
        for point in points:
            point_id = int(point.id)
            if point_id == query_id:
                continue

            payload = point.payload or {}
            neighbors.append(
                Neighbor(
                    point_id=point_id,
                    label=str(payload.get("label", "unknown")),
                    image_path=str(payload.get("image_path", "")),
                    score=float(point.score),
                )
            )
            if len(neighbors) == limit:
                break

        if len(neighbors) < limit:
            raise RuntimeError(
                f"Qdrant je za ID={query_id} vratio samo {len(neighbors)} validnih "
                f"suseda, a traženo je {limit}."
            )
        return neighbors

    def prepare_batch(
        self,
        queries: list[tuple[int, np.ndarray]],
        limit: int,
    ) -> None:
        """Pošalji Qdrant upite paketno i keširaj rezultate za punu analizu."""
        self._neighbor_cache.clear()
        total_batches = math.ceil(len(queries) / self.batch_size)

        for batch_number, start in enumerate(
            range(0, len(queries), self.batch_size), start=1
        ):
            batch = queries[start : start + self.batch_size]
            requests = [
                self.query_request_class(
                    query=vector.astype(float).tolist(),
                    limit=limit + 3,
                    with_payload=["label", "image_path"],
                    with_vector=False,
                    params=self.search_params,
                    filter=self.query_filter,
                )
                for _, vector in batch
            ]
            responses = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=requests,
            )

            if len(responses) != len(batch):
                raise RuntimeError(
                    "Qdrant batch odgovor nema isti broj rezultata kao poslati upiti."
                )

            for (query_id, _), response in zip(batch, responses):
                self._neighbor_cache[query_id] = self._convert_result(
                    query_id=query_id,
                    points=response.points,
                    limit=limit,
                )

            print(
                f"Qdrant batch: {batch_number}/{total_batches} "
                f"({min(start + len(batch), len(queries))}/{len(queries)} upita)"
            )

    def search(self, query_id: int, query_vector: np.ndarray, limit: int) -> list[Neighbor]:
        cached = self._neighbor_cache.get(query_id)
        if cached is not None and len(cached) >= limit:
            return cached[:limit]

        # Single-query fallback keeps interactive inspection compatible with older clients.
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.astype(float).tolist(),
            limit=limit + 3,
            with_payload=["label", "image_path"],
            with_vectors=False,
            search_params=self.search_params,
            query_filter=self.query_filter,
        )
        return self._convert_result(query_id, result.points, limit)


def build_backend(
    backend_name: str,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    qdrant_batch_size: int = 100,
    labeled_only: bool = False,
) -> NeighborBackend:
    if backend_name == "local":
        return LocalNeighborBackend(embeddings, metadata)
    if backend_name == "qdrant":
        return QdrantNeighborBackend(
            batch_size=qdrant_batch_size,
            labeled_only=labeled_only,
        )
    raise ValueError(f"Nepoznat backend: {backend_name}")


def weighted_knn_prediction(neighbors: list[Neighbor]) -> dict[str, Any]:
    if not neighbors:
        raise ValueError("Za klasifikaciju je potreban bar jedan sused.")

    weighted_votes: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    score_sums: dict[str, float] = defaultdict(float)

    for neighbor in neighbors:
        # Negative cosine similarity must not contribute a negative class vote.
        weight = max(neighbor.score, 0.0) + 1e-9
        weighted_votes[neighbor.label] += weight
        counts[neighbor.label] += 1
        score_sums[neighbor.label] += neighbor.score

    labels = sorted(weighted_votes)
    ordered = sorted(
        labels,
        key=lambda label: (
            -weighted_votes[label],
            -counts[label],
            -(score_sums[label] / counts[label]),
            label,
        ),
    )

    predicted = ordered[0]
    total_weight = sum(weighted_votes.values())
    top_weight = weighted_votes[predicted]
    second_weight = weighted_votes[ordered[1]] if len(ordered) > 1 else 0.0

    return {
        "predicted_label": predicted,
        "prediction_confidence": top_weight / total_weight,
        "vote_margin": (top_weight - second_weight) / total_weight,
        "weighted_votes": dict(sorted(weighted_votes.items())),
        "neighbor_counts": dict(sorted(counts.items())),
    }


DIAGNOSIS_TEXT = {
    "possible_annotation_issue": (
        "Svi veoma slični susedi pripadaju drugoj klasi. Ovo nije dokaz greške u "
        "datasetu, ali je dobar kandidat za ručnu proveru anotacije i same slike."
    ),
    "class_confusion": (
        "Većina najbližih CLIP vektora pripada predviđenoj, a ne stvarnoj klasi. "
        "Embedding prostor lokalno meša ove dve klase."
    ),
    "boundary_case": (
        "U susedstvu postoje i stvarna i predviđena klasa. Slika se nalazi blizu "
        "granice klasa i mala razlika u sličnosti menja odluku."
    ),
    "ambiguous_or_outlier": (
        "Susedi su pomešani ili nisu dovoljno slični. Slika može biti netipična, "
        "vizuelno nejasna ili van gustog dela svoje klase."
    ),
}


def diagnose_error(
    true_label: str,
    predicted_label: str,
    neighbors: list[Neighbor],
    annotation_similarity: float = 0.92,
    low_similarity: float = 0.55,
) -> dict[str, Any]:
    k = len(neighbors)
    labels = [neighbor.label for neighbor in neighbors]
    counts = Counter(labels)
    predicted_support = counts[predicted_label] / k
    true_support = counts[true_label] / k
    top_score = neighbors[0].score
    predicted_scores = [n.score for n in neighbors if n.label == predicted_label]
    predicted_average_score = (
        float(np.mean(predicted_scores)) if predicted_scores else float("nan")
    )

    if (
        predicted_support == 1.0
        and true_support == 0.0
        and predicted_average_score >= annotation_similarity
    ):
        diagnosis = "possible_annotation_issue"
    elif top_score < low_similarity:
        diagnosis = "ambiguous_or_outlier"
    elif predicted_support >= 0.6:
        diagnosis = "class_confusion"
    elif true_support > 0:
        diagnosis = "boundary_case"
    else:
        diagnosis = "ambiguous_or_outlier"

    return {
        "diagnosis": diagnosis,
        "diagnosis_explanation": DIAGNOSIS_TEXT[diagnosis],
        "predicted_neighbor_support": predicted_support,
        "true_neighbor_support": true_support,
        "top_neighbor_score": top_score,
        "average_predicted_neighbor_score": predicted_average_score,
    }


def class_order(metadata: pd.DataFrame) -> list[str]:
    labels = set(metadata["label"].astype(str))
    canonical = [label for label in CANONICAL_STL10_CLASSES if label in labels]
    extras = sorted(labels - set(canonical))
    return canonical + extras


def analyze_dataset(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    backend: NeighborBackend,
    k: int,
    annotation_similarity: float,
    low_similarity: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[Neighbor]]]:
    if k < 1:
        raise ValueError("k mora biti najmanje 1.")
    if k >= len(metadata):
        raise ValueError("k mora biti manji od broja slika.")

    prediction_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    error_neighbors: dict[int, list[Neighbor]] = {}

    prepare_batch = getattr(backend, "prepare_batch", None)
    if callable(prepare_batch):
        print("Šaljem Qdrant upite paketno...")
        prepare_batch(
            [
                (int(metadata.iloc[index]["id"]), embeddings[index])
                for index in range(len(metadata))
            ],
            k,
        )

    total = len(metadata)
    for index, row in metadata.iterrows():
        point_id = int(row["id"])
        neighbors = backend.search(point_id, embeddings[index], k)
        prediction = weighted_knn_prediction(neighbors)
        predicted_label = str(prediction["predicted_label"])
        true_label = str(row["label"])
        is_correct = predicted_label == true_label

        base_row = {
            "id": point_id,
            "image_path": str(row["image_path"]),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "is_correct": is_correct,
            "prediction_confidence": float(prediction["prediction_confidence"]),
            "vote_margin": float(prediction["vote_margin"]),
            "top_neighbor_id": neighbors[0].point_id,
            "top_neighbor_label": neighbors[0].label,
            "top_neighbor_score": neighbors[0].score,
            "neighbor_labels": ", ".join(n.label for n in neighbors),
        }
        prediction_rows.append(base_row)

        if not is_correct:
            diagnosis = diagnose_error(
                true_label=true_label,
                predicted_label=predicted_label,
                neighbors=neighbors,
                annotation_similarity=annotation_similarity,
                low_similarity=low_similarity,
            )
            error_rows.append({**base_row, **diagnosis})
            error_neighbors[point_id] = neighbors

        processed = index + 1
        if processed % 100 == 0 or processed == total:
            print(f"Analizirano: {processed}/{total}")

    predictions = pd.DataFrame(prediction_rows)
    errors = pd.DataFrame(error_rows)
    if not errors.empty:
        errors = errors.sort_values(
            ["diagnosis", "prediction_confidence", "top_neighbor_score"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    return predictions, errors, error_neighbors


def confusion_matrix(predictions: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    matrix = pd.crosstab(
        predictions["true_label"],
        predictions["predicted_label"],
        rownames=["true_label"],
        colnames=["predicted_label"],
    )
    return matrix.reindex(index=labels, columns=labels, fill_value=0)


def per_class_metrics(predictions: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        class_rows = predictions[predictions["true_label"] == label]
        correct = int(class_rows["is_correct"].sum())
        support = len(class_rows)
        rows.append(
            {
                "label": label,
                "correct": correct,
                "errors": support - correct,
                "support": support,
                "accuracy": correct / support if support else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def most_common_confusions(predictions: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    errors = predictions[~predictions["is_correct"]]
    if errors.empty:
        return []
    grouped = (
        errors.groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "true_label", "predicted_label"], ascending=[False, True, True])
    )
    return grouped.head(limit).to_dict(orient="records")


def make_neighbor_details(
    errors: pd.DataFrame,
    error_neighbors: dict[int, list[Neighbor]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, error in errors.iterrows():
        query_id = int(error["id"])
        for rank, neighbor in enumerate(error_neighbors[query_id], start=1):
            rows.append(
                {
                    "query_id": query_id,
                    "query_true_label": error["true_label"],
                    "query_predicted_label": error["predicted_label"],
                    "rank": rank,
                    "neighbor_id": neighbor.point_id,
                    "neighbor_label": neighbor.label,
                    "neighbor_image_path": neighbor.image_path,
                    "score": neighbor.score,
                }
            )
    return pd.DataFrame(rows)


def safe_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(number):
        return "-"
    return f"{number:.{digits}f}"


def relative_image_source(image_path: str, output_dir: Path) -> str | None:
    path = Path(image_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return None
    return Path(os.path.relpath(path, output_dir)).as_posix()


def image_block(image_path: str, alt: str, output_dir: Path) -> str:
    source = relative_image_source(image_path, output_dir)
    if source is None:
        return (
            '<div class="missing-image">Slika nije lokalno preuzeta.<br>'
            f'<code>{html.escape(image_path)}</code></div>'
        )
    return f'<img src="{html.escape(source)}" alt="{html.escape(alt)}">'


def dataframe_table(frame: pd.DataFrame, percent_columns: set[str] | None = None) -> str:
    percent_columns = percent_columns or set()
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column, value in row.items():
            if column in percent_columns:
                text = f"{float(value) * 100:.2f}%"
            elif isinstance(value, (float, np.floating)):
                text = safe_float(value)
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def confusion_html(matrix: pd.DataFrame) -> str:
    header = "<th>stvarna ↓ / predikcija →</th>" + "".join(
        f"<th>{html.escape(label)}</th>" for label in matrix.columns
    )
    rows = []
    for true_label, row in matrix.iterrows():
        cells = [f"<th>{html.escape(str(true_label))}</th>"]
        for predicted_label, value in row.items():
            class_name = "diag" if true_label == predicted_label else ("error" if value else "")
            cells.append(f'<td class="{class_name}">{int(value)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table class="matrix"><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def generate_html_report(
    output_dir: Path,
    summary: dict[str, Any],
    matrix: pd.DataFrame,
    metrics: pd.DataFrame,
    errors: pd.DataFrame,
    error_neighbors: dict[int, list[Neighbor]],
    html_error_limit: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnosis_counts = summary["diagnosis_counts"]
    diagnosis_items = "".join(
        f"<li><strong>{html.escape(name)}</strong>: {count}</li>"
        for name, count in diagnosis_counts.items()
    ) or "<li>Nema pogrešnih klasifikacija.</li>"

    confusion_rows = summary["most_common_confusions"]
    if confusion_rows:
        common_confusions = pd.DataFrame(confusion_rows)
        common_confusions_html = dataframe_table(common_confusions)
    else:
        common_confusions_html = "<p>Nema pogrešnih klasifikacija.</p>"

    cards = []
    for _, error in errors.head(html_error_limit).iterrows():
        point_id = int(error["id"])
        query_image = image_block(
            str(error["image_path"]),
            f"Query slika {point_id}",
            output_dir,
        )
        neighbor_cards = []
        for rank, neighbor in enumerate(error_neighbors[point_id], start=1):
            neighbor_image = image_block(
                neighbor.image_path,
                f"Sused {neighbor.point_id}",
                output_dir,
            )
            neighbor_cards.append(
                f"""
                <div class="neighbor">
                  {neighbor_image}
                  <div><strong>#{rank} · ID {neighbor.point_id}</strong></div>
                  <div>label: {html.escape(neighbor.label)}</div>
                  <div>score: {neighbor.score:.4f}</div>
                </div>
                """
            )

        cards.append(
            f"""
            <section class="error-card">
              <div class="query">
                {query_image}
                <div>
                  <h3>ID {point_id}: {html.escape(str(error['true_label']))} → {html.escape(str(error['predicted_label']))}</h3>
                  <p><strong>Dijagnoza:</strong> {html.escape(str(error['diagnosis']))}</p>
                  <p>{html.escape(str(error['diagnosis_explanation']))}</p>
                  <p>confidence: {safe_float(error['prediction_confidence'])} · margin: {safe_float(error['vote_margin'])}</p>
                </div>
              </div>
              <div class="neighbors">{"".join(neighbor_cards)}</div>
            </section>
            """
        )

    generated_at = html.escape(summary["generated_at"])
    document = f"""<!doctype html>
<html lang="sr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Analiza grešaka modela</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#64748b; --line:#dbe3ef; --blue:#2563eb; --red:#fee2e2; --green:#dcfce7; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--ink); background:#f6f8fc; line-height:1.5; }}
    main {{ max-width:1200px; margin:0 auto; padding:32px 20px 64px; }}
    h1,h2,h3 {{ line-height:1.2; }}
    .muted {{ color:var(--muted); }}
    .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:24px 0; }}
    .stat,.panel,.error-card {{ background:white; border:1px solid var(--line); border-radius:14px; box-shadow:0 6px 18px rgba(23,32,51,.05); }}
    .stat {{ padding:18px; }} .stat strong {{ display:block; font-size:1.7rem; color:var(--blue); }}
    .panel {{ padding:20px; margin:18px 0; }}
    .table-wrap {{ overflow:auto; }} table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
    th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:center; white-space:nowrap; }} th {{ background:#eff4fb; }}
    .matrix td.diag {{ background:var(--green); font-weight:700; }} .matrix td.error {{ background:var(--red); font-weight:700; }}
    .error-card {{ padding:20px; margin:20px 0; }} .query {{ display:grid; grid-template-columns:180px 1fr; gap:20px; align-items:start; }}
    img {{ width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:10px; background:#e8edf5; }}
    .missing-image {{ min-height:120px; display:grid; place-content:center; text-align:center; padding:10px; border:1px dashed #94a3b8; border-radius:10px; color:var(--muted); background:#f8fafc; }}
    .neighbors {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:12px; margin-top:18px; }}
    .neighbor {{ border:1px solid var(--line); border-radius:10px; padding:10px; font-size:.86rem; }}
    code {{ overflow-wrap:anywhere; }}
    @media (max-width:650px) {{ .query {{ grid-template-columns:1fr; }} .query img {{ max-width:220px; }} }}
  </style>
</head>
<body><main>
  <h1>Analiza grešaka modela pomoću Qdranta</h1>
  <p class="muted">CLIP embedding + weighted k-NN (k={summary['k']}, backend={html.escape(summary['backend'])}) · generisano {generated_at}</p>
  <div class="summary">
    <div class="stat"><strong>{summary['total_images']}</strong>slika</div>
    <div class="stat"><strong>{summary['correct_predictions']}</strong>tačno</div>
    <div class="stat"><strong>{summary['error_count']}</strong>grešaka</div>
    <div class="stat"><strong>{summary['accuracy'] * 100:.2f}%</strong>tačnost</div>
  </div>
  <section class="panel"><h2>Šta je urađeno</h2><p>Za svaku sliku pronađeno je k najbližih drugih slika po cosine sličnosti. Susedi glasaju ponderisano svojim score-om. Kod pogrešne predikcije prikazani su susedi koji objašnjavaju odluku i heuristička dijagnoza. Oznaka <em>possible_annotation_issue</em> znači samo kandidat za ručnu proveru, ne dokaz pogrešne anotacije.</p></section>
  <section class="panel"><h2>Vrste pronađenih grešaka</h2><ul>{diagnosis_items}</ul></section>
  <section class="panel"><h2>Najčešće zamene klasa</h2>{common_confusions_html}</section>
  <section class="panel"><h2>Tačnost po klasama</h2>{dataframe_table(metrics, {'accuracy'})}</section>
  <section class="panel"><h2>Matrica konfuzije</h2><p class="muted">Red je stvarna klasa, kolona je predikcija.</p>{confusion_html(matrix)}</section>
  <h2>Primeri pogrešnih klasifikacija</h2>
  {"".join(cards) if cards else '<p>Nema grešaka za prikaz.</p>'}
</main></body></html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def save_analysis(
    output_dir: Path,
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
    error_neighbors: dict[int, list[Neighbor]],
    backend_name: str,
    k: int,
    html_error_limit: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = class_order(predictions.rename(columns={"true_label": "label"}))
    matrix = confusion_matrix(predictions, labels)
    metrics = per_class_metrics(predictions, labels)

    diagnosis_counts = (
        errors["diagnosis"].value_counts().sort_index().to_dict()
        if not errors.empty
        else {}
    )
    correct_count = int(predictions["is_correct"].sum())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "k": k,
        "total_images": len(predictions),
        "correct_predictions": correct_count,
        "error_count": len(predictions) - correct_count,
        "accuracy": correct_count / len(predictions),
        "diagnosis_counts": diagnosis_counts,
        "most_common_confusions": most_common_confusions(predictions),
    }

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    errors.to_csv(output_dir / "errors.csv", index=False)
    make_neighbor_details(errors, error_neighbors).to_csv(
        output_dir / "error_neighbors.csv", index=False
    )
    matrix.to_csv(output_dir / "confusion_matrix.csv")
    metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = generate_html_report(
        output_dir=output_dir,
        summary=summary,
        matrix=matrix,
        metrics=metrics,
        errors=errors,
        error_neighbors=error_neighbors,
        html_error_limit=html_error_limit,
    )
    summary["report_path"] = str(report_path)
    return summary


def print_single_analysis(
    row: pd.Series,
    neighbors: list[Neighbor],
    annotation_similarity: float,
    low_similarity: float,
) -> bool:
    prediction = weighted_knn_prediction(neighbors)
    true_label = str(row["label"])
    predicted_label = str(prediction["predicted_label"])
    is_correct = true_label == predicted_label

    print()
    print("QUERY SLIKA")
    print(f"ID: {int(row['id'])}")
    print(f"Putanja: {row['image_path']}")
    print(f"Stvarna klasa: {true_label}")
    print(f"Predikcija: {predicted_label}")
    print(f"Tačno: {is_correct}")
    print(f"Confidence glasanja: {prediction['prediction_confidence']:.4f}")
    print(f"Vote margin: {prediction['vote_margin']:.4f}")
    print()
    print("NAJBLIŽI SUSEDI")
    for rank, neighbor in enumerate(neighbors, start=1):
        print(
            f"{rank}. ID={neighbor.point_id} | label={neighbor.label} | "
            f"score={neighbor.score:.4f} | {neighbor.image_path}"
        )

    if not is_correct:
        diagnosis = diagnose_error(
            true_label=true_label,
            predicted_label=predicted_label,
            neighbors=neighbors,
            annotation_similarity=annotation_similarity,
            low_similarity=low_similarity,
        )
        print()
        print("TUMAČENJE GREŠKE")
        print(f"Kategorija: {diagnosis['diagnosis']}")
        print(diagnosis["diagnosis_explanation"])
        print(
            f"Podrška stvarne klase među susedima: "
            f"{diagnosis['true_neighbor_support']:.0%}"
        )
        print(
            f"Podrška predviđene klase među susedima: "
            f"{diagnosis['predicted_neighbor_support']:.0%}"
        )
    return is_correct


def find_demo_error(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    backend: NeighborBackend,
    k: int,
) -> tuple[int, list[Neighbor]]:
    for index, row in metadata.iterrows():
        neighbors = backend.search(int(row["id"]), embeddings[index], k)
        prediction = weighted_knn_prediction(neighbors)
        if prediction["predicted_label"] != str(row["label"]):
            return index, neighbors
    raise RuntimeError("Nije pronađena pogrešno klasifikovana slika.")


def command_validate(args: argparse.Namespace) -> None:
    embeddings, metadata = load_inputs()
    backend = build_backend(args.backend, embeddings, metadata)
    backend.validate(len(metadata))

    labels = class_order(metadata)
    print("VALIDACIJA USPEŠNA")
    print(f"Backend: {backend.name}")
    print(f"Broj validnih slika: {len(metadata)}")
    print(f"Dimenzija embeddinga: {embeddings.shape[1]}")
    print(f"Broj klasa: {len(labels)}")
    print(f"Klase: {', '.join(labels)}")


def command_analyze(args: argparse.Namespace) -> None:
    all_embeddings, all_metadata = load_inputs()
    embeddings, metadata = select_analysis_sample(all_embeddings, all_metadata)

    if args.backend == "qdrant":
        backend = build_backend(
            args.backend,
            embeddings,
            metadata,
            qdrant_batch_size=args.qdrant_batch_size,
            labeled_only=True,
        )
        backend.validate(len(all_metadata))
    else:
        backend = build_backend(args.backend, embeddings, metadata)
        backend.validate(len(metadata))

    print(f"Pokrećem analizu: backend={backend.name}, k={args.k}")
    print(
        f"Analizira se {len(metadata)} labeliranih slika "
        f"({ANALYSIS_IMAGES_PER_CLASS} po klasi)."
    )
    predictions, errors, error_neighbors = analyze_dataset(
        embeddings=embeddings,
        metadata=metadata,
        backend=backend,
        k=args.k,
        annotation_similarity=args.annotation_similarity,
        low_similarity=args.low_similarity,
    )
    summary = save_analysis(
        output_dir=Path(args.output_dir),
        predictions=predictions,
        errors=errors,
        error_neighbors=error_neighbors,
        backend_name=backend.name,
        k=args.k,
        html_error_limit=args.html_error_limit,
    )

    print()
    print("ANALIZA ZAVRŠENA")
    print(f"Ukupno slika: {summary['total_images']}")
    print(f"Tačno: {summary['correct_predictions']}")
    print(f"Greške: {summary['error_count']}")
    print(f"Tačnost: {summary['accuracy']:.2%}")
    print(f"Dijagnoze: {summary['diagnosis_counts']}")
    print(f"HTML izveštaj: {summary['report_path']}")


def command_inspect(args: argparse.Namespace) -> None:
    embeddings, metadata = load_inputs()
    backend = build_backend(args.backend, embeddings, metadata)
    backend.validate(len(metadata))

    matches = metadata.index[metadata["id"] == args.id].tolist()
    if not matches:
        raise KeyError(f"Slika sa ID={args.id} ne postoji.")
    index = matches[0]
    neighbors = backend.search(args.id, embeddings[index], args.k)
    print_single_analysis(
        row=metadata.iloc[index],
        neighbors=neighbors,
        annotation_similarity=args.annotation_similarity,
        low_similarity=args.low_similarity,
    )


def command_demo(args: argparse.Namespace) -> None:
    embeddings, metadata = load_inputs()
    backend = build_backend(args.backend, embeddings, metadata)
    backend.validate(len(metadata))

    if args.id is None:
        index, neighbors = find_demo_error(embeddings, metadata, backend, args.k)
    else:
        matches = metadata.index[metadata["id"] == args.id].tolist()
        if not matches:
            raise KeyError(f"Slika sa ID={args.id} ne postoji.")
        index = matches[0]
        neighbors = backend.search(args.id, embeddings[index], args.k)

    print("DEMO: objašnjenje jedne klasifikacije")
    print_single_analysis(
        row=metadata.iloc[index],
        neighbors=neighbors,
        annotation_similarity=args.annotation_similarity,
        low_similarity=args.low_similarity,
    )


def command_compare(args: argparse.Namespace) -> None:
    embeddings, metadata = load_inputs()
    local_backend = LocalNeighborBackend(embeddings, metadata)
    qdrant_backend = QdrantNeighborBackend()
    qdrant_backend.validate(len(metadata))

    sample_size = min(args.sample_size, len(metadata))
    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(len(metadata), size=sample_size, replace=False).tolist())
    identical_predictions = 0
    identical_neighbor_ids = 0

    for index in indices:
        row = metadata.iloc[index]
        point_id = int(row["id"])
        local_neighbors = local_backend.search(point_id, embeddings[index], args.k)
        qdrant_neighbors = qdrant_backend.search(point_id, embeddings[index], args.k)

        local_prediction = weighted_knn_prediction(local_neighbors)["predicted_label"]
        qdrant_prediction = weighted_knn_prediction(qdrant_neighbors)["predicted_label"]
        identical_predictions += int(local_prediction == qdrant_prediction)
        identical_neighbor_ids += int(
            [n.point_id for n in local_neighbors] == [n.point_id for n in qdrant_neighbors]
        )

    print("POREĐENJE QDRANT I LOKALNE COSINE PRETRAGE")
    print(f"Provereno slika: {sample_size}")
    print(f"Iste k-NN predikcije: {identical_predictions}/{sample_size}")
    print(f"Identičan redosled suseda: {identical_neighbor_ids}/{sample_size}")
    if identical_predictions != sample_size:
        raise RuntimeError("Qdrant i lokalna pretraga nisu dali iste predikcije za sve uzorke.")


def add_shared_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["qdrant", "local"], default="qdrant")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--annotation-similarity",
        type=float,
        default=0.92,
        help="Minimalna prosečna sličnost za oprezni annotation flag.",
    )
    parser.add_argument(
        "--low-similarity",
        type=float,
        default=0.55,
        help="Ispod ove top-1 sličnosti slika se tretira kao mogući outlier.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analiza grešaka CLIP + weighted k-NN modela"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Proveri ulazne fajlove i backend.")
    validate_parser.add_argument("--backend", choices=["qdrant", "local"], default="qdrant")

    analyze_parser = subparsers.add_parser("analyze", help="Analiziraj svih 1000 slika.")
    add_shared_analysis_arguments(analyze_parser)
    analyze_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    analyze_parser.add_argument(
        "--qdrant-batch-size",
        type=int,
        default=100,
        help="Broj Qdrant upita u jednom HTTP zahtevu (podrazumevano 100).",
    )
    analyze_parser.add_argument(
        "--html-error-limit",
        type=int,
        default=30,
        help="Maksimalan broj error kartica u HTML-u; CSV uvek sadrži sve greške.",
    )

    inspect_parser = subparsers.add_parser("inspect", help="Detaljno analiziraj zadati ID.")
    inspect_parser.add_argument("id", type=int)
    add_shared_analysis_arguments(inspect_parser)

    demo_parser = subparsers.add_parser("demo", help="Prikaži jednu grešku i njene susede.")
    demo_parser.add_argument("--id", type=int, default=None)
    add_shared_analysis_arguments(demo_parser)

    compare_parser = subparsers.add_parser(
        "compare-backends",
        help="Uporedi Qdrant rezultate sa tačnom lokalnom cosine pretragom.",
    )
    compare_parser.add_argument("--sample-size", type=int, default=30)
    compare_parser.add_argument("--k", type=int, default=5)
    compare_parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        command_validate(args)
    elif args.command == "analyze":
        command_analyze(args)
    elif args.command == "inspect":
        command_inspect(args)
    elif args.command == "demo":
        command_demo(args)
    elif args.command == "compare-backends":
        command_compare(args)


if __name__ == "__main__":
    main()
