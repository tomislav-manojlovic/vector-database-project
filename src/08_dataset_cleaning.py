"""Varijanta 3: pronalaženje duplikata i bezbedno čišćenje dataseta.

Qdrant se koristi za pronalaženje veoma sličnih CLIP embeddinga. Skripta nikad
ne briše originalne slike ili originalne metadata fajlove. Ona pravi izveštaj,
predloge za ručni pregled i novu, odvojenu očišćenu verziju dataseta.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings.npy"
EMBEDDINGS_METADATA_PATH = (
    ROOT_DIR / "data" / "embeddings" / "embeddings_metadata.csv"
)
EMBEDDING_CONFIG_PATH = ROOT_DIR / "data" / "embeddings" / "embedding_config.json"
FULL_METADATA_PATH = ROOT_DIR / "data" / "metadata.csv"
DEFAULT_REPORT_DIR = ROOT_DIR / "reports" / "variant3_dataset_cleaning"
DEFAULT_CLEANED_DIR = ROOT_DIR / "data" / "cleaned"
CLEANING_IMAGES_PER_CLASS = 100

REQUIRED_COLUMNS = {
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
class SimilarPair:
    left_id: int
    right_id: int
    left_label: str
    right_label: str
    left_image_path: str
    right_image_path: str
    score: float


class SimilarityBackend(Protocol):
    name: str

    def validate(self, expected_count: int) -> None:
        ...

    def find_pairs(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        threshold: float,
        top_k: int,
    ) -> list[SimilarPair]:
        ...


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Pronađen je embedding sa L2 normom jednakom nuli.")
    return (vectors / norms).astype(np.float32)


def load_inputs() -> tuple[np.ndarray, pd.DataFrame]:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Nedostaje: {EMBEDDINGS_PATH}")
    if not EMBEDDINGS_METADATA_PATH.exists():
        raise FileNotFoundError(f"Nedostaje: {EMBEDDINGS_METADATA_PATH}")

    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(EMBEDDINGS_METADATA_PATH)
    missing = REQUIRED_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata fajlu nedostaju kolone: {sorted(missing)}")

    metadata = metadata[metadata["status"] == "ok"].copy()
    metadata["id"] = metadata["id"].astype(int)
    metadata["embedding_index"] = metadata["embedding_index"].astype(int)
    metadata["label"] = metadata["label"].astype(str)
    metadata["image_path"] = metadata["image_path"].astype(str)
    metadata = metadata.sort_values("embedding_index").reset_index(drop=True)

    if embeddings.ndim != 2:
        raise ValueError("embeddings.npy mora biti 2D matrica.")
    if metadata.empty:
        raise ValueError("Nema redova sa status='ok'.")
    if metadata["id"].duplicated().any():
        raise ValueError("Metadata sadrži duplirane ID-eve.")
    if metadata["embedding_index"].duplicated().any():
        raise ValueError("Metadata sadrži duplirane embedding_index vrednosti.")

    indices = metadata["embedding_index"].to_numpy(dtype=int)
    if indices.min() < 0 or indices.max() >= len(embeddings):
        raise ValueError("Neki embedding_index izlazi iz granica embeddings.npy.")

    selected = normalize_rows(embeddings[indices].astype(np.float32))
    return selected, metadata


def select_cleaning_sample(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    images_per_class: int = CLEANING_IMAGES_PER_CLASS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Izaberi uravnotežen uzorak labeliranih slika za interaktivnu analizu."""
    labeled_mask = metadata["is_labeled"].astype(str).str.lower().isin({"true", "1"})
    labeled = metadata[labeled_mask]
    if labeled.empty:
        raise ValueError("Nema labeliranih slika za analizu čišćenja.")

    sampled = labeled.groupby("label", sort=False).head(images_per_class).copy()
    sampled_embeddings = embeddings[sampled.index.to_numpy(dtype=int)]
    return sampled_embeddings, sampled.reset_index(drop=True)


def pair_from_rows(left: pd.Series, right: pd.Series, score: float) -> SimilarPair:
    if int(left["id"]) > int(right["id"]):
        left, right = right, left
    return SimilarPair(
        left_id=int(left["id"]),
        right_id=int(right["id"]),
        left_label=str(left["label"]),
        right_label=str(right["label"]),
        left_image_path=str(left["image_path"]),
        right_image_path=str(right["image_path"]),
        score=float(score),
    )


class LocalSimilarityBackend:
    """Tačna NumPy cosine pretraga za offline proveru."""

    name = "local"

    def validate(self, expected_count: int) -> None:
        if expected_count < 2:
            raise RuntimeError("Za traženje parova potrebne su bar dve slike.")

    def find_pairs(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        threshold: float,
        top_k: int,
    ) -> list[SimilarPair]:
        similarities = embeddings @ embeddings.T
        left_indices, right_indices = np.triu_indices(len(embeddings), k=1)
        scores = similarities[left_indices, right_indices]
        selected = np.flatnonzero(scores >= threshold)

        pairs = [
            pair_from_rows(
                metadata.iloc[int(left_indices[index])],
                metadata.iloc[int(right_indices[index])],
                float(scores[index]),
            )
            for index in selected
        ]
        return sorted(pairs, key=lambda pair: (-pair.score, pair.left_id, pair.right_id))


class QdrantSimilarityBackend:
    """Batch cosine pretraga kroz Qdrant kolekciju."""

    name = "qdrant"

    def __init__(
        self,
        client: Any | None = None,
        collection_name: str | None = None,
        batch_size: int = 100,
        allowed_ids: list[int] | None = None,
    ):
        try:
            from qdrant_client.models import Filter, HasIdCondition, QueryRequest, SearchParams
        except ImportError as exc:
            raise ImportError(
                "Instaliraj zavisnosti: python -m pip install -r requirements.txt"
            ) from exc

        if client is None:
            from qdrant_common import COLLECTION_NAME, get_qdrant_client

            client = get_qdrant_client()
            collection_name = collection_name or COLLECTION_NAME

        if batch_size < 1:
            raise ValueError("batch_size mora biti najmanje 1.")

        self.client = client
        self.collection_name = collection_name or "stl10_clip_images"
        self.batch_size = batch_size
        self.query_request_class = QueryRequest
        self.search_params = SearchParams(exact=True)
        self.query_filter = (
            Filter(must=[HasIdCondition(has_id=allowed_ids)])
            if allowed_ids
            else None
        )

    def validate(self, expected_count: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            raise RuntimeError(
                f"Qdrant kolekcija '{self.collection_name}' ne postoji."
            )
        count = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count
        if count != expected_count:
            raise RuntimeError(
                f"Qdrant ima {count} pointova, a metadata {expected_count}. "
                "Uradi čist import pre analize."
            )

    def find_pairs(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        threshold: float,
        top_k: int,
    ) -> list[SimilarPair]:
        if top_k < 2:
            raise ValueError("top_k mora biti najmanje 2.")

        metadata_by_id = metadata.set_index("id", drop=False)
        unique_pairs: dict[tuple[int, int], SimilarPair] = {}
        total_batches = math.ceil(len(metadata) / self.batch_size)

        for batch_number, start in enumerate(
            range(0, len(metadata), self.batch_size), start=1
        ):
            end = min(start + self.batch_size, len(metadata))
            requests = [
                self.query_request_class(
                    query=embeddings[index].astype(float).tolist(),
                    score_threshold=threshold,
                    limit=top_k,
                    with_payload=["label", "image_path"],
                    with_vector=False,
                    params=self.search_params,
                    filter=self.query_filter,
                )
                for index in range(start, end)
            ]
            responses = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=requests,
            )
            if len(responses) != end - start:
                raise RuntimeError("Neispravan broj Qdrant batch odgovora.")

            for index, response in zip(range(start, end), responses):
                query_row = metadata.iloc[index]
                query_id = int(query_row["id"])
                points = response.points

                if (
                    len(points) == top_k
                    and points[-1].score is not None
                    and float(points[-1].score) >= threshold
                ):
                    raise RuntimeError(
                        f"ID={query_id} ima najmanje {top_k} rezultata iznad praga. "
                        "Ponovi sa većim --top-k."
                    )

                for point in points:
                    neighbor_id = int(point.id)
                    if neighbor_id == query_id:
                        continue
                    if neighbor_id not in metadata_by_id.index:
                        continue

                    key = tuple(sorted((query_id, neighbor_id)))
                    neighbor_row = metadata_by_id.loc[neighbor_id]
                    candidate = pair_from_rows(
                        query_row,
                        neighbor_row,
                        float(point.score),
                    )
                    previous = unique_pairs.get(key)
                    if previous is None or candidate.score > previous.score:
                        unique_pairs[key] = candidate

            print(
                f"Qdrant batch: {batch_number}/{total_batches} "
                f"({end}/{len(metadata)} slika)"
            )

        return sorted(
            unique_pairs.values(),
            key=lambda pair: (-pair.score, pair.left_id, pair.right_id),
        )


def validate_thresholds(
    candidate_threshold: float,
    probable_threshold: float,
    very_likely_threshold: float,
) -> None:
    if not 0 <= candidate_threshold <= 1:
        raise ValueError("candidate threshold mora biti između 0 i 1.")
    if not candidate_threshold <= probable_threshold <= very_likely_threshold <= 1:
        raise ValueError(
            "Mora važiti: candidate <= probable <= very-likely <= 1."
        )


def pair_category(
    score: float,
    probable_threshold: float,
    very_likely_threshold: float,
) -> str:
    if score >= very_likely_threshold:
        return "very_likely_duplicate"
    if score >= probable_threshold:
        return "probable_duplicate"
    return "very_similar"


class UnionFind:
    def __init__(self, values: set[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def build_groups(
    pairs: list[SimilarPair],
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    very_likely_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "group_id",
        "point_id",
        "label",
        "image_path",
        "representative_id",
        "similarity_to_representative",
        "group_size",
        "labels_in_group",
        "has_label_conflict",
        "recommended_action",
    ]
    summary_columns = [
        "group_id",
        "group_size",
        "representative_id",
        "labels_in_group",
        "has_label_conflict",
        "maximum_pair_score",
        "minimum_pair_score",
        "remove_candidates",
    ]
    if not pairs:
        return pd.DataFrame(columns=group_columns), pd.DataFrame(columns=summary_columns)

    nodes = {pair.left_id for pair in pairs} | {pair.right_id for pair in pairs}
    union_find = UnionFind(nodes)
    for pair in pairs:
        union_find.union(pair.left_id, pair.right_id)

    components: dict[int, list[int]] = {}
    for point_id in sorted(nodes):
        components.setdefault(union_find.find(point_id), []).append(point_id)

    metadata_by_id = metadata.set_index("id", drop=False)
    index_by_id = {int(row.id): index for index, row in metadata.iterrows()}
    pair_scores = {
        tuple(sorted((pair.left_id, pair.right_id))): pair.score for pair in pairs
    }
    member_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    sorted_components = sorted(components.values(), key=lambda ids: (min(ids), len(ids)))
    for group_id, point_ids in enumerate(sorted_components, start=1):
        vector_indices = [index_by_id[point_id] for point_id in point_ids]
        group_vectors = embeddings[vector_indices]
        similarities = group_vectors @ group_vectors.T
        average_similarities = (
            (similarities.sum(axis=1) - 1.0) / (len(point_ids) - 1)
        )
        best_average = float(average_similarities.max())
        representative_candidates = [
            point_ids[index]
            for index, value in enumerate(average_similarities)
            if np.isclose(float(value), best_average, atol=1e-7)
        ]
        representative_id = min(representative_candidates)
        representative_index = point_ids.index(representative_id)

        labels = [str(metadata_by_id.loc[point_id]["label"]) for point_id in point_ids]
        unique_labels = sorted(set(labels))
        has_label_conflict = len(unique_labels) > 1
        labels_text = ", ".join(unique_labels)

        relevant_scores = [
            score
            for (left_id, right_id), score in pair_scores.items()
            if left_id in point_ids and right_id in point_ids
        ]
        remove_candidates = 0

        for member_index, point_id in enumerate(point_ids):
            row = metadata_by_id.loc[point_id]
            similarity_to_representative = float(
                similarities[member_index, representative_index]
            )
            if point_id == representative_id:
                action = "keep"
            elif has_label_conflict:
                action = "review"
            elif similarity_to_representative >= very_likely_threshold:
                action = "remove_candidate"
                remove_candidates += 1
            else:
                action = "review"

            member_rows.append(
                {
                    "group_id": group_id,
                    "point_id": point_id,
                    "label": str(row["label"]),
                    "image_path": str(row["image_path"]),
                    "representative_id": representative_id,
                    "similarity_to_representative": similarity_to_representative,
                    "group_size": len(point_ids),
                    "labels_in_group": labels_text,
                    "has_label_conflict": has_label_conflict,
                    "recommended_action": action,
                }
            )

        group_rows.append(
            {
                "group_id": group_id,
                "group_size": len(point_ids),
                "representative_id": representative_id,
                "labels_in_group": labels_text,
                "has_label_conflict": has_label_conflict,
                "maximum_pair_score": max(relevant_scores),
                "minimum_pair_score": min(relevant_scores),
                "remove_candidates": remove_candidates,
            }
        )

    return pd.DataFrame(member_rows, columns=group_columns), pd.DataFrame(
        group_rows, columns=summary_columns
    )


def pairs_dataframe(
    pairs: list[SimilarPair],
    probable_threshold: float,
    very_likely_threshold: float,
) -> pd.DataFrame:
    columns = [
        "left_id",
        "right_id",
        "left_label",
        "right_label",
        "same_label",
        "left_image_path",
        "right_image_path",
        "score",
        "category",
    ]
    rows = []
    for pair in pairs:
        rows.append(
            {
                "left_id": pair.left_id,
                "right_id": pair.right_id,
                "left_label": pair.left_label,
                "right_label": pair.right_label,
                "same_label": pair.left_label == pair.right_label,
                "left_image_path": pair.left_image_path,
                "right_image_path": pair.right_image_path,
                "score": pair.score,
                "category": pair_category(
                    pair.score, probable_threshold, very_likely_threshold
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def relative_image_source(image_path: str, report_dir: Path) -> str | None:
    path = Path(image_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return None
    return Path(os.path.relpath(path, report_dir)).as_posix()


def image_html(image_path: str, report_dir: Path, alt: str) -> str:
    source = relative_image_source(image_path, report_dir)
    if source is None:
        return (
            '<div class="missing">Slika nije lokalno preuzeta.<br>'
            f"<code>{html.escape(image_path)}</code></div>"
        )
    return f'<img src="{html.escape(source)}" alt="{html.escape(alt)}">'


def generate_report_html(
    report_dir: Path,
    summary: dict[str, Any],
    pair_frame: pd.DataFrame,
    group_members: pd.DataFrame,
    group_summary: pd.DataFrame,
    html_group_limit: int,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    for _, row in pair_frame.head(100).iterrows():
        pair_rows.append(
            "<tr>"
            f"<td>{int(row['left_id'])}</td><td>{html.escape(str(row['left_label']))}</td>"
            f"<td>{int(row['right_id'])}</td><td>{html.escape(str(row['right_label']))}</td>"
            f"<td>{float(row['score']):.4f}</td><td>{html.escape(str(row['category']))}</td>"
            "</tr>"
        )

    group_cards = []
    for _, group in group_summary.head(html_group_limit).iterrows():
        group_id = int(group["group_id"])
        members = group_members[group_members["group_id"] == group_id]
        cards = []
        for _, member in members.iterrows():
            picture = image_html(
                str(member["image_path"]), report_dir, f"ID {int(member['point_id'])}"
            )
            cards.append(
                f"""
                <div class="member">
                  {picture}
                  <strong>ID {int(member['point_id'])} · {html.escape(str(member['label']))}</strong>
                  <span>prema reprezentantu: {float(member['similarity_to_representative']):.4f}</span>
                  <span>predlog: {html.escape(str(member['recommended_action']))}</span>
                </div>
                """
            )
        conflict = "DA" if bool(group["has_label_conflict"]) else "ne"
        group_cards.append(
            f"""
            <section class="group">
              <h3>Grupa {group_id} · {int(group['group_size'])} slike</h3>
              <p>Reprezentant: ID {int(group['representative_id'])} · klase: {html.escape(str(group['labels_in_group']))} · konflikt labela: {conflict}</p>
              <div class="members">{"".join(cards)}</div>
            </section>
            """
        )

    table_body = "".join(pair_rows) or '<tr><td colspan="6">Nema parova.</td></tr>'
    document = f"""<!doctype html>
<html lang="sr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Varijanta 3 — čišćenje dataseta</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f6f8fc;color:#172033;line-height:1.5}}main{{max-width:1200px;margin:auto;padding:28px 18px 60px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}}.stat,.panel,.group{{background:white;border:1px solid #dbe3ef;border-radius:13px;padding:18px;box-shadow:0 5px 16px #1720330d}}.stat strong{{display:block;font-size:1.65rem;color:#2563eb}}
.panel,.group{{margin:18px 0}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #dbe3ef;padding:8px;text-align:center}}th{{background:#edf3fb}}.table-wrap{{overflow:auto}}
.members{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}.member{{border:1px solid #dbe3ef;border-radius:10px;padding:10px;display:grid;gap:5px;font-size:.88rem}}img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px}}.missing{{min-height:130px;border:1px dashed #94a3b8;border-radius:8px;display:grid;place-content:center;text-align:center;color:#64748b;padding:8px}}code{{overflow-wrap:anywhere}}.note{{color:#64748b}}
</style></head><body><main>
<h1>Pronalaženje duplikata i veoma sličnih slika</h1>
<p class="note">CLIP embedding + {html.escape(summary['backend'])} cosine pretraga · prag kandidata {summary['candidate_threshold']:.2f}</p>
<div class="stats"><div class="stat"><strong>{summary['total_images']}</strong>slika</div><div class="stat"><strong>{summary['candidate_pairs']}</strong>parova</div><div class="stat"><strong>{summary['groups']}</strong>grupa</div><div class="stat"><strong>{summary['recommended_removals']}</strong>stroga predloga za uklanjanje</div></div>
<section class="panel"><h2>Bezbednosno pravilo</h2><p>Originalne slike i metadata se ne brišu. <em>remove_candidate</em> je samo strogi predlog za novu kopiju dataseta. Stavke <em>review</em> zahtevaju pregled slika.</p></section>
<section class="panel"><h2>Kategorije</h2><ul><li>very_likely_duplicate: score ≥ {summary['very_likely_threshold']:.2f}</li><li>probable_duplicate: score ≥ {summary['probable_threshold']:.2f}</li><li>very_similar: score ≥ {summary['candidate_threshold']:.2f}</li></ul></section>
<section class="panel"><h2>Parovi kandidata</h2><div class="table-wrap"><table><thead><tr><th>Levi ID</th><th>Labela</th><th>Desni ID</th><th>Labela</th><th>Score</th><th>Kategorija</th></tr></thead><tbody>{table_body}</tbody></table></div></section>
<h2>Grupe za pregled</h2>{"".join(group_cards) if group_cards else '<p>Nema grupa.</p>'}
</main></body></html>"""
    path = report_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


def save_analysis(
    report_dir: Path,
    backend_name: str,
    metadata: pd.DataFrame,
    pairs: list[SimilarPair],
    pair_frame: pd.DataFrame,
    group_members: pd.DataFrame,
    group_summary: pd.DataFrame,
    candidate_threshold: float,
    probable_threshold: float,
    very_likely_threshold: float,
    html_group_limit: int,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    recommended_removals = int(
        (group_members["recommended_action"] == "remove_candidate").sum()
    )
    category_counts = pair_frame["category"].value_counts().sort_index().to_dict()
    cross_label_pairs = int((~pair_frame["same_label"]).sum())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "total_images": len(metadata),
        "candidate_threshold": candidate_threshold,
        "probable_threshold": probable_threshold,
        "very_likely_threshold": very_likely_threshold,
        "candidate_pairs": len(pairs),
        "groups": len(group_summary),
        "images_in_groups": int(len(group_members)),
        "recommended_removals": recommended_removals,
        "potential_cleaned_count": len(metadata) - recommended_removals,
        "cross_label_pairs": cross_label_pairs,
        "category_counts": category_counts,
    }

    pair_frame.to_csv(report_dir / "candidate_pairs.csv", index=False)
    group_members.to_csv(report_dir / "group_members.csv", index=False)
    group_summary.to_csv(report_dir / "group_summary.csv", index=False)

    decisions = group_members[
        [
            "group_id",
            "point_id",
            "label",
            "image_path",
            "representative_id",
            "similarity_to_representative",
            "recommended_action",
        ]
    ].copy()
    decisions["final_action"] = decisions["recommended_action"].map(
        {"keep": "keep", "remove_candidate": "remove", "review": "review"}
    )
    decisions.to_csv(report_dir / "review_decisions.csv", index=False)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = generate_report_html(
        report_dir,
        summary,
        pair_frame,
        group_members,
        group_summary,
        html_group_limit,
    )
    summary["report_path"] = str(report_path)
    return summary


def build_clean_dataset(
    report_dir: Path,
    output_dir: Path,
    decisions_path: Path | None,
) -> dict[str, Any]:
    group_members_path = report_dir / "group_members.csv"
    summary_path = report_dir / "summary.json"
    if not group_members_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Prvo pokreni komandu analyze.")

    group_members = pd.read_csv(group_members_path)
    if decisions_path is None:
        removed_ids = set(
            group_members.loc[
                group_members["recommended_action"] == "remove_candidate", "point_id"
            ].astype(int)
        )
        decision_source = "automatic_conservative_recommendations"
    else:
        decisions = pd.read_csv(decisions_path)
        required = {"point_id", "final_action"}
        if not required.issubset(decisions.columns):
            raise ValueError(f"Decisions fajl mora imati kolone: {sorted(required)}")
        allowed = {"keep", "remove", "review"}
        invalid = set(decisions["final_action"].astype(str)) - allowed
        if invalid:
            raise ValueError(f"Nepoznate final_action vrednosti: {sorted(invalid)}")
        removed_ids = set(
            decisions.loc[decisions["final_action"] == "remove", "point_id"].astype(int)
        )
        decision_source = str(decisions_path)

    embeddings, embedding_metadata = load_inputs()
    all_ids = set(embedding_metadata["id"].astype(int))
    unknown = removed_ids - all_ids
    if unknown:
        raise ValueError(f"Decisions sadrži nepoznate ID-eve: {sorted(unknown)}")

    keep_mask = ~embedding_metadata["id"].isin(removed_ids)
    cleaned_metadata = embedding_metadata[keep_mask].copy().reset_index(drop=True)
    cleaned_vectors = embeddings[keep_mask.to_numpy()].astype(np.float32)
    cleaned_metadata["embedding_index"] = np.arange(len(cleaned_metadata), dtype=int)
    cleaned_metadata["status"] = "ok"

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", cleaned_vectors)
    cleaned_metadata.to_csv(output_dir / "embeddings_metadata.csv", index=False)

    if FULL_METADATA_PATH.exists():
        full_metadata = pd.read_csv(FULL_METADATA_PATH)
        full_metadata[~full_metadata["id"].isin(removed_ids)].to_csv(
            output_dir / "metadata.csv", index=False
        )

    if EMBEDDING_CONFIG_PATH.exists():
        config = json.loads(EMBEDDING_CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        config = {}
    config.update(
        {
            "number_of_images": len(cleaned_metadata),
            "cleaned": True,
            "source_number_of_images": len(embedding_metadata),
            "removed_images": len(removed_ids),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    (output_dir / "embedding_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    before_counts = embedding_metadata["label"].value_counts().sort_index().to_dict()
    after_counts = cleaned_metadata["label"].value_counts().sort_index().to_dict()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_source": decision_source,
        "original_count": len(embedding_metadata),
        "cleaned_count": len(cleaned_metadata),
        "removed_count": len(removed_ids),
        "removed_ids": sorted(removed_ids),
        "class_counts_before": before_counts,
        "class_counts_after": after_counts,
        "original_files_modified": False,
    }
    (output_dir / "cleaning_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def verify_cleaned(output_dir: Path) -> dict[str, Any]:
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "embeddings_metadata.csv"
    manifest_path = output_dir / "cleaning_manifest.json"
    for path in (embeddings_path, metadata_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Nedostaje očišćeni fajl: {path}")

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Neispravan oblik očišćenih embeddinga: {embeddings.shape}")
    if len(embeddings) != len(metadata):
        raise ValueError("Broj očišćenih embeddinga i metadata redova se ne poklapa.")
    if metadata["id"].duplicated().any():
        raise ValueError("Očišćeni metadata sadrži duplirane ID-eve.")
    expected_indices = np.arange(len(metadata), dtype=int)
    if not np.array_equal(metadata["embedding_index"].to_numpy(dtype=int), expected_indices):
        raise ValueError("Očišćeni embedding_index nije kompaktan od 0 do N-1.")
    removed_ids = set(int(value) for value in manifest["removed_ids"])
    if removed_ids & set(metadata["id"].astype(int)):
        raise ValueError("Neki uklonjeni ID je i dalje u očišćenim metapodacima.")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Očišćeni embedding vektori nisu L2-normalizovani.")

    return {
        "cleaned_count": len(metadata),
        "removed_count": len(removed_ids),
        "embedding_shape": list(embeddings.shape),
        "class_counts": metadata["label"].value_counts().sort_index().to_dict(),
    }


def create_backend(
    name: str,
    batch_size: int,
    allowed_ids: list[int] | None = None,
) -> SimilarityBackend:
    if name == "local":
        return LocalSimilarityBackend()
    if name == "qdrant":
        return QdrantSimilarityBackend(
            batch_size=batch_size,
            allowed_ids=allowed_ids,
        )
    raise ValueError(f"Nepoznat backend: {name}")


def command_validate(args: argparse.Namespace) -> None:
    embeddings, metadata = load_inputs()
    backend = create_backend(args.backend, args.batch_size)
    backend.validate(len(metadata))
    print("VALIDACIJA USPEŠNA")
    print(f"Backend: {backend.name}")
    print(f"Broj slika: {len(metadata)}")
    print(f"Embedding dimenzija: {embeddings.shape[1]}")
    print(f"Broj klasa: {metadata['label'].nunique()}")


def command_analyze(args: argparse.Namespace) -> None:
    validate_thresholds(args.threshold, args.probable_threshold, args.very_likely_threshold)
    all_embeddings, all_metadata = load_inputs()
    embeddings, metadata = select_cleaning_sample(all_embeddings, all_metadata)

    if args.backend == "qdrant":
        backend = create_backend(
            args.backend,
            args.batch_size,
            allowed_ids=metadata["id"].astype(int).tolist(),
        )
        backend.validate(len(all_metadata))
    else:
        backend = create_backend(args.backend, args.batch_size)
        backend.validate(len(metadata))

    print(
        f"Tražim parove: backend={backend.name}, threshold={args.threshold:.4f}"
    )
    print(
        f"Analizira se {len(metadata)} labeliranih slika "
        f"({CLEANING_IMAGES_PER_CLASS} po klasi)."
    )
    pairs = backend.find_pairs(
        embeddings, metadata, threshold=args.threshold, top_k=args.top_k
    )
    pair_frame = pairs_dataframe(
        pairs, args.probable_threshold, args.very_likely_threshold
    )
    group_members, group_summary = build_groups(
        pairs, embeddings, metadata, args.very_likely_threshold
    )
    summary = save_analysis(
        report_dir=Path(args.report_dir),
        backend_name=backend.name,
        metadata=metadata,
        pairs=pairs,
        pair_frame=pair_frame,
        group_members=group_members,
        group_summary=group_summary,
        candidate_threshold=args.threshold,
        probable_threshold=args.probable_threshold,
        very_likely_threshold=args.very_likely_threshold,
        html_group_limit=args.html_group_limit,
    )

    print()
    print("ANALIZA ZAVRŠENA")
    print(f"Broj slika: {summary['total_images']}")
    print(f"Parovi kandidati: {summary['candidate_pairs']}")
    print(f"Grupe: {summary['groups']}")
    print(f"Kategorije: {summary['category_counts']}")
    print(f"Konflikti labela: {summary['cross_label_pairs']}")
    print(f"Strogi predlozi za uklanjanje: {summary['recommended_removals']}")
    print(f"HTML izveštaj: {summary['report_path']}")


def command_inspect_group(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    members_path = report_dir / "group_members.csv"
    pairs_path = report_dir / "candidate_pairs.csv"
    if not members_path.exists() or not pairs_path.exists():
        raise FileNotFoundError("Prvo pokreni analyze.")
    members = pd.read_csv(members_path)
    selected = members[members["group_id"] == args.group_id]
    if selected.empty:
        raise KeyError(f"Grupa {args.group_id} ne postoji.")
    ids = set(selected["point_id"].astype(int))
    pairs = pd.read_csv(pairs_path)
    selected_pairs = pairs[
        pairs["left_id"].isin(ids) & pairs["right_id"].isin(ids)
    ]

    print(f"GRUPA {args.group_id}")
    for _, row in selected.iterrows():
        print(
            f"ID={int(row['point_id'])} | label={row['label']} | "
            f"sim_to_rep={float(row['similarity_to_representative']):.4f} | "
            f"predlog={row['recommended_action']} | {row['image_path']}"
        )
    print("PAROVI U GRUPI")
    for _, row in selected_pairs.iterrows():
        print(
            f"{int(row['left_id'])} <-> {int(row['right_id'])} | "
            f"score={float(row['score']):.4f} | {row['category']}"
        )


def command_build_clean(args: argparse.Namespace) -> None:
    decisions_path = Path(args.decisions) if args.decisions else None
    manifest = build_clean_dataset(
        report_dir=Path(args.report_dir),
        output_dir=Path(args.output_dir),
        decisions_path=decisions_path,
    )
    print("OČIŠĆENA KOPIJA JE NAPRAVLJENA")
    print(f"Originalno: {manifest['original_count']}")
    print(f"Očišćeno: {manifest['cleaned_count']}")
    print(f"Uklonjeno iz kopije: {manifest['removed_count']}")
    print(f"Uklonjeni ID-evi: {manifest['removed_ids']}")
    print("Originalni fajlovi nisu menjani.")


def command_verify_cleaned(args: argparse.Namespace) -> None:
    result = verify_cleaned(Path(args.output_dir))
    print("PROVERA OČIŠĆENOG DATASETA USPEŠNA")
    print(f"Broj slika: {result['cleaned_count']}")
    print(f"Uklonjeno: {result['removed_count']}")
    print(f"Oblik embedding matrice: {result['embedding_shape']}")
    print(f"Broj po klasama: {result['class_counts']}")


def command_compare(args: argparse.Namespace) -> None:
    embeddings, metadata = load_inputs()
    local_backend = LocalSimilarityBackend()
    qdrant_backend = QdrantSimilarityBackend(batch_size=args.batch_size)
    qdrant_backend.validate(len(metadata))
    local_pairs = local_backend.find_pairs(
        embeddings, metadata, args.threshold, args.top_k
    )
    qdrant_pairs = qdrant_backend.find_pairs(
        embeddings, metadata, args.threshold, args.top_k
    )
    local_keys = {(pair.left_id, pair.right_id) for pair in local_pairs}
    qdrant_keys = {(pair.left_id, pair.right_id) for pair in qdrant_pairs}
    print("POREĐENJE BACKENDA")
    print(f"Lokalni parovi: {len(local_keys)}")
    print(f"Qdrant parovi: {len(qdrant_keys)}")
    print(f"Identični skupovi: {local_keys == qdrant_keys}")
    if local_keys != qdrant_keys:
        raise RuntimeError(
            f"Razlika: samo local={sorted(local_keys-qdrant_keys)}, "
            f"samo Qdrant={sorted(qdrant_keys-local_keys)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Varijanta 3: Qdrant analiza duplikata i čišćenje dataseta"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--backend", choices=["qdrant", "local"], default="qdrant")
    validate_parser.add_argument("--batch-size", type=int, default=100)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--backend", choices=["qdrant", "local"], default="qdrant")
    analyze_parser.add_argument("--threshold", type=float, default=0.94)
    analyze_parser.add_argument("--probable-threshold", type=float, default=0.95)
    analyze_parser.add_argument("--very-likely-threshold", type=float, default=0.97)
    analyze_parser.add_argument("--top-k", type=int, default=50)
    analyze_parser.add_argument("--batch-size", type=int, default=100)
    analyze_parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    analyze_parser.add_argument("--html-group-limit", type=int, default=30)

    inspect_parser = subparsers.add_parser("inspect-group")
    inspect_parser.add_argument("group_id", type=int)
    inspect_parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))

    build_clean_parser = subparsers.add_parser("build-clean-dataset")
    build_clean_parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    build_clean_parser.add_argument("--output-dir", default=str(DEFAULT_CLEANED_DIR))
    build_clean_parser.add_argument(
        "--decisions",
        default=None,
        help="Opcioni review_decisions.csv sa final_action kolonom.",
    )

    verify_parser = subparsers.add_parser("verify-cleaned")
    verify_parser.add_argument("--output-dir", default=str(DEFAULT_CLEANED_DIR))

    compare_parser = subparsers.add_parser("compare-backends")
    compare_parser.add_argument("--threshold", type=float, default=0.94)
    compare_parser.add_argument("--top-k", type=int, default=50)
    compare_parser.add_argument("--batch-size", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        command_validate(args)
    elif args.command == "analyze":
        command_analyze(args)
    elif args.command == "inspect-group":
        command_inspect_group(args)
    elif args.command == "build-clean-dataset":
        command_build_clean(args)
    elif args.command == "verify-cleaned":
        command_verify_cleaned(args)
    elif args.command == "compare-backends":
        command_compare(args)


if __name__ == "__main__":
    main()
