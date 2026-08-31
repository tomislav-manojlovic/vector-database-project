# HNSW benchmark v2.2

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    QueryRequest,
    SearchParams,
)

from qdrant_common import COLLECTION_NAME, get_qdrant_client


VERSION = "2.2"
HNSW_EF_VALUES = [16, 64, 128]

ROOT_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings.npy"
METADATA_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings_metadata.csv"
REPORT_PATH = ROOT_DIR / "reports" / "hnsw_benchmark.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="NumPy exact, Qdrant exact i Qdrant HNSW benchmark."
    )
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(query_count, seed):
    stored_embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    metadata = pd.read_csv(METADATA_PATH)
    metadata = metadata[metadata["status"] == "ok"].copy()
    metadata["embedding_index"] = metadata["embedding_index"].astype(int)
    metadata = metadata.sort_values("embedding_index").reset_index(drop=True)

    embedding_indices = metadata["embedding_index"].to_numpy()
    embeddings = np.asarray(
        stored_embeddings[embedding_indices], dtype=np.float32
    )

    if "is_labeled" in metadata.columns:
        labeled_mask = (
            metadata["is_labeled"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
            .to_numpy()
        )
    else:
        labeled_mask = (metadata["label"] != "unlabeled").to_numpy()

    labeled_positions = np.flatnonzero(labeled_mask)
    if query_count > len(labeled_positions):
        raise ValueError("Nema dovoljno labeled slika za trazeni broj upita.")

    rng = np.random.default_rng(seed)
    query_positions = rng.choice(
        labeled_positions, size=query_count, replace=False
    )

    queries = []
    for position in query_positions:
        row = metadata.iloc[position]
        queries.append(
            {
                "id": int(row["id"]),
                "label": str(row["label"]),
                "position": int(position),
                "vector": embeddings[position],
            }
        )

    return embeddings, metadata, queries


def label_filter(label):
    return Filter(
        must=[FieldCondition(key="label", match=MatchValue(value=label))]
    )


def select_top_ids(scores, candidate_ids, top_k):
    selected = np.argpartition(scores, -top_k)[-top_k:]
    selected = selected[np.argsort(scores[selected])[::-1]]
    return candidate_ids[selected].astype(int).tolist()


def numpy_exact(embeddings, metadata, queries, top_k, use_filter):
    ids = metadata["id"].to_numpy(dtype=int)
    labels = metadata["label"].astype(str).to_numpy()
    results = {}
    started = time.perf_counter()

    if not use_filter:
        query_matrix = np.vstack([query["vector"] for query in queries])
        similarities = query_matrix @ embeddings.T

        for row_index, query in enumerate(queries):
            similarities[row_index, query["position"]] = -np.inf
            results[query["id"]] = select_top_ids(
                similarities[row_index], ids, top_k
            )
    else:
        positions_by_label = {
            value: np.flatnonzero(labels == value)
            for value in np.unique(labels)
        }

        for query in queries:
            positions = positions_by_label[query["label"]]
            scores = embeddings[positions] @ query["vector"]
            scores[positions == query["position"]] = -np.inf
            results[query["id"]] = select_top_ids(
                scores, ids[positions], top_k
            )

    mean_ms = (time.perf_counter() - started) * 1000 / len(queries)
    return results, [mean_ms] * len(queries)


def qdrant_search(
    client,
    queries,
    top_k,
    batch_size,
    use_filter,
    exact,
    hnsw_ef=None,
):
    results = {}
    times = []

    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        requests = [
            QueryRequest(
                query=query["vector"].astype(float).tolist(),
                filter=(label_filter(query["label"]) if use_filter else None),
                params=SearchParams(exact=exact, hnsw_ef=hnsw_ef),
                limit=top_k + 1,
                with_payload=False,
                with_vector=False,
            )
            for query in batch
        ]

        started = time.perf_counter()
        responses = client.query_batch_points(
            collection_name=COLLECTION_NAME,
            requests=requests,
        )
        mean_ms = (time.perf_counter() - started) * 1000 / len(batch)
        times.extend([mean_ms] * len(batch))

        for query, response in zip(batch, responses):
            results[query["id"]] = [
                int(point.id)
                for point in response.points
                if int(point.id) != query["id"]
            ][:top_k]

    return results, times


def recall(exact_results, tested_results, queries, top_k):
    values = [
        len(
            set(exact_results[query["id"]])
            & set(tested_results[query["id"]])
        )
        / top_k
        for query in queries
    ]
    return float(np.mean(values))


def make_row(search_type, hnsw_ef, filter_name, queries, top_k, score, times):
    values = np.asarray(times, dtype=float)
    return {
        "search_type": search_type,
        "hnsw_ef": hnsw_ef,
        "filter": filter_name,
        "queries": len(queries),
        "k": top_k,
        "recall_at_k": score,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def run_mode(client, embeddings, metadata, queries, args, use_filter):
    filter_name = "label" if use_filter else "none"
    rows = []

    exact_results, numpy_times = numpy_exact(
        embeddings, metadata, queries, args.top_k, use_filter
    )
    rows.append(
        make_row(
            "numpy_exact", "", filter_name, queries,
            args.top_k, 1.0, numpy_times
        )
    )

    qdrant_exact_results, qdrant_exact_times = qdrant_search(
        client, queries, args.top_k, args.batch_size,
        use_filter, exact=True
    )
    exact_recall = recall(
        exact_results, qdrant_exact_results, queries, args.top_k
    )
    exact_row = make_row(
        "qdrant_exact", "", filter_name, queries,
        args.top_k, exact_recall, qdrant_exact_times
    )
    rows.append(exact_row)
    print(
        f"filter={filter_name}, Qdrant exact: "
        f"recall@{args.top_k}={exact_recall:.4f}, "
        f"prosecno={exact_row['mean_ms']:.2f} ms"
    )

    for hnsw_ef in HNSW_EF_VALUES:
        hnsw_results, hnsw_times = qdrant_search(
            client, queries, args.top_k, args.batch_size,
            use_filter, exact=False, hnsw_ef=hnsw_ef
        )
        hnsw_recall = recall(
            exact_results, hnsw_results, queries, args.top_k
        )
        row = make_row(
            "hnsw", hnsw_ef, filter_name, queries,
            args.top_k, hnsw_recall, hnsw_times
        )
        rows.append(row)
        print(
            f"filter={filter_name}, ef={hnsw_ef}: "
            f"recall@{args.top_k}={hnsw_recall:.4f}, "
            f"prosecno={row['mean_ms']:.2f} ms"
        )

    return rows


def main():
    args = parse_args()
    if args.queries < 1 or args.top_k < 1 or args.batch_size < 1:
        raise ValueError("queries, top-k i batch-size moraju biti pozitivni.")

    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(f"Kolekcija '{COLLECTION_NAME}' ne postoji.")

    info = client.get_collection(COLLECTION_NAME)
    indexed = int(info.indexed_vectors_count or 0)
    points = int(info.points_count or 0)
    if indexed == 0:
        raise RuntimeError("HNSW indeks jos nije napravljen.")
    if points and indexed / points < 0.9:
        raise RuntimeError("HNSW indeksiranje jos nije dovoljno zavrseno.")

    print(
        f"HNSW benchmark v{VERSION}: "
        "NumPy exact + Qdrant exact + Qdrant HNSW"
    )

    embeddings, metadata, queries = load_data(args.queries, args.seed)
    rows = []
    rows.extend(run_mode(client, embeddings, metadata, queries, args, False))
    rows.extend(run_mode(client, embeddings, metadata, queries, args, True))

    results = pd.DataFrame(rows)
    required_types = {"numpy_exact", "qdrant_exact", "hnsw"}
    if set(results["search_type"]) != required_types:
        raise RuntimeError("Benchmark nije generisao sve tipove pretrage.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_PATH, index=False)

    print()
    print(results.to_string(index=False))
    print(f"Rezultati: {REPORT_PATH}")
    print("Qdrant vremena su prosecna vremena po upitu unutar batch-a.")


if __name__ == "__main__":
    main()