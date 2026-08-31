import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client.models import FieldCondition, Filter, MatchValue, SearchParams

from qdrant_common import get_qdrant_client, COLLECTION_NAME


ROOT_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings.npy"
METADATA_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings_metadata.csv"
REPORT_DIR = ROOT_DIR / "reports"
REPORT_PATH = REPORT_DIR / "hnsw_benchmark.csv"

HNSW_EF_VALUES = [16, 32, 64, 128]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Poredi Qdrant exact i HNSW pretragu pomocu Recall@k i vremena."
    )
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_queries(number_of_queries, seed):
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    metadata = pd.read_csv(METADATA_PATH)

    metadata = metadata[metadata["status"] == "ok"].copy()

    if "is_labeled" in metadata.columns:
        labeled = (
            metadata["is_labeled"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )
        metadata = metadata[labeled].copy()
    else:
        metadata = metadata[metadata["label"] != "unlabeled"].copy()

    if number_of_queries > len(metadata):
        raise ValueError(
            f"Trazeno je {number_of_queries} upita, a postoji samo "
            f"{len(metadata)} labeled slika."
        )

    sample = metadata.sample(n=number_of_queries, random_state=seed)
    sample = sample.reset_index(drop=True)

    queries = []
    for _, row in sample.iterrows():
        embedding_index = int(row["embedding_index"])
        queries.append(
            {
                "id": int(row["id"]),
                "label": str(row["label"]),
                "vector": np.asarray(
                    embeddings[embedding_index], dtype=np.float32
                ).tolist(),
            }
        )

    return queries


def make_label_filter(label):
    return Filter(
        must=[
            FieldCondition(
                key="label",
                match=MatchValue(value=label),
            )
        ]
    )


def search(client, query, top_k, exact, hnsw_ef=None, use_filter=False):
    query_filter = make_label_filter(query["label"]) if use_filter else None

    params = SearchParams(
        exact=exact,
        hnsw_ef=hnsw_ef,
    )

    start = time.perf_counter()

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query["vector"],
        query_filter=query_filter,
        search_params=params,
        limit=top_k + 1,
        with_payload=False,
        with_vectors=False,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    ids = [
        int(point.id)
        for point in result.points
        if int(point.id) != query["id"]
    ][:top_k]

    return ids, elapsed_ms


def recall_at_k(exact_ids, ann_ids, top_k):
    return len(set(exact_ids) & set(ann_ids)) / top_k


def summarize_times(times):
    values = np.asarray(times, dtype=float)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def run_mode(client, queries, top_k, use_filter):
    filter_name = "label" if use_filter else "none"

    print()
    print(f"=== Filter: {filter_name} ===")
    print("Racunam exact ground truth...")

    exact_results = {}
    exact_times = []

    for index, query in enumerate(queries, start=1):
        ids, elapsed_ms = search(
            client=client,
            query=query,
            top_k=top_k,
            exact=True,
            use_filter=use_filter,
        )
        exact_results[query["id"]] = ids
        exact_times.append(elapsed_ms)

        if index % 25 == 0:
            print(f"Exact: {index}/{len(queries)}")

    exact_stats = summarize_times(exact_times)

    rows = [
        {
            "search_type": "exact",
            "hnsw_ef": "",
            "filter": filter_name,
            "queries": len(queries),
            "k": top_k,
            "recall_at_k": 1.0,
            **exact_stats,
        }
    ]

    for hnsw_ef in HNSW_EF_VALUES:
        print(f"Testiram HNSW hnsw_ef={hnsw_ef}...")

        recalls = []
        times = []

        for query in queries:
            ann_ids, elapsed_ms = search(
                client=client,
                query=query,
                top_k=top_k,
                exact=False,
                hnsw_ef=hnsw_ef,
                use_filter=use_filter,
            )

            recalls.append(
                recall_at_k(
                    exact_results[query["id"]],
                    ann_ids,
                    top_k,
                )
            )
            times.append(elapsed_ms)

        stats = summarize_times(times)

        row = {
            "search_type": "hnsw",
            "hnsw_ef": hnsw_ef,
            "filter": filter_name,
            "queries": len(queries),
            "k": top_k,
            "recall_at_k": float(np.mean(recalls)),
            **stats,
        }
        rows.append(row)

        print(
            f"  Recall@{top_k}: {row['recall_at_k']:.4f} | "
            f"median: {row['median_ms']:.3f} ms | "
            f"p95: {row['p95_ms']:.3f} ms"
        )

    return rows


def main():
    args = parse_args()

    if args.queries < 1:
        raise ValueError("--queries mora biti najmanje 1.")
    if args.top_k < 1:
        raise ValueError("--top-k mora biti najmanje 1.")

    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Kolekcija '{COLLECTION_NAME}' ne postoji."
        )

    info = client.get_collection(COLLECTION_NAME)
    indexed_count = int(info.indexed_vectors_count or 0)

    if indexed_count == 0:
        raise RuntimeError(
            "indexed_vectors_count je 0. HNSW indeks jos nije spreman."
        )

    print(f"Kolekcija: {COLLECTION_NAME}")
    print(f"Indeksirano vektora: {indexed_count}")
    print(
        f"Ucitavam {args.queries} labeled upita "
        f"(seed={args.seed}, k={args.top_k})..."
    )

    queries = load_queries(args.queries, args.seed)

    rows = []
    rows.extend(
        run_mode(
            client=client,
            queries=queries,
            top_k=args.top_k,
            use_filter=False,
        )
    )
    rows.extend(
        run_mode(
            client=client,
            queries=queries,
            top_k=args.top_k,
            use_filter=True,
        )
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(REPORT_PATH, index=False)

    print()
    print("=== REZULTATI ===")
    print(results.to_string(index=False))
    print()
    print(f"Rezultati sacuvani u: {REPORT_PATH}")


if __name__ == "__main__":
    main()
