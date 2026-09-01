# HNSW benchmark v3.0: poređenje m i hnsw_ef parametara

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client.models import (
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    QueryRequest,
    SearchParams,
)

from qdrant_common import COLLECTION_NAME, get_qdrant_client


VERSION = "3.0"

ROOT_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings.npy"
METADATA_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings_metadata.csv"
REPORT_PATH = ROOT_DIR / "reports" / "hnsw_m_benchmark.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="NumPy exact, Qdrant exact i Qdrant HNSW benchmark."
    )
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument(
        "--hnsw-ef-values",
        type=int,
        nargs="+",
        default=[16, 64, 128],
    )
    parser.add_argument("--reindex-timeout", type=int, default=1800)
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument(
        "--keep-last-m",
        action="store_true",
        help="Ne vraćaj originalnu m vrednost nakon benchmarka.",
    )
    return parser.parse_args()


def collection_m(info):
    return int(info.config.hnsw_config.m)


def wait_for_index(client, expected_m, timeout, poll_interval):
    """Sačekaj da nova HNSW konfiguracija bude primenjena i indeks spreman."""
    deadline = time.monotonic() + timeout
    ready_checks = 0

    while time.monotonic() < deadline:
        info = client.get_collection(COLLECTION_NAME)
        status = getattr(info.status, "value", str(info.status)).lower()
        points = int(info.points_count or 0)
        indexed = int(info.indexed_vectors_count or 0)
        indexed_ratio = indexed / points if points else 0.0

        ready = (
            collection_m(info) == expected_m
            and status.endswith("green")
            and indexed_ratio >= 0.9
        )
        ready_checks = ready_checks + 1 if ready else 0
        if ready_checks >= 2:
            print(f"HNSW indeks je spreman: m={expected_m}, indexed={indexed}/{points}")
            return

        print(
            f"Čekam indeks: m={collection_m(info)}, status={status}, "
            f"indexed={indexed}/{points}"
        )
        time.sleep(poll_interval)

    raise TimeoutError(
        f"HNSW indeks za m={expected_m} nije završen za {timeout} sekundi."
    )


def set_hnsw_m(client, m, timeout, poll_interval):
    current = collection_m(client.get_collection(COLLECTION_NAME))
    if current == m:
        print(f"HNSW već koristi m={m}.")
        return

    print(f"Menjam HNSW m: {current} -> {m}. Qdrant ponovo gradi indeks...")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        hnsw_config=HnswConfigDiff(m=m),
        timeout=timeout,
    )
    time.sleep(poll_interval)
    wait_for_index(client, m, timeout, poll_interval)


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


def make_row(search_type, m, hnsw_ef, filter_name, queries, top_k, score, times):
    values = np.asarray(times, dtype=float)
    return {
        "search_type": search_type,
        "m": m,
        "hnsw_ef": hnsw_ef,
        "filter": filter_name,
        "queries": len(queries),
        "k": top_k,
        "recall_at_k": score,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def run_mode(client, embeddings, metadata, queries, args, use_filter, m):
    filter_name = "label" if use_filter else "none"
    rows = []

    exact_results, numpy_times = numpy_exact(
        embeddings, metadata, queries, args.top_k, use_filter
    )
    rows.append(
        make_row(
            "numpy_exact", m, "", filter_name, queries,
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
        "qdrant_exact", m, "", filter_name, queries,
        args.top_k, exact_recall, qdrant_exact_times
    )
    rows.append(exact_row)
    print(
        f"m={m}, filter={filter_name}, Qdrant exact: "
        f"recall@{args.top_k}={exact_recall:.4f}, "
        f"prosecno={exact_row['mean_ms']:.2f} ms"
    )

    for hnsw_ef in args.hnsw_ef_values:
        hnsw_results, hnsw_times = qdrant_search(
            client, queries, args.top_k, args.batch_size,
            use_filter, exact=False, hnsw_ef=hnsw_ef
        )
        hnsw_recall = recall(
            exact_results, hnsw_results, queries, args.top_k
        )
        row = make_row(
            "hnsw", m, hnsw_ef, filter_name, queries,
            args.top_k, hnsw_recall, hnsw_times
        )
        rows.append(row)
        print(
            f"m={m}, filter={filter_name}, ef={hnsw_ef}: "
            f"recall@{args.top_k}={hnsw_recall:.4f}, "
            f"prosecno={row['mean_ms']:.2f} ms"
        )

    return rows


def main():
    args = parse_args()
    if args.queries < 1 or args.top_k < 1 or args.batch_size < 1:
        raise ValueError("queries, top-k i batch-size moraju biti pozitivni.")
    if any(value < 2 for value in args.m_values):
        raise ValueError("Sve m vrednosti moraju biti najmanje 2.")
    if any(value < 1 for value in args.hnsw_ef_values):
        raise ValueError("Sve hnsw-ef vrednosti moraju biti pozitivne.")
    if args.reindex_timeout < 1 or args.poll_interval < 1:
        raise ValueError("Timeout i poll interval moraju biti pozitivni.")

    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(f"Kolekcija '{COLLECTION_NAME}' ne postoji.")

    info = client.get_collection(COLLECTION_NAME)
    original_m = collection_m(info)
    indexed = int(info.indexed_vectors_count or 0)
    points = int(info.points_count or 0)
    if indexed == 0:
        raise RuntimeError("HNSW indeks jos nije napravljen.")
    if points and indexed / points < 0.9:
        raise RuntimeError("HNSW indeksiranje jos nije dovoljno zavrseno.")

    print(
        f"HNSW benchmark v{VERSION}: "
        f"m={args.m_values}, hnsw_ef={args.hnsw_ef_values}"
    )

    embeddings, metadata, queries = load_data(args.queries, args.seed)
    rows = []

    try:
        for m in args.m_values:
            print()
            print(f"=== BENCHMARK ZA m={m} ===")
            set_hnsw_m(
                client,
                m=m,
                timeout=args.reindex_timeout,
                poll_interval=args.poll_interval,
            )
            rows.extend(
                run_mode(client, embeddings, metadata, queries, args, False, m)
            )
            rows.extend(
                run_mode(client, embeddings, metadata, queries, args, True, m)
            )
    finally:
        if not args.keep_last_m:
            current_m = collection_m(client.get_collection(COLLECTION_NAME))
            if current_m != original_m:
                print()
                print(f"Vraćam originalnu vrednost m={original_m}...")
                set_hnsw_m(
                    client,
                    m=original_m,
                    timeout=args.reindex_timeout,
                    poll_interval=args.poll_interval,
                )

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
    print(f"Originalna m vrednost kolekcije: {original_m}")


if __name__ == "__main__":
    main()
