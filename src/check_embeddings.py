import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check generated embedding files.")
    parser.add_argument("--embeddings", default="data/embeddings/embeddings.npy")
    parser.add_argument("--metadata", default="data/embeddings/embeddings_metadata.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings_path = Path(args.embeddings)
    metadata_path = Path(args.metadata)

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file does not exist: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    ok_metadata = metadata[metadata["status"] == "ok"] if "status" in metadata.columns else metadata

    embedding_count = int(embeddings.shape[0])
    embedding_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 and embedding_count > 0 else 0
    metadata_count = int(len(ok_metadata))
    average_norm = float(np.linalg.norm(embeddings, axis=1).mean()) if embedding_count > 0 else 0.0
    counts_match = embedding_count == metadata_count

    print(f"Number of embeddings: {embedding_count}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Metadata rows with status ok: {metadata_count}")
    print(f"Total metadata rows: {len(metadata)}")
    print()
    print("First 5 examples:")
    print(metadata.head(5).to_string(index=False))
    print()
    print(f"Embedding count matches metadata rows: {counts_match}")
    print(f"Average embedding L2 norm: {average_norm:.6f}")

    if counts_match and abs(average_norm - 1.0) < 1e-3:
        print("Embeddings are ready for Qdrant import.")
    else:
        raise ValueError("Embedding files need attention before Qdrant import.")


if __name__ == "__main__":
    main()
