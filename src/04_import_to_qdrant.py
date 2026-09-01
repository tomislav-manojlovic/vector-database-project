from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client.models import PointStruct
from tqdm import tqdm

from qdrant_common import (
    get_qdrant_client,
    COLLECTION_NAME,
    VECTOR_SIZE,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

EMBEDDINGS_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings.npy"
METADATA_PATH = ROOT_DIR / "data" / "embeddings" / "embeddings_metadata.csv"

BATCH_SIZE = 1000
PARALLEL_WORKERS = 4


def validate_inputs(embeddings, metadata):
    if embeddings.ndim != 2:
        raise ValueError("embeddings.npy mora biti 2D matrica oblika (N, D).")

    if embeddings.shape[1] != VECTOR_SIZE:
        raise ValueError(
            f"Ocekivana dimenzija embeddinga je {VECTOR_SIZE}, "
            f"a dobijeno je {embeddings.shape[1]}."
        )

    required_columns = {
        "id",
        "image_path",
        "label",
        "label_id",
        "is_labeled",
        "split",
        "embedding_index",
        "status",
    }
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(f"Nedostaju kolone u metadata fajlu: {missing_columns}")

    if metadata["id"].duplicated().any():
        raise ValueError("Postoje duplirani id-jevi u metadata fajlu.")

    max_index = metadata["embedding_index"].max()
    if max_index >= len(embeddings):
        raise ValueError("Neki embedding_index izlazi van granica embeddings.npy fajla.")


def build_point(row, embeddings):
    embedding_index = int(row.embedding_index)
    image_id = int(row.id)

    vector = embeddings[embedding_index].astype(float).tolist()

    payload = {
        "id": image_id,
        "image_path": str(row.image_path),
        "label": str(row.label),
        "label_id": int(row.label_id),
        "is_labeled": bool(row.is_labeled),
        "split": str(row.split),
    }

    return PointStruct(
        id=image_id,
        vector=vector,
        payload=payload,
    )


def main():
    print("Ucitavam embeddinge...")
    embeddings = np.load(EMBEDDINGS_PATH)

    print("Ucitavam metadata...")
    metadata = pd.read_csv(METADATA_PATH)

    metadata = metadata[metadata["status"] == "ok"].copy()
    metadata["embedding_index"] = metadata["embedding_index"].astype(int)

    print(f"Broj embeddinga u .npy fajlu: {len(embeddings)}")
    print(f"Broj validnih metadata redova: {len(metadata)}")

    validate_inputs(embeddings, metadata)

    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Kolekcija '{COLLECTION_NAME}' ne postoji. "
            "Prvo pokreni src/02_create_collection.py."
        )

    points = (
        build_point(row, embeddings)
        for row in metadata.itertuples(index=False)
    )
    points = tqdm(points, total=len(metadata), desc="Import u Qdrant", unit="point")

    print(
        f"Krecem import: batch_size={BATCH_SIZE}, "
        f"parallel={PARALLEL_WORKERS}"
    )

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=BATCH_SIZE,
        parallel=PARALLEL_WORKERS,
        max_retries=3,
        wait=True,
    )

    print(f"Gotovo. Ukupno importovano pointova: {len(metadata)}")


if __name__ == "__main__":
    main()