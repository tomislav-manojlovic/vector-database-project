import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "08_dataset_cleaning.py"
SPEC = importlib.util.spec_from_file_location("dataset_cleaning", MODULE_PATH)
dataset_cleaning = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = dataset_cleaning
SPEC.loader.exec_module(dataset_cleaning)


def sample_data():
    vectors = dataset_cleaning.normalize_rows(
        np.array(
            [
                [1.0, 0.0],
                [0.999, 0.001],
                [0.7, 0.7],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    metadata = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "image_path": ["1.jpg", "2.jpg", "3.jpg", "4.jpg"],
            "label": ["cat", "cat", "cat", "dog"],
            "embedding_index": [0, 1, 2, 3],
            "status": ["ok"] * 4,
        }
    )
    return vectors, metadata


class DatasetCleaningTests(unittest.TestCase):
    def test_local_backend_returns_unique_pairs_without_self_matches(self):
        vectors, metadata = sample_data()
        backend = dataset_cleaning.LocalSimilarityBackend()
        pairs = backend.find_pairs(vectors, metadata, threshold=0.95, top_k=10)

        keys = [(pair.left_id, pair.right_id) for pair in pairs]
        self.assertEqual(keys, [(1, 2)])
        self.assertTrue(all(left != right for left, right in keys))

    def test_pair_categories_use_configured_thresholds(self):
        self.assertEqual(
            dataset_cleaning.pair_category(0.975, 0.95, 0.97),
            "very_likely_duplicate",
        )
        self.assertEqual(
            dataset_cleaning.pair_category(0.955, 0.95, 0.97),
            "probable_duplicate",
        )
        self.assertEqual(
            dataset_cleaning.pair_category(0.945, 0.95, 0.97), "very_similar"
        )

    def test_same_label_high_similarity_member_is_removal_candidate(self):
        vectors, metadata = sample_data()
        pairs = dataset_cleaning.LocalSimilarityBackend().find_pairs(
            vectors, metadata, threshold=0.95, top_k=10
        )
        members, groups = dataset_cleaning.build_groups(
            pairs, vectors, metadata, very_likely_threshold=0.97
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            set(members["recommended_action"]), {"keep", "remove_candidate"}
        )

    def test_label_conflict_never_gets_automatic_removal(self):
        vectors, metadata = sample_data()
        metadata.loc[metadata["id"] == 2, "label"] = "dog"
        pairs = dataset_cleaning.LocalSimilarityBackend().find_pairs(
            vectors, metadata, threshold=0.95, top_k=10
        )
        members, groups = dataset_cleaning.build_groups(
            pairs, vectors, metadata, very_likely_threshold=0.97
        )

        self.assertTrue(bool(groups.iloc[0]["has_label_conflict"]))
        self.assertNotIn("remove_candidate", set(members["recommended_action"]))

    def test_qdrant_backend_matches_local_backend_in_memory(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        vectors, metadata = sample_data()
        client = QdrantClient(location=":memory:")
        collection = "cleaning_test"
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=int(row["id"]),
                vector=vectors[index].astype(float).tolist(),
                payload={"label": row["label"], "image_path": row["image_path"]},
            )
            for index, row in metadata.iterrows()
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        qdrant_backend = dataset_cleaning.QdrantSimilarityBackend(
            client=client,
            collection_name=collection,
            batch_size=2,
        )

        qdrant_backend.validate(len(metadata))
        qdrant_pairs = qdrant_backend.find_pairs(
            vectors, metadata, threshold=0.95, top_k=4
        )
        local_pairs = dataset_cleaning.LocalSimilarityBackend().find_pairs(
            vectors, metadata, threshold=0.95, top_k=4
        )

        self.assertEqual(
            {(pair.left_id, pair.right_id) for pair in qdrant_pairs},
            {(pair.left_id, pair.right_id) for pair in local_pairs},
        )


if __name__ == "__main__":
    unittest.main()
