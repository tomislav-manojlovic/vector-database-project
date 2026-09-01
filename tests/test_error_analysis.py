import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "07_error_analysis.py"
SPEC = importlib.util.spec_from_file_location("error_analysis", MODULE_PATH)
error_analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = error_analysis
SPEC.loader.exec_module(error_analysis)


Neighbor = error_analysis.Neighbor


class ErrorAnalysisTests(unittest.TestCase):
    def test_local_backend_excludes_query_image_and_sorts_by_similarity(self):
        embeddings = error_analysis.normalize_rows(
            np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        metadata = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "image_path": ["1.jpg", "2.jpg", "3.jpg"],
                "label": ["cat", "cat", "dog"],
            }
        )
        backend = error_analysis.LocalNeighborBackend(embeddings, metadata)

        neighbors = backend.search(query_id=1, query_vector=embeddings[0], limit=2)

        self.assertEqual([neighbor.point_id for neighbor in neighbors], [2, 3])
        self.assertNotIn(1, [neighbor.point_id for neighbor in neighbors])
        self.assertGreater(neighbors[0].score, neighbors[1].score)

    def test_weighted_knn_uses_cosine_score_as_vote_weight(self):
        neighbors = [
            Neighbor(2, "cat", "2.jpg", 0.95),
            Neighbor(3, "dog", "3.jpg", 0.60),
            Neighbor(4, "dog", "4.jpg", 0.20),
        ]

        result = error_analysis.weighted_knn_prediction(neighbors)

        self.assertEqual(result["predicted_label"], "cat")
        self.assertGreater(result["prediction_confidence"], 0.5)

    def test_class_confusion_diagnosis(self):
        neighbors = [
            Neighbor(2, "deer", "2.jpg", 0.88),
            Neighbor(3, "deer", "3.jpg", 0.86),
            Neighbor(4, "deer", "4.jpg", 0.84),
            Neighbor(5, "cat", "5.jpg", 0.82),
            Neighbor(6, "bird", "6.jpg", 0.80),
        ]

        result = error_analysis.diagnose_error("cat", "deer", neighbors)

        self.assertEqual(result["diagnosis"], "class_confusion")
        self.assertEqual(result["predicted_neighbor_support"], 0.6)
        self.assertEqual(result["true_neighbor_support"], 0.2)

    def test_annotation_issue_is_only_a_cautious_high_similarity_flag(self):
        neighbors = [
            Neighbor(index, "dog", f"{index}.jpg", score)
            for index, score in enumerate([0.97, 0.96, 0.95, 0.94, 0.93], start=2)
        ]

        result = error_analysis.diagnose_error("cat", "dog", neighbors)

        self.assertEqual(result["diagnosis"], "possible_annotation_issue")
        self.assertIn("nije dokaz", result["diagnosis_explanation"])

    def test_qdrant_backend_with_in_memory_qdrant(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = QdrantClient(location=":memory:")
        collection = "test_images"
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=1, vector=[1.0, 0.0], payload={"label": "cat", "image_path": "1.jpg", "is_labeled": True}),
            PointStruct(id=2, vector=[0.9, 0.1], payload={"label": "cat", "image_path": "2.jpg", "is_labeled": True}),
            PointStruct(id=3, vector=[0.0, 1.0], payload={"label": "dog", "image_path": "3.jpg", "is_labeled": True}),
            PointStruct(id=4, vector=[1.0, 0.0], payload={"label": "unlabeled", "image_path": "4.jpg", "is_labeled": False}),
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        backend = error_analysis.QdrantNeighborBackend(
            client=client,
            collection_name=collection,
            batch_size=2,
            labeled_only=True,
        )

        backend.validate(expected_count=4)
        backend.prepare_batch(
            [
                (1, np.array([1.0, 0.0], dtype=np.float32)),
                (2, np.array([0.9, 0.1], dtype=np.float32)),
                (3, np.array([0.0, 1.0], dtype=np.float32)),
            ],
            limit=2,
        )
        neighbors = backend.search(
            query_id=1,
            query_vector=np.array([1.0, 0.0], dtype=np.float32),
            limit=2,
        )

        self.assertEqual([neighbor.point_id for neighbor in neighbors], [2, 3])
        self.assertNotIn(1, [neighbor.point_id for neighbor in neighbors])


if __name__ == "__main__":
    unittest.main()
