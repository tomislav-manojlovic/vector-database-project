import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
UI_DIR = ROOT_DIR / "ui"

for directory in (SRC_DIR, UI_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from qdrant_common import COLLECTION_NAME, VECTOR_SIZE, get_qdrant_client


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ui_server = load_module("real_ui_server", UI_DIR / "server.py")
error_analysis = load_module("real_error_analysis", SRC_DIR / "07_error_analysis.py")


class RealDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        metadata_path = ROOT_DIR / "data" / "embeddings" / "embeddings_metadata.csv"
        embeddings_path = ROOT_DIR / "data" / "embeddings" / "embeddings.npy"

        cls.metadata = pd.read_csv(metadata_path)
        cls.metadata = cls.metadata[cls.metadata["status"] == "ok"].reset_index(drop=True)
        cls.embeddings = np.load(embeddings_path, mmap_mode="r")
        cls.client = get_qdrant_client()

        if not cls.client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(f"Qdrant kolekcija '{COLLECTION_NAME}' ne postoji.")

    def test_collection_matches_real_metadata(self):
        database_count = int(
            self.client.count(collection_name=COLLECTION_NAME, exact=True).count
        )
        demo_count = len(ui_server.demo_points())
        self.assertEqual(database_count, len(self.metadata) + demo_count)

        info = self.client.get_collection(COLLECTION_NAME)
        vectors_config = info.config.params.vectors
        self.assertEqual(int(vectors_config.size), VECTOR_SIZE)

        payload_schema = info.payload_schema or {}
        for field_name in ("label", "split", "is_labeled"):
            self.assertIn(field_name, payload_schema)

        print(
            f"Stvarna kolekcija: {database_count} pointova "
            f"({len(self.metadata)} iz dataseta, {demo_count} demo)."
        )

    def test_real_point_matches_metadata_and_embedding(self):
        row = self.metadata.iloc[0]
        point_id = int(row["id"])
        points = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_vectors=True,
            with_payload=True,
        )

        self.assertEqual(len(points), 1)
        point = points[0]
        self.assertEqual(str(point.payload["label"]), str(row["label"]))
        self.assertEqual(str(point.payload["image_path"]), str(row["image_path"]))

        stored_vector = np.asarray(ui_server.get_vector(point), dtype=np.float32)
        expected_vector = np.asarray(
            self.embeddings[int(row["embedding_index"])], dtype=np.float32
        )
        np.testing.assert_allclose(stored_vector, expected_vector, rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(stored_vector)), 1.0, places=5)

        print(f"Stvarni point ID={point_id}: payload i 512D vektor su usklađeni.")

    def test_real_filter_and_similarity_search(self):
        row = self.metadata.iloc[0]
        point_id = int(row["id"])
        label = str(row["label"])

        filtered = ui_server.filter_points(label, 5)
        self.assertEqual(len(filtered), 5)
        self.assertTrue(all(point["label"] == label for point in filtered))

        similar = ui_server.similar_points(point_id, 5)
        self.assertEqual(len(similar), 5)
        self.assertNotIn(point_id, [point["id"] for point in similar])
        scores = [float(point["score"]) for point in similar]
        self.assertEqual(scores, sorted(scores, reverse=True))

        print(
            f"Stvarni upiti: label filter '{label}' i top-5 pretraga za ID={point_id} rade."
        )

    def test_real_weighted_knn_query(self):
        labeled = self.metadata[
            self.metadata["is_labeled"].astype(str).str.lower().isin({"true", "1"})
        ]
        row = labeled.iloc[0]
        point_id = int(row["id"])
        vector = np.asarray(
            self.embeddings[int(row["embedding_index"])], dtype=np.float32
        )

        backend = error_analysis.QdrantNeighborBackend(
            client=self.client,
            collection_name=COLLECTION_NAME,
            labeled_only=True,
        )
        neighbors = backend.search(point_id, vector, limit=5)
        prediction = error_analysis.weighted_knn_prediction(neighbors)

        self.assertEqual(len(neighbors), 5)
        self.assertNotIn(point_id, [neighbor.point_id for neighbor in neighbors])
        self.assertTrue(all(neighbor.label != "unlabeled" for neighbor in neighbors))
        self.assertIn(
            prediction["predicted_label"], error_analysis.CANONICAL_STL10_CLASSES
        )
        self.assertGreaterEqual(prediction["prediction_confidence"], 0.0)
        self.assertLessEqual(prediction["prediction_confidence"], 1.0)

        print(
            f"Stvarni weighted k-NN za ID={point_id}: "
            f"{row['label']} -> {prediction['predicted_label']}."
        )


if __name__ == "__main__":
    unittest.main()
