import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ui" / "server.py"
SPEC = importlib.util.spec_from_file_location("ui_server", MODULE_PATH)
ui_server = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ui_server
SPEC.loader.exec_module(ui_server)


class UIServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        cls.client = QdrantClient(location=":memory:")
        cls.client.create_collection(
            collection_name=ui_server.COLLECTION_NAME,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        cls.client.upsert(
            collection_name=ui_server.COLLECTION_NAME,
            points=[
                PointStruct(id=1, vector=[1, 0, 0, 0], payload={"id": 1, "label": "cat", "image_path": "cat.jpg"}),
                PointStruct(id=2, vector=[0.9, 0.1, 0, 0], payload={"id": 2, "label": "cat", "image_path": "cat2.jpg"}),
                PointStruct(id=3, vector=[0, 1, 0, 0], payload={"id": 3, "label": "dog", "image_path": "dog.jpg"}),
            ],
            wait=True,
        )
        ui_server.get_qdrant_client = lambda: cls.client

    def tearDown(self):
        ui_server.cleanup_demo_points()

    def test_status_lookup_filter_and_similarity(self):
        status = ui_server.qdrant_status()
        self.assertTrue(status["connected"])
        self.assertEqual(status["count"], 3)
        self.assertEqual(ui_server.get_point(1)["label"], "cat")
        self.assertEqual(len(ui_server.filter_points("cat", 10)), 2)
        similar = ui_server.similar_points(1, 2)
        self.assertEqual(similar[0]["id"], 2)
        self.assertNotIn(1, [point["id"] for point in similar])

    def test_safe_crud_lifecycle(self):
        created = ui_server.create_demo_point(1, 9_000_001, "demo-cat")
        self.assertTrue(created["is_demo"])
        self.assertEqual(created["vector_dimension"], 4)

        updated = ui_server.update_demo_point(9_000_001, "reviewed", True)
        self.assertTrue(updated["payload"]["reviewed"])

        deleted = ui_server.delete_demo_point(9_000_001)
        self.assertTrue(deleted["deleted"])
        with self.assertRaises(KeyError):
            ui_server.get_point(9_000_001)

    def test_original_point_cannot_be_updated_or_deleted(self):
        with self.assertRaises(PermissionError):
            ui_server.update_demo_point(1, "reviewed", True)
        with self.assertRaises(PermissionError):
            ui_server.delete_demo_point(1)

    def test_reports_are_available_to_ui(self):
        errors = ui_server.error_data(5)
        cleaning = ui_server.cleaning_data()
        self.assertIsNotNone(errors["summary"])
        self.assertGreater(len(errors["errors"]), 0)
        self.assertIsNotNone(cleaning["summary"])
        self.assertGreater(len(cleaning["groups"]), 0)


if __name__ == "__main__":
    unittest.main()
