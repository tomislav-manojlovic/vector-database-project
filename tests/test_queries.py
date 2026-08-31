import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_PATH = SRC_DIR / "06_queries.py"
SPEC = importlib.util.spec_from_file_location("queries", MODULE_PATH)
queries = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = queries

model_utils_stub = ModuleType("model_utils")
model_utils_stub.extract_embedding = lambda *_args, **_kwargs: None
model_utils_stub.load_embedding_model = lambda *_args, **_kwargs: None
model_utils_stub.resolve_device = lambda device: device

qdrant_common_stub = ModuleType("qdrant_common")
qdrant_common_stub.get_qdrant_client = lambda: None
qdrant_common_stub.COLLECTION_NAME = "test_collection"
qdrant_common_stub.VECTOR_SIZE = 512

qdrant_client_stub = ModuleType("qdrant_client")
qdrant_models_stub = ModuleType("qdrant_client.models")
for model_name in [
    "FieldCondition",
    "Filter",
    "MatchValue",
    "PointStruct",
    "PointIdsList",
]:
    setattr(qdrant_models_stub, model_name, object)
qdrant_client_stub.models = qdrant_models_stub

with patch.dict(
    sys.modules,
    {
        "model_utils": model_utils_stub,
        "qdrant_common": qdrant_common_stub,
        "qdrant_client": qdrant_client_stub,
        "qdrant_client.models": qdrant_models_stub,
    },
):
    SPEC.loader.exec_module(queries)


class FakeQdrantClient:
    def __init__(self):
        self.query_limit = None

    def retrieve(self, **_kwargs):
        return [SimpleNamespace(id=1, vector=[1.0, 0.0], payload={})]

    def query_points(self, **kwargs):
        self.query_limit = kwargs["limit"]
        return SimpleNamespace(
            points=[
                SimpleNamespace(id=1, score=1.0, payload={"image_path": "query.jpg"}),
                SimpleNamespace(id=2, score=0.9, payload={"image_path": "second.jpg"}),
                SimpleNamespace(id=3, score=0.8, payload={"image_path": "third.jpg"}),
            ]
        )


class QueryTests(unittest.TestCase):
    def test_similar_excludes_query_point_and_keeps_top_k_neighbors(self):
        client = FakeQdrantClient()

        with patch.object(queries, "get_qdrant_client", return_value=client):
            result = queries.search_similar_by_id(point_id=1, top_k=2)

        self.assertEqual(client.query_limit, 3)
        self.assertEqual([point.id for point in result], [2, 3])
        self.assertNotIn(1, [point.id for point in result])


if __name__ == "__main__":
    unittest.main()
