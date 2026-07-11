from pathlib import Path
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient


ROOT_DIR = Path(__file__).resolve().parents[1]

# Load shared settings from either supported local environment file.
for env_path in [ROOT_DIR / ".env", ROOT_DIR / "infra" / ".env"]:
    if env_path.exists():
        load_dotenv(env_path, override=False)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "stl10_clip_images")

VECTOR_SIZE = 512
DISTANCE = "Cosine"


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
