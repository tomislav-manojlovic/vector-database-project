import argparse
from typing import Any

from pathlib import Path
from datetime import datetime, timezone
import time

try:
    from model_utils import extract_embedding, load_embedding_model, resolve_device
except ModuleNotFoundError:
    from src.model_utils import extract_embedding, load_embedding_model, resolve_device

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    PointIdsList,
)

from qdrant_common import get_qdrant_client, COLLECTION_NAME, VECTOR_SIZE


TEST_POINT_ID = 999999


def get_vector_from_record(record: Any):
    vector = record.vector

    # Unnamed-vector collections return a flat vector.
    if isinstance(vector, list):
        return vector

    # Named-vector collections may return a mapping instead.
    if isinstance(vector, dict):
        return next(iter(vector.values()))

    raise RuntimeError("Ne mogu da pročitam vector iz Qdrant record-a.")


def check():
    client = get_qdrant_client()

    exists = client.collection_exists(COLLECTION_NAME)
    print(f"Kolekcija: {COLLECTION_NAME}")
    print(f"Postoji: {exists}")

    if not exists:
        return

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )

    print(f"Broj pointova: {count.count}")


def get_by_id(point_id: int, with_vector: bool = False):
    client = get_qdrant_client()

    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_vectors=with_vector,
    )

    if not points:
        print(f"Point sa ID={point_id} ne postoji.")
        return None

    point = points[0]

    print(f"ID: {point.id}")
    print(f"Payload: {point.payload}")

    if with_vector:
        vector = get_vector_from_record(point)
        print(f"Vector dimenzija: {len(vector)}")
        print(f"Prvih 5 vrednosti vectora: {vector[:5]}")

    return point


def filter_by_label(label: str, limit: int):
    client = get_qdrant_client()

    q_filter = Filter(
        must=[
            FieldCondition(
                key="label",
                match=MatchValue(value=label),
            )
        ]
    )

    points, next_page = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=q_filter,
        limit=limit,
        with_vectors=False,
    )

    print(f"Prikazujem do {limit} pointova za label='{label}':")

    for point in points:
        print(f"ID: {point.id} | Payload: {point.payload}")

    print(f"Next page offset: {next_page}")


def search_similar_by_id(point_id: int, top_k: int, label_filter: str | None = None):
    client = get_qdrant_client()

    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_vectors=True,
    )

    if not points:
        print(f"Point sa ID={point_id} ne postoji.")
        return

    query_vector = get_vector_from_record(points[0])

    q_filter = None
    if label_filter:
        q_filter = Filter(
            must=[
                FieldCondition(
                    key="label",
                    match=MatchValue(value=label_filter),
                )
            ]
        )

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=q_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    print(f"Top {top_k} najsličnijih za ID={point_id}:")
    if label_filter:
        print(f"Filter label='{label_filter}'")

    for scored_point in result.points:
        print(
            f"ID: {scored_point.id} | "
            f"Score: {scored_point.score:.4f} | "
            f"Payload: {scored_point.payload}"
        )


def update_payload(point_id: int, key: str, value: str):
    client = get_qdrant_client()

    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={key: value},
        points=[point_id],
    )

    print(f"Payload update urađen za ID={point_id}: {key}={value}")

    get_by_id(point_id)


def delete_test_point():
    client = get_qdrant_client()

    # Reuse a stored vector so the temporary point respects the collection schema.
    source_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1],
        with_vectors=True,
    )

    if not source_points:
        raise RuntimeError("Ne postoji point ID=1, ne mogu da napravim test point.")

    vector = get_vector_from_record(source_points[0])

    test_point = PointStruct(
        id=TEST_POINT_ID,
        vector=vector,
        payload={
            "id": TEST_POINT_ID,
            "image_path": "test/delete_demo.jpg",
            "label": "test",
            "note": "temporary point for delete demo",
        },
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[test_point],
    )

    print(f"Test point ubačen: ID={TEST_POINT_ID}")

    before_delete = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[TEST_POINT_ID],
        with_vectors=False,
    )

    print(f"Postoji pre brisanja: {len(before_delete) == 1}")

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[TEST_POINT_ID]),
    )

    after_delete = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[TEST_POINT_ID],
        with_vectors=False,
    )

    print(f"Postoji posle brisanja: {len(after_delete) == 1}")

def delete_by_id(point_id: int, force: bool = False):
    client = get_qdrant_client()

    existing = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_vectors=False,
    )

    if not existing:
        print(f"Point sa ID={point_id} ne postoji.")
        return

    point = existing[0]

    print("Point koji će biti obrisan:")
    print(f"ID: {point.id}")
    print(f"Payload: {point.payload}")

    if not force:
        confirmation = input(
            f"Da li sigurno želiš da obrišeš point ID={point_id}? Ukucaj 'yes' za potvrdu: "
        )

        if confirmation != "yes":
            print("Brisanje otkazano.")
            return

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[point_id]),
    )

    after_delete = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_vectors=False,
    )

    if after_delete:
        print(f"Brisanje nije uspelo. Point ID={point_id} i dalje postoji.")
    else:
        print(f"Point ID={point_id} je uspešno obrisan.")



def generate_manual_point_id() -> int:
    return int(time.time() * 1000)


def create_from_image(
    image_path: str,
    label: str,
    point_id: int | None = None,
    device: str = "auto",
    model_name: str = "openai/clip-vit-base-patch32",
    ):
    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Kolekcija '{COLLECTION_NAME}' ne postoji.")
        print("Prvo pokreni import deo Člana 4.")
        return

    path = Path(image_path)

    if not path.exists():
        print(f"Slika ne postoji: {path}")
        return

    if not path.is_file():
        print(f"Putanja nije fajl: {path}")
        return

    if point_id is None:
        point_id = generate_manual_point_id()

    existing = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_vectors=False,
    )

    if existing:
        print(f"Point sa ID={point_id} već postoji.")
        print("Izaberi drugi ID ili pokreni bez --id da se automatski generiše novi.")
        return

    resolved_device = resolve_device(device)

    print(f"Učitavam CLIP model: {model_name}")
    print(f"Device: {resolved_device}")

    model, processor = load_embedding_model(model_name, resolved_device)

    print(f"Pravim embedding za sliku: {path}")

    vector = extract_embedding(
        image_path=str(path),
        model=model,
        processor=processor,
        device=resolved_device,
    )

    if len(vector) != VECTOR_SIZE:
        raise ValueError(
            f"Embedding dimenzija nije ispravna. "
            f"Očekivano: {VECTOR_SIZE}, dobijeno: {len(vector)}"
        )

    payload = {
        "id": point_id,
        "image_path": str(path).replace("\\", "/"),
        "label": label,
        "source": "cli_create_from_image",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    point = PointStruct(
        id=point_id,
        vector=vector.astype(float).tolist(),
        payload=payload,
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point],
    )

    print("Novi point je uspešno dodat u Qdrant.")
    print(f"ID: {point_id}")
    print(f"Payload: {payload}")

    print()
    print("Provera dodatog pointa:")
    get_by_id(point_id, with_vector=True)

def main():
    parser = argparse.ArgumentParser(description="Qdrant upiti i CRUD demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check")

    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("id", type=int)
    get_parser.add_argument("--with-vector", action="store_true")

    filter_parser = subparsers.add_parser("filter")
    filter_parser.add_argument("label", type=str)
    filter_parser.add_argument("--limit", type=int, default=10)

    similar_parser = subparsers.add_parser("similar")
    similar_parser.add_argument("id", type=int)
    similar_parser.add_argument("--top-k", type=int, default=5)
    similar_parser.add_argument("--label", type=str, default=None)

    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("id", type=int)
    update_parser.add_argument("key", type=str)
    update_parser.add_argument("value", type=str)

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("id", type=int)
    delete_parser.add_argument(
        "--yes",
        action="store_true",
        help="Briše point bez dodatne potvrde.",
    )

    create_image_parser = subparsers.add_parser("create-from-image")
    create_image_parser.add_argument("image_path", type=str)
    create_image_parser.add_argument("label", type=str)
    create_image_parser.add_argument("--id", type=int, default=None)
    create_image_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    create_image_parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
    )

    subparsers.add_parser("delete-test")

    args = parser.parse_args()

    if args.command == "check":
        check()
    elif args.command == "get":
        get_by_id(args.id, with_vector=args.with_vector)
    elif args.command == "filter":
        filter_by_label(args.label, args.limit)
    elif args.command == "similar":
        search_similar_by_id(args.id, args.top_k, args.label)
    elif args.command == "update":
        update_payload(args.id, args.key, args.value)
    elif args.command == "delete-test":
        delete_test_point()
    elif args.command == "delete":
        delete_by_id(args.id, force=args.yes)
    elif args.command == "create-from-image":
        create_from_image(
            image_path=args.image_path,
            label=args.label,
            point_id=args.id,
            device=args.device,
            model_name=args.model_name,
        )


if __name__ == "__main__":
    main()
