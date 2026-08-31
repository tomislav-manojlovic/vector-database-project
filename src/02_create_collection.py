import sys

from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

from qdrant_common import (
    get_qdrant_client,
    COLLECTION_NAME,
    VECTOR_SIZE,
)


def main():
    client = get_qdrant_client()

    recreate = "--recreate" in sys.argv

    if client.collection_exists(COLLECTION_NAME):
        if recreate:
            print(f"Kolekcija '{COLLECTION_NAME}' vec postoji. Brisem je...")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"Kolekcija '{COLLECTION_NAME}' vec postoji.")
            print("Ako zelis da je obrises i napravis ponovo, pokreni sa --recreate.")
            return

    print(f"Pravim kolekciju '{COLLECTION_NAME}'...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print("Kolekcija napravljena.")

    print("Pravim payload index za polje 'label'...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="label",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    print("Pravim payload index za polje 'split'...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="split",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    print("Pravim payload index za polje 'is_labeled'...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="is_labeled",
        field_schema=PayloadSchemaType.BOOL,
    )

    print("Payload indeksi napravljeni.")
    print("Gotovo.")


if __name__ == "__main__":
    main()
