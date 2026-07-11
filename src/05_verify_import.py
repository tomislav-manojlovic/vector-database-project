from qdrant_common import get_qdrant_client, COLLECTION_NAME


def main():
    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Kolekcija '{COLLECTION_NAME}' ne postoji.")
        return

    count_result = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )

    print(f"Broj pointova u kolekciji: {count_result.count}")

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=5,
        with_vectors=False,
    )

    print("Prvih nekoliko pointova:")

    for point in points:
        print("ID:", point.id)
        print("Payload:", point.payload)
        print("---")


if __name__ == "__main__":
    main()