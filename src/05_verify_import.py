from qdrant_common import get_qdrant_client, COLLECTION_NAME


EXPECTED_POINTS = 113_000


def enum_value(value):
    return getattr(value, "value", value)


def main():
    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        raise SystemExit(f"Kolekcija '{COLLECTION_NAME}' ne postoji.")

    verification_errors = []

    count_result = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )
    exact_count = int(count_result.count)

    info = client.get_collection(COLLECTION_NAME)

    points_count = int(info.points_count or 0)
    indexed_vectors_count = int(info.indexed_vectors_count or 0)
    status = str(enum_value(info.status))
    optimizer_status = str(enum_value(info.optimizer_status))

    print(f"Kolekcija: {COLLECTION_NAME}")
    print(f"Tacno pointova: {exact_count}")
    print(f"Qdrant points_count: {points_count}")
    print(f"Indeksirano vektora (HNSW): {indexed_vectors_count}")
    print(f"Status kolekcije: {status}")
    print(f"Optimizer status: {optimizer_status}")

    if exact_count:
        indexed_percent = indexed_vectors_count / exact_count * 100
        print(f"Priblizno indeksirano: {indexed_percent:.2f}%")

    print()

    if exact_count != EXPECTED_POINTS:
        verification_errors.append(
            f"Ocekivano je {EXPECTED_POINTS} pointova, a trenutno ih ima {exact_count}."
        )
        print(f"GRESKA: {verification_errors[-1]}")
    else:
        print(f"OK: U kolekciji je svih {EXPECTED_POINTS} pointova.")

    if indexed_vectors_count == 0:
        print(
            "HNSW NIJE napravljen: indexed_vectors_count je 0. "
            "Proveri indexing_threshold ili sacekaj optimizaciju."
        )
    elif status.lower() == "green":
        print("OK: HNSW indeks postoji i kolekcija je spremna za pretragu.")
    else:
        print(
            "HNSW indeks se gradi ili optimizacija jos traje. "
            "Ponovo pokreni proveru kada status postane green."
        )

    print()
    print("Payload indeksi:")
    payload_schema = info.payload_schema or {}

    for field_name in ("label", "split", "is_labeled"):
        if field_name in payload_schema:
            print(f"  OK: {field_name}")
        else:
            print(f"  NEDOSTAJE: {field_name}")
            verification_errors.append(f"Nedostaje payload indeks: {field_name}.")

    print()
    print("Prvih nekoliko pointova:")

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=5,
        with_vectors=False,
        with_payload=True,
    )

    for point in points:
        print("ID:", point.id)
        print("Payload:", point.payload)
        print("---")

    if verification_errors:
        print()
        print("VERIFIKACIJA NIJE USPESNA:")
        for error in verification_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print()
    print("VERIFIKACIJA JE USPESNA.")


if __name__ == "__main__":
    main()
