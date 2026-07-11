from pathlib import Path
import shutil
import pandas as pd
from torchvision.datasets import STL10

DATASET_NAME = "STL10"
LIMIT_PER_CLASS = 100

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
IMAGES_DIR = ROOT_DIR / "data" / "images" / "stl10"
METADATA_PATH = ROOT_DIR / "data" / "metadata.csv"
SAMPLE_METADATA_PATH = ROOT_DIR / "data" / "metadata_sample.csv"

CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def main():
    print("Preuzimam/učitavam STL-10 dataset...")

    dataset = STL10(
        root=str(RAW_DIR),
        split="train",
        download=True
    )

    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    counters = {class_name: 0 for class_name in CLASS_NAMES}
    rows = []
    image_id = 1

    for source_index in range(len(dataset)):
        image, label_id = dataset[source_index]
        label = CLASS_NAMES[label_id]

        if counters[label] >= LIMIT_PER_CLASS:
            continue

        counters[label] += 1

        class_dir = IMAGES_DIR / label
        class_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{label}_{counters[label]:04d}.jpg"
        image_path = class_dir / filename

        image.convert("RGB").save(image_path, quality=95)

        relative_path = image_path.relative_to(ROOT_DIR).as_posix()

        rows.append({
            "id": image_id,
            "image_path": relative_path,
            "label": label,
            "label_id": label_id,
            "dataset": DATASET_NAME,
            "split": "train",
            "source_index": source_index,
            "filename": filename
        })

        image_id += 1

        if all(count >= LIMIT_PER_CLASS for count in counters.values()):
            break

    metadata = pd.DataFrame(rows)
    metadata.to_csv(METADATA_PATH, index=False)

    sample_metadata = metadata.groupby("label").head(3)
    sample_metadata.to_csv(SAMPLE_METADATA_PATH, index=False)

    print("Gotovo.")
    print(f"Ukupan broj slika: {len(metadata)}")
    print(f"Metadata fajl: {METADATA_PATH}")
    print(f"Sample metadata fajl: {SAMPLE_METADATA_PATH}")
    print()
    print("Broj slika po klasama:")
    print(metadata["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()