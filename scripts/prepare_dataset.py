from pathlib import Path
import shutil
import pandas as pd
from torchvision.datasets import STL10

DATASET_NAME = "STL10"
SPLITS = ("train", "test", "unlabeled")
EXPECTED_SPLIT_SIZES = {
    "train": 5_000,
    "test": 8_000,
    "unlabeled": 100_000,
}

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

    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    image_id = 1

    for split in SPLITS:
        dataset = STL10(
            root=str(RAW_DIR),
            split=split,
            download=True,
        )

        expected_size = EXPECTED_SPLIT_SIZES[split]
        if len(dataset) != expected_size:
            raise RuntimeError(
                f"Split '{split}' ima {len(dataset)} slika, očekivano {expected_size}."
            )

        print(f"Obrađujem split '{split}': {len(dataset)} slika...")

        for source_index in range(len(dataset)):
            image, label_id = dataset[source_index]
            label_id = int(label_id)
            is_labeled = label_id >= 0
            label = CLASS_NAMES[label_id] if is_labeled else "unlabeled"

            split_dir = IMAGES_DIR / split
            image_dir = split_dir / label if is_labeled else split_dir
            image_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{label}_{source_index + 1:06d}.jpg"
            image_path = image_dir / filename
            image.convert("RGB").save(image_path, quality=95)

            rows.append({
                "id": image_id,
                "image_path": image_path.relative_to(ROOT_DIR).as_posix(),
                "label": label,
                "label_id": label_id,
                "is_labeled": is_labeled,
                "dataset": DATASET_NAME,
                "split": split,
                "source_index": source_index,
                "filename": filename,
            })

            image_id += 1

            if (source_index + 1) % 10_000 == 0:
                print(f"  Sačuvano: {source_index + 1}/{len(dataset)}")

    metadata = pd.DataFrame(rows)
    metadata.to_csv(METADATA_PATH, index=False)

    sample_metadata = metadata.groupby(["split", "label"], sort=False).head(3)
    sample_metadata.to_csv(SAMPLE_METADATA_PATH, index=False)

    print("Gotovo.")
    print(f"Ukupan broj slika: {len(metadata)}")
    print(f"Metadata fajl: {METADATA_PATH}")
    print(f"Sample metadata fajl: {SAMPLE_METADATA_PATH}")
    print()
    print("Broj slika po splitu i labeli:")
    print(metadata.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
