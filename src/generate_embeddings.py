import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from model_utils import (
        image_features_to_tensor,
        load_embedding_model,
        normalize_embedding,
        open_image,
        resolve_device,
    )
except ModuleNotFoundError:
    from src.model_utils import (
        image_features_to_tensor,
        load_embedding_model,
        normalize_embedding,
        open_image,
        resolve_device,
    )


REQUIRED_COLUMNS = {"id", "image_path", "label"}
OUTPUT_FILES = ("embeddings.npy", "embeddings_metadata.csv", "embedding_config.json")


def get_tqdm():
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "The tqdm package is required for the progress bar. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate normalized image embeddings for later Qdrant import."
    )
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--images-root", default="data/images")
    parser.add_argument("--output-dir", default="data/embeddings")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def validate_metadata(metadata: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(metadata.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"metadata.csv is missing required columns: {missing_columns}")


def path_parts_lower(path: Path) -> tuple[str, ...]:
    return tuple(part.lower() for part in path.parts)


def resolve_image_path(raw_path: str, images_root: Path, project_root: Path, metadata_dir: Path) -> Path:
    """Resolve image_path without duplicating data/images when it is already present."""
    image_path = Path(str(raw_path).strip())

    if image_path.is_absolute():
        return image_path

    candidates = [
        Path.cwd() / image_path,
        project_root / image_path,
        metadata_dir / image_path,
    ]

    image_parts = path_parts_lower(image_path)
    root_parts = path_parts_lower(images_root)
    root_name = images_root.name.lower()

    if root_name not in image_parts and not image_parts[: len(root_parts)] == root_parts:
        candidates.append(images_root / image_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1] if candidates else image_path


def output_image_path(image_path: Path, project_root: Path) -> str:
    try:
        return image_path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(image_path)


def warn_if_outputs_exist(output_dir: Path) -> None:
    existing = [output_dir / filename for filename in OUTPUT_FILES if (output_dir / filename).exists()]
    if existing:
        print("Warning: the following output files already exist and will be overwritten:")
        for path in existing:
            print(f"  {path}")


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract_batch(paths: list[Path], model, processor, device: str) -> np.ndarray:
    images = [open_image(path) for path in paths]

    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        features = model.get_image_features(**inputs)

    vectors = image_features_to_tensor(features, model).cpu().numpy().astype(np.float32)
    normalized = np.vstack([normalize_embedding(vector) for vector in vectors])
    return normalized.astype(np.float32)


def main() -> None:
    args = parse_args()
    tqdm = get_tqdm()
    project_root = Path(__file__).resolve().parents[1]
    metadata_path = Path(args.metadata)
    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)

    if not metadata_path.is_absolute():
        metadata_path = project_root / metadata_path
    if not images_root.is_absolute():
        images_root = project_root / images_root
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    warn_if_outputs_exist(output_dir)

    metadata = pd.read_csv(metadata_path)
    validate_metadata(metadata)

    print(f"Loading model: {args.model_name}")
    print(f"Using device: {device}")
    model, processor = load_embedding_model(args.model_name, device)

    pending_rows = []
    skipped_rows = []

    for _, row in metadata.iterrows():
        image_path = resolve_image_path(
            str(row["image_path"]),
            images_root=images_root,
            project_root=project_root,
            metadata_dir=metadata_path.parent,
        )
        row_data = {
            "id": row["id"],
            "image_path": output_image_path(image_path, project_root),
            "label": row["label"],
        }

        if not image_path.exists():
            row_data["embedding_index"] = -1
            row_data["status"] = "missing"
            skipped_rows.append(row_data)
            print(f"Skipping missing image: {image_path}")
            continue

        pending_rows.append({**row_data, "resolved_path": image_path})

    embeddings = []
    output_rows = []

    for batch in tqdm(
        list(batched(pending_rows, args.batch_size)),
        desc="Generating embeddings",
        unit="batch",
    ):
        try:
            batch_paths = [item["resolved_path"] for item in batch]
            batch_embeddings = extract_batch(batch_paths, model, processor, device)
        except ValueError as exc:
            print(f"Batch failed, retrying images one by one. Reason: {exc}")
            for item in batch:
                try:
                    vector = extract_batch([item["resolved_path"]], model, processor, device)[0]
                except ValueError as image_exc:
                    skipped_rows.append(
                        {
                            "id": item["id"],
                            "image_path": item["image_path"],
                            "label": item["label"],
                            "embedding_index": -1,
                            "status": f"skipped: {image_exc}",
                        }
                    )
                    print(f"Skipping invalid image: {item['resolved_path']} ({image_exc})")
                    continue

                embedding_index = len(embeddings)
                embeddings.append(vector)
                output_rows.append(
                    {
                        "id": item["id"],
                        "image_path": item["image_path"],
                        "label": item["label"],
                        "embedding_index": embedding_index,
                        "status": "ok",
                    }
                )
            continue

        for item, vector in zip(batch, batch_embeddings):
            embedding_index = len(embeddings)
            embeddings.append(vector)
            output_rows.append(
                {
                    "id": item["id"],
                    "image_path": item["image_path"],
                    "label": item["label"],
                    "embedding_index": embedding_index,
                    "status": "ok",
                }
            )

    if embeddings:
        embeddings_matrix = np.vstack(embeddings).astype(np.float32)
        embedding_dim = int(embeddings_matrix.shape[1])
    else:
        embeddings_matrix = np.empty((0, 0), dtype=np.float32)
        embedding_dim = 0

    embeddings_path = output_dir / "embeddings.npy"
    metadata_output_path = output_dir / "embeddings_metadata.csv"
    config_path = output_dir / "embedding_config.json"

    np.save(embeddings_path, embeddings_matrix)
    pd.DataFrame(output_rows + skipped_rows).to_csv(metadata_output_path, index=False)

    config = {
        "model_name": args.model_name,
        "embedding_dim": embedding_dim,
        "normalize": True,
        "distance_metric": "cosine",
        "number_of_images": int(embeddings_matrix.shape[0]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Valid embeddings: {embeddings_matrix.shape[0]}")
    print(f"Skipped images: {len(skipped_rows)}")
    print(f"Embeddings file: {embeddings_path}")
    print(f"Metadata file: {metadata_output_path}")
    print(f"Config file: {config_path}")


if __name__ == "__main__":
    main()
