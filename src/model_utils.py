from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


def resolve_device(device: str) -> str:
    """Return a concrete torch device name from auto/cpu/cuda input."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested, but it is not available. Falling back to CPU.")
        return "cpu"
    return device


def load_embedding_model(model_name: str, device: str):
    """Load a pretrained CLIP image encoder and processor."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError(
            "The transformers package is required for the default CLIP model. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    resolved_device = resolve_device(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()
    return model, processor


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    """Normalize one embedding vector to L2 norm 1."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def image_features_to_tensor(features, model=None) -> torch.Tensor:
    """Convert CLIP image feature output variants to a tensor."""
    if isinstance(features, torch.Tensor):
        return features

    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds

    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        pooled_output = features.pooler_output
        if model is not None and hasattr(model, "visual_projection"):
            projection = model.visual_projection
            expected_dim = getattr(projection, "in_features", None)
            if expected_dim is None or pooled_output.shape[-1] == expected_dim:
                return projection(pooled_output)
        return pooled_output

    if isinstance(features, (tuple, list)) and features:
        first = features[0]
        if isinstance(first, torch.Tensor):
            return first

    raise TypeError(f"Unsupported model output type for image embeddings: {type(features)!r}")


def open_image(image_path: str | Path) -> Image.Image:
    """Open an image as RGB and raise a clear error for invalid files."""
    path = Path(image_path)
    try:
        return Image.open(path).convert("RGB")
    except FileNotFoundError as exc:
        raise ValueError(f"Image file does not exist: {path}") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Image file cannot be opened or is corrupted: {path}") from exc


def extract_embedding(image_path: str, model, processor, device: str) -> np.ndarray:
    """Extract one normalized CLIP image embedding from a file path."""
    resolved_device = resolve_device(device)
    image = open_image(image_path)

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
        features = model.get_image_features(**inputs)

    vector = image_features_to_tensor(features, model).cpu().numpy()[0].astype(np.float32)
    return normalize_embedding(vector).astype(np.float32)
