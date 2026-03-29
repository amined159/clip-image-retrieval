import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"
OUTPUT_PATH = OUTPUT_DIR / "image_embeddings.npz"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model = model.to(device)
    model.eval()

    metadata = load_metadata()

    ids = []
    filenames = []
    descriptions = []
    tags_joined = []
    embeddings = []

    print("Embedding images...")
    with torch.no_grad():
        for item in tqdm(metadata):
            filename = item["filename"]
            image_path = IMAGES_DIR / filename

            if not image_path.exists():
                print(f"[warn] Missing file, skipped: {image_path}")
                continue

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                print(f"[warn] Unsupported extension, skipped: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            ids.append(item["id"])
            filenames.append(filename)
            descriptions.append(item.get("description", ""))
            tags_joined.append(" | ".join(item.get("tags", [])))
            embeddings.append(image_features.cpu().numpy()[0])

    if not embeddings:
        raise ValueError("No embeddings were created. Check your images and metadata.")

    embeddings = np.stack(embeddings).astype(np.float32)

    np.savez(
        OUTPUT_PATH,
        ids=np.array(ids),
        filenames=np.array(filenames),
        descriptions=np.array(descriptions),
        tags=np.array(tags_joined),
        embeddings=embeddings,
    )

    print(f"\nSaved embeddings to: {OUTPUT_PATH}")
    print(f"Total embedded images: {len(ids)}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()