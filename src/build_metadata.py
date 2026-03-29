import os
import json
import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata.json"

MODEL_NAME = "gpt-5.4-mini"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 6,
        },
        "mood": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
        "use_cases": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "composition": {"type": "string"},
    },
    "required": ["description", "tags", "mood", "use_cases", "composition"],
    "additionalProperties": False,
}


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def build_prompt() -> str:
    return (
        "You are annotating images for a cross-modal image retrieval dataset.\n\n"
        "Return metadata for the image in strict JSON.\n\n"
        "Rules:\n"
        "- description: exactly one concise sentence\n"
        "- tags: 3 to 6 short semantic tags\n"
        "- mood: 1 to 3 mood words\n"
        "- use_cases: 2 to 4 short retrieval use cases\n"
        "- composition: one short phrase describing framing/composition\n"
        "- Focus on visible content and abstract semantic meaning\n"
        "- Do not invent backstory\n"
        "- Do not mention camera brands, watermarks, or text overlays\n"
        "- Keep wording concise and reusable for retrieval/evaluation\n"
    )


def annotate_image(client: OpenAI, image_path: Path) -> Dict[str, Any]:
    data_url = image_to_data_url(image_path)

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_prompt()},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "image_metadata",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )

    return json.loads(response.output_text)


def load_existing_metadata() -> List[Dict[str, Any]]:
    if OUTPUT_PATH.exists():
        return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    return []


def next_image_id(index: int) -> str:
    return f"img_{index:03d}"


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

    image_paths = sorted(
        [p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    if not image_paths:
        raise ValueError(f"No supported images found in {IMAGES_DIR}")

    client = OpenAI(api_key=api_key)

    existing = load_existing_metadata()
    existing_by_filename = {item["filename"]: item for item in existing}

    results: List[Dict[str, Any]] = []
    running_index = 1

    for image_path in image_paths:
        if image_path.name in existing_by_filename:
            item = existing_by_filename[image_path.name]
            if "id" not in item or not item["id"]:
                item["id"] = next_image_id(running_index)
            results.append(item)
            print(f"[skip] {image_path.name}")
            running_index += 1
            continue

        print(f"[annotate] {image_path.name}")
        metadata = annotate_image(client, image_path)

        item = {
            "id": next_image_id(running_index),
            "filename": image_path.name,
            "description": metadata["description"],
            "tags": metadata["tags"],
            "mood": metadata["mood"],
            "use_cases": metadata["use_cases"],
            "composition": metadata["composition"],
        }
        results.append(item)
        running_index += 1

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved metadata to: {OUTPUT_PATH}")
    print(f"Total images annotated: {len(results)}")


if __name__ == "__main__":
    main()