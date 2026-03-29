import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import open_clip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_embeddings.npz"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def load_embeddings() -> Dict[str, Any]:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return {
        "ids": data["ids"],
        "filenames": data["filenames"],
        "descriptions": data["descriptions"],
        "tags": data["tags"],
        "embeddings": data["embeddings"].astype(np.float32),
    }


def load_metadata_map() -> Dict[str, Dict[str, Any]]:
    if not METADATA_PATH.exists():
        return {}
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {item["id"]: item for item in metadata}


def get_text_embedding(query: str, model, tokenizer, device: str) -> np.ndarray:
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()[0].astype(np.float32)


def softmax_sample_topk(scores: np.ndarray, top_k: int = 5, temperature: float = 0.5):
    top_idx = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_idx]

    if len(top_scores) > 1 and (top_scores[0] - top_scores[1]) > 0.08:
        return int(top_idx[0]), {
            "method": "argmax",
            "candidate_indices": top_idx.tolist(),
            "candidate_scores": top_scores.tolist(),
        }

    scaled = top_scores / temperature
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / probs.sum()

    chosen_local = np.random.choice(len(top_idx), p=probs)
    chosen_global = int(top_idx[chosen_local])

    return chosen_global, {
        "method": "sample_topk",
        "candidate_indices": top_idx.tolist(),
        "candidate_scores": top_scores.tolist(),
        "candidate_probabilities": probs.tolist(),
    }


def retrieve(
    query: str,
    top_k: int = 5,
    sample_one: bool = False,
    temperature: float = 0.5,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    db = load_embeddings()
    metadata_map = load_metadata_map()

    query_embedding = get_text_embedding(query, model, tokenizer, device)

    image_embeddings = db["embeddings"]
    scores = image_embeddings @ query_embedding

    ranked_idx = np.argsort(scores)[::-1]

    results = []
    for idx in ranked_idx[:top_k]:
        image_id = str(db["ids"][idx])
        meta = metadata_map.get(image_id, {})

        results.append({
            "rank": len(results) + 1,
            "id": image_id,
            "filename": str(db["filenames"][idx]),
            "score": float(scores[idx]),
            "description": meta.get("description", str(db["descriptions"][idx])),
            "tags": meta.get("tags", str(db["tags"][idx]).split(" | ") if str(db["tags"][idx]) else []),
            "mood": meta.get("mood", []),
            "use_cases": meta.get("use_cases", []),
            "composition": meta.get("composition", ""),
        })

    output: Dict[str, Any] = {
        "query": query,
        "top_k": results,
    }

    if sample_one:
        chosen_idx, sampling_info = softmax_sample_topk(
            scores=scores,
            top_k=min(5, len(scores)),
            temperature=temperature,
        )

        image_id = str(db["ids"][chosen_idx])
        meta = metadata_map.get(image_id, {})

        output["selected"] = {
            "id": image_id,
            "filename": str(db["filenames"][chosen_idx]),
            "score": float(scores[chosen_idx]),
            "description": meta.get("description", str(db["descriptions"][chosen_idx])),
            "tags": meta.get("tags", str(db["tags"][chosen_idx]).split(" | ") if str(db["tags"][chosen_idx]) else []),
            "mood": meta.get("mood", []),
            "use_cases": meta.get("use_cases", []),
            "composition": meta.get("composition", ""),
        }
        output["sampling"] = sampling_info

    return output


def main():
    parser = argparse.ArgumentParser(description="Retrieve images from text using CLIP embeddings.")
    parser.add_argument("query", type=str, help="Text query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--sample_one", action="store_true", help="Sample one result from top-5")
    parser.add_argument("--temperature", type=float, default=0.5, help="Softmax temperature for sampling")
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    args = parser.parse_args()

    result = retrieve(
        query=args.query,
        top_k=args.top_k,
        sample_one=args.sample_one,
        temperature=args.temperature,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"\nQuery: {result['query']}\n")
    print("Top results:")
    for item in result["top_k"]:
        print(f"  {item['rank']}. {item['filename']}  score={item['score']:.4f}")
        print(f"     description: {item['description']}")
        print(f"     tags: {', '.join(item['tags'])}")
        if item["mood"]:
            print(f"     mood: {', '.join(item['mood'])}")
        if item["use_cases"]:
            print(f"     use_cases: {', '.join(item['use_cases'])}")
        if item["composition"]:
            print(f"     composition: {item['composition']}")
        print()

    if "selected" in result:
        picked = result["selected"]
        print("Selected image:")
        print(f"  {picked['filename']}  score={picked['score']:.4f}")
        print(f"  method: {result['sampling']['method']}")


if __name__ == "__main__":
    main()