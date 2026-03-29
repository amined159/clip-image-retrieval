import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import open_clip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_embeddings.npz"
EVAL_QUERIES_PATH = PROJECT_ROOT / "data" / "eval_queries.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation_results.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def load_embeddings() -> Dict[str, Any]:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")

    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)

    return {
        "ids": data["ids"],
        "filenames": data["filenames"],
        "descriptions": data["descriptions"] if "descriptions" in data else None,
        "tags": data["tags"] if "tags" in data else None,
        "embeddings": data["embeddings"].astype(np.float32),
    }


def load_eval_queries() -> List[Dict[str, Any]]:
    if not EVAL_QUERIES_PATH.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_QUERIES_PATH}")

    items = json.loads(EVAL_QUERIES_PATH.read_text(encoding="utf-8"))

    validated = []
    for i, item in enumerate(items, start=1):
        if "query" not in item:
            raise ValueError(f"Missing 'query' in eval item #{i}")
        if "relevant_image_ids" not in item:
            raise ValueError(f"Missing 'relevant_image_ids' in eval item #{i}")
        if not isinstance(item["relevant_image_ids"], list) or not item["relevant_image_ids"]:
            raise ValueError(f"'relevant_image_ids' must be a non-empty list in eval item #{i}")

        validated.append({
            "query": item["query"],
            "relevant_image_ids": [str(x) for x in item["relevant_image_ids"]],
            "source_query": item.get("source_query"),
        })

    return validated


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, tokenizer, device


def get_text_embedding(query: str, model, tokenizer, device: str) -> np.ndarray:
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()[0].astype(np.float32)


def rank_images_for_query(
    query: str,
    db: Dict[str, Any],
    model,
    tokenizer,
    device: str,
):
    query_embedding = get_text_embedding(query, model, tokenizer, device)
    scores = db["embeddings"] @ query_embedding
    ranked_idx = np.argsort(scores)[::-1]

    ranked_results = []
    for rank, idx in enumerate(ranked_idx, start=1):
        ranked_results.append({
            "rank": rank,
            "image_id": str(db["ids"][idx]),
            "filename": str(db["filenames"][idx]),
            "score": float(scores[idx]),
        })

    return ranked_results


def first_relevant_rank(
    ranked_results: List[Dict[str, Any]],
    relevant_ids: List[str],
) -> Optional[int]:
    relevant_set = set(relevant_ids)
    for item in ranked_results:
        if item["image_id"] in relevant_set:
            return item["rank"]
    return None


def evaluate():
    db = load_embeddings()
    eval_queries = load_eval_queries()
    model, tokenizer, device = load_model()

    print(f"Using device: {device}")
    print(f"Loaded {len(db['ids'])} image embeddings")
    print(f"Loaded {len(eval_queries)} evaluation queries")

    recall_at_1_hits = 0
    recall_at_5_hits = 0
    reciprocal_ranks = []

    detailed_results = []

    for idx, item in enumerate(eval_queries, start=1):
        query = item["query"]
        relevant_ids = item["relevant_image_ids"]
        source_query = item.get("source_query")

        ranked_results = rank_images_for_query(
            query=query,
            db=db,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        best_rank = first_relevant_rank(ranked_results, relevant_ids)

        hit_at_1 = best_rank == 1
        hit_at_5 = best_rank is not None and best_rank <= 5
        rr = 1.0 / best_rank if best_rank is not None else 0.0

        recall_at_1_hits += int(hit_at_1)
        recall_at_5_hits += int(hit_at_5)
        reciprocal_ranks.append(rr)

        top5 = ranked_results[:5]
        for row in top5:
            row["is_relevant"] = row["image_id"] in set(relevant_ids)

        detailed_results.append({
            "query_index": idx,
            "query": query,
            "source_query": source_query,
            "relevant_image_ids": relevant_ids,
            "best_rank": best_rank,
            "hit@1": hit_at_1,
            "hit@5": hit_at_5,
            "reciprocal_rank": rr,
            "top5": top5,
        })

    n = len(eval_queries)
    metrics = {
        "num_queries": n,
        "recall@1": recall_at_1_hits / n if n else 0.0,
        "recall@5": recall_at_5_hits / n if n else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }

    output = {
        "model": {
            "model_name": MODEL_NAME,
            "pretrained": PRETRAINED,
        },
        "metrics": metrics,
        "results": detailed_results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Evaluation Metrics ===")
    print(f"num_queries: {metrics['num_queries']}")
    print(f"recall@1:   {metrics['recall@1']:.4f}")
    print(f"recall@5:   {metrics['recall@5']:.4f}")
    print(f"mrr:        {metrics['mrr']:.4f}")

    print(f"\nSaved detailed evaluation to: {OUTPUT_PATH}")

    print("\n=== Sample Per-query Results (first 5) ===")
    for result in detailed_results[:5]:
        print(f"\nQuery: {result['query']}")
        if result["source_query"]:
            print(f"Source query: {result['source_query']}")
        print(f"Relevant IDs: {result['relevant_image_ids']}")
        print(f"Best rank: {result['best_rank']}")
        print(f"Hit@1: {result['hit@1']}")
        print(f"Hit@5: {result['hit@5']}")
        print("Top 5:")
        for row in result["top5"]:
            flag = "✅" if row["is_relevant"] else " "
            print(
                f"  {row['rank']}. {row['filename']} "
                f"(id={row['image_id']}, score={row['score']:.4f}) {flag}"
            )


if __name__ == "__main__":
    evaluate()