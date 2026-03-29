"""
FastAPI serving layer for the CLIP image retrieval pipeline.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

Example:
    curl -X POST http://localhost:8000/retrieve \
         -H "Content-Type: application/json" \
         -d '{"query": "a person reflecting on their identity", "top_k": 3}'

Health:
    curl http://localhost:8000/health
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths / Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_embeddings.npz"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

app = FastAPI(title="CLIP Image Retrieval API", version="1.0.0")

# Globals loaded at startup
model = None
tokenizer = None
device = None
image_embeddings = None
image_ids = None
image_filenames = None
metadata_by_id = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metadata_map() -> dict[str, dict[str, Any]]:
    if not METADATA_PATH.exists():
        return {}

    items = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in items}


def get_text_embedding(query: str) -> torch.Tensor:
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return text_features


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_resources():
    global model, tokenizer, device
    global image_embeddings, image_ids, image_filenames, metadata_by_id

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_loaded, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model_loaded = model_loaded.to(device).eval()
    tok = open_clip.get_tokenizer(MODEL_NAME)

    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)

    embeddings_np = data["embeddings"].astype(np.float32)
    ids_np = data["ids"]
    filenames_np = data["filenames"]

    embeddings_t = torch.tensor(embeddings_np, dtype=torch.float32).to(device)
    embeddings_t = torch.nn.functional.normalize(embeddings_t, dim=-1)

    model = model_loaded
    tokenizer = tok
    image_embeddings = embeddings_t
    image_ids = [str(x) for x in ids_np.tolist()]
    image_filenames = [str(x) for x in filenames_np.tolist()]
    metadata_by_id = load_metadata_map()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RetrievalRequest(BaseModel):
    query: str = Field(..., description="Natural language retrieval query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class RetrievalResult(BaseModel):
    image_id: str
    filename: str
    score: float
    description: str | None = None
    tags: list[str] = []
    mood: list[str] = []
    use_cases: list[str] = []
    composition: str | None = None


class RetrievalResponse(BaseModel):
    query: str
    model_name: str
    pretrained: str
    num_images: int
    results: list[RetrievalResult]
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve(request: RetrievalRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    t0 = time.perf_counter()

    text_emb = get_text_embedding(request.query)
    scores = (image_embeddings @ text_emb.T).squeeze(-1)

    k = min(request.top_k, len(image_ids))
    top = scores.topk(k)
    top_indices = top.indices.tolist()

    results = []
    for i in top_indices:
        image_id = image_ids[i]
        filename = image_filenames[i]
        meta = metadata_by_id.get(image_id, {})

        results.append(
            RetrievalResult(
                image_id=image_id,
                filename=filename,
                score=float(scores[i].item()),
                description=meta.get("description"),
                tags=meta.get("tags", []),
                mood=meta.get("mood", []),
                use_cases=meta.get("use_cases", []),
                composition=meta.get("composition"),
            )
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    return RetrievalResponse(
        query=request.query,
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
        num_images=len(image_ids),
        results=results,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/metadata/{image_id}")
def get_metadata(image_id: str):
    item = metadata_by_id.get(image_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Image ID not found: {image_id}")
    return item


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "device": device,
        "num_images": len(image_ids) if image_ids is not None else 0,
        "metadata_loaded": len(metadata_by_id) if metadata_by_id is not None else 0,
    }