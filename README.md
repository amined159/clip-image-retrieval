# CLIP Image Retrieval

A **cross-modal image retrieval system** using CLIP to match natural language queries with semantically relevant images.

---

## Overview

This project implements a full **text-to-image retrieval pipeline**:

> Given a textual query describing a concept, retrieve the most semantically relevant image from a dataset.

Unlike simple keyword matching, this system leverages **CLIP embeddings** to capture **abstract and symbolic meaning**, enabling queries such as:

* *“a person questioning their identity”*
* *“feeling trapped and powerless”*
* *“uncertainty before transformation”*

---

## Pipeline

```text
Images → Metadata → CLIP Embeddings → Retrieval → Evaluation → Analysis
```

### 1. Dataset

* Local image collection (~40 images)
* Focus: abstract, symbolic, psychological scenes
* Not included in the repository (see setup)

---

### 2. Metadata Generation

Each image is annotated using an LLM:

* description
* tags
* mood
* use cases
* composition

Script:

```bash
python src/build_metadata.py
```

Output:

```
data/metadata.json
```

---

### 3. Image Embeddings (CLIP)

Images are encoded using OpenCLIP:

* Model: `ViT-B-32`
* Pretrained on large-scale image-text data

Script:

```bash
python src/embed_images.py
```

Output:

```
data/embeddings/image_embeddings.npz
```

---

### 4. Retrieval

Given a query:

* encode text using CLIP
* compute cosine similarity
* rank images

Script:

```bash
python src/retrieve.py "a person reflecting on their identity"
```

Options:

```bash
--top_k 5
--sample_one
--json
```

---

### 5. Evaluation Dataset

Queries are generated in two steps:

1. Mechanical generation from metadata
2. Natural language rewriting using an LLM

Scripts:

```bash
python src/build_eval_queries.py
python src/rewrite_eval_queries.py
```

Output:

```
data/eval_queries.json
```

---

### 6. Evaluation

Metrics:

* Recall@1
* Recall@5
* Mean Reciprocal Rank (MRR)

Script:

```bash
python src/evaluate.py
```

Output:

```
data/evaluation_results.json
```

---

### 7. Analysis

Notebook includes:

* similarity distribution
* PCA / t-SNE projections

---

## Results

| Metric   | Score |
| -------- | ----: |
| Recall@1 |  0.82 |
| Recall@5 |  0.94 |
| MRR      |  0.87 |
| Queries  |   117 |

---

## Key Findings

### Strong retrieval performance

CLIP effectively captures semantic relationships between text and images, even for abstract concepts.

---

### Main limitation: dataset structure

Error analysis shows that failures are primarily due to:

* **low semantic separability**
* high overlap between images in embedding space
* similar visual and symbolic features

---

### Additional factor: query reformulation

Natural language rewriting increases ambiguity:

* broader queries
* multiple plausible matches
* reduced discriminative precision

---

### Conclusion

> The main limitation is not the model, but the dataset: many images occupy overlapping regions in embedding space, making fine-grained retrieval inherently difficult.

---

## Possible Improvements

* Multi-label evaluation (allow multiple correct images)
* Fine-tuning CLIP on the dataset

---

## Installation

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Set API key (for metadata + query rewriting)

```bash
export OPENAI_API_KEY="your_key_here"
```

---

## Project Structure

```text
clip-image-retrieval/
├── data/
│   ├── metadata.json
│   ├── eval_queries.json
│
├── src/
│   ├── build_metadata.py
│   ├── embed_images.py
│   ├── retrieve.py
│   ├── build_eval_queries.py
│   ├── rewrite_eval_queries.py
│   └── evaluate.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── requirements.txt
└── README.md
```

---

## Notes

* `data/images/` is not included (local dataset)
* embeddings are recomputed locally
* evaluation is reproducible from metadata + queries

---

## Positioning

This project demonstrates:

* multimodal representation learning
* retrieval system design
* evaluation methodology
* error analysis and interpretation

---

## Summary

This repository presents a **complete, reproducible pipeline** for semantic image retrieval using CLIP, along with a **quantitative and qualitative analysis** of its strengths and limitations.
