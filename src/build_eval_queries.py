import json
import re
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval_queries_draft.json"


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_description(desc: str) -> str:
    desc = normalize_text(desc)
    if desc.endswith("."):
        desc = desc[:-1]
    return desc


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def make_tag_query(tags: List[str]) -> str | None:
    tags = [t.strip() for t in tags if t.strip()]
    if len(tags) >= 2:
        return f"{tags[0]} and {tags[1]}"
    if len(tags) == 1:
        return tags[0]
    return None


def make_use_case_query(use_cases: List[str]) -> str | None:
    use_cases = [u.strip() for u in use_cases if u.strip()]
    if len(use_cases) >= 2:
        return f"{use_cases[0]} and {use_cases[1]}"
    if len(use_cases) == 1:
        return use_cases[0]
    return None


def make_description_query(description: str) -> str | None:
    description = clean_description(description)
    if not description:
        return None

    prefixes = [
        "an image of",
        "a scene of",
        "a visual of",
    ]

    lowered = description.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            description = description[len(prefix):].strip()
            break

    return description


def build_queries_for_image(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_id = item["id"]
    description = item.get("description", "")
    tags = item.get("tags", [])
    use_cases = item.get("use_cases", [])

    candidate_queries = []

    q1 = make_tag_query(tags)
    if q1:
        candidate_queries.append(q1)

    q2 = make_use_case_query(use_cases)
    if q2:
        candidate_queries.append(q2)

    q3 = make_description_query(description)
    if q3:
        candidate_queries.append(q3)

    candidate_queries = unique_preserve_order(candidate_queries)

    return [
        {
            "query": q,
            "relevant_image_ids": [image_id]
        }
        for q in candidate_queries
    ]


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    eval_queries = []
    for item in metadata:
        eval_queries.extend(build_queries_for_image(item))

    OUTPUT_PATH.write_text(
        json.dumps(eval_queries, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Saved {len(eval_queries)} evaluation queries to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()