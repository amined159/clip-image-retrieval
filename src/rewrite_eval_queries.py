import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "eval_queries_draft.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval_queries.json"

MODEL_NAME = "gpt-5.4-mini"
SLEEP_SECONDS = 0.2

SYSTEM_PROMPT = """
You rewrite retrieval queries into natural language.

Your job:
- Rewrite the input query into a short, natural, human-like search query.
- Preserve the original meaning.
- Do not add new concepts, objects, emotions, or symbolism.
- Do not mention filenames, image IDs, tags, metadata, or implementation details.
- Do not make the query overly poetic, overly specific, or too vague.
- Keep it concise: ideally 4 to 12 words.
- Return only valid JSON matching the schema.
""".strip()

SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_query": {"type": "string"}
    },
    "required": ["rewritten_query"],
    "additionalProperties": False
}


def load_draft_queries() -> List[Dict[str, Any]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Draft eval file not found: {INPUT_PATH}")
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def rewrite_query(client: OpenAI, query: str) -> str:
    user_prompt = f"""
Rewrite this retrieval query into natural language while preserving its meaning.

Original query:
{query}
""".strip()

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "rewritten_eval_query",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )

    data = json.loads(response.output_text)
    rewritten = data["rewritten_query"].strip()

    if not rewritten:
        raise ValueError("Empty rewritten query returned by model.")

    return rewritten


def deduplicate_queries(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []

    for item in items:
        key = (
            item["query"].strip().lower(),
            tuple(item["relevant_image_ids"]),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    draft_items = load_draft_queries()

    rewritten_items: List[Dict[str, Any]] = []

    for i, item in enumerate(draft_items, start=1):
        original_query = item["query"]
        relevant_ids = item["relevant_image_ids"]

        print(f"[{i}/{len(draft_items)}] rewriting: {original_query}")
        rewritten_query = rewrite_query(client, original_query)

        rewritten_items.append({
            "query": rewritten_query,
            "relevant_image_ids": relevant_ids,
            "source_query": original_query,
        })

        time.sleep(SLEEP_SECONDS)

    # Deduplicate on rewritten query + relevant ids
    rewritten_items = deduplicate_queries(rewritten_items)

    OUTPUT_PATH.write_text(
        json.dumps(rewritten_items, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nSaved rewritten evaluation queries to: {OUTPUT_PATH}")
    print(f"Total queries: {len(rewritten_items)}")


if __name__ == "__main__":
    main()