"""Run the two assignment checks against the live API (usage is billed)."""

import asyncio
import json

from rag import UNKNOWN_ANSWER, get_rag_response
from settings import BASE_DIR


async def main():
    cases = [
        ("answerable", "Do I need programming experience?"),
        ("unanswerable", "Which university accredits the course certificate?"),
    ]
    results = []
    for name, query in cases:
        response = await get_rag_response(query)
        if name == "answerable":
            passed = response.text != UNKNOWN_ANSWER and "data/prerequisites.md" in response.references
        else:
            passed = response.text == UNKNOWN_ANSWER and response.references == []
        results.append({"case": name, "query": query, "passed": passed,
                        "response": response.model_dump()})
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
        print(response.model_dump_json(indent=2))
    (BASE_DIR / "rag_check_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not all(row["passed"] for row in results):
        raise SystemExit("One or more checks failed. Inspect rag_check_results.json.")
    print("Review the answerable text against the source as well: citations alone do not prove correctness.")


if __name__ == "__main__":
    asyncio.run(main())
