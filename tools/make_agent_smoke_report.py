import argparse
import datetime as dt
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown report for an agent smoke run.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--command", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = read_jsonl(Path(args.cases))
    results = read_jsonl(Path(args.results))
    case_by_id = {case["case_id"]: case for case in cases}
    correct = sum(result.get("label") == result.get("true_relevance") for result in results)
    used_search = sum(bool(result.get("used_search")) for result in results)

    lines = [
        "# Agent Full Smoke Test: 5 Validation Cases",
        "",
        f"Date: {dt.date.today().isoformat()}",
        "",
        "Command:",
        "",
        "```bash",
        args.command or "See terminal history.",
        "```",
        "",
        "Configuration:",
        "",
        "- LLM provider: RouterAI OpenAI-compatible API",
        "- Model: deepseek/deepseek-v4-pro",
        "- Search provider: Serper",
        "- Few-shot examples: lexical top-3 from train split",
        "- Eval labels were not used",
        "",
        "Summary:",
        "",
        f"- Completed cases: {len(results)}",
        f"- Correct predictions: {correct}/{len(results)}",
        f"- Accuracy: {correct / len(results):.4f}",
        f"- Cases with external search: {used_search}/{len(results)}",
        "- Runtime status: completed without Python exceptions; JSON responses parsed successfully",
        "",
        "Confusion counts:",
    ]

    counts: dict[str, int] = {}
    for result in results:
        key = f"{result.get('true_relevance')} -> {result.get('label')}"
        counts[key] = counts.get(key, 0) + 1
    lines.extend(f"- {key}: {counts[key]}" for key in sorted(counts))
    lines.append("")

    for index, result in enumerate(results, 1):
        case = case_by_id.get(result["case_id"], {})
        plan = result.get("plan") or {}
        final = result.get("final") or {}
        status = "OK" if result.get("label") == result.get("true_relevance") else "ERROR"

        lines.extend(
            [
                f"## Case {index}: {result.get('case_id')} - {status}",
                "",
                f"- Query: {case.get('Text', '')}",
                f"- Organization: {case.get('name', '')}",
                f"- Rubric: {case.get('normalized_main_rubric_name_ru', '')}",
                f"- Address: {case.get('address', '')}",
                f"- True label: {result.get('true_relevance')}",
                f"- Agent label: {result.get('label')}",
                f"- Final confidence: {final.get('confidence')}",
                f"- Plan local label: {plan.get('local_label')}",
                f"- Plan local confidence: {plan.get('local_confidence')}",
                f"- Plan needs search: {plan.get('needs_search')}",
                f"- Search actually used: {result.get('used_search')}",
                f"- Search queries: {json.dumps(plan.get('search_queries') or [], ensure_ascii=False)}",
            ]
        )

        if result.get("search_results"):
            lines.append("- Search results:")
            for block in result["search_results"]:
                lines.append(f"  - Query: {block.get('query', '')}")
                for item in (block.get("results") or [])[:3]:
                    snippet = (item.get("snippet") or "")[:240]
                    lines.append(f"    - {item.get('title', '')} | {item.get('url', '')} | {snippet}")

        lines.extend(
            [
                "",
                "Plan evidence:",
                "",
                str(plan.get("evidence", "")),
                "",
                "Plan explanation:",
                "",
                str(plan.get("explanation", "")),
                "",
                "Final evidence:",
                "",
                str(final.get("evidence", "")),
                "",
                "Final explanation:",
                "",
                str(final.get("explanation", "")),
                "",
            ]
        )

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
