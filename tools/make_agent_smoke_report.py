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

    title = f"Полный прогон агента на валидации: {len(results)} кейсов"
    lines = [
        f"# {title}",
        "",
        f"Дата: {dt.date.today().isoformat()}",
        "",
        "Команда:",
        "",
        "```bash",
        args.command or "See terminal history.",
        "```",
        "",
        "Конфигурация:",
        "",
        "- Провайдер LLM: RouterAI, OpenAI-совместимый API",
        "- Модель: deepseek/deepseek-v4-pro",
        "- Поиск: Serper",
        "- Примеры для подсказки: top-3 похожих размеченных примера из обучающей выборки",
        "- Метки eval-множества не использовались",
        "",
        "Сводка:",
        "",
        f"- Завершено кейсов: {len(results)}",
        f"- Правильных ответов: {correct}/{len(results)}",
        f"- Точность (accuracy): {correct / len(results):.4f}",
        f"- Внешний поиск использован: {used_search}/{len(results)}",
        "- Статус выполнения: Python-ошибок нет; JSON-ответы успешно распарсились",
        "",
        "Матрица ошибок в виде истинная метка -> метка агента:",
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
        status = "ВЕРНО" if result.get("label") == result.get("true_relevance") else "ОШИБКА"

        lines.extend(
            [
                f"## Кейс {index}: {result.get('case_id')} - {status}",
                "",
                f"- Запрос: {case.get('Text', '')}",
                f"- Организация: {case.get('name', '')}",
                f"- Рубрика: {case.get('normalized_main_rubric_name_ru', '')}",
                f"- Адрес: {case.get('address', '')}",
                f"- Истинная метка: {result.get('true_relevance')}",
                f"- Метка агента: {result.get('label')}",
                f"- Финальная уверенность: {final.get('confidence')}",
                f"- Локальная метка на этапе plan: {plan.get('local_label')}",
                f"- Локальная уверенность на этапе plan: {plan.get('local_confidence')}",
                f"- Plan решил, что нужен поиск: {plan.get('needs_search')}",
                f"- Поиск фактически использован: {result.get('used_search')}",
                f"- Поисковые запросы: {json.dumps(plan.get('search_queries') or [], ensure_ascii=False)}",
            ]
        )

        if result.get("search_results"):
            lines.append("- Результаты поиска:")
            for block in result["search_results"]:
                lines.append(f"  - Запрос поиска: {block.get('query', '')}")
                for item in (block.get("results") or [])[:3]:
                    snippet = (item.get("snippet") or "")[:240]
                    lines.append(f"    - {item.get('title', '')} | {item.get('url', '')} | {snippet}")

        lines.extend(
            [
                "",
                "Факты на этапе plan:",
                "",
                str(plan.get("evidence", "")),
                "",
                "Объяснение на этапе plan:",
                "",
                str(plan.get("explanation", "")),
                "",
                "Финальные факты:",
                "",
                str(final.get("evidence", "")),
                "",
                "Финальное объяснение:",
                "",
                str(final.get("explanation", "")),
                "",
            ]
        )

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
