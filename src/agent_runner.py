import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LABEL_VALUES = [0.0, 0.1, 1.0]
DEFAULT_AGENT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_BASE_URL = "https://routerai.ru/api/v1"


SYSTEM_PROMPT = """
Ты оцениваешь релевантность организаций на Яндекс.Картах широким пользовательским запросам.

Классы:
- 1.0: организация явно подходит запросу; услуга, категория или важное свойство подтверждены.
- 0.1: организация частично подходит: близкая категория, но нет нужного свойства; услуга смежная; релевантность слабая или неоднозначная.
- 0.0: организация не подходит запросу или нет достаточных оснований считать её подходящей.

Будь строгим. Если пользователь ищет конкретное свойство ("веранда", "живая музыка", "мероприятия", "бесплатное занятие"), не ставь 1.0 без прямого или очень сильного подтверждения.
Возвращай только JSON без markdown.
""".strip()


PLAN_PROMPT = """
Оцени карточку организации по локальным данным и реши, нужен ли внешний поиск.

Верни JSON строго такого вида:
{
  "local_label": 0.0,
  "local_confidence": 0.0,
  "needs_search": true,
  "search_queries": ["..."],
  "evidence": "краткие факты из локальных данных",
  "explanation": "краткое объяснение"
}

Правила:
- local_label должен быть одним из 0.0, 0.1, 1.0.
- local_confidence от 0 до 1.
- needs_search=true, если локальных данных не хватает для проверки важного свойства запроса.
- search_queries максимум 2, на русском, с названием/адресом организации и важным свойством из запроса.
""".strip()


FINAL_PROMPT = """
Прими финальное решение о релевантности организации запросу.

Верни JSON строго такого вида:
{
  "label": 0.0,
  "confidence": 0.0,
  "used_search": false,
  "evidence": "краткие факты",
  "explanation": "почему выбран этот класс"
}

label должен быть одним из 0.0, 0.1, 1.0.
confidence от 0 до 1.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cached LLM/search agent on relevance cases.")
    parser.add_argument("--cases", help="JSONL with validation/eval cases.")
    parser.add_argument("--out", help="JSONL output cache/results.")
    parser.add_argument("--env-file", default=".env", help="Optional dotenv-style file with API settings.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--search-provider",
        choices=["none", "serper", "tavily"],
        default=None,
    )
    parser.add_argument("--serper-api-key", default=None)
    parser.add_argument("--tavily-api-key", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all cases.")
    parser.add_argument("--max-search-queries", type=int, default=2)
    parser.add_argument("--max-search-results", type=int, default=4)
    parser.add_argument("--fewshot-path", default="", help="Optional JSONL with labeled examples for few-shot context.")
    parser.add_argument("--fewshot-k", type=int, default=0, help="Number of similar labeled examples to add.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--submission-csv", default="", help="Optional CSV path with permalink, Text, relevance.")
    parser.add_argument("--check-search", action="store_true", help="Test configured search provider and exit.")
    parser.add_argument("--search-query", default="cafe terrace Moscow", help="Query for --check-search.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without adding a dependency on python-dotenv."""
    if not path.exists():
        return

    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)


def configure_args(args: argparse.Namespace) -> argparse.Namespace:
    load_env_file(Path(args.env_file).expanduser())
    args.model = args.model or os.getenv("AGENT_MODEL") or DEFAULT_AGENT_MODEL
    args.base_url = (
        args.base_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("ROUTERAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    args.api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ROUTERAI_API_KEY")
    args.search_provider = str(args.search_provider or os.getenv("SEARCH_PROVIDER") or "none").lower()
    if args.search_provider not in {"none", "serper", "tavily"}:
        raise ValueError("SEARCH_PROVIDER must be one of: none, serper, tavily.")
    args.serper_api_key = args.serper_api_key or os.getenv("SERPER_API_KEY")
    args.tavily_api_key = args.tavily_api_key or os.getenv("TAVILY_API_KEY")
    return args


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Cannot parse {path} line {line_no}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def existing_case_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        case_id = row.get("case_id")
        if case_id is not None:
            ids.add(str(case_id))
    return ids


def ensure_case_ids(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for idx, case in enumerate(cases):
        item = dict(case)
        if not item.get("case_id"):
            permalink = item.get("permalink")
            if permalink is not None:
                item["case_id"] = f"case_{idx}_{permalink}"
            else:
                item["case_id"] = f"case_{idx}"
        normalized.append(item)
    return normalized


def http_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 90) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc


def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    base_url: str,
    api_key: Optional[str],
    temperature: float,
) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or ROUTERAI_API_KEY is required for LLM calls.")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    response = http_json(url, headers, payload)
    content = response["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Some OpenAI-compatible gateways ignore response_format.
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
    return {"parsed": parsed, "raw_response": response}


def case_to_text(case: Dict[str, Any]) -> str:
    fields = [
        ("Запрос", case.get("Text")),
        ("Название", case.get("name")),
        ("Рубрика", case.get("normalized_main_rubric_name_ru") or case.get("rubric")),
        ("Адрес", case.get("address")),
        ("Товары и услуги", case.get("prices_summarized")),
        ("Отзывы и описание", case.get("reviews_summarized")),
        ("Предсказание трансформера", case.get("transformer_predicted_relevance")),
        ("Уверенность трансформера", case.get("transformer_confidence")),
        ("Отрыв top-1 от top-2", case.get("transformer_margin")),
    ]
    return "\n".join(f"{name}: {'' if value is None else value}" for name, value in fields)


def label_from_case(case: Dict[str, Any]) -> Optional[float]:
    for key in ("true_relevance", "relevance", "relevance_new", "label"):
        label = normalize_label(case.get(key))
        if label is not None:
            return label
    return None


def token_set(case: Dict[str, Any]) -> set:
    text = " ".join(
        str(case.get(key) or "")
        for key in (
            "Text",
            "name",
            "normalized_main_rubric_name_ru",
            "rubric",
            "address",
            "prices_summarized",
            "reviews_summarized",
        )
    ).lower()
    return set(re.findall(r"[\wа-яё]+", text, flags=re.IGNORECASE))


def same_case(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    if left.get("case_id") and right.get("case_id") and str(left["case_id"]) == str(right["case_id"]):
        return True
    return (
        left.get("permalink") is not None
        and right.get("permalink") is not None
        and str(left.get("permalink")) == str(right.get("permalink"))
        and str(left.get("Text")) == str(right.get("Text"))
    )


def build_fewshot_index(examples: List[Dict[str, Any]]) -> List[tuple]:
    indexed = []
    for example in examples:
        label = label_from_case(example)
        if label is None:
            continue
        indexed.append((example, label, token_set(example)))
    return indexed


def select_fewshot_examples(case: Dict[str, Any], examples: List[Any], k: int) -> List[Dict[str, Any]]:
    if k <= 0 or not examples:
        return []
    case_tokens = token_set(case)
    scored = []
    for item in examples:
        if isinstance(item, tuple):
            example, label, example_tokens = item
        else:
            example = item
            label = label_from_case(example)
            example_tokens = token_set(example)
        if label is None or same_case(case, example):
            continue
        overlap = len(case_tokens & example_tokens)
        if overlap == 0:
            continue
        union = len(case_tokens | example_tokens) or 1
        score = overlap / union
        scored.append((score, overlap, example))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:k]]


def fewshot_to_text(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return "Few-shot примеры не используются."
    chunks = []
    for idx, example in enumerate(examples, 1):
        label = label_from_case(example)
        chunks.append(
            "\n".join(
                [
                    f"Пример {idx}:",
                    f"Запрос: {example.get('Text', '')}",
                    f"Название: {example.get('name', '')}",
                    f"Рубрика: {example.get('normalized_main_rubric_name_ru') or example.get('rubric') or ''}",
                    f"Адрес: {example.get('address', '')}",
                    f"Правильная метка: {label}",
                ]
            )
        )
    return "\n\n".join(chunks)


def normalize_label(value: Any) -> Optional[float]:
    try:
        label = float(value)
    except (TypeError, ValueError):
        return None
    return label if label in LABEL_VALUES else None


def search_serper(query: str, api_key: str, max_results: int) -> List[Dict[str, str]]:
    response = http_json(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        payload={"q": query, "hl": "ru", "gl": "ru", "num": max_results},
        timeout=45,
    )
    results = []
    for item in response.get("organic", [])[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )
    return results


def search_tavily(query: str, api_key: str, max_results: int) -> List[Dict[str, str]]:
    response = http_json(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload={
            "query": query,
            "topic": "general",
            "max_results": max_results,
        },
        timeout=45,
    )
    results = []
    for item in response.get("results", [])[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
        )
    return results


def run_search(
    queries: Iterable[str],
    provider: str,
    serper_api_key: Optional[str],
    tavily_api_key: Optional[str],
    max_results: int,
) -> List[Dict[str, Any]]:
    search_results = []
    if provider == "none":
        return search_results

    for query in queries:
        query = str(query).strip()
        if not query:
            continue
        if provider == "serper":
            if not serper_api_key:
                raise RuntimeError("SERPER_API_KEY is required for search_provider=serper.")
            results = search_serper(query, serper_api_key, max_results)
        elif provider == "tavily":
            if not tavily_api_key:
                raise RuntimeError("TAVILY_API_KEY is required for search_provider=tavily.")
            results = search_tavily(query, tavily_api_key, max_results)
        else:
            raise ValueError(f"Unknown search provider: {provider}")
        search_results.append({"query": query, "results": results})
    return search_results


def search_results_to_text(search_results: List[Dict[str, Any]]) -> str:
    if not search_results:
        return "Внешний поиск не использовался или не дал результатов."
    chunks = []
    for block in search_results:
        chunks.append(f"Запрос поиска: {block['query']}")
        for idx, item in enumerate(block.get("results", []), 1):
            chunks.append(
                f"{idx}. {item.get('title', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Snippet: {item.get('snippet', '')}"
            )
    return "\n\n".join(chunks)


def run_case(case: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    case_text = case_to_text(case)
    fewshot_examples = select_fewshot_examples(
        case,
        getattr(args, "fewshot_examples", []),
        getattr(args, "fewshot_k", 0),
    )
    fewshot_text = fewshot_to_text(fewshot_examples)
    plan_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PLAN_PROMPT + "\n\nПохожие размеченные примеры:\n" + fewshot_text + "\n\nКарточка:\n" + case_text,
        },
    ]
    plan = call_llm(
        plan_messages,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    plan_json = plan["parsed"]
    queries = plan_json.get("search_queries", []) or []
    if not isinstance(queries, list):
        queries = []
    queries = queries[: args.max_search_queries]

    search_results = []
    if plan_json.get("needs_search") and args.search_provider != "none":
        search_results = run_search(
            queries,
            provider=args.search_provider,
            serper_api_key=args.serper_api_key,
            tavily_api_key=args.tavily_api_key,
            max_results=args.max_search_results,
        )

    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                FINAL_PROMPT
                + "\n\nКарточка:\n"
                + case_text
                + "\n\nПохожие размеченные примеры:\n"
                + fewshot_text
                + "\n\nПредварительная оценка:\n"
                + json.dumps(plan_json, ensure_ascii=False, indent=2)
                + "\n\nРезультаты поиска:\n"
                + search_results_to_text(search_results)
            ),
        },
    ]
    final = call_llm(
        final_messages,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    final_json = final["parsed"]
    label = normalize_label(final_json.get("label"))

    return {
        "case_id": str(case.get("case_id")),
        "permalink": case.get("permalink"),
        "Text": case.get("Text"),
        "label": label,
        "true_relevance": case.get("true_relevance"),
        "transformer_predicted_relevance": case.get("transformer_predicted_relevance"),
        "transformer_confidence": case.get("transformer_confidence"),
        "transformer_margin": case.get("transformer_margin"),
        "used_search": bool(search_results),
        "search_provider": args.search_provider,
        "search_results": search_results,
        "fewshot_examples": [
            {
                "Text": item.get("Text"),
                "name": item.get("name"),
                "rubric": item.get("normalized_main_rubric_name_ru") or item.get("rubric"),
                "label": label_from_case(item),
            }
            for item in fewshot_examples
        ],
        "plan": plan_json,
        "final": final_json,
    }


def evaluate(cases: List[Dict[str, Any]], results_path: Path) -> None:
    results = read_jsonl(results_path)
    case_by_id = {str(case.get("case_id")): case for case in cases}
    joined = []
    for result in results:
        case = case_by_id.get(str(result.get("case_id")))
        if not case:
            continue
        y_true = normalize_label(case.get("true_relevance"))
        y_agent = normalize_label(result.get("label"))
        y_base = normalize_label(case.get("transformer_predicted_relevance"))
        if y_true is None or y_agent is None:
            continue
        joined.append((y_true, y_agent, y_base))

    if not joined:
        print("No completed labeled results to evaluate.")
        return

    total = len(joined)
    agent_acc = sum(y_true == y_agent for y_true, y_agent, _ in joined) / total
    baseline_joined = [(y_true, y_base) for y_true, _, y_base in joined if y_base is not None]
    print(f"Completed labeled cases: {total}")
    print(f"Agent accuracy on completed cases: {agent_acc:.4f}")
    if baseline_joined:
        base_acc = sum(y_true == y_base for y_true, y_base in baseline_joined) / len(baseline_joined)
        print(f"Baseline accuracy on comparable cases: {base_acc:.4f}")
    else:
        print("Baseline accuracy is not available: cases do not contain baseline predictions.")

    print("\nPer-class counts (true, agent):")
    counts: Dict[str, int] = {}
    for y_true, y_agent, _ in joined:
        key = f"{y_true}->{y_agent}"
        counts[key] = counts.get(key, 0) + 1
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")


def write_submission_csv(results_path: Path, submission_path: Path) -> None:
    results = read_jsonl(results_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["permalink", "Text", "relevance"])
        writer.writeheader()
        for row in results:
            label = normalize_label(row.get("label"))
            if label is None:
                continue
            writer.writerow(
                {
                    "permalink": row.get("permalink"),
                    "Text": row.get("Text"),
                    "relevance": label,
                }
            )
    print(f"Saved submission CSV: {submission_path}")


def main() -> None:
    args = configure_args(parse_args())
    if args.check_search:
        if args.search_provider == "none":
            raise SystemExit("Set SEARCH_PROVIDER=serper or SEARCH_PROVIDER=tavily for --check-search.")
        search_results = run_search(
            [args.search_query],
            provider=args.search_provider,
            serper_api_key=args.serper_api_key,
            tavily_api_key=args.tavily_api_key,
            max_results=args.max_search_results,
        )
        print(json.dumps(search_results, ensure_ascii=False, indent=2))
        return

    if not args.cases or not args.out:
        raise SystemExit("--cases and --out are required unless --check-search is used.")

    cases_path = Path(args.cases)
    out_path = Path(args.out)
    cases = ensure_case_ids(read_jsonl(cases_path))
    if args.limit:
        cases = cases[: args.limit]
    args.fewshot_examples = build_fewshot_index(read_jsonl(Path(args.fewshot_path))) if args.fewshot_path else []

    done_ids = set() if args.overwrite else existing_case_ids(out_path)
    print(f"Loaded cases: {len(cases)}")
    print(f"Already completed: {len(done_ids)}")

    for idx, case in enumerate(cases, 1):
        case_id = str(case.get("case_id"))
        if case_id in done_ids:
            continue
        print(f"[{idx}/{len(cases)}] case_id={case_id}")
        try:
            result = run_case(case, args)
        except Exception as exc:
            result = {
                "case_id": case_id,
                "label": None,
                "error": repr(exc),
            }
        append_jsonl(out_path, result)
        if args.sleep:
            time.sleep(args.sleep)

    if args.evaluate:
        evaluate(cases, out_path)

    if args.submission_csv:
        write_submission_csv(out_path, Path(args.submission_csv))


if __name__ == "__main__":
    main()
