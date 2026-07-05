import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_VALUES = {0.0, 0.1, 1.0}
TEXT_FIELDS = [
    "Text",
    "address",
    "name",
    "normalized_main_rubric_name_ru",
    "permalink",
    "prices_summarized",
    "reviews_summarized",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare JSONL files for the LLM/search agent.")
    parser.add_argument("--train-path", default="data/data_for_train.jsonl")
    parser.add_argument("--eval-path", default="data/data_for_eval.jsonl")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--validation-cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Cannot parse {path} line {line_no}: {exc}") from exc
    return pd.DataFrame(rows)


def check_labels(df: pd.DataFrame, column: str) -> None:
    labels = {float(value) for value in df[column].dropna().unique()}
    unknown = sorted(labels - LABEL_VALUES)
    if unknown:
        raise ValueError(f"Unknown labels in {column}: {unknown}")


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def make_case(row: pd.Series, case_id: str, include_label: bool) -> Dict[str, Any]:
    item = {field: json_safe(row.get(field)) for field in TEXT_FIELDS}
    item["case_id"] = case_id
    if include_label:
        item["true_relevance"] = float(row["relevance"])
    return item


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sample_validation(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(df):
        return df.reset_index(drop=True)
    sampled, _ = train_test_split(
        df,
        train_size=limit,
        random_state=seed,
        stratify=df["relevance"],
    )
    return sampled.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    train_df = load_jsonl(Path(args.train_path))
    eval_df = load_jsonl(Path(args.eval_path))
    check_labels(train_df, "relevance")

    train_part, valid_part = train_test_split(
        train_df,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=train_df["relevance"],
    )
    train_part = train_part.reset_index(drop=True)
    valid_part = sample_validation(valid_part, args.validation_cases, args.seed)

    fewshot_rows = (
        make_case(row, f"train_{idx}_{json_safe(row.get('permalink'))}", include_label=True)
        for idx, row in train_part.iterrows()
    )
    validation_rows = (
        make_case(row, f"valid_{idx}_{json_safe(row.get('permalink'))}", include_label=True)
        for idx, row in valid_part.iterrows()
    )
    eval_rows = (
        make_case(row, f"eval_{idx}_{json_safe(row.get('permalink'))}", include_label=False)
        for idx, row in eval_df.reset_index(drop=True).iterrows()
    )

    fewshot_count = write_jsonl(out_dir / "agent_fewshot_train_examples.jsonl", fewshot_rows)
    validation_count = write_jsonl(out_dir / "agent_validation_cases.jsonl", validation_rows)
    eval_count = write_jsonl(out_dir / "agent_eval_cases.jsonl", eval_rows)

    print(f"Saved few-shot train examples: {fewshot_count}")
    print(f"Saved validation cases: {validation_count}")
    print(f"Saved eval cases without labels: {eval_count}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
