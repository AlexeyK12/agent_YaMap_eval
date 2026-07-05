import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC


LABEL_VALUES = [0.0, 0.1, 1.0]
LABEL_TO_ID = {0.0: 0, 0.1: 1, 1.0: 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_IDS = [LABEL_TO_ID[label] for label in LABEL_VALUES]
LABEL_NAMES = [str(label) for label in LABEL_VALUES]
TEXT_COLUMNS = [
    "Text",
    "name",
    "normalized_main_rubric_name_ru",
    "address",
    "prices_summarized",
    "reviews_summarized",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF baseline for Yandex Maps relevance classification."
    )
    parser.add_argument("--train-path", default="data/data_for_train.jsonl")
    parser.add_argument("--eval-path", default="data/data_for_eval.jsonl")
    parser.add_argument("--out-dir", default="artifacts/baseline_tfidf")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--max-word-features", type=int, default=160_000)
    parser.add_argument("--max-char-features", type=int, default=160_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Use balanced only after checking validation metrics.",
    )
    parser.add_argument(
        "--classifier",
        choices=["linear_svc", "ridge"],
        default="linear_svc",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Debug option: train on a stratified sample of this size.",
    )
    parser.add_argument(
        "--predict-eval",
        action="store_true",
        help="After validation, train on all train data and predict eval labels.",
    )
    parser.add_argument(
        "--score-eval",
        action="store_true",
        help="Score eval if labels are present. Do not use this for calibration.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    rows = []
    pending_variants = []
    pending_start_line = None
    last_error = None
    recovered_multiline = 0

    def try_parse(text: str):
        return json.loads(text)

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue

            if pending_variants:
                next_variants = []
                for pending in pending_variants:
                    for sep in ("\\n", "\n", ""):
                        candidate = pending + sep + line
                        try:
                            rows.append(try_parse(candidate))
                            pending_variants = []
                            pending_start_line = None
                            recovered_multiline += 1
                            break
                        except json.JSONDecodeError as exc:
                            last_error = exc
                            next_variants.append(candidate)
                    if not pending_variants:
                        break
                if pending_variants:
                    pending_variants = list(dict.fromkeys(next_variants))[:30]
                    too_long = line_no - pending_start_line > 200
                    too_large = max(len(item) for item in pending_variants) > 2_000_000
                    if too_long or too_large:
                        preview = pending_variants[0][:500]
                        raise ValueError(
                            f"Cannot recover JSON object in {path}, starting at line "
                            f"{pending_start_line}. Last error: {last_error}. "
                            f"Preview: {preview!r}. Re-upload the file: it is likely corrupted."
                        ) from last_error
                continue

            try:
                rows.append(try_parse(line))
            except json.JSONDecodeError as exc:
                pending_variants = [line]
                pending_start_line = line_no
                last_error = exc

    if pending_variants:
        preview = pending_variants[0][:500]
        raise ValueError(
            f"Unfinished JSON object in {path}, starting at line {pending_start_line}. "
            f"Last error: {last_error}. Preview: {preview!r}. "
            f"Re-upload the file: it is likely truncated or corrupted."
        ) from last_error

    if recovered_multiline:
        print(f"Recovered {recovered_multiline} multiline JSON records from {path}.")
    return pd.DataFrame(rows)


def build_model_text(row: pd.Series) -> str:
    parts = []

    def add(label: str, value, repeat: int = 1) -> None:
        if pd.isna(value):
            return
        value = str(value).strip()
        if not value:
            return
        parts.extend([f"{label}: {value}"] * repeat)

    # The query, rubric, and explicit service/price text carry the strongest signal.
    add("query", row.get("Text"), repeat=4)
    add("rubric", row.get("normalized_main_rubric_name_ru"), repeat=3)
    add("name", row.get("name"), repeat=2)
    add("prices", row.get("prices_summarized"), repeat=2)
    add("reviews", row.get("reviews_summarized"), repeat=1)
    add("address", row.get("address"), repeat=1)
    return "\n".join(parts)


def add_model_text(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in TEXT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["model_text"] = df.apply(build_model_text, axis=1)
    return df


def encode_labels(labels: pd.Series) -> pd.Series:
    encoded = labels.map(LABEL_TO_ID)
    if encoded.isna().any():
        unknown = sorted(labels.loc[encoded.isna()].unique())
        raise ValueError(f"Unknown labels: {unknown}")
    return encoded.astype(int)


def decode_labels(label_ids) -> list:
    return [ID_TO_LABEL[int(label_id)] for label_id in label_ids]


def make_pipeline(args: argparse.Namespace) -> Pipeline:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=0.98,
        max_features=args.max_word_features,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=args.min_df,
        max_features=args.max_char_features,
        sublinear_tf=True,
    )
    class_weight = None if args.class_weight == "none" else args.class_weight
    if args.classifier == "linear_svc":
        classifier = LinearSVC(
            C=args.c,
            class_weight=class_weight,
            dual="auto",
            max_iter=args.max_iter,
            random_state=args.seed,
        )
    elif args.classifier == "ridge":
        classifier = RidgeClassifier(
            alpha=args.alpha,
            class_weight=class_weight,
            random_state=args.seed,
        )
    else:
        raise ValueError(f"Unknown classifier: {args.classifier}")
    return Pipeline(
        [
            ("features", FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])),
            ("classifier", classifier),
        ]
    )


def maybe_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if not sample_size or sample_size >= len(df):
        return df

    _, sampled = train_test_split(
        df,
        test_size=sample_size,
        random_state=seed,
        stratify=encode_labels(df["relevance"]),
    )
    return sampled.reset_index(drop=True)


def save_validation_outputs(
    out_dir: Path,
    valid_df: pd.DataFrame,
    y_true_ids,
    y_pred_ids,
    args: argparse.Namespace,
) -> dict:
    accuracy = accuracy_score(y_true_ids, y_pred_ids)
    report = classification_report(
        y_true_ids,
        y_pred_ids,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_ids, y_pred_ids, labels=LABEL_IDS)

    metrics = {
        "accuracy": accuracy,
        "valid_rows": int(len(valid_df)),
        "labels": LABEL_VALUES,
        "class_distribution_valid": {
            str(ID_TO_LABEL[int(key)]): int(value)
            for key, value in pd.Series(y_true_ids).value_counts().sort_index().items()
        },
        "args": vars(args),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABEL_VALUES],
        columns=[f"pred_{label}" for label in LABEL_VALUES],
    )
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8")

    pred_df = valid_df.copy()
    pred_df["predicted_relevance"] = decode_labels(y_pred_ids)
    pred_df["is_correct"] = pred_df["relevance"] == pred_df["predicted_relevance"]

    columns = [
        "Text",
        "name",
        "normalized_main_rubric_name_ru",
        "address",
        "prices_summarized",
        "reviews_summarized",
        "relevance",
        "predicted_relevance",
        "is_correct",
    ]
    pred_df[columns].to_csv(out_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    pred_df.loc[~pred_df["is_correct"], columns].head(300).to_csv(
        out_dir / "errors_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Validation accuracy: {accuracy:.4f}")
    print(report)
    print(f"Saved validation artifacts to: {out_dir}")
    return metrics


def predict_eval(
    model: Pipeline,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    score_eval: bool,
) -> None:
    model.fit(train_df["model_text"], encode_labels(train_df["relevance"]))
    eval_pred = model.predict(eval_df["model_text"])

    pred_df = eval_df.copy()
    pred_df["predicted_relevance"] = decode_labels(eval_pred)
    pred_df.to_csv(out_dir / "eval_predictions.csv", index=False, encoding="utf-8-sig")

    if score_eval and "relevance_new" in pred_df.columns:
        y_true_ids = encode_labels(pred_df["relevance_new"])
        y_pred_ids = encode_labels(pred_df["predicted_relevance"])
        accuracy = accuracy_score(y_true_ids, y_pred_ids)
        report = classification_report(
            y_true_ids,
            y_pred_ids,
            labels=LABEL_IDS,
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
        (out_dir / "eval_score.txt").write_text(
            f"Eval accuracy: {accuracy:.4f}\n\n{report}",
            encoding="utf-8",
        )
        print(f"Eval accuracy: {accuracy:.4f}")

    print(f"Saved eval predictions to: {out_dir / 'eval_predictions.csv'}")


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_path)
    eval_path = Path(args.eval_path)
    out_dir = Path(args.out_dir)

    data = load_jsonl(train_path)
    data = maybe_sample(data, args.sample_size, args.seed)
    data = add_model_text(data)

    train_df, valid_df = train_test_split(
        data,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=encode_labels(data["relevance"]),
    )

    model = make_pipeline(args)
    model.fit(train_df["model_text"], encode_labels(train_df["relevance"]))
    valid_pred = model.predict(valid_df["model_text"])
    save_validation_outputs(out_dir, valid_df, encode_labels(valid_df["relevance"]), valid_pred, args)

    if args.predict_eval:
        eval_df = add_model_text(load_jsonl(eval_path))
        predict_eval(model, data, eval_df, out_dir, args.score_eval)


if __name__ == "__main__":
    main()
