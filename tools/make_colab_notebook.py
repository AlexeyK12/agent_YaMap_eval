import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "yandex_maps_relevance_colab.ipynb"
AGENT_RUNNER_PATH = ROOT / "src" / "agent_runner.py"
AGENT_RUNNER_CODE = AGENT_RUNNER_PATH.read_text(encoding="utf-8")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
# DLS final project: Yandex Maps relevance

Ноутбук для Colab/GPU. Задача: предсказывать релевантность пары `запрос - организация`.

Классы:

- `0.0` - нерелевантно
- `0.1` - частично релевантно
- `1.0` - релевантно

Основная метрика из задания - `accuracy`.

Важно: `relevance_new` в eval не используем для подбора параметров, промптов или анализа ошибок. Для настройки используем только train/valid split из `data_for_train.jsonl`.
"""
    ),
    md(
        """
## 0. Подготовка

1. Включи GPU: `Runtime -> Change runtime type -> T4 GPU` или лучше.
2. Надёжный вариант: загрузи готовый `yandex_maps_data.zip` в `/content/`. Ноутбук проверит SHA-256 и сам распакует его в `/content/data/`.
3. Альтернативно можно загрузить отдельные файлы:
   - `/content/data/data_for_train.jsonl`
   - `/content/data/data_for_eval.jsonl`
4. Запускай ячейки сверху вниз.

Если файлы лежат на Google Drive, можно поменять `DATA_DIR` в конфиге ниже.
"""
    ),
    code(
        """
!pip -q install -U "transformers>=4.44" "datasets>=2.20" "accelerate>=0.33" "evaluate>=0.4" "sentencepiece"
"""
    ),
    code(
        """
from pathlib import Path
import hashlib
import json
import os
import random
import zipfile
import warnings

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = Path("/content/data")
DATA_ZIP_PATH = Path("/content/yandex_maps_data.zip")
EXPECTED_DATA_ZIP_SHA256 = "19aec81812909d2616fa9f5aca448003a833f4a99b437242710f5b2d05a892e1"

if DATA_ZIP_PATH.exists():
    actual_sha256 = hashlib.sha256(DATA_ZIP_PATH.read_bytes()).hexdigest()
    if actual_sha256 != EXPECTED_DATA_ZIP_SHA256:
        raise ValueError(
            f"Поврежден ZIP: SHA256={actual_sha256}, ожидается {EXPECTED_DATA_ZIP_SHA256}. "
            "Загрузите yandex_maps_data.zip заново."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA_ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)
    print("Extracted verified data archive to", DATA_DIR)

TRAIN_PATH = DATA_DIR / "data_for_train.jsonl"
EVAL_PATH = DATA_DIR / "data_for_eval.jsonl"
OUTPUT_DIR = Path("/content/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_SIZE = 0.2

LABEL_VALUES = [0.0, 0.1, 1.0]
LABEL_TO_ID = {0.0: 0, 0.1: 1, 1.0: 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
LABEL_IDS = [LABEL_TO_ID[x] for x in LABEL_VALUES]
LABEL_NAMES = [str(x) for x in LABEL_VALUES]

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Train path:", TRAIN_PATH)
print("Eval path:", EVAL_PATH)
"""
    ),
    md(
        """
## 1. Загрузка данных и общий препроцессинг

Функция `build_model_text` собирает один текст из запроса и всех полей организации. Один и тот же текст потом используется для TF-IDF и трансформера.
"""
    ),
    code(
        """
def load_jsonl(path: Path, skip_bad_tail: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл: {path}. Загрузите данные в /content/data/ или поменяйте DATA_DIR."
        )
    rows = []
    pending_variants = []
    pending_start_line = None
    last_error = None
    recovered_multiline = 0

    def try_parse(text: str):
        return json.loads(text)

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip()
            if not line:
                continue

            if pending_variants:
                next_variants = []
                for pending in pending_variants:
                    # Try common transfer/copy corruptions:
                    # 1) escaped "\\n" became a physical line break inside a JSON string
                    # 2) object was split on regular JSON whitespace
                    # 3) object was wrapped without a separator
                    for sep in ("\\\\n", " ", ""):
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
                            f"Preview: {preview!r}. Перезагрузите файл: он, вероятно, поврежден."
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
        message = (
            f"Unfinished JSON object in {path}, starting at line {pending_start_line}. "
            f"Last error: {last_error}. Preview: {preview!r}. "
            f"Перезагрузите файл: он, вероятно, обрезан или поврежден."
        )
        if skip_bad_tail:
            print("WARNING:", message)
            print("Skipped the unfinished tail record and kept", len(rows), "complete rows.")
        else:
            raise ValueError(message) from last_error

    if recovered_multiline:
        print(f"Recovered {recovered_multiline} multiline JSON records from {path}.")
    return pd.DataFrame(rows)


def clean_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def build_model_text(row: pd.Series) -> str:
    query = clean_value(row.get("Text"))
    name = clean_value(row.get("name"))
    rubric = clean_value(row.get("normalized_main_rubric_name_ru"))
    address = clean_value(row.get("address"))
    prices = clean_value(row.get("prices_summarized"))
    reviews = clean_value(row.get("reviews_summarized"))

    return (
        f"Запрос пользователя: {query}\\n"
        f"Название организации: {name}\\n"
        f"Рубрика: {rubric}\\n"
        f"Адрес: {address}\\n"
        f"Товары и услуги: {prices}\\n"
        f"Отзывы и описание: {reviews}"
    )


def encode_labels(labels: pd.Series) -> pd.Series:
    encoded = labels.map(LABEL_TO_ID)
    if encoded.isna().any():
        unknown = sorted(labels.loc[encoded.isna()].unique())
        raise ValueError(f"Unknown labels: {unknown}")
    return encoded.astype(int)


def decode_label_ids(label_ids) -> list:
    return [ID_TO_LABEL[int(label_id)] for label_id in label_ids]


train_data = load_jsonl(TRAIN_PATH)
eval_data = load_jsonl(EVAL_PATH)

EXPECTED_TRAIN_ROWS = 34_094
EXPECTED_EVAL_ROWS = 570

if len(train_data) != EXPECTED_TRAIN_ROWS or len(eval_data) != EXPECTED_EVAL_ROWS:
    raise ValueError(
        "Загружены неполные или поврежденные данные. "
        f"Получено train={len(train_data)}, eval={len(eval_data)}; "
        f"ожидается train={EXPECTED_TRAIN_ROWS}, eval={EXPECTED_EVAL_ROWS}. "
        "Перезагрузите исходные JSONL-файлы перед обучением."
    )

train_data["text"] = train_data.apply(build_model_text, axis=1)
train_data["label_id"] = encode_labels(train_data["relevance"])
eval_data["text"] = eval_data.apply(build_model_text, axis=1)

print("Train shape:", train_data.shape)
print("Eval shape:", eval_data.shape)
display(train_data.head(3))
display(train_data["relevance"].value_counts().sort_index())
"""
    ),
    code(
        """
train_df, valid_df = train_test_split(
    train_data,
    test_size=VALID_SIZE,
    random_state=SEED,
    stratify=train_data["label_id"],
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print("Train split:", train_df.shape)
print("Valid split:", valid_df.shape)
display(valid_df["relevance"].value_counts().sort_index())

valid_df[["Text", "name", "normalized_main_rubric_name_ru", "relevance"]].to_csv(
    OUTPUT_DIR / "valid_split_preview.csv",
    index=False,
    encoding="utf-8-sig",
)
"""
    ),
    md(
        """
## 2. Быстрый TF-IDF baseline

Это CPU-бейзлайн и контрольная точка. На локальном запуске лучшая конфигурация дала около `0.5969` accuracy на таком же split: TF-IDF word/char + `RidgeClassifier`.
"""
    ),
    code(
        """
def make_tfidf_pipeline() -> Pipeline:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=160_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\\b\\w+\\b",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=160_000,
        sublinear_tf=True,
    )
    return Pipeline(
        [
            ("features", FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])),
            ("classifier", RidgeClassifier(alpha=1.0, random_state=SEED)),
        ]
    )


tfidf_model = make_tfidf_pipeline()
tfidf_model.fit(train_df["text"], train_df["label_id"])
tfidf_valid_pred_ids = tfidf_model.predict(valid_df["text"])
tfidf_valid_pred_labels = decode_label_ids(tfidf_valid_pred_ids)

tfidf_accuracy = accuracy_score(valid_df["label_id"], tfidf_valid_pred_ids)
print(f"TF-IDF validation accuracy: {tfidf_accuracy:.4f}")
print(
    classification_report(
        valid_df["label_id"],
        tfidf_valid_pred_ids,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
)

tfidf_cm = pd.DataFrame(
    confusion_matrix(valid_df["label_id"], tfidf_valid_pred_ids, labels=LABEL_IDS),
    index=[f"true_{x}" for x in LABEL_VALUES],
    columns=[f"pred_{x}" for x in LABEL_VALUES],
)
display(tfidf_cm)

tfidf_valid_out = valid_df.copy()
tfidf_valid_out["predicted_relevance"] = tfidf_valid_pred_labels
tfidf_valid_out["is_correct"] = (
    tfidf_valid_out["relevance"] == tfidf_valid_out["predicted_relevance"]
)
tfidf_valid_out.to_csv(
    OUTPUT_DIR / "tfidf_valid_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)
tfidf_valid_out.loc[~tfidf_valid_out["is_correct"]].head(300).to_csv(
    OUTPUT_DIR / "tfidf_errors_sample.csv",
    index=False,
    encoding="utf-8-sig",
)
"""
    ),
    md(
        """
## 3. GPU transformer baseline

По умолчанию стоит лёгкая модель `cointegrated/rubert-tiny2`, чтобы быстро проверить пайплайн. Для финального эксперимента на GPU можно заменить на более сильную модель, например:

- `DeepPavlov/rubert-base-cased`
- `ai-forever/ruBert-base`
- `xlm-roberta-base`

Начни с tiny-модели, убедись, что всё работает, затем меняй `MODEL_NAME`.
"""
    ),
    code(
        """
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import inspect

MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LENGTH = 384
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
USE_CLASS_WEIGHTS = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def make_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


hf_train = make_hf_dataset(train_df)
hf_valid = make_hf_dataset(valid_df)

tokenized_train = hf_train.map(tokenize_batch, batched=True, remove_columns=["text"])
tokenized_valid = hf_valid.map(tokenize_batch, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def trainer_tokenizer_kwargs():
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        return {"processing_class": tokenizer}
    if "tokenizer" in trainer_params:
        return {"tokenizer": tokenizer}
    return {}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred_ids = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred_ids),
        "macro_f1": f1_score(labels, pred_ids, average="macro"),
    }


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None,
        )
        return (loss, outputs) if return_outputs else loss


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_VALUES),
    id2label={i: str(ID_TO_LABEL[i]) for i in LABEL_IDS},
    label2id={str(label): label_id for label, label_id in LABEL_TO_ID.items()},
)

class_weights = None
if USE_CLASS_WEIGHTS:
    counts = np.bincount(train_df["label_id"], minlength=len(LABEL_VALUES))
    weights = counts.sum() / (len(LABEL_VALUES) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print("Class weights:", class_weights)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "transformer_checkpoints"),
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=SEED,
)

trainer_common_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    **trainer_tokenizer_kwargs(),
)

if USE_CLASS_WEIGHTS:
    trainer = WeightedTrainer(**trainer_common_kwargs, class_weights=class_weights)
else:
    trainer = Trainer(**trainer_common_kwargs)

trainer.train()
trainer.save_model(str(OUTPUT_DIR / "best_transformer_model"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "best_transformer_model"))
"""
    ),
    code(
        """
transformer_valid_output = trainer.predict(tokenized_valid)
transformer_valid_pred_ids = np.argmax(transformer_valid_output.predictions, axis=-1)
transformer_valid_pred_labels = decode_label_ids(transformer_valid_pred_ids)

transformer_accuracy = accuracy_score(valid_df["label_id"], transformer_valid_pred_ids)
print(f"Transformer validation accuracy: {transformer_accuracy:.4f}")
print(
    classification_report(
        valid_df["label_id"],
        transformer_valid_pred_ids,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
)

transformer_cm = pd.DataFrame(
    confusion_matrix(valid_df["label_id"], transformer_valid_pred_ids, labels=LABEL_IDS),
    index=[f"true_{x}" for x in LABEL_VALUES],
    columns=[f"pred_{x}" for x in LABEL_VALUES],
)
display(transformer_cm)

transformer_valid_out = valid_df.copy()
transformer_valid_out["predicted_relevance"] = transformer_valid_pred_labels
transformer_valid_out["is_correct"] = (
    transformer_valid_out["relevance"] == transformer_valid_out["predicted_relevance"]
)
transformer_valid_out.to_csv(
    OUTPUT_DIR / "transformer_valid_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)
transformer_valid_out.loc[~transformer_valid_out["is_correct"]].head(300).to_csv(
    OUTPUT_DIR / "transformer_errors_sample.csv",
    index=False,
    encoding="utf-8-sig",
)
"""
    ),
    md(
        """
## 3.1. Калибровка логитов на validation

Если модель почти не выбирает класс `0.1`, можно подобрать небольшие bias-поправки к логитам классов на validation. Это не использует eval и не меняет веса модели.
"""
    ),
    code(
        """
def predict_with_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(logits + bias.reshape(1, -1), axis=-1)


def tune_logit_bias(logits: np.ndarray, labels: np.ndarray, step: float = 0.1, limit: float = 2.0):
    # Adding the same constant to all logits changes nothing, so fix bias for class 0.0 at zero.
    grid = np.round(np.arange(-limit, limit + 1e-9, step), 4)
    best = {
        "accuracy": accuracy_score(labels, np.argmax(logits, axis=-1)),
        "bias": np.zeros(logits.shape[1], dtype=float),
    }
    for bias_01 in grid:
        for bias_10 in grid:
            bias = np.array([0.0, bias_01, bias_10], dtype=float)
            pred_ids = predict_with_bias(logits, bias)
            acc = accuracy_score(labels, pred_ids)
            if acc > best["accuracy"]:
                best = {"accuracy": acc, "bias": bias}
    return best


BEST_LOGIT_BIAS_INFO = tune_logit_bias(
    transformer_valid_output.predictions,
    valid_df["label_id"].to_numpy(),
    step=0.1,
    limit=2.0,
)
BEST_LOGIT_BIAS = BEST_LOGIT_BIAS_INFO["bias"]

print("Raw transformer accuracy:", f"{transformer_accuracy:.4f}")
print("Best calibrated accuracy:", f"{BEST_LOGIT_BIAS_INFO['accuracy']:.4f}")
print("Best logit bias:", BEST_LOGIT_BIAS.tolist())

calibrated_valid_pred_ids = predict_with_bias(transformer_valid_output.predictions, BEST_LOGIT_BIAS)
calibrated_valid_pred_labels = decode_label_ids(calibrated_valid_pred_ids)

print(
    classification_report(
        valid_df["label_id"],
        calibrated_valid_pred_ids,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
)

calibrated_cm = pd.DataFrame(
    confusion_matrix(valid_df["label_id"], calibrated_valid_pred_ids, labels=LABEL_IDS),
    index=[f"true_{x}" for x in LABEL_VALUES],
    columns=[f"pred_{x}" for x in LABEL_VALUES],
)
display(calibrated_cm)

calibrated_valid_out = valid_df.copy()
calibrated_valid_out["predicted_relevance"] = calibrated_valid_pred_labels
calibrated_valid_out["is_correct"] = (
    calibrated_valid_out["relevance"] == calibrated_valid_out["predicted_relevance"]
)
calibrated_valid_out.to_csv(
    OUTPUT_DIR / "transformer_calibrated_valid_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

(OUTPUT_DIR / "transformer_logit_bias.json").write_text(
    json.dumps(
        {
            "raw_accuracy": float(transformer_accuracy),
            "calibrated_accuracy": float(BEST_LOGIT_BIAS_INFO["accuracy"]),
            "bias": BEST_LOGIT_BIAS.tolist(),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
"""
    ),
    md(
        """
## 3.2. Усиленный режим: ensemble трансформеров

Этот блок обучает несколько сильных моделей на одном train/valid split, сохраняет validation/eval logits, подбирает веса ансамбля и class-bias на validation, затем пишет `submission_ensemble.csv`.

Можно пропустить разделы `2`, `3` и `3.1`, если цель - только ensemble. Перед этим должны быть выполнены только подготовка, загрузка данных и split из раздела `1`.

По умолчанию включены две русские BERT-модели. `xlm-roberta-base` добавлен как опциональный третий участник: он медленнее, поэтому сначала лучше прогнать первые две.
"""
    ),
    code(
        """
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import inspect

RUN_STRONG_ENSEMBLE = True

STRONG_MODEL_CONFIGS = [
    {
        "name": "ai-forever/ruBert-base",
        "slug": "ai_forever_rubert_base",
        "enabled": True,
        "max_length": 384,
        "learning_rate": 2e-5,
        "epochs": 3,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "weight_decay": 0.01,
        "force_retrain": False,
    },
    {
        "name": "DeepPavlov/rubert-base-cased",
        "slug": "deeppavlov_rubert_base_cased",
        "enabled": True,
        "max_length": 384,
        "learning_rate": 2e-5,
        "epochs": 3,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "weight_decay": 0.01,
        "force_retrain": False,
    },
    {
        "name": "xlm-roberta-base",
        "slug": "xlm_roberta_base",
        "enabled": False,
        "max_length": 384,
        "learning_rate": 1.5e-5,
        "epochs": 3,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "weight_decay": 0.01,
        "force_retrain": False,
    },
]

ENSEMBLE_DIR = OUTPUT_DIR / "strong_ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred_ids = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred_ids),
        "macro_f1": f1_score(labels, pred_ids, average="macro"),
    }


def predict_with_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(logits + bias.reshape(1, -1), axis=-1)


def tune_logit_bias(logits: np.ndarray, labels: np.ndarray, step: float = 0.1, limit: float = 2.0):
    # Adding the same constant to all logits changes nothing, so fix bias for class 0.0 at zero.
    grid = np.round(np.arange(-limit, limit + 1e-9, step), 4)
    best = {
        "accuracy": accuracy_score(labels, np.argmax(logits, axis=-1)),
        "bias": np.zeros(logits.shape[1], dtype=float),
    }
    for bias_01 in grid:
        for bias_10 in grid:
            bias = np.array([0.0, bias_01, bias_10], dtype=float)
            pred_ids = predict_with_bias(logits, bias)
            acc = accuracy_score(labels, pred_ids)
            if acc > best["accuracy"]:
                best = {"accuracy": acc, "bias": bias}
    return best


def training_args_compat(**kwargs):
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def trainer_processing_kwargs(tokenizer_obj):
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        return {"processing_class": tokenizer_obj}
    if "tokenizer" in trainer_params:
        return {"tokenizer": tokenizer_obj}
    return {}


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def make_weight_grid(n_models: int, step: float = 0.1):
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)

    def rec(prefix, remaining, slots_left):
        if slots_left == 1:
            yield prefix + [round(remaining, 4)]
            return
        for value in values:
            if value <= remaining + 1e-9:
                yield from rec(prefix + [float(value)], round(remaining - float(value), 4), slots_left - 1)

    yield from rec([], 1.0, n_models)


def combine_probabilities(logits_list, weights):
    probs = np.stack([softmax_np(logits) for logits in logits_list], axis=0)
    return np.tensordot(np.array(weights, dtype=float), probs, axes=(0, 0))


def tune_ensemble(valid_logits_list, labels, weight_step=0.1, bias_step=0.1, bias_limit=2.0):
    best = None
    for weights in make_weight_grid(len(valid_logits_list), step=weight_step):
        probs = combine_probabilities(valid_logits_list, weights)
        scores = np.log(np.clip(probs, 1e-12, 1.0))
        raw_pred = np.argmax(scores, axis=-1)
        raw_acc = accuracy_score(labels, raw_pred)
        bias_info = tune_logit_bias(scores, labels, step=bias_step, limit=bias_limit)
        item = {
            "weights": weights,
            "raw_accuracy": float(raw_acc),
            "accuracy": float(bias_info["accuracy"]),
            "bias": bias_info["bias"],
        }
        if best is None or item["accuracy"] > best["accuracy"]:
            best = item
    return best


def fit_or_load_ensemble_member(cfg):
    slug = cfg["slug"]
    model_dir = ENSEMBLE_DIR / slug
    valid_logits_path = ENSEMBLE_DIR / f"{slug}_valid_logits.npy"
    eval_logits_path = ENSEMBLE_DIR / f"{slug}_eval_logits.npy"
    score_path = ENSEMBLE_DIR / f"{slug}_score.json"

    if (
        valid_logits_path.exists()
        and eval_logits_path.exists()
        and score_path.exists()
        and not cfg.get("force_retrain", False)
    ):
        print(f"[load] {cfg['name']} logits")
        valid_logits = np.load(valid_logits_path)
        eval_logits = np.load(eval_logits_path)
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score["valid_logits"] = valid_logits
        score["eval_logits"] = eval_logits
        return score

    print(f"[train] {cfg['name']}")
    tokenizer_local = AutoTokenizer.from_pretrained(cfg["name"])

    def tokenize_local(batch):
        return tokenizer_local(
            batch["text"],
            truncation=True,
            max_length=cfg.get("max_length", 384),
        )

    hf_train_local = Dataset.from_pandas(
        train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )
    hf_valid_local = Dataset.from_pandas(
        valid_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )
    hf_eval_local = Dataset.from_pandas(eval_data[["text"]], preserve_index=False)

    tokenized_train_local = hf_train_local.map(tokenize_local, batched=True, remove_columns=["text"])
    tokenized_valid_local = hf_valid_local.map(tokenize_local, batched=True, remove_columns=["text"])
    tokenized_eval_local = hf_eval_local.map(tokenize_local, batched=True, remove_columns=["text"])

    collator_local = DataCollatorWithPadding(tokenizer=tokenizer_local)
    model_local = AutoModelForSequenceClassification.from_pretrained(
        cfg["name"],
        num_labels=len(LABEL_VALUES),
        id2label={i: str(ID_TO_LABEL[i]) for i in LABEL_IDS},
        label2id={str(label): label_id for label, label_id in LABEL_TO_ID.items()},
    )

    args_local = training_args_compat(
        output_dir=str(ENSEMBLE_DIR / f"{slug}_checkpoints"),
        learning_rate=cfg.get("learning_rate", 2e-5),
        per_device_train_batch_size=cfg.get("train_batch_size", 8),
        per_device_eval_batch_size=cfg.get("eval_batch_size", 16),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=cfg.get("epochs", 3),
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.06),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 250),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 250),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        optim="adamw_torch",
    )

    trainer_local = Trainer(
        model=model_local,
        args=args_local,
        train_dataset=tokenized_train_local,
        eval_dataset=tokenized_valid_local,
        data_collator=collator_local,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        **trainer_processing_kwargs(tokenizer_local),
    )

    trainer_local.train()
    trainer_local.save_model(str(model_dir))
    tokenizer_local.save_pretrained(str(model_dir))

    valid_logits = trainer_local.predict(tokenized_valid_local).predictions
    eval_logits = trainer_local.predict(tokenized_eval_local).predictions

    np.save(valid_logits_path, valid_logits)
    np.save(eval_logits_path, eval_logits)

    raw_pred_ids = np.argmax(valid_logits, axis=-1)
    raw_acc = accuracy_score(valid_df["label_id"], raw_pred_ids)
    bias_info = tune_logit_bias(valid_logits, valid_df["label_id"].to_numpy(), step=0.1, limit=2.0)

    score = {
        "name": cfg["name"],
        "slug": slug,
        "raw_accuracy": float(raw_acc),
        "calibrated_accuracy": float(bias_info["accuracy"]),
        "bias": bias_info["bias"].tolist(),
        "valid_logits_path": str(valid_logits_path),
        "eval_logits_path": str(eval_logits_path),
        "model_dir": str(model_dir),
    }
    score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")

    del trainer_local, model_local, tokenizer_local
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    score["valid_logits"] = valid_logits
    score["eval_logits"] = eval_logits
    return score


ENSEMBLE_RECORDS = []

if RUN_STRONG_ENSEMBLE:
    for cfg in STRONG_MODEL_CONFIGS:
        if not cfg.get("enabled", True):
            print(f"[skip] {cfg['name']}")
            continue
        ENSEMBLE_RECORDS.append(fit_or_load_ensemble_member(cfg))

    model_score_rows = [
        {
            "name": item["name"],
            "slug": item["slug"],
            "raw_accuracy": item["raw_accuracy"],
            "calibrated_accuracy": item["calibrated_accuracy"],
            "bias": item["bias"],
        }
        for item in ENSEMBLE_RECORDS
    ]
    model_scores_df = pd.DataFrame(model_score_rows)
    model_scores_df.to_csv(ENSEMBLE_DIR / "model_scores.csv", index=False, encoding="utf-8-sig")
    display(model_scores_df)
else:
    print("RUN_STRONG_ENSEMBLE=False. Set it to True to train the ensemble.")
"""
    ),
    code(
        """
if RUN_STRONG_ENSEMBLE and ENSEMBLE_RECORDS:
    valid_logits_list = [item["valid_logits"] for item in ENSEMBLE_RECORDS]
    eval_logits_list = [item["eval_logits"] for item in ENSEMBLE_RECORDS]
    y_valid = valid_df["label_id"].to_numpy()

    BEST_ENSEMBLE_INFO = tune_ensemble(
        valid_logits_list,
        y_valid,
        weight_step=0.1,
        bias_step=0.1,
        bias_limit=2.0,
    )
    BEST_ENSEMBLE_WEIGHTS = np.array(BEST_ENSEMBLE_INFO["weights"], dtype=float)
    BEST_ENSEMBLE_BIAS = np.array(BEST_ENSEMBLE_INFO["bias"], dtype=float)

    ensemble_valid_probs = combine_probabilities(valid_logits_list, BEST_ENSEMBLE_WEIGHTS)
    ensemble_valid_scores = np.log(np.clip(ensemble_valid_probs, 1e-12, 1.0))
    ensemble_valid_pred_ids = predict_with_bias(ensemble_valid_scores, BEST_ENSEMBLE_BIAS)
    ensemble_valid_pred_labels = decode_label_ids(ensemble_valid_pred_ids)

    print("Best ensemble raw accuracy:", f"{BEST_ENSEMBLE_INFO['raw_accuracy']:.4f}")
    print("Best ensemble calibrated accuracy:", f"{BEST_ENSEMBLE_INFO['accuracy']:.4f}")
    print("Models:", [item["name"] for item in ENSEMBLE_RECORDS])
    print("Weights:", BEST_ENSEMBLE_WEIGHTS.tolist())
    print("Bias:", BEST_ENSEMBLE_BIAS.tolist())

    print(
        classification_report(
            valid_df["label_id"],
            ensemble_valid_pred_ids,
            labels=LABEL_IDS,
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    ensemble_cm = pd.DataFrame(
        confusion_matrix(valid_df["label_id"], ensemble_valid_pred_ids, labels=LABEL_IDS),
        index=[f"true_{x}" for x in LABEL_VALUES],
        columns=[f"pred_{x}" for x in LABEL_VALUES],
    )
    display(ensemble_cm)

    ensemble_valid_out = valid_df.copy()
    ensemble_valid_out["predicted_relevance"] = ensemble_valid_pred_labels
    ensemble_valid_out["is_correct"] = (
        ensemble_valid_out["relevance"] == ensemble_valid_out["predicted_relevance"]
    )
    ensemble_valid_out.to_csv(
        OUTPUT_DIR / "ensemble_valid_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ensemble_valid_out.loc[~ensemble_valid_out["is_correct"]].head(300).to_csv(
        OUTPUT_DIR / "ensemble_errors_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ensemble_eval_probs = combine_probabilities(eval_logits_list, BEST_ENSEMBLE_WEIGHTS)
    ensemble_eval_scores = np.log(np.clip(ensemble_eval_probs, 1e-12, 1.0))
    ensemble_eval_pred_ids = predict_with_bias(ensemble_eval_scores, BEST_ENSEMBLE_BIAS)
    ensemble_eval_pred_labels = decode_label_ids(ensemble_eval_pred_ids)

    ensemble_eval_out = eval_data.copy()
    ensemble_eval_out["predicted_relevance"] = ensemble_eval_pred_labels
    ensemble_eval_out.to_csv(
        OUTPUT_DIR / "ensemble_eval_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    submission_ensemble = pd.DataFrame(
        {
            "permalink": eval_data.get("permalink"),
            "Text": eval_data.get("Text"),
            "relevance": ensemble_eval_pred_labels,
        }
    )
    submission_ensemble.to_csv(
        OUTPUT_DIR / "submission_ensemble.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ensemble_config = {
        "models": [
            {
                "name": item["name"],
                "slug": item["slug"],
                "raw_accuracy": item["raw_accuracy"],
                "calibrated_accuracy": item["calibrated_accuracy"],
                "single_model_bias": item["bias"],
            }
            for item in ENSEMBLE_RECORDS
        ],
        "ensemble_raw_accuracy": float(BEST_ENSEMBLE_INFO["raw_accuracy"]),
        "ensemble_calibrated_accuracy": float(BEST_ENSEMBLE_INFO["accuracy"]),
        "weights": BEST_ENSEMBLE_WEIGHTS.tolist(),
        "bias": BEST_ENSEMBLE_BIAS.tolist(),
    }
    (ENSEMBLE_DIR / "ensemble_config.json").write_text(
        json.dumps(ensemble_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display(ensemble_eval_out["predicted_relevance"].value_counts().sort_index())
    print("Saved:", OUTPUT_DIR / "submission_ensemble.csv")
    print("Saved:", ENSEMBLE_DIR / "ensemble_config.json")
else:
    print("No ensemble predictions. Run the previous cell with RUN_STRONG_ENSEMBLE=True.")
"""
    ),
    md(
        """
## 3.3. Validation subset для LLM-агента

Агента запускаем не на всех примерах, а на спорных случаях: низкая уверенность, маленький margin между top-1/top-2, предсказание `0.1` или расхождение raw/calibrated предсказаний.

Файл `agent_validation_cases.jsonl` содержит true labels, потому что это validation subset из train. На eval этот механизм не используем для калибровки.
"""
    ),
    code(
        """
def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def json_safe(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


AGENT_CASE_LIMIT = 300
AGENT_CONFIDENCE_THRESHOLD = 0.72
AGENT_MARGIN_THRESHOLD = 0.18

if "ensemble_valid_scores" in globals() and "BEST_ENSEMBLE_BIAS" in globals():
    agent_selector_base = "strong_ensemble"
    raw_logits = ensemble_valid_scores
    calibration_bias = BEST_ENSEMBLE_BIAS
else:
    agent_selector_base = "single_transformer"
    raw_logits = transformer_valid_output.predictions
    calibration_bias = BEST_LOGIT_BIAS

raw_pred_ids = np.argmax(raw_logits, axis=-1)
calibrated_logits = raw_logits + calibration_bias.reshape(1, -1)
calibrated_probs = softmax_np(calibrated_logits)
calibrated_pred_ids = np.argmax(calibrated_logits, axis=-1)

sorted_probs = np.sort(calibrated_probs, axis=1)
confidence = sorted_probs[:, -1]
margin = sorted_probs[:, -1] - sorted_probs[:, -2]

agent_pool = valid_df.copy()
agent_pool["case_id"] = [f"valid_{i}_{row.permalink}" for i, row in agent_pool.iterrows()]
agent_pool["raw_transformer_predicted_relevance"] = decode_label_ids(raw_pred_ids)
agent_pool["transformer_predicted_relevance"] = decode_label_ids(calibrated_pred_ids)
agent_pool["transformer_confidence"] = confidence
agent_pool["transformer_margin"] = margin
agent_pool["raw_calibrated_disagree"] = raw_pred_ids != calibrated_pred_ids
agent_pool["agent_selector_base"] = agent_selector_base
agent_pool["is_uncertain"] = (
    (agent_pool["transformer_predicted_relevance"] == 0.1)
    | (agent_pool["transformer_confidence"] < AGENT_CONFIDENCE_THRESHOLD)
    | (agent_pool["transformer_margin"] < AGENT_MARGIN_THRESHOLD)
    | agent_pool["raw_calibrated_disagree"]
)

agent_cases_df = (
    agent_pool.loc[agent_pool["is_uncertain"]]
    .sort_values(["transformer_margin", "transformer_confidence"], ascending=[True, True])
    .head(AGENT_CASE_LIMIT)
    .reset_index(drop=True)
)

agent_fields = [
    "case_id",
    "Text",
    "name",
    "normalized_main_rubric_name_ru",
    "address",
    "prices_summarized",
    "reviews_summarized",
    "relevance",
    "raw_transformer_predicted_relevance",
    "transformer_predicted_relevance",
    "transformer_confidence",
    "transformer_margin",
    "raw_calibrated_disagree",
    "agent_selector_base",
]

agent_cases_path = OUTPUT_DIR / "agent_validation_cases.jsonl"
with agent_cases_path.open("w", encoding="utf-8") as f:
    for _, row in agent_cases_df.iterrows():
        item = {field: json_safe(row.get(field)) for field in agent_fields}
        item["true_relevance"] = item.pop("relevance")
        f.write(json.dumps(item, ensure_ascii=False) + "\\n")

print("Agent validation cases:", len(agent_cases_df))
print("Agent selector base:", agent_selector_base)
print("Saved:", agent_cases_path)
print("True distribution:")
display(agent_cases_df["relevance"].value_counts().sort_index())
print("Transformer prediction distribution on agent subset:")
display(agent_cases_df["transformer_predicted_relevance"].value_counts().sort_index())
display(
    agent_cases_df[
        [
            "Text",
            "name",
            "normalized_main_rubric_name_ru",
            "relevance",
            "transformer_predicted_relevance",
            "transformer_confidence",
            "transformer_margin",
        ]
    ].head(10)
)

fewshot_fields = [
    "Text",
    "name",
    "normalized_main_rubric_name_ru",
    "address",
    "prices_summarized",
    "reviews_summarized",
    "relevance",
    "permalink",
]

fewshot_path = OUTPUT_DIR / "agent_fewshot_train_examples.jsonl"
with fewshot_path.open("w", encoding="utf-8") as f:
    for _, row in train_data[fewshot_fields].iterrows():
        item = {field: json_safe(row.get(field)) for field in fewshot_fields}
        f.write(json.dumps(item, ensure_ascii=False) + "\\n")

agent_eval_fields = [
    "case_id",
    "Text",
    "name",
    "normalized_main_rubric_name_ru",
    "address",
    "prices_summarized",
    "reviews_summarized",
    "permalink",
]

agent_eval_df = eval_data.copy().reset_index(drop=True)
agent_eval_df["case_id"] = [f"eval_{i}_{row.permalink}" for i, row in agent_eval_df.iterrows()]
agent_eval_path = OUTPUT_DIR / "agent_eval_cases.jsonl"
with agent_eval_path.open("w", encoding="utf-8") as f:
    for _, row in agent_eval_df.iterrows():
        item = {field: json_safe(row.get(field)) for field in agent_eval_fields}
        f.write(json.dumps(item, ensure_ascii=False) + "\\n")

print("Few-shot train examples:", len(train_data), "Saved:", fewshot_path)
print("Agent eval cases without labels:", len(agent_eval_df), "Saved:", agent_eval_path)
"""
    ),
    md(
        """
## 3.4. LLM/search agent runner

Runner ниже работает с OpenAI-compatible API. Можно использовать OpenAI напрямую или совместимый gateway.

Минимальные env-переменные:

```python
os.environ["ROUTERAI_API_KEY"] = "..."
os.environ["ROUTERAI_BASE_URL"] = "https://routerai.ru/api/v1"
os.environ["AGENT_MODEL"] = "deepseek/deepseek-v4-pro"
```

Для OpenAI-compatible gateway можно добавить:

```python
Path("/content/.env").write_text(
    "ROUTERAI_API_KEY=...\\n"
    "ROUTERAI_BASE_URL=https://routerai.ru/api/v1\\n"
    "AGENT_MODEL=deepseek/deepseek-v4-pro\\n"
    "SEARCH_PROVIDER=none\\n",
    encoding="utf-8",
)
```

Поиск опционален:

- без поиска: `SEARCH_PROVIDER=none`;
- Serper: `SEARCH_PROVIDER=serper`, `SERPER_API_KEY=...`;
- Tavily: `SEARCH_PROVIDER=tavily`, `TAVILY_API_KEY=...`.
"""
    ),
    code(
        f"""
agent_runner_code = {json.dumps(AGENT_RUNNER_CODE, ensure_ascii=False)}
Path("/content/agent_runner.py").write_text(agent_runner_code, encoding="utf-8")
print("Wrote /content/agent_runner.py")
"""
    ),
    code(
        """
import subprocess

# Запуск выключен по умолчанию, чтобы случайно не потратить API-бюджет.
RUN_AGENT = False

# Пример настройки:
# Лучше загрузить /content/.env с ROUTERAI_API_KEY, чем вписывать ключ в notebook.
# os.environ["ROUTERAI_API_KEY"] = "..."
# os.environ["ROUTERAI_BASE_URL"] = "https://routerai.ru/api/v1"
# os.environ["AGENT_MODEL"] = "deepseek/deepseek-v4-pro"
# os.environ["SEARCH_PROVIDER"] = "none"

agent_results_path = OUTPUT_DIR / "agent_validation_results.jsonl"
agent_eval_results_path = OUTPUT_DIR / "agent_eval_results.jsonl"
agent_submission_path = OUTPUT_DIR / "submission_agent.csv"

if RUN_AGENT:
    cmd = [
        "python",
        "/content/agent_runner.py",
        "--env-file",
        "/content/.env",
        "--cases",
        str(OUTPUT_DIR / "agent_validation_cases.jsonl"),
        "--out",
        str(agent_results_path),
        "--limit",
        "50",
        "--search-provider",
        os.getenv("SEARCH_PROVIDER", "none"),
        "--fewshot-path",
        str(OUTPUT_DIR / "agent_fewshot_train_examples.jsonl"),
        "--fewshot-k",
        "3",
        "--evaluate",
    ]
    subprocess.run(cmd, check=True)
else:
    print("RUN_AGENT=False. Set RUN_AGENT=True after configuring API keys.")
    print("Cases:", OUTPUT_DIR / "agent_validation_cases.jsonl")
    print("Results will be saved to:", agent_results_path)

# Eval запуск выключен отдельно: он может стоить денег.
# Включать только после выбора дешевого провайдера/модели.
RUN_AGENT_ON_EVAL = False

if RUN_AGENT_ON_EVAL:
    cmd = [
        "python",
        "/content/agent_runner.py",
        "--env-file",
        "/content/.env",
        "--cases",
        str(OUTPUT_DIR / "agent_eval_cases.jsonl"),
        "--out",
        str(agent_eval_results_path),
        "--search-provider",
        os.getenv("SEARCH_PROVIDER", "none"),
        "--fewshot-path",
        str(OUTPUT_DIR / "agent_fewshot_train_examples.jsonl"),
        "--fewshot-k",
        "3",
        "--submission-csv",
        str(agent_submission_path),
    ]
    subprocess.run(cmd, check=True)
else:
    print("RUN_AGENT_ON_EVAL=False. Agent eval cases:", OUTPUT_DIR / "agent_eval_cases.jsonl")
    print("Agent submission will be saved to:", agent_submission_path)
"""
    ),
    md(
        """
## 4. Предсказания на eval

Эта секция сохраняет предсказания. Не используем `relevance_new` для выбора параметров.

Если усиленный ensemble выше отработал успешно, основной кандидат для отправки - `submission_ensemble.csv`. Одиночный `submission_transformer.csv` ниже остается запасным вариантом.

Для финального прогона можно поставить `TRAIN_FINAL_TRANSFORMER_ON_FULL_TRAIN = True`, чтобы переобучить трансформер на всём train после выбора настроек на validation.
"""
    ),
    code(
        """
# TF-IDF eval predictions: train on all train data.
tfidf_full = make_tfidf_pipeline()
tfidf_full.fit(train_data["text"], train_data["label_id"])
tfidf_eval_pred_ids = tfidf_full.predict(eval_data["text"])
tfidf_eval_pred_labels = decode_label_ids(tfidf_eval_pred_ids)

tfidf_eval_out = eval_data.copy()
tfidf_eval_out["predicted_relevance"] = tfidf_eval_pred_labels
tfidf_eval_out.to_csv(
    OUTPUT_DIR / "tfidf_eval_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

display(tfidf_eval_out["predicted_relevance"].value_counts().sort_index())
"""
    ),
    code(
        """
TRAIN_FINAL_TRANSFORMER_ON_FULL_TRAIN = False

prediction_trainer = trainer
prediction_tokenizer = tokenizer

if TRAIN_FINAL_TRANSFORMER_ON_FULL_TRAIN:
    full_hf_train = make_hf_dataset(train_data)
    tokenized_full_train = full_hf_train.map(tokenize_batch, batched=True, remove_columns=["text"])

    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_VALUES),
        id2label={i: str(ID_TO_LABEL[i]) for i in LABEL_IDS},
        label2id={str(label): label_id for label, label_id in LABEL_TO_ID.items()},
    )

    final_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "final_transformer_checkpoints"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    prediction_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=tokenized_full_train,
        data_collator=data_collator,
        **trainer_tokenizer_kwargs(),
    )
    prediction_trainer.train()
    prediction_trainer.save_model(str(OUTPUT_DIR / "final_transformer_model"))

eval_hf = Dataset.from_pandas(eval_data[["text"]], preserve_index=False)
tokenized_eval = eval_hf.map(tokenize_batch, batched=True, remove_columns=["text"])
transformer_eval_output = prediction_trainer.predict(tokenized_eval)
if "BEST_LOGIT_BIAS" in globals():
    transformer_eval_pred_ids = predict_with_bias(transformer_eval_output.predictions, BEST_LOGIT_BIAS)
else:
    transformer_eval_pred_ids = np.argmax(transformer_eval_output.predictions, axis=-1)
transformer_eval_pred_labels = decode_label_ids(transformer_eval_pred_ids)

transformer_eval_out = eval_data.copy()
transformer_eval_out["predicted_relevance"] = transformer_eval_pred_labels
transformer_eval_out.to_csv(
    OUTPUT_DIR / "transformer_eval_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

submission = pd.DataFrame(
    {
        "permalink": eval_data.get("permalink"),
        "Text": eval_data.get("Text"),
        "relevance": transformer_eval_pred_labels,
    }
)
submission.to_csv(
    OUTPUT_DIR / "submission_transformer.csv",
    index=False,
    encoding="utf-8-sig",
)

display(transformer_eval_out["predicted_relevance"].value_counts().sort_index())
print("Saved:", OUTPUT_DIR / "transformer_eval_predictions.csv")
print("Saved:", OUTPUT_DIR / "submission_transformer.csv")
"""
    ),
    md(
        """
## 5. Архив результатов

Для сдачи и отчёта обычно достаточно лёгкого архива `final_artifacts.zip`: он не включает гигабайтные веса моделей, но содержит submission, validation metrics/predictions и конфиг ансамбля.

Полный архив всего `/content/outputs` можно сделать отдельно, если нужно сохранить обученные чекпойнты.
"""
    ),
    code(
        """
import shutil

FINAL_ARTIFACTS_DIR = Path("/content/final_artifacts")
shutil.rmtree(FINAL_ARTIFACTS_DIR, ignore_errors=True)
FINAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

artifact_paths = [
    OUTPUT_DIR / "submission_ensemble.csv",
    OUTPUT_DIR / "ensemble_eval_predictions.csv",
    OUTPUT_DIR / "ensemble_valid_predictions.csv",
    OUTPUT_DIR / "ensemble_errors_sample.csv",
    OUTPUT_DIR / "submission_transformer.csv",
    OUTPUT_DIR / "transformer_eval_predictions.csv",
    OUTPUT_DIR / "transformer_calibrated_valid_predictions.csv",
    OUTPUT_DIR / "transformer_logit_bias.json",
    OUTPUT_DIR / "valid_split_preview.csv",
    OUTPUT_DIR / "strong_ensemble" / "model_scores.csv",
    OUTPUT_DIR / "strong_ensemble" / "ensemble_config.json",
]

for src in artifact_paths:
    if src.exists():
        rel = src.relative_to(OUTPUT_DIR)
        dst = FINAL_ARTIFACTS_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print("copied:", src)
    else:
        print("skip missing:", src)

shutil.make_archive("/content/final_artifacts", "zip", FINAL_ARTIFACTS_DIR)
print("Light archive:", "/content/final_artifacts.zip")

# Если нужно сохранить вообще всё, включая большие чекпойнты моделей, включи:
CREATE_FULL_OUTPUTS_ARCHIVE = False
if CREATE_FULL_OUTPUTS_ARCHIVE:
    shutil.make_archive("/content/yandex_maps_outputs", "zip", OUTPUT_DIR)
    print("Full archive:", "/content/yandex_maps_outputs.zip")
"""
    ),
    md(
        """
## 6. Что делать дальше

1. Сравнить TF-IDF и transformer на validation.
2. По `*_errors_sample.csv` понять, какие типы ошибок повторяются.
3. Если `submission_ensemble.csv` создан и validation accuracy выше одиночной модели, использовать его как основной submit.
4. Агентский слой оставить как отдельный эксперимент: запускать только на спорных случаях и сравнивать с transformer/ensemble на validation.
"""
    ),
]


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
        },
        "language_info": {
            "name": "python",
        },
        "accelerator": "GPU",
    },
    "cells": cells,
}


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
