# DLS final project: Yandex Maps relevance

Рабочая задача: предсказывать релевантность организации запросу на Яндекс.Картах.

Классы:

- `0.0` - нерелевантно
- `0.1` - частично релевантно
- `1.0` - релевантно

Основная метрика из задания - `accuracy`.

## Quick Start For Review

1. Create an environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Download the task data from the original Yandex Disk link and put files here:

```text
data/data_for_train.jsonl
data/data_for_eval.jsonl
```

The raw data files are intentionally ignored by git because `data_for_train.jsonl` is larger than the normal GitHub file limit.

3. Create a private config:

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill:

```text
ROUTERAI_API_KEY=...
SERPER_API_KEY=...
SEARCH_PROVIDER=serper
```

4. Prepare agent JSONL files:

```bash
python src/prepare_agent_cases.py --validation-cases 100
```

This writes:

```text
outputs/agent_fewshot_train_examples.jsonl
outputs/agent_validation_cases.jsonl
outputs/agent_eval_cases.jsonl
```

5. Check search only:

```bash
python src/agent_runner.py --check-search --search-provider serper --search-query "cafe terrace Moscow"
```

6. Run a small validation agent smoke test:

```bash
python src/agent_runner.py \
  --cases outputs/agent_validation_cases.jsonl \
  --out outputs/agent_validation_results_serper_10.jsonl \
  --limit 10 \
  --search-provider serper \
  --fewshot-path outputs/agent_fewshot_train_examples.jsonl \
  --fewshot-k 3 \
  --evaluate
```

7. Run agent on eval after validation is acceptable:

```bash
python src/agent_runner.py \
  --cases outputs/agent_eval_cases.jsonl \
  --out outputs/agent_eval_results_serper.jsonl \
  --search-provider serper \
  --fewshot-path outputs/agent_fewshot_train_examples.jsonl \
  --fewshot-k 3 \
  --submission-csv outputs/submission_agent.csv
```

## Baseline

Первый воспроизводимый бейзлайн - TF-IDF по полям запроса и организации + `LinearSVC`.

```bash
python src/baseline_tfidf.py --classifier ridge --out-dir artifacts/baseline_tfidf_ridge
```

Артефакты сохраняются в `artifacts/baseline_tfidf/`:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.csv`
- `validation_predictions.csv`
- `errors_sample.csv`

Для быстрого smoke-test запуска:

```bash
python src/baseline_tfidf.py --sample-size 5000 --out-dir artifacts/baseline_tfidf_smoke
```

Для генерации предсказаний на eval после локальной проверки:

```bash
python src/baseline_tfidf.py --classifier ridge --predict-eval
```

Важно: `relevance_new` в eval не используем для подбора промптов/параметров. Для анализа ошибок и калибровки используем только train/valid split из `data_for_train.jsonl`.

## Colab notebook

Основной ноутбук для GPU:

```text
notebooks/yandex_maps_relevance_colab.ipynb
```

В Colab надёжнее загрузить архив в `/content/`:

```text
/content/yandex_maps_data.zip
```

Ноутбук проверит SHA-256 и распакует данные сам. Альтернативно можно загрузить отдельные файлы в `/content/data/`:

```text
/content/data/data_for_train.jsonl
/content/data/data_for_eval.jsonl
```

Ноутбук содержит TF-IDF baseline, GPU fine-tuning transformer baseline, усиленный ensemble нескольких трансформеров и экспорт eval-предсказаний.

Усиленный режим пишет:

- `/content/outputs/submission_ensemble.csv`
- `/content/outputs/ensemble_valid_predictions.csv`
- `/content/outputs/strong_ensemble/model_scores.csv`
- `/content/outputs/strong_ensemble/ensemble_config.json`

Также в ноутбук добавлен agent layer:

- selection спорных validation-примеров в `agent_validation_cases.jsonl`;
- запись `/content/agent_runner.py`;
- cached LLM runner с опциональным поиском Serper/Tavily;
- сравнение agent accuracy с transformer accuracy на одном validation subset.

## Agent application

Опциональное локальное demo-приложение агента:

```text
app/agent_web_app.py
```

Запуск:

```bash
python app/agent_web_app.py --host 127.0.0.1 --port 7860
```

Открыть:

```text
http://127.0.0.1:7860
```

Это не обязательная часть сдачи. Основной агентский артефакт проекта - `src/agent_runner.py` и соответствующие ячейки в Colab.

Приложение принимает одну пару `запрос + карточка организации`, решает, нужен ли поиск, опционально ходит в Serper/Tavily и возвращает дискретную оценку `0.0`, `0.1` или `1.0` с объяснением.

## Project scoring checklist

По оригинальному заданию из Colab:

| Требование | Баллы | Что сделано |
| --- | ---: | --- |
| Бейзлайн без агента | 2 | TF-IDF, ruBERT, 2-model ensemble; best validation accuracy `0.7067` |
| Агент с возможностью обращаться к поисковой строке | 4 | `src/agent_runner.py`: OpenAI-compatible LLM, plan/final steps, Serper/Tavily search |
| Показать, помогает агент или нет | 4 | validation smoke tests на 30 и 100 спорных кейсах; local-only агент не дал стабильного прироста |
| Доп. идея | optional | lexical few-shot retrieval из train через `--fewshot-path` и `--fewshot-k` |

## Current results

Validation split: stratified random 80/20 from `data_for_train.jsonl`, seed `42`.

| Run | Classifier | Class weight | Accuracy |
| --- | --- | --- | --- |
| `artifacts/baseline_tfidf` | `LinearSVC` | none | `0.5872` |
| `artifacts/baseline_tfidf_balanced` | `LinearSVC` | balanced | `0.5810` |
| `artifacts/baseline_tfidf_ridge` | `RidgeClassifier` | none | `0.5969` |
| `artifacts/baseline_tfidf_ridge_balanced` | `RidgeClassifier` | balanced | `0.5794` |

Best local baseline so far: `RidgeClassifier`, accuracy `0.5969`.

Weak spot: class `0.1`; it is often confused with both `0.0` and `1.0`.

Colab transformer runs:

| Model | Accuracy | Notes |
| --- | --- | --- |
| `cointegrated/rubert-tiny2` | `0.6269` | Stronger than TF-IDF, weak recall for `0.1` |
| `ai-forever/ruBert-base` | `0.6913` | Best run so far; `0.1` recall improved to `0.2442` |
| `ai-forever/ruBert-base` + logit calibration | `0.6988` | Strong single-model baseline; validation bias `[0.0, -0.8, 0.0]` |
| `ai-forever/ruBert-base` + `DeepPavlov/rubert-base-cased` ensemble | `0.7067` | Current best; weights `[0.7, 0.3]`, bias `[0.0, -0.8, -0.5]` |

Optional next model to try in the ensemble: `xlm-roberta-base`.

Current submission candidate:

```text
/content/outputs/submission_ensemble.csv
```

Ensemble submission distribution: `0.0` - 280, `0.1` - 20, `1.0` - 270.

Manual Codex-agent smoke test on 30 transformer-uncertain validation cases:

| Model | Accuracy |
| --- | ---: |
| Raw transformer | `0.3333` |
| Calibrated transformer | `0.4333` |
| Codex-agent manual judge | `0.6333` |

Details: `reports/codex_agent_validation/summary_30.md`.

Manual Codex-agent test on the next 100 transformer-uncertain validation cases:

| Model | Accuracy |
| --- | ---: |
| Raw transformer | `0.3500` |
| Calibrated transformer | `0.4500` |
| Codex-agent manual judge | `0.4100` |

Details: `reports/codex_agent_validation/summary_100.md`.

## RouterAI configuration

The agent runner is OpenAI-compatible and now defaults to RouterAI:

```text
base_url: https://routerai.ru/api/v1
model: deepseek/deepseek-v4-pro
```

Create a private `.env` from the template and paste the real key there:

```powershell
Copy-Item .env.example .env
notepad .env
```

Minimum `.env` values:

```text
ROUTERAI_API_KEY=...
ROUTERAI_BASE_URL=https://routerai.ru/api/v1
AGENT_MODEL=deepseek/deepseek-v4-pro
SEARCH_PROVIDER=none
```

`.env` is ignored by git. Keep `SEARCH_PROVIDER=none` for the first validation run; enable Serper/Tavily only after the cheap smoke test looks useful.

## Next steps

1. Use `/content/outputs/submission_ensemble.csv` as the current best submission candidate.
2. Optional: try adding `xlm-roberta-base` to the ensemble if there is enough GPU time.
3. Keep the agent result in the report as a checked negative/limited result unless search-enabled or few-shot agent validation improves it.

Local/Colab runner:

```bash
python /content/agent_runner.py \
  --env-file /content/.env \
  --cases /content/outputs/agent_validation_cases.jsonl \
  --out /content/outputs/agent_validation_results.jsonl \
  --limit 50 \
  --search-provider none \
  --fewshot-path /content/outputs/agent_fewshot_train_examples.jsonl \
  --fewshot-k 3 \
  --evaluate
```

Agent run on eval, after provider/model choice:

```bash
python /content/agent_runner.py \
  --env-file /content/.env \
  --cases /content/outputs/agent_eval_cases.jsonl \
  --out /content/outputs/agent_eval_results.jsonl \
  --search-provider serper \
  --fewshot-path /content/outputs/agent_fewshot_train_examples.jsonl \
  --fewshot-k 3 \
  --submission-csv /content/outputs/submission_agent.csv
```
