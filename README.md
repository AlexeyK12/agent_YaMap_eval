# Финальный проект DLS: релевантность организаций на Яндекс.Картах

## Итог проекта

В репозитории две основные части:

1. **Сильный бейзлайн без агента:** ансамбль дообученных русскоязычных трансформерных моделей.
   Лучшая точность на валидации: **0.7067**.
   Основной файл для отправки: `notebooks/submission_ensemble.csv`.

2. **LLM-агент с поиском:** OpenAI-совместимая LLM через RouterAI + поиск Serper + несколько похожих размеченных примеров из обучающей выборки.
   Полный прогон агента на валидации: **20 кейсов, 10 правильных, точность 0.5000, поиск использован в 7/20 кейсов, ошибок выполнения/разбора JSON: 0**.
   Подробный отчёт: `reports/agent_full_validation_20.md`.

Главный вывод: агент реализован и технически работает, включая внешний поиск, но **не побил** сильный трансформерный бейзлайн на валидации. Поэтому итоговая позиция проекта: построен сильный бейзлайн, реализован агент с возможностью поиска, и показано, что в этой задаче агент не дал стабильного прироста качества.

Рабочая задача: предсказывать релевантность организации запросу на Яндекс.Картах.

Классы:

- `0.0` - нерелевантно
- `0.1` - частично релевантно
- `1.0` - релевантно

Основная метрика из задания - точность (`accuracy`).

## Быстрый Запуск Для Проверки

1. Установить зависимости:

```bash
python -m pip install -r requirements.txt
```

2. Скачать данные по ссылке из задания и положить файлы сюда:

```text
data/data_for_train.jsonl
data/data_for_eval.jsonl
```

Сырые данные специально игнорируются git, потому что `data_for_train.jsonl` больше обычного лимита GitHub.

3. Создать приватный конфиг:

```powershell
Copy-Item .env.example .env
notepad .env
```

Заполнить:

```text
ROUTERAI_API_KEY=...
SERPER_API_KEY=...
SEARCH_PROVIDER=serper
```

4. Подготовить JSONL-файлы для агента:

```bash
python src/prepare_agent_cases.py --validation-cases 100
```

Команда создаст:

```text
outputs/agent_fewshot_train_examples.jsonl
outputs/agent_validation_cases.jsonl
outputs/agent_eval_cases.jsonl
```

5. Проверить только поисковый инструмент:

```bash
python src/agent_runner.py --check-search --search-provider serper --search-query "кафе с верандой Москва"
```

6. Запустить короткую проверку агента на валидации:

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

7. Запустить агента на проверочном eval-множестве только после приемлемой проверки на валидации:

```bash
python src/agent_runner.py \
  --cases outputs/agent_eval_cases.jsonl \
  --out outputs/agent_eval_results_serper.jsonl \
  --search-provider serper \
  --fewshot-path outputs/agent_fewshot_train_examples.jsonl \
  --fewshot-k 3 \
  --submission-csv outputs/submission_agent.csv
```

## Бейзлайн

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

Для быстрой проверочной тренировки:

```bash
python src/baseline_tfidf.py --sample-size 5000 --out-dir artifacts/baseline_tfidf_smoke
```

Для генерации предсказаний на проверочном eval-множестве после локальной проверки:

```bash
python src/baseline_tfidf.py --classifier ridge --predict-eval
```

Важно: `relevance_new` в eval-файле не используем для подбора промптов/параметров. Для анализа ошибок и калибровки используем только разбиение обучающей выборки на train/valid из `data_for_train.jsonl`.

## Colab-ноутбук

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

Ноутбук содержит TF-IDF бейзлайн, дообучение трансформерного бейзлайна на GPU, усиленный ансамбль нескольких трансформерных моделей и экспорт предсказаний для проверочного eval-множества.

Усиленный режим пишет:

- `/content/outputs/submission_ensemble.csv`
- `/content/outputs/ensemble_valid_predictions.csv`
- `/content/outputs/strong_ensemble/model_scores.csv`
- `/content/outputs/strong_ensemble/ensemble_config.json`

Также в ноутбук добавлен слой агента:

- отбор спорных валидационных примеров в `agent_validation_cases.jsonl`;
- запись `/content/agent_runner.py`;
- кешируемый скрипт LLM-агента с опциональным поиском Serper/Tavily;
- сравнение точности агента с точностью трансформерного бейзлайна на одной валидационной подвыборке.

## Приложение Агента

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

## Чеклист По Баллам

По оригинальному заданию из Colab:

| Требование | Баллы | Что сделано |
| --- | ---: | --- |
| Бейзлайн без агента | 2 | TF-IDF, ruBERT, ансамбль из 2 моделей; лучшая точность на валидации `0.7067` |
| Агент с возможностью обращаться к поисковой строке | 4 | `src/agent_runner.py`: OpenAI-совместимая LLM, этапы планирования и финального решения, поиск Serper/Tavily |
| Показать, помогает агент или нет | 4 | проверки на 30 и 100 спорных валидационных кейсах; агент только по локальным полям не дал стабильного прироста |
| Доп. идея | опционально | лексический подбор похожих размеченных примеров из обучающей выборки через `--fewshot-path` и `--fewshot-k` |

## Текущие Результаты

Валидационное разбиение: стратифицированное случайное разбиение 80/20 из `data_for_train.jsonl`, seed `42`.

| Запуск | Классификатор | Веса классов | Точность |
| --- | --- | --- | --- |
| `artifacts/baseline_tfidf` | `LinearSVC` | none | `0.5872` |
| `artifacts/baseline_tfidf_balanced` | `LinearSVC` | balanced | `0.5810` |
| `artifacts/baseline_tfidf_ridge` | `RidgeClassifier` | none | `0.5969` |
| `artifacts/baseline_tfidf_ridge_balanced` | `RidgeClassifier` | balanced | `0.5794` |

Лучший локальный бейзлайн: `RidgeClassifier`, точность `0.5969`.

Слабое место: класс `0.1`; он часто путается и с `0.0`, и с `1.0`.

Запуски трансформерных моделей в Colab:

| Модель | Точность | Комментарий |
| --- | --- | --- |
| `cointegrated/rubert-tiny2` | `0.6269` | Лучше TF-IDF, но слабая полнота для `0.1` |
| `ai-forever/ruBert-base` | `0.6913` | Сильный одиночный ruBERT; полнота для `0.1` выросла до `0.2442` |
| `ai-forever/ruBert-base` + калибровка логитов | `0.6988` | Сильный одиночный бейзлайн; смещение классов на валидации `[0.0, -0.8, 0.0]` |
| `ai-forever/ruBert-base` + `DeepPavlov/rubert-base-cased` в ансамбле | `0.7067` | Текущий лучший результат; веса `[0.7, 0.3]`, смещение `[0.0, -0.8, -0.5]` |

Опциональная следующая модель для ансамбля: `xlm-roberta-base`.

Текущий основной файл для отправки:

```text
/content/outputs/submission_ensemble.csv
```

Распределение предсказаний в файле ансамбля: `0.0` - 280, `0.1` - 20, `1.0` - 270.

Ручная проверка Codex-agent на 30 спорных валидационных кейсах трансформерного бейзлайна:

| Модель | Точность |
| --- | ---: |
| Некалиброванный трансформер | `0.3333` |
| Калиброванный трансформер | `0.4333` |
| Ручная оценка Codex-agent | `0.6333` |

Подробности: `reports/codex_agent_validation/summary_30.md`.

Ручная проверка Codex-agent на следующих 100 спорных валидационных кейсах:

| Модель | Точность |
| --- | ---: |
| Некалиброванный трансформер | `0.3500` |
| Калиброванный трансформер | `0.4500` |
| Ручная оценка Codex-agent | `0.4100` |

Подробности: `reports/codex_agent_validation/summary_100.md`.

## Конфигурация RouterAI

Скрипт агента использует OpenAI-совместимый API и по умолчанию настроен на RouterAI:

```text
base_url: https://routerai.ru/api/v1
model: deepseek/deepseek-v4-pro
```

Создать приватный `.env` из шаблона и вставить реальные ключи:

```powershell
Copy-Item .env.example .env
notepad .env
```

Минимальные значения `.env`:

```text
ROUTERAI_API_KEY=...
ROUTERAI_BASE_URL=https://routerai.ru/api/v1
AGENT_MODEL=deepseek/deepseek-v4-pro
SEARCH_PROVIDER=none
```

`.env` игнорируется git. Для полноценного агента с поиском используется `SEARCH_PROVIDER=serper`; для дешёвой проверки без поиска можно временно поставить `SEARCH_PROVIDER=none`.

## Дальше

1. Использовать `notebooks/submission_ensemble.csv` как основной файл для отправки.
2. Агент оставить как реализованную LLM-часть проекта с поиском и как отрицательный результат по качеству.
3. Опционально пробовать `xlm-roberta-base` в ансамбле, если будет дополнительное GPU-время.

Запуск агента на валидации:

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

Запуск агента на eval-множестве после проверки на валидации:

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
