# Финальный проект DLS: релевантность организаций на Яндекс.Картах

Репозиторий содержит решение финального проекта по оценке релевантности организаций из Яндекс.Карт широким пользовательским запросам. 

## Резюме

- Лучший результат на валидации: **точность 0.7067**.
- Лучшая модель: ансамбль `ai-forever/ruBert-base` и `DeepPavlov/rubert-base-cased`.
- Итоговые предсказания для `data_for_eval.jsonl`: `notebooks/submission_ensemble.csv`.
- В `notebooks/submission_ensemble.csv` 570 строк и три колонки: `permalink`, `Text`, `relevance`.
- Агент с внешним поиском реализован в `src/agent_runner.py`.
- Агент проверен на 20 валидационных примерах: **10/20 правильных, точность 0.5000**.
- Вывод по агенту: технически работает и использует поиск, но на проверенной валидации не улучшает качество относительно сильного трансформерного бейзлайна.

## Постановка задачи

Нужно предсказать релевантность организации широкому пользовательскому запросу.

Классы:

- `0.0` - нерелевантно;
- `0.1` - частично релевантно;
- `1.0` - релевантно.

Основная метрика из задания: `accuracy`.

Для обучения и настройки использовался только `data_for_train.jsonl`. Колонка `relevance_new` из `data_for_eval.jsonl` не использовалась для подбора моделей, промптов, калибровки или анализа ошибок.

## Данные и признаки

Вход для моделей строился из полей карточки организации:

- `Text` - пользовательский запрос;
- `name` - название организации;
- `normalized_main_rubric_name_ru` - основная рубрика;
- `address` - адрес;
- `prices_summarized` - краткое описание товаров и услуг;
- `reviews_summarized` - краткое описание отзывов.

Для локального запуска исходные данные должны лежать здесь:

```text
data/data_for_train.jsonl
data/data_for_eval.jsonl
```

## Эксперименты

### Классический бейзлайн

Первый бейзлайн построен на TF-IDF признаках и линейных классификаторах.

| Модель | Точность на валидации |
| --- | ---: |
| `LinearSVC` | `0.5872` |
| `LinearSVC` с весами классов | `0.5810` |
| `RidgeClassifier` | `0.5969` |
| `RidgeClassifier` с весами классов | `0.5794` |

Лучший классический бейзлайн: `RidgeClassifier`, точность `0.5969`.

### Трансформерные модели

Основное улучшение качества дало дообучение русскоязычных трансформеров на видеокарте в ноутбуке `notebooks/yandex_maps_relevance_colab.ipynb`.

| Подход | Точность на валидации |
| --- | ---: |
| `cointegrated/rubert-tiny2` | `0.6269` |
| `ai-forever/ruBert-base` | `0.6913` |
| `ai-forever/ruBert-base` + калибровка логитов | `0.6988` |
| Ансамбль `ai-forever/ruBert-base` + `DeepPavlov/rubert-base-cased` | `0.7067` |

Итоговая конфигурация ансамбля:

```text
models: ai-forever/ruBert-base, DeepPavlov/rubert-base-cased
weights: [0.7, 0.3]
logit_bias: [0.0, -0.8, -0.5]
```

Распределение предсказанных классов в `notebooks/submission_ensemble.csv`:

```text
0.0: 280
0.1: 20
1.0: 270
```

## Агентная часть

По заданию был реализован агент, который не обязан сразу выдавать ответ, а может использовать внешний поиск.

Реализация: `src/agent_runner.py`.

Компоненты агента:

- языковая модель через RouterAI;
- поиск через Serper;
- похожие размеченные примеры из обучающей выборки;
- кеширование результатов;
- строгий JSON-ответ с одним из классов `0.0`, `0.1`, `1.0`.

Проверенная конфигурация:

```text
model: deepseek/deepseek-v4-pro
search_provider: serper
fewshot_k: 3
```

Результат полного прогона на 20 валидационных примерах:

```text
Правильно: 10/20
Точность: 0.5000
Поиск использован: 7/20
Ошибок выполнения: 0
Ошибок разбора JSON: 0
```

Подробный отчёт по примерам: `reports/agent_full_validation_20.md`.

Агентная часть закрывает требование проекта про возможность обращения к поиску. По качеству на проверенной валидации агент уступил трансформерному ансамблю, поэтому итоговые предсказания для `data_for_eval.jsonl` сделаны ансамблем.

## Состав репозитория

```text
src/baseline_tfidf.py              классический TF-IDF бейзлайн
src/prepare_agent_cases.py         подготовка JSONL-файлов для агента
src/agent_runner.py                агент с RouterAI, Serper/Tavily и кешем

app/agent_web_app.py               опциональный локальный интерфейс к агенту
app/README.md                      запуск локального интерфейса

notebooks/yandex_maps_relevance_colab.ipynb
notebooks/submission_ensemble.csv  итоговые предсказания для data_for_eval.jsonl
notebooks/submission_transformer.csv

reports/agent_full_validation_20.md подробный прогон агента
reports/agent_full_smoke_5.md      короткая техническая проверка агента

.env.example                       шаблон переменных окружения
requirements.txt                   зависимости
```

В репозитории нет приватных ключей API и сырых данных. `.env`, `data/`, `outputs/` и большие артефакты обучения исключены из git.

## Воспроизведение

Установка зависимостей:

```bash
python -m pip install -r requirements.txt
```

Запуск классического бейзлайна:

```bash
python src/baseline_tfidf.py --classifier ridge --out-dir artifacts/baseline_tfidf_ridge
```

Подготовка файлов для агента:

```bash
python src/prepare_agent_cases.py --validation-cases 100
```

Проверка поиска Serper:

```bash
python src/agent_runner.py --check-search --search-provider serper --search-query "кафе с верандой Москва"
```

Запуск агента на валидационных примерах:

```bash
python src/agent_runner.py \
  --cases outputs/agent_validation_cases.jsonl \
  --out outputs/agent_validation_results_serper.jsonl \
  --limit 20 \
  --search-provider serper \
  --fewshot-path outputs/agent_fewshot_train_examples.jsonl \
  --fewshot-k 3 \
  --evaluate \
  --overwrite
```

Запуск локального интерфейса к агенту:

```bash
python app/agent_web_app.py --host 127.0.0.1 --port 7860
```

Для запуска агента нужен `.env` с ключами:

```text
ROUTERAI_API_KEY=...
ROUTERAI_BASE_URL=https://routerai.ru/api/v1
AGENT_MODEL=deepseek/deepseek-v4-pro
SEARCH_PROVIDER=serper
SERPER_API_KEY=...
```

`.env` не добавляется в git.

## Вывод

Сильнейшее решение в проекте - трансформерный ансамбль с точностью `0.7067` на валидации. Агент с поиском реализован и проверен, но в текущей конфигурации не дал прироста качества. Это важный результат проекта: для данной задачи локально дообученный классификатор оказался надёжнее, чем более дорогой агентный подход с поиском.
