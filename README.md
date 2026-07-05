# Финальный проект DLS: релевантность организаций на Яндекс.Картах

Проект решает задачу классификации: нужно определить, насколько организация из Яндекс.Карт релевантна широкому пользовательскому запросу вроде «ресторан с верандой» или «романтичный джаз-бар».

Классы:

- `0.0` - нерелевантно;
- `0.1` - частично релевантно;
- `1.0` - релевантно.

Основная метрика из задания - точность (`accuracy`).

## Итог

В репозитории есть две основные части:

1. **Сильный бейзлайн без агента**: ансамбль двух дообученных русскоязычных моделей семейства BERT.
   Лучший результат на валидации: **0.7067**.
   Основной файл для отправки: `notebooks/submission_ensemble.csv`.

2. **Агент на языковой модели с внешним поиском**: модель вызывается через RouterAI, поиск выполняется через Serper, в промпт добавляются похожие размеченные примеры из обучающей выборки.
   Полный прогон агента на 20 валидационных примерах: **10/20 правильных, точность 0.5000, поиск использован в 7/20 примеров, ошибок выполнения и разбора JSON: 0**.
   Подробный отчёт: `reports/agent_full_validation_20.md`.

Главный вывод: агент технически работает и умеет обращаться к поиску, но на проверенных валидационных примерах не побил сильный трансформерный бейзлайн. Поэтому финальный файл для отправки основан на ансамбле трансформеров, а агентная часть показывает выполнение требования проекта и отрицательный результат по качеству.

## Результаты

| Подход | Точность на валидации | Комментарий |
| --- | ---: | --- |
| TF-IDF + `RidgeClassifier` | `0.5969` | Быстрый классический бейзлайн |
| `cointegrated/rubert-tiny2` | `0.6269` | Лёгкая трансформерная модель |
| `ai-forever/ruBert-base` | `0.6913` | Сильная одиночная модель |
| `ai-forever/ruBert-base` + калибровка логитов | `0.6988` | Лучший одиночный трансформер |
| Ансамбль `ai-forever/ruBert-base` + `DeepPavlov/rubert-base-cased` | `0.7067` | Лучший результат проекта |
| RouterAI + Serper, агент на языковой модели | `0.5000` на 20 примерах | Работает технически, но не даёт прироста качества |

Финальный файл для отправки:

```text
notebooks/submission_ensemble.csv
```

Распределение классов в этом файле:

```text
0.0: 280
0.1: 20
1.0: 270
```

## Зачем обучали модель на видеокарте

Дообучение трансформеров на видеокарте было нужно не как лишний эксперимент, а как основа проекта:

- это сильный бейзлайн без агента;
- он даёт лучший результат проекта: `0.7067`;
- по нему формируется основной файл для отправки;
- с ним сравнивается агентная часть;
- именно сильный бейзлайн позволяет честно показать, что агент с поиском не дал стабильного прироста.

## Состав репозитория

```text
src/baseline_tfidf.py              TF-IDF бейзлайн
src/prepare_agent_cases.py         подготовка JSONL-файлов для агента
src/agent_runner.py                агент с RouterAI, Serper/Tavily и кешем результатов

notebooks/yandex_maps_relevance_colab.ipynb
notebooks/submission_ensemble.csv  лучший файл для отправки
notebooks/submission_transformer.csv

reports/experiment_report.md       подробный отчёт по экспериментам
reports/agent_full_validation_20.md отчёт по полному прогону агента
reports/agent_full_smoke_5.md      короткая техническая проверка агента

.env.example                       шаблон ключей без секретов
requirements.txt                   зависимости
```

В репозиторий не добавляются:

- `.env` с реальными ключами;
- сырые данные из `data/`;
- временные результаты из `outputs/`;
- большие артефакты обучения.

## Данные

Нужно скачать данные по ссылке из задания и положить файлы так:

```text
data/data_for_train.jsonl
data/data_for_eval.jsonl
```

Колонка `relevance_new` из файла контрольной выборки не использовалась для подбора параметров, промптов или анализа ошибок. Для настройки использовалось только разбиение `data_for_train.jsonl` на обучающую и валидационную части.

## Установка

```bash
python -m pip install -r requirements.txt
```

## Ключи API

Создать приватный `.env` из шаблона:

```powershell
Copy-Item .env.example .env
notepad .env
```

Минимальный `.env` для агента:

```text
ROUTERAI_API_KEY=...
ROUTERAI_BASE_URL=https://routerai.ru/api/v1
AGENT_MODEL=deepseek/deepseek-v4-pro

SEARCH_PROVIDER=serper
SERPER_API_KEY=...
```

`.env` игнорируется git.

## Запуск бейзлайна

TF-IDF бейзлайн:

```bash
python src/baseline_tfidf.py --classifier ridge --out-dir artifacts/baseline_tfidf_ridge
```

Быстрая проверка на подвыборке:

```bash
python src/baseline_tfidf.py --sample-size 5000 --out-dir artifacts/baseline_tfidf_smoke
```

## Запуск трансформерного ансамбля

Основной сценарий обучения на видеокарте находится в ноутбуке:

```text
notebooks/yandex_maps_relevance_colab.ipynb
```

В ноутбуке обучаются трансформерные модели, подбирается калибровка, строится ансамбль и сохраняется файл для отправки.

Лучший уже сохранённый файл:

```text
notebooks/submission_ensemble.csv
```

## Запуск агента

Сначала подготовить JSONL-файлы для агента:

```bash
python src/prepare_agent_cases.py --validation-cases 100
```

Будут созданы:

```text
outputs/agent_fewshot_train_examples.jsonl
outputs/agent_validation_cases.jsonl
outputs/agent_eval_cases.jsonl
```

Проверить только поиск Serper:

```bash
python src/agent_runner.py --check-search --search-provider serper --search-query "кафе с верандой Москва"
```

Запустить агента на валидационных примерах:

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

Запустить агента на контрольной выборке:

```bash
python src/agent_runner.py \
  --cases outputs/agent_eval_cases.jsonl \
  --out outputs/agent_eval_results_serper.jsonl \
  --search-provider serper \
  --fewshot-path outputs/agent_fewshot_train_examples.jsonl \
  --fewshot-k 3 \
  --submission-csv outputs/submission_agent.csv
```

## Проверка агента

Проведён полный сквозной прогон агента:

```text
Модель: RouterAI / deepseek/deepseek-v4-pro
Поиск: Serper
Примеры: 20 валидационных примеров
Правильно: 10/20
Точность: 0.5000
Поиск использован: 7/20
Ошибок выполнения: 0
Ошибок разбора JSON: 0
```

Подробности по каждому примеру:

```text
reports/agent_full_validation_20.md
```

## Финальный вывод

Лучшее качество даёт трансформерный ансамбль, поэтому для отправки выбран:

```text
notebooks/submission_ensemble.csv
```

Агент с поиском реализован и проверен, но на валидационных примерах не показал улучшения относительно сильного бейзлайна. Это соответствует варианту задания: не только построить агента, но и показать, помогает он в задаче или нет.
