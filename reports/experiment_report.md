# Experiment Report Notes

## Task

We solve relevance classification for Yandex Maps organizations. Each example is a pair:

- user query, for example `ресторан с верандой`;
- organization card fields from Yandex Maps.

Target classes:

- `0.0` - irrelevant;
- `0.1` - partially relevant;
- `1.0` - relevant.

Main metric from the task statement: `accuracy`.

Important protocol detail: `relevance_new` in eval was not used for model selection, calibration, prompt tuning, or error analysis. All tuning was done on a stratified train/validation split from `data_for_train.jsonl`.

## Data Split

Training data: `data/data_for_train.jsonl`, 34,094 rows.

Validation split: stratified random 80/20 split from train, seed `42`.

Validation size: 6,819 rows.

Validation class distribution:

| Class | Count |
| --- | ---: |
| `0.0` | 2,815 |
| `0.1` | 913 |
| `1.0` | 3,091 |

Eval data: `data/data_for_eval.jsonl`, 570 rows. Eval labels were not used during tuning.

## Baseline 1: TF-IDF + Linear Models

Features: concatenated text from query and organization fields:

- `Text`;
- `name`;
- `normalized_main_rubric_name_ru`;
- `address`;
- `prices_summarized`;
- `reviews_summarized`.

Vectorization:

- word TF-IDF, 1-2 grams;
- char TF-IDF, 3-5 grams.

Classifier results:

| Model | Class weight | Validation accuracy |
| --- | --- | ---: |
| `LinearSVC` | none | `0.5872` |
| `LinearSVC` | balanced | `0.5810` |
| `RidgeClassifier` | none | `0.5969` |
| `RidgeClassifier` | balanced | `0.5794` |

Best classical baseline: `RidgeClassifier`, accuracy `0.5969`.

Observation: the class `0.1` is the hardest one. Balanced class weights improve its recall but reduce the main metric, accuracy.

## Baseline 2: Transformer Fine-Tuning

The same concatenated text representation was used as input. Fine-tuning was done in Colab with GPU.

Main training settings for the successful run:

- model: `ai-forever/ruBert-base`;
- max length: `384`;
- epochs: `3`;
- learning rate: `2e-5`;
- train batch size: `16`;
- eval batch size: `32`;
- metric for best model: validation accuracy.

Transformer results:

| Model | Validation accuracy | Notes |
| --- | ---: | --- |
| `cointegrated/rubert-tiny2` | `0.6269` | Better than TF-IDF, but almost ignores class `0.1` |
| `ai-forever/ruBert-base` | `0.6913` | Strong improvement over TF-IDF and tiny RuBERT |
| `ai-forever/ruBert-base` + logit calibration | `0.6988` | Strong single-model baseline |
| `ai-forever/ruBert-base` + `DeepPavlov/rubert-base-cased` ensemble | `0.7067` | Current best validation result |

## Best Raw Transformer Metrics

Model: `ai-forever/ruBert-base`.

Validation accuracy: `0.6913`.

Classification report:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `0.0` | `0.7199` | `0.7339` | `0.7268` | 2,815 |
| `0.1` | `0.3845` | `0.2442` | `0.2987` | 913 |
| `1.0` | `0.7198` | `0.7845` | `0.7508` | 3,091 |

Confusion matrix:

|  | pred `0.0` | pred `0.1` | pred `1.0` |
| --- | ---: | ---: | ---: |
| true `0.0` | 2,066 | 193 | 556 |
| true `0.1` | 302 | 223 | 388 |
| true `1.0` | 502 | 164 | 2,425 |

## Logit Calibration

After fine-tuning, a simple validation-only calibration was applied: add a fixed bias to class logits before `argmax`.

Best validation bias:

```python
[0.0, -0.8, 0.0]
```

Raw transformer accuracy: `0.6913`.

Calibrated transformer accuracy: `0.6988`.

Calibrated classification report:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `0.0` | `0.7042` | `0.7620` | `0.7320` | 2,815 |
| `0.1` | `0.5311` | `0.1216` | `0.1979` | 913 |
| `1.0` | `0.7040` | `0.8117` | `0.7540` | 3,091 |

Calibrated confusion matrix:

|  | pred `0.0` | pred `0.1` | pred `1.0` |
| --- | ---: | ---: | ---: |
| true `0.0` | 2,145 | 57 | 613 |
| true `0.1` | 360 | 111 | 442 |
| true `1.0` | 541 | 41 | 2,509 |

Interpretation: calibration improves the target metric, accuracy, by making the model more conservative about class `0.1`. This lowers recall for `0.1`, but increases precision and improves total accuracy.

## Strong Transformer Ensemble

The next strongest non-agent route was an ensemble of two independently fine-tuned Russian BERT models on the same train/validation split.

Single model validation results inside the ensemble run:

| Model | Raw accuracy | Calibrated accuracy | Single-model bias |
| --- | ---: | ---: | --- |
| `ai-forever/ruBert-base` | `0.7024` | `0.7044` | `[0.0, -0.6, 0.0]` |
| `DeepPavlov/rubert-base-cased` | `0.6875` | `0.6906` | `[0.0, -0.3, -0.1]` |

Best ensemble configuration:

```python
weights = [0.7, 0.3]
bias = [0.0, -0.8, -0.5]
```

Ensemble validation accuracy:

```text
raw ensemble:        0.7022
calibrated ensemble: 0.7067
```

Classification report:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `0.0` | `0.6870` | `0.8085` | `0.7428` | 2,815 |
| `0.1` | `0.6579` | `0.1095` | `0.1878` | 913 |
| `1.0` | `0.7284` | `0.7904` | `0.7581` | 3,091 |

Confusion matrix:

|  | pred `0.0` | pred `0.1` | pred `1.0` |
| --- | ---: | ---: | ---: |
| true `0.0` | 2,276 | 29 | 510 |
| true `0.1` | 412 | 100 | 401 |
| true `1.0` | 625 | 23 | 2,443 |

The ensemble improves the previous best validation accuracy from `0.6988` to `0.7067`.

## Current Submission

Current best submission file:

```text
/content/outputs/submission_ensemble.csv
```

Format:

| Column | Meaning |
| --- | --- |
| `permalink` | organization id |
| `Text` | user query |
| `relevance` | predicted class |

Rows: 570.

Prediction distribution:

| Predicted class | Count |
| --- | ---: |
| `0.0` | 280 |
| `0.1` | 20 |
| `1.0` | 270 |

## Conclusions So Far

The classical TF-IDF baseline is useful and fast, but the fine-tuned Russian transformer gives a large improvement:

```text
0.5969 -> 0.6913 -> 0.6988 -> 0.7067
TF-IDF -> ruBert-base -> ruBert-base calibration -> 2-model ensemble
```

The main remaining challenge is class `0.1`. It is semantically fuzzy and often sits between clearly irrelevant and clearly relevant cases. Optimizing for accuracy pushes the model to predict `0.1` conservatively.

## Project Scoring Checklist

Original Colab scoring:

| Requirement | Points | Implementation |
| --- | ---: | --- |
| Baseline without agent | 2 | TF-IDF, fine-tuned transformer, calibrated 2-model ensemble |
| Agent with search-line capability | 4 | OpenAI-compatible LLM agent with optional Serper/Tavily search |
| Beat baseline or show agent does not help | 4 | Validation experiments on uncertain cases; local-only LLM judge was not stable enough |
| Optional improvement idea | optional | lexical few-shot retrieval over labeled train examples |

## Implemented Agent Layer

Implemented files:

```text
src/agent_runner.py
notebooks/yandex_maps_relevance_colab.ipynb
```

The agent pipeline:

- receives one pair `(query, organization card)`;
- optionally receives similar labeled train examples as few-shot context;
- performs a first LLM call to assess local evidence and decide whether search is needed;
- optionally forms search queries and calls Serper or Tavily;
- performs a final LLM call to produce one of `0.0`, `0.1`, `1.0`;
- appends every result to JSONL cache and can resume safely;
- can evaluate validation accuracy when true labels are present;
- can export eval predictions to submission CSV.

The Colab notebook creates:

```text
/content/outputs/agent_validation_cases.jsonl
/content/outputs/agent_eval_cases.jsonl
/content/outputs/agent_fewshot_train_examples.jsonl
```

Important eval protocol:

- `agent_eval_cases.jsonl` intentionally excludes `relevance_new`;
- eval labels are not used for prompt tuning, calibration, or error analysis.

Case selection criteria:

- calibrated transformer predicts `0.1`;
- calibrated confidence below `0.72`;
- margin between top-1 and top-2 probabilities below `0.18`;
- raw and calibrated transformer predictions disagree.

Default subset size: 300 most uncertain validation cases, sorted by margin and confidence.

RouterAI/OpenAI-compatible configuration is expected in `.env`:

```text
ROUTERAI_API_KEY=...
ROUTERAI_BASE_URL=https://routerai.ru/api/v1
AGENT_MODEL=deepseek/deepseek-v4-pro
SEARCH_PROVIDER=none
```

The real `.env` file is ignored by git; only `.env.example` is stored in the project.

Suggested validation run without search:

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

Suggested search-enabled run:

```bash
python /content/agent_runner.py \
  --env-file /content/.env \
  --cases /content/outputs/agent_validation_cases.jsonl \
  --out /content/outputs/agent_validation_results_search.jsonl \
  --limit 50 \
  --search-provider serper \
  --fewshot-path /content/outputs/agent_fewshot_train_examples.jsonl \
  --fewshot-k 3 \
  --evaluate
```

Suggested eval run after choosing a cheap provider/model:

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

## Optional Agent Application

In addition to the required batch runner, a minimal local web application was added as an optional demo:

```text
app/agent_web_app.py
app/README.md
```

Run command:

```bash
python app/agent_web_app.py --host 127.0.0.1 --port 7860
```

The application accepts one query and one organization card, then runs the same LLM/search agent pipeline as `src/agent_runner.py`:

1. local card assessment;
2. decision whether external search is needed;
3. optional Serper/Tavily search;
4. final discrete label: `0.0`, `0.1`, or `1.0`;
5. evidence and explanation JSON.

This matches the project clarification that the agent may evaluate one pair `(query, organization)` rather than implement a full recommendation product.

## Preliminary Codex-Agent Smoke Test

Before using an external API, a tiny blinded smoke test was run manually with Codex reasoning in this chat.

Files:

```text
reports/codex_agent_smoke/cases_blind.jsonl
reports/codex_agent_smoke/truth.jsonl
reports/codex_agent_smoke/predictions_codex.jsonl
reports/codex_agent_smoke/summary.md
```

Protocol:

1. Select 12 difficult validation cases from TF-IDF errors.
2. Hide true labels.
3. Predict labels with Codex reasoning.
4. Reveal labels and compare.

Results:

| Metric | Value |
| --- | ---: |
| Cases | 12 |
| Codex-agent accuracy | `0.3333` |
| TF-IDF accuracy on same selected cases | `0.0000` |

Interpretation:

This is not a final agent result because the sample is tiny and intentionally biased toward TF-IDF failures. Still, it shows an important risk: raw LLM judgment can disagree with assessor labels. Therefore the agent needs strict prompting, few-shot calibration from train, and careful validation before being applied to eval.

## Codex-Agent on Transformer-Uncertain Validation Cases

After generating `/content/outputs/agent_validation_cases.jsonl` in Colab, a second blinded manual Codex smoke test was run on the first 30 uncertain validation cases.

Files:

```text
reports/codex_agent_validation/cases_blind_30.jsonl
reports/codex_agent_validation/truth_30.jsonl
reports/codex_agent_validation/predictions_codex_30.jsonl
reports/codex_agent_validation/joined_results_30.jsonl
reports/codex_agent_validation/summary_30.md
```

Results on the same 30 cases:

| Model | Accuracy |
| --- | ---: |
| Raw transformer | `0.3333` |
| Calibrated transformer | `0.4333` |
| Codex-agent manual judge | `0.6333` |

Interpretation:

This supports the selective-agent approach. The LLM judge is not necessarily better globally, but on low-margin/low-confidence transformer cases it can recover many examples by applying semantic reasoning over the card fields. The next step is to replace the manual Codex pass with the cached API runner on 50-100 validation cases.

## Extended Codex-Agent Validation on 100 Uncertain Cases

The next blinded manual Codex pass was run on another 100 transformer-uncertain validation cases from the same Colab export.

Files:

```text
reports/codex_agent_validation/cases_blind_100.jsonl
reports/codex_agent_validation/truth_100.jsonl
reports/codex_agent_validation/predictions_codex_100.jsonl
reports/codex_agent_validation/joined_results_100.jsonl
reports/codex_agent_validation/summary_100.md
```

Results on the same 100 cases:

| Model | Accuracy |
| --- | ---: |
| Raw transformer | `0.3500` |
| Calibrated transformer | `0.4500` |
| Codex-agent manual judge | `0.4100` |

Class distributions:

| Source | `0.0` | `0.1` | `1.0` |
| --- | ---: | ---: | ---: |
| True labels | 35 | 29 | 36 |
| Calibrated transformer | 35 | 21 | 44 |
| Codex-agent | 11 | 58 | 31 |

Interpretation:

The larger sample did not confirm the 30-case improvement. The manual Codex judge overused the cautious `0.1` class, which helped on some ambiguous cases but hurt overall: it fixed 22 calibrated-transformer errors and broke 26 cases that the transformer had correct.

This is a useful negative result for the final report. We implemented the agent layer and validated it on difficult examples, but local-field-only LLM judging is not reliable enough to replace the calibrated transformer. Further agent work should focus on few-shot calibration and optional web search for properties absent from the card fields.

## Strong Transformer Ensemble Notebook

The Colab notebook was extended and run with a stronger non-agent route:

```text
notebooks/yandex_maps_relevance_colab.ipynb
```

New outputs:

```text
/content/outputs/submission_ensemble.csv
/content/outputs/ensemble_valid_predictions.csv
/content/outputs/ensemble_errors_sample.csv
/content/outputs/strong_ensemble/model_scores.csv
/content/outputs/strong_ensemble/ensemble_config.json
```

Default ensemble members:

| Model | Status |
| --- | --- |
| `ai-forever/ruBert-base` | enabled |
| `DeepPavlov/rubert-base-cased` | enabled |
| `xlm-roberta-base` | optional, disabled by default |

The ensemble block saves validation/eval logits for every model, resumes from saved logits on rerun, tunes ensemble weights on validation, then tunes a class logit/probability bias on validation.

Final ensemble result:

```text
best calibrated validation accuracy: 0.7067
weights: [0.7, 0.3]
bias: [0.0, -0.8, -0.5]
eval prediction distribution: 0.0 = 280, 0.1 = 20, 1.0 = 270
```

This is the current best result without API costs.
