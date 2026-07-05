# Final Project Checklist

## Task Interpretation

Original task:

- build a baseline;
- build an LLM agent for Yandex Maps relevance judging;
- the agent receives one pair: user query + organization card;
- the agent must be able to use search before deciding;
- labels are discrete: `0.0`, `0.1`, `1.0`;
- do not use eval labels for prompt tuning, calibration, or error analysis.

Clarification from project chat:

- a full recommendation product with frontend/backend is not required;
- an agent system that assigns a relevance score to one pair is sufficient.

## Ready Artifacts

Strong baseline / submission:

```text
notebooks/yandex_maps_relevance_colab.ipynb
/content/outputs/submission_ensemble.csv
/content/outputs/strong_ensemble/ensemble_config.json
/content/outputs/strong_ensemble/model_scores.csv
```

Current best validation result:

```text
2-model ensemble accuracy: 0.7067
models: ai-forever/ruBert-base + DeepPavlov/rubert-base-cased
weights: [0.7, 0.3]
bias: [0.0, -0.8, -0.5]
```

Agent:

```text
src/agent_runner.py
/content/outputs/agent_validation_cases.jsonl
/content/outputs/agent_eval_cases.jsonl
/content/outputs/agent_fewshot_train_examples.jsonl
```

Report:

```text
reports/experiment_report.md
reports/codex_agent_validation/summary_30.md
reports/codex_agent_validation/summary_100.md
```

Optional demo, not required:

```text
app/agent_web_app.py
```

## Scoring Coverage

| Requirement | Status |
| --- | --- |
| Baseline without agent | Done: TF-IDF, ruBERT, calibrated ensemble |
| Agent with search capability | Done: OpenAI-compatible runner + Serper/Tavily |
| Show whether agent helps | Done: validation tests on uncertain cases |
| Optional improvement | Done: lexical few-shot retrieval from train |

## Agent Validation Commands

RouterAI config:

```text
ROUTERAI_API_KEY=...
ROUTERAI_BASE_URL=https://routerai.ru/api/v1
AGENT_MODEL=deepseek/deepseek-v4-pro
SEARCH_PROVIDER=none
```

Put these values into `/content/.env` in Colab or into local `.env`. The real `.env` file must not be committed.

Without search:

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

With search:

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

## Agent Eval Command

Run only after choosing a cheap provider/model:

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

`agent_eval_cases.jsonl` excludes `relevance_new`, so this does not leak eval labels into the agent.

## Remaining Decision

Choose provider/model for the paid agent run.

Recommended cheap first pass:

```text
model: deepseek/deepseek-v4-pro
search-provider: none
fewshot-k: 3
limit on validation first: 30-50
```

Then, only if validation behavior is acceptable:

```text
search-provider: serper or tavily
run on eval
```
