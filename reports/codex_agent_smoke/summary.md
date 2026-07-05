# Codex-Agent Smoke Test

This is a small manual smoke test using Codex reasoning in this chat, without OpenAI API calls.

Protocol:

1. Select 12 difficult validation cases from TF-IDF errors.
2. Hide true labels before manual Codex classification.
3. Save Codex predictions to `predictions_codex.jsonl`.
4. Reveal labels from `truth.jsonl` and compare.

Files:

- `cases_blind.jsonl` - input cases without labels.
- `truth.jsonl` - hidden labels and TF-IDF predictions.
- `predictions_codex.jsonl` - Codex predictions made before reading `truth.jsonl`.

Results:

| Metric | Value |
| --- | ---: |
| Cases | 12 |
| Codex-agent accuracy | `0.3333` |
| TF-IDF accuracy on same cases | `0.0000` |

Class distributions:

| Source | `0.0` | `0.1` | `1.0` |
| --- | ---: | ---: | ---: |
| True labels | 7 | 2 | 3 |
| Codex predictions | 3 | 5 | 4 |
| TF-IDF predictions | 3 | 1 | 8 |

Observation:

Codex reasoning corrected some obviously wrong TF-IDF cases, for example irrelevant school/autoglass or attraction/food queries. However, it also disagreed with assessor labels on several cases where the dataset logic is stricter or noisier than natural human judgment.

Conclusion:

A raw LLM judge is not enough. For the agent layer, we need calibration with few-shot examples from train and a strict prompt aligned with assessor labels. The agent should be evaluated only on a validation subset and should not replace the fine-tuned transformer globally.
