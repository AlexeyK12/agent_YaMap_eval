# Codex-Agent Validation Smoke Test

Input file from Colab:

```text
C:/Users/alkud/Downloads/agent_validation_cases.jsonl
```

Protocol:

1. Use the first 30 cases from the transformer-uncertain validation subset.
2. Split into:
   - `cases_blind_30.jsonl` without labels;
   - `truth_30.jsonl` with true labels and transformer predictions.
3. Predict labels with Codex reasoning from card fields only.
4. Reveal labels and compare against raw/calibrated transformer on the same 30 cases.

Results:

| Model | Accuracy on same 30 cases |
| --- | ---: |
| Raw transformer | `0.3333` |
| Calibrated transformer | `0.4333` |
| Codex-agent manual judge | `0.6333` |

Class distributions:

| Source | `0.0` | `0.1` | `1.0` |
| --- | ---: | ---: | ---: |
| True labels | 13 | 6 | 11 |
| Calibrated transformer | 10 | 7 | 13 |
| Codex-agent | 5 | 13 | 12 |

Interpretation:

On this intentionally uncertain validation slice, Codex reasoning substantially outperformed the calibrated transformer. This does not prove full-dataset improvement, but it supports the agent design: run an LLM/agent only on uncertain cases rather than replacing the transformer globally.

Main caveat:

The sample is small and manually judged. The next step is to run the API-based cached agent on 50-100 uncertain validation cases with the same prompt and optional search.
