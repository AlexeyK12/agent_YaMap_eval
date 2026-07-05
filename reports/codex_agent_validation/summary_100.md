# Codex-Agent Validation Test on 100 Uncertain Cases

Input file from Colab:

```text
C:/Users/alkud/Downloads/agent_validation_cases.jsonl
```

Protocol:

1. Use cases 31-130 from the transformer-uncertain validation subset.
2. Split into:
   - `cases_blind_100.jsonl` without labels;
   - `truth_100.jsonl` with true labels and transformer predictions.
3. Predict labels with Codex reasoning from card fields only.
4. Save predictions before revealing labels.
5. Reveal labels and compare against raw/calibrated transformer on the same 100 cases.

Results:

| Model | Accuracy on same 100 cases |
| --- | ---: |
| Raw transformer | `0.3500` |
| Calibrated transformer | `0.4500` |
| Codex-agent manual judge | `0.4100` |

Class distributions:

| Source | `0.0` | `0.1` | `1.0` |
| --- | ---: | ---: | ---: |
| True labels | 35 | 29 | 36 |
| Raw transformer | 17 | 61 | 22 |
| Calibrated transformer | 35 | 21 | 44 |
| Codex-agent | 11 | 58 | 31 |

Codex-agent confusion matrix:

|  | pred `0.0` | pred `0.1` | pred `1.0` |
| --- | ---: | ---: | ---: |
| true `0.0` | 7 | 19 | 9 |
| true `0.1` | 3 | 19 | 7 |
| true `1.0` | 1 | 20 | 15 |

Comparison against calibrated transformer:

| Outcome | Count |
| --- | ---: |
| Both correct | 19 |
| Agent fixed transformer error | 22 |
| Agent broke transformer correct case | 26 |
| Both wrong | 33 |

Interpretation:

The larger blinded slice does not confirm the 30-case improvement. The manual Codex judge overused the cautious `0.1` class and missed many cases where assessor labels were either strict `0.0` or permissive `1.0`.

This is still useful for the project report: it shows a realistic negative agent result. The agent layer is implemented and tested, but a local-field-only LLM judge should not replace the calibrated transformer. The next agent iteration should add few-shot examples and optional web search, then rerun the same validation protocol.

Generated files:

```text
reports/codex_agent_validation/cases_blind_100.jsonl
reports/codex_agent_validation/truth_100.jsonl
reports/codex_agent_validation/predictions_codex_100.jsonl
reports/codex_agent_validation/joined_results_100.jsonl
reports/codex_agent_validation/summary_100.md
```
