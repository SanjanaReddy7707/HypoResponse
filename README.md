# HypoResponse — Hypothesis-Driven Multi-Agent Incident Root-Cause Analysis

A framework for automated incident root-cause analysis using hypothesis-driven reasoning with multi-agent collaboration. It includes a synthetic benchmark, RAG corpus, mock tools, multiple baselines, ablation studies, safety gating, evaluation metrics, and LoRA fine-tuning code.

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

### Run Evaluation

```bash
# Smoke test (stub LLM, no API key needed)
python run_eval.py --llm stub --systems all --runs 1 --out results/smoke.jsonl

# Pilot run with Anthropic
export ANTHROPIC_API_KEY=...
python run_eval.py --llm anthropic --systems hypo_response,single_agent --runs 1 --limit 5 --out results/pilot.jsonl

# Full experiments
bash scripts/run_all.sh
```

Model defaults to `claude-haiku-5-5` (override with `--model` / `HYPO_MODEL`). Use `--verifier-model` to run the Verifier on a different model to reduce correlated errors with the Investigator.

## Project Layout

```
hypo/scenarios.py   30 test incidents (5 categories × [3 easy, 2 ambiguous, 1 adversarial]) + synthetic train set
hypo/env.py         Mock tools (query_logs, query_metrics, get_deployments, execute_rollback), approval oracle, safety accounting
hypo/rag.py         TF-IDF retriever over data/corpus (5 runbooks + 5 postmortems), citation-faithfulness heuristic
hypo/systems.py     All systems + ablations (registry at bottom)
hypo/metrics.py     All metrics + bootstrap CIs and paired bootstrap
hypo/classifier.py  TF-IDF / zero-shot / cached (LoRA) pre-screen classifiers
classifier/         eval_classifiers.py, make_lora_data.py, finetune_lora.py
run_eval.py         Resumable evaluation runner
analyze.py          Tables + plots generation
tests/              Unit tests
report/             Project report
data/corpus/        5 runbooks + 5 postmortems
data/scenarios.json Benchmark scenarios
```

## Systems

| Name | Description |
|---|---|
| `single_agent` | **Baseline 1**: One ReAct LLM, all 4 tools incl. rollback, no RAG/verifier |
| `rag_only` | **Baseline 2**: Retrieval + one LLM call, no tools |
| `react_rag` | ReAct + retrieval tool (Roy et al., FSE'24 style) |
| `rcacopilot` | RCACopilot-style: zero-shot category → predefined handler → retrieval → one LLM call |
| `sop_flow` | Flow-of-Action-style: runbook as SOP + judge agent |
| `mabc_vote` | mABC-style: 3 independent ReAct runs, majority vote |
| `lats_style` | LATS-RCA-style: parallel hypotheses, per-branch evidence, self-scored |
| `hypo_response` | **Ours**: classifier prior → Investigator (RAG + hypotheses + tools) ↔ Verifier → Recovery Planner → safety gate → approval → rollback |
| `abl_*` | Ablations: no verifier / self-verification / no RAG / no classifier / no safety gate |

## Metrics

Evaluated using metrics defined in `hypo/metrics.py`:

- **Root-cause accuracy** (category-level and strict)
- **Diagnostic efficiency**: steps to correct diagnosis, wrong hypotheses before correct
- **Tool usage**: call validity, relevance, required-tool coverage
- **RAG quality**: citation faithfulness (lexical overlap heuristic)
- **Safety**: unsafe attempts, unapproved state changes, harmful rollbacks/proposals

## TF-IDF Pre-Screen Classifier

The TF-IDF classifier achieves 40% accuracy from alert text alone (vs. 20% chance baseline), providing a useful prior for hypothesis generation.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
- Anthropic API key for LLM-based evaluation
