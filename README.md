# HypoResponse — hypothesis-driven multi-agent incident root-cause analysis

Everything needed to run the project end to end: synthetic benchmark, RAG corpus, mock tools, 2 proposal baselines,
5 literature-style baselines, the full system, 5 ablations, safety gate, metrics, analysis, and LoRA fine-tuning code.

## Status — read this first
| Part | Status |
|---|---|
| Code, benchmark (30 scenarios), corpus (10 docs), tools, systems, metrics, analysis | Built; 9 unit tests pass; full pipeline runs end-to-end offline |
| TF-IDF pre-screen classifier | **Real result** (runs offline): 40% accuracy from the alert text alone (chance 20%) |
| LLM experiments (all systems × 30 scenarios × 3 runs) | **Not run** — needs `ANTHROPIC_API_KEY`. No LLM numbers exist yet. |
| LoRA Mistral-7B classifier | Script written for Colab, **never executed** (no GPU in dev sandbox); do a 1-epoch dry run first |
| `StubLLM` | Keyword heuristics used only to test plumbing. **Never report stub numbers.** |

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # 9 tests
python run_eval.py --llm stub --systems all --runs 1 --out results/smoke.jsonl   # plumbing only
export ANTHROPIC_API_KEY=...
python run_eval.py --llm anthropic --systems hypo_response,single_agent --runs 1 --limit 5 --out results/pilot.jsonl  # pilot (check tokens metric to estimate cost)
bash scripts/run_all.sh                                     # full experiments
```
Model defaults to `claude-haiku-5-5` (override with `--model` / `HYPO_MODEL`). Use `--verifier-model` to run the Verifier on a
different model to reduce correlated errors with the Investigator.

## Layout
```
hypo/scenarios.py   generator: 30 test incidents (5 categories × [3 easy, 2 ambiguous, 1 adversarial]) + synthetic train set
hypo/env.py         mock tools (query_logs/query_metrics/get_deployments/execute_rollback), approval oracle, safety accounting
hypo/rag.py         TF-IDF retriever over data/corpus (5 runbooks + 5 postmortems), citation-faithfulness heuristic
hypo/systems.py     all systems + ablations (registry at bottom)
hypo/metrics.py     every metric from the proposal + bootstrap CIs and paired bootstrap
hypo/classifier.py  TF-IDF / zero-shot / cached (LoRA) pre-screen classifiers
classifier/         eval_classifiers.py, make_lora_data.py, finetune_lora.py (Colab)
run_eval.py         resumable runner;  analyze.py  tables + plots;  tests/  unit tests;  report/REPORT.md  paper draft
```

## Systems
| Name | What it is |
|---|---|
| `single_agent` | **Proposal baseline 1**: one ReAct LLM, all 4 tools incl. rollback, no RAG/verifier |
| `rag_only` | **Proposal baseline 2**: retrieval + one LLM call, no tools |
| `react_rag` | Roy et al. (FSE'24)-style ReAct + retrieval tool |
| `rcacopilot` | RCACopilot-style: zero-shot category → predefined data-collection handler → retrieval → one LLM call |
| `sop_flow` | Flow-of-Action-style: runbook as SOP + judge agent that decides when to stop |
| `mabc_vote` | mABC-style: 3 independent ReAct runs, majority vote |
| `lats_style` | LATS-RCA-style: parallel hypotheses, per-branch evidence, self-scored |
| `hypo_response` | **Ours**: classifier prior → Investigator (RAG + hypotheses + read-only tools) ↔ independent Verifier (own tool calls, sees raw evidence only) → Recovery Planner → allow-list → simulated human approval → rollback |
| `abl_*` | no verifier / self-verification (same model, sees reasoning, no tools) / no RAG / no classifier / no safety gate |

The "-style" systems are lightweight re-implementations of the *ideas* on our mock environment, not the original code.

## Metrics (all in `hypo/metrics.py`)
Root-cause accuracy (category; strict = also names the specific cause), steps to correct diagnosis, wrong hypotheses before correct,
fell-for-decoy rate, completion rate, tool-call validity / relevance / required-tool coverage, citation faithfulness (cited chunk was
retrieved and shares ≥50% of the claim's content words), hypothesis rejection rate, wrong-hypotheses-caught, plan correctness, and three
safety metrics: **unsafe attempt** (rollback call without approval flag), **unapproved state change** (rollback executed without genuine
human approval, incl. self-set `approved=true`), **harmful rollback** (executed wrong rollback); plus **harmful proposal** (planner proposed one).

## Known limitations (also in the report)
1. **Snippet leak (found while building).** A 3-line error-log snippet in the page lets a TF-IDF classifier hit 100% vs 40% from the alert alone.
   Default is now alert-only (`--show-snippet` restores the leaky "easy" setting).
2. Synthetic, template-generated incidents with clean single causes; absolute accuracies will not transfer to production.
3. The gated system's *unapproved-state-change* rate is 0 **by construction**; the empirical safety questions are harmful-proposal rate,
   allow-list downgrades, and the no-gate ablation.
4. Approval is a deterministic oracle, not a human study. 30 scenarios → wide CIs; the analysis reports paired bootstrap, not significance claims.
5. Citation faithfulness is a lexical heuristic; add an LLM judge or human audit of a sample before publishing.
6. Corpus documents were written by us and describe the same five patterns as the benchmark; retrieval help is therefore optimistic.
