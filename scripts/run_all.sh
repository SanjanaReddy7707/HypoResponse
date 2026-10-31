#!/usr/bin/env bash
# Full experimental pipeline. Requires ANTHROPIC_API_KEY. Run a cheap pilot first:
#   python run_eval.py --llm anthropic --systems hypo_response,single_agent --runs 1 --limit 5 --out results/pilot.jsonl
set -euo pipefail
MODEL=${MODEL:-claude-haiku-5-5}
VERIFIER_MODEL=${VERIFIER_MODEL:-}     # e.g. a different model family/size for the Verifier
cd "$(dirname "$0")/.."

python -m hypo.scenarios data/scenarios.json
python -m pytest -q tests

# Fine-tuning capability: classifiers on alert-only pre-screen (+ leaky snippet reference)
python classifier/eval_classifiers.py --which tfidf
python classifier/eval_classifiers.py --which zeroshot --model "$MODEL"
# (Colab) python classifier/make_lora_data.py 200 && python classifier/finetune_lora.py   -> results/classifier_preds_lora.json
# python classifier/eval_classifiers.py --which cache --cache results/classifier_preds_lora.json

VARG=""; [ -n "$VERIFIER_MODEL" ] && VARG="--verifier-model $VERIFIER_MODEL"

# 1) Main comparison + ablations, 3 runs, alert-only page
python run_eval.py --llm anthropic --model "$MODEL" $VARG --systems all --runs 3 --out results/raw.jsonl
python analyze.py --raw results/raw.jsonl --out results/summary

# 2) Robustness: confident WRONG triage prior given to everyone (tests whether verification rescues bad first beliefs)
python run_eval.py --llm anthropic --model "$MODEL" $VARG --systems all --runs 3 --inject-wrong-prior --out results/raw_inject.jsonl
python analyze.py --raw results/raw_inject.jsonl --injected --out results/summary_inject

# 3) Easy setting (leaky snippet) for reference
python run_eval.py --llm anthropic --model "$MODEL" $VARG --systems main --runs 1 --show-snippet --out results/raw_snippet.jsonl
