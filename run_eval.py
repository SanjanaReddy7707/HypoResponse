#!/usr/bin/env python
"""Run systems on scenarios and append per-run rows to a JSONL file (resumable).

Real run:
  export ANTHROPIC_API_KEY=...
  python run_eval.py --llm anthropic --model claude-haiku-5-5 --systems all --runs 3 --out results/raw.jsonl
Pipeline smoke test (NOT results):
  python run_eval.py --llm stub --systems all --runs 1 --out results/smoke.jsonl
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hypo import scenarios as S
from hypo.classifier import CacheClassifier, TfidfClassifier, ZeroShotClassifier
from hypo.llm import AnthropicLLM, StubLLM
from hypo.metrics import run_metrics
from hypo.rag import Retriever
from hypo.systems import Ctx, build_registry

MAIN = ["single_agent", "rag_only", "react_rag", "rcacopilot", "sop_flow", "mabc_vote", "lats_style", "hypo_response"]
ABL = ["abl_no_verifier", "abl_self_verify", "abl_no_rag", "abl_no_classifier", "abl_no_safety_gate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["anthropic", "stub"], default="stub")
    ap.add_argument("--model", default=None)
    ap.add_argument("--verifier-model", default=None, help="different model for the Verifier (reduces correlated errors)")
    ap.add_argument("--systems", default="main", help="'main', 'ablations', 'all' or comma list")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--scenarios", default="data/scenarios.json")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--classifier", choices=["tfidf", "zeroshot", "cache", "none"], default="tfidf")
    ap.add_argument("--classifier-cache", default="results/classifier_preds_lora.json")
    ap.add_argument("--inject-wrong-prior", action="store_true",
                    help="give every system a confident WRONG triage prior (the decoy category) on scenarios that have a decoy")
    ap.add_argument("--show-snippet", action="store_true", help="include leaky error-line snippet in the page (easy setting)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/raw.jsonl")
    a = ap.parse_args()

    from hypo import config
    config.SHOW_SNIPPET = a.show_snippet
    reg = build_registry()
    groups = {"main": MAIN, "ablations": ABL, "all": MAIN + ABL}
    names = []
    for tok in a.systems.split(","):
        names += groups.get(tok, [tok])
    unknown = [n for n in names if n not in reg]
    if unknown:
        raise SystemExit(f"unknown systems: {unknown}; available: {sorted(reg)}")
    scs = S.load(a.scenarios)[: a.limit]
    retriever = Retriever(a.corpus)

    clf_shared = None
    if a.classifier == "tfidf":
        clf_shared = TfidfClassifier().fit(S.generate_train_set(120))
    elif a.classifier == "cache":
        clf_shared = CacheClassifier(a.classifier_cache)

    forced = (lambda sc: (sc["ground_truth"]["decoy_category"], 0.9) if sc["ground_truth"]["decoy_category"] else None) \
        if a.inject_wrong_prior else None

    def make_llm(model):
        return StubLLM() if a.llm == "stub" else AnthropicLLM(model)

    def job(system, sc, run):
        llm = make_llm(a.model)
        vllm = make_llm(a.verifier_model) if a.verifier_model else None
        clf = clf_shared
        if a.classifier == "zeroshot":
            clf = ZeroShotClassifier(llm)
        ctx = Ctx(llm=llm, retriever=retriever, verifier_llm=vllm, classifier=clf, forced_prior=forced)
        r = reg[system](ctx, sc)
        met = run_metrics(sc, r)
        return {"system": system, "scenario_id": sc["id"], "run": run, "category": sc["category"],
                "difficulty": sc["difficulty"], "injected_prior": bool(forced), "snippet": a.show_snippet, "model": llm.model,
                "metrics": met, "final": r["final"], "verifier_log": r["verifier_log"], "plan": r["plan"],
                "hypotheses": r["hypotheses"], "safety": r["safety"], "rollbacks": r["rollbacks"],
                "env_calls": r["env_calls"]}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            x = json.loads(line)
            done.add((x["system"], x["scenario_id"], x["run"], x["injected_prior"]))
    todo = [(s, sc, r) for r in range(a.runs) for s in names for sc in scs
            if (s, sc["id"], r, bool(forced)) not in done]
    print(f"{len(todo)} jobs ({len(done)} already done)")
    lock = threading.Lock()
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as fh:
        futs = {ex.submit(job, *t): t for t in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            s, sc, r = futs[fu]
            try:
                row = fu.result()
            except Exception as e:  # keep going; failed runs are simply not recorded and will be retried on resume
                print(f"FAILED {s} {sc['id']} run{r}: {e}")
                continue
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            if i % 20 == 0:
                print(f"{i}/{len(todo)}")
    print("done ->", out)


if __name__ == "__main__":
    main()
