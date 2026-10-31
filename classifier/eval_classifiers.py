#!/usr/bin/env python
"""Evaluate pre-screening classifiers on the 30 test scenarios (capability: fine-tuning).

  python classifier/eval_classifiers.py --which tfidf                     # runs offline, real result
  python classifier/eval_classifiers.py --which zeroshot --model claude-haiku-5-5
  python classifier/eval_classifiers.py --which cache --cache results/classifier_preds_lora.json
Writes results/classifier_preds_<which>.json ({scenario_id: [category, confidence]}).
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hypo import scenarios as S
from hypo.classifier import CacheClassifier, TfidfClassifier, ZeroShotClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--which", choices=["tfidf", "zeroshot", "cache"], required=True)
ap.add_argument("--model", default=None)
ap.add_argument("--cache", default="results/classifier_preds_lora.json")
ap.add_argument("--show-snippet", action="store_true")
ap.add_argument("--n-train", type=int, default=120, help="synthetic training incidents per category (tfidf)")
a = ap.parse_args()
from hypo import config
config.SHOW_SNIPPET = a.show_snippet

test = S.load("data/scenarios.json")
if a.which == "tfidf":
    clf = TfidfClassifier().fit(S.generate_train_set(a.n_train))
elif a.which == "zeroshot":
    from hypo.llm import AnthropicLLM
    clf = ZeroShotClassifier(AnthropicLLM(a.model))
else:
    clf = CacheClassifier(a.cache)

preds = {s["id"]: clf.predict(s) for s in test}
acc = sum(preds[s["id"]][0] == s["category"] for s in test) / len(test)
print(f"{a.which}: accuracy {acc:.3f} on {len(test)} test scenarios")
for d in ["easy", "ambiguous", "adversarial"]:
    sub = [s for s in test if s["difficulty"] == d]
    print(f"  {d:12s} {sum(preds[s['id']][0]==s['category'] for s in sub)}/{len(sub)}")
conf = Counter((s["category"], preds[s["id"]][0]) for s in test if preds[s["id"]][0] != s["category"])
print("  confusions (true -> pred):", dict(conf))
Path("results").mkdir(exist_ok=True)
Path(f"results/classifier_preds_{a.which}{'_snippet' if a.show_snippet else ''}.json").write_text(json.dumps({k: [v[0], v[1]] for k, v in preds.items()}))
