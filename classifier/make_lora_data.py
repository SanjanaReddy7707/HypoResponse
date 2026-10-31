#!/usr/bin/env python
"""Write synthetic train/val/test JSONL for the LoRA classifier. Test = the 30 benchmark scenarios (never trained on)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hypo import scenarios as S
from hypo.classifier import text_of

from hypo import config
config.SHOW_SNIPPET = '--show-snippet' in sys.argv
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
tr = S.generate_train_set(n, seed=1000)
va = S.generate_train_set(40, seed=2000)
te = S.load("data/scenarios.json")
Path("data").mkdir(exist_ok=True)
for name, ds in [("train", tr), ("val", va), ("test", te)]:
    with open(f"data/clf_{name}.jsonl", "w") as f:
        for s in ds:
            f.write(json.dumps({"id": s["id"], "text": text_of(s), "label": S.CATEGORIES.index(s["category"])}) + "\n")
    print(name, len(ds))
