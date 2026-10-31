"""Incident pre-screening classifiers (capability: fine-tuning).

TfidfClassifier   - cheap trained baseline on synthetic data (runs anywhere)
ZeroShotClassifier- zero-shot LLM comparison point
CacheClassifier   - loads predictions produced elsewhere (e.g. the LoRA Mistral-7B run on Colab)
"""
from __future__ import annotations

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from . import config
from .llm import parse_json
from .scenarios import CATEGORIES


def text_of(sc) -> str:
    return sc["alert"] + ("\n" + "\n".join(sc["snippet"]) if config.SHOW_SNIPPET else "")


class TfidfClassifier:
    def __init__(self):
        self.pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                                  LogisticRegression(max_iter=2000, C=5.0))

    def fit(self, scs):
        self.pipe.fit([text_of(s) for s in scs], [s["category"] for s in scs])
        return self

    def predict(self, sc):
        p = self.pipe.predict_proba([text_of(sc)])[0]
        i = int(p.argmax())
        return str(self.pipe.classes_[i]), float(p[i])


class ZeroShotClassifier:
    def __init__(self, llm):
        self.llm = llm

    def predict(self, sc):
        sysmsg = ("Classify the production incident into exactly one category: " + ", ".join(CATEGORIES) +
                  '. Reply JSON: {"category": "...", "confidence": 0-1}')
        j = parse_json(self.llm.chat(sysmsg, [{"role": "user", "content": text_of(sc)}], role="classify")) or {}
        c = j.get("category")
        return (c if c in CATEGORIES else CATEGORIES[0]), float(j.get("confidence", 0.5) or 0.5)


class CacheClassifier:
    def __init__(self, path):
        self.preds = json.load(open(path))

    def predict(self, sc):
        c, p = self.preds[sc["id"]]
        return c, float(p)
