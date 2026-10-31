"""Tiny dependency-light RAG over the 10-document corpus (TF-IDF over section chunks)."""
from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class Retriever:
    def __init__(self, corpus_dir="data/corpus"):
        self.chunks = []
        for p in sorted(Path(corpus_dir).glob("*.md")):
            text = p.read_text()
            title = text.splitlines()[0].lstrip("# ").strip()
            doc_id = p.stem.split("_")[0]
            parts = re.split(r"\n(?=## )", text)
            for n, part in enumerate(parts[1:], start=1):
                self.chunks.append({"chunk_id": f"{doc_id}#{n}", "doc_id": doc_id, "title": title,
                                    "text": f"{title}\n{part.strip()}"})
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.mat = self.vec.fit_transform([c["text"] for c in self.chunks])

    def search(self, query: str, k: int = 4, doc_prefix: str | None = None) -> list[dict]:
        sims = linear_kernel(self.vec.transform([query]), self.mat)[0]
        order = sims.argsort()[::-1]
        out = []
        for i in order:
            c = self.chunks[i]
            if doc_prefix and not c["doc_id"].startswith(doc_prefix):
                continue
            out.append({**c, "score": float(sims[i])})
            if len(out) == k:
                break
        return out


def format_chunks(chunks) -> str:
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


_WORD = re.compile(r"[a-z0-9_\-\.]{4,}")


def citation_faithfulness(citations, retrieved_by_id) -> float | None:
    """Fraction of citations that (a) point at a chunk actually retrieved in this run and
    (b) whose claim shares >=50% of its content words with that chunk. None if no citations."""
    if not citations:
        return None
    ok = 0
    for c in citations:
        if not isinstance(c, dict):
            continue
        chunk = retrieved_by_id.get(c.get("chunk_id"))
        if not chunk:
            continue
        claim = set(_WORD.findall(str(c.get("claim", "")).lower()))
        if not claim:
            continue
        words = set(_WORD.findall(chunk["text"].lower()))
        if len(claim & words) / len(claim) >= 0.5:
            ok += 1
    return ok / len(citations)
