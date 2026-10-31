"""LLM backends.

AnthropicLLM  - real runs (set ANTHROPIC_API_KEY; model via --model or HYPO_MODEL).
StubLLM       - deterministic keyword heuristics used ONLY to test the pipeline offline.
                Its outputs are NOT experimental results and must never be reported.
"""
from __future__ import annotations

import json
import os
import random
import re
import time


def parse_json(text: str):
    """Extract the first balanced JSON object from model output (tolerates code fences / prose)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class LLM:
    def __init__(self, model: str):
        self.model = model
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0

    def chat(self, system, messages, role="agent", temperature=0.0, max_tokens=900) -> str:
        raise NotImplementedError

    def usage(self):
        return {"calls": self.calls, "in_tokens": self.in_tokens, "out_tokens": self.out_tokens}


class AnthropicLLM(LLM):
    def __init__(self, model: str | None = None):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic()
        super().__init__(model or os.environ.get("HYPO_MODEL", "claude-haiku-5-5"))

    def chat(self, system, messages, role="agent", temperature=0.0, max_tokens=900):
        kwargs = dict(model=self.model, system=system, messages=messages, max_tokens=max_tokens,
                      temperature=temperature)
        last = None
        for attempt in range(6):
            try:
                r = self.client.messages.create(**kwargs)
                self.calls += 1
                self.in_tokens += r.usage.input_tokens
                self.out_tokens += r.usage.output_tokens
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            except Exception as e:  # rate limits, transient errors, unsupported params
                last = e
                if "temperature" in str(e).lower() and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed after retries: {last}")


# ------------------------------------------------------------------------------------------------
KEYWORDS = {
    "bad_deployment": ["nullpointer", "keyerror", "typeerror", "json decode", "build=v"],
    "database_slowdown": ["slow query", "lock wait", "replica lag", "pool after", "query timeout"],
    "memory_leak": ["outofmemory", "oomkilled", "full gc", "heap", "evictions=0", "descriptors", "listeners"],
    "third_party_api_failure": ["upstream", "circuit breaker", "returned 503", "timeout after 5000ms calling"],
    "config_change": ["config reload", "config version=", "max_size=", "rate_limit", "feature flag", "http.client.timeout"],
}


def guess_category(text: str) -> str:
    t = text.lower()
    scores = {c: sum(t.count(k) for k in kws) for c, kws in KEYWORDS.items()}
    if "config version=" in t:
        scores["config_change"] += 3
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "bad_deployment"


class StubLLM(LLM):
    """Offline heuristic stand-in. Role is inferred from the FINAL_SCHEMA / ROLE markers in the prompt."""

    def __init__(self, seed: int = 0):
        super().__init__("stub")
        self.rng = random.Random(seed)

    def chat(self, system, messages, role="agent", temperature=0.0, max_tokens=900):
        self.calls += 1
        first = messages[0]["content"]
        obs = "\n".join(m["content"] for m in messages if m["content"].startswith("OBSERVATION"))
        svc_m = re.search(r"Affected service: (\S+)", first)
        svc = svc_m.group(1) if svc_m else "unknown"
        cid_m = re.search(r"\[(RB-\d+#\d+|PM-\d+#\d+)\]", first)
        cites = [{"chunk_id": cid_m.group(1), "claim": "runbook guidance"}] if cid_m else []

        if role == "classify":
            return json.dumps({"category": guess_category(first), "confidence": 0.6})
        if role == "hypothesize":
            cats = list(KEYWORDS)
            g = guess_category(first)
            others = [c for c in cats if c != g][:2]
            return json.dumps({"hypotheses": [{"category": c, "rationale": "stub"} for c in [g] + others]})
        if role == "judge":
            return json.dumps({"stop": obs.count("OBSERVATION") >= 3 or len(obs) > 1500})
        if role == "verify_self":
            m = re.search(r"Hypothesis under review: category=(\w+)", first)
            claimed = m.group(1) if m else "bad_deployment"
            raw = first.split("RAW OBSERVATIONS", 1)[-1]
            cat = guess_category(raw)
            ok = claimed == cat
            return json.dumps({"final": {"verdict": "support" if ok else "reject", "reasons": ["stub self-check"],
                                         "contradictions": [] if ok else [f"evidence suggests {cat}"], "suggested_checks": []}})
        if role == "rag_only":
            return json.dumps({"final": self._diag(guess_category(first), first, cites)})
        if role == "plan":
            cat = re.search(r"Verified category: (\w+)", first)
            cat = cat.group(1) if cat else "bad_deployment"
            m = re.search(r"service=(\S+) .*?previous_version=(\S+)", first) or None
            prev = re.search(r"previous_version=([\w\.\-]+)", first)
            if cat in ("bad_deployment", "config_change") and prev:
                return json.dumps({"action": {"type": "rollback", "service": svc, "version": prev.group(1),
                                              "note": "revert"}})
            return json.dumps({"action": {"type": "mitigate", "service": svc, "version": None, "note": "manual"}})

        # --- tool-using roles ("react*") ---
        n_obs = sum(1 for m in messages if m["content"].startswith("OBSERVATION"))
        schema = re.search(r"FINAL_SCHEMA: (\w+)", system)
        schema = schema.group(1) if schema else "diagnosis"
        plan = [("get_deployments", {"service": svc, "time_range": "last_12h"}),
                ("query_logs", {"service": svc, "time_range": "last_12h"}),
                ("query_metrics", {"service": svc, "metric_name": "memory_usage"}),
                ("query_metrics", {"service": svc, "metric_name": "error_rate"})]
        limit = 1 if schema == "branch" else (2 if schema == "verdict" else len(plan))
        if n_obs < limit:
            tool, args = plan[n_obs]
            cat = guess_category(obs + first) if obs else "bad_deployment"
            return json.dumps({"thought": "gather", "hypothesis": {"category": cat, "confidence": 0.5},
                               "action": {"tool": tool, "args": args}})
        cat = guess_category(obs)
        if schema == "verdict":
            m = re.search(r"Hypothesis under review: category=(\w+)", first)
            claimed = m.group(1) if m else cat
            ok = claimed == cat
            return json.dumps({"final": {"verdict": "support" if ok else "reject",
                                         "reasons": ["stub check"], "contradictions": [] if ok else [f"evidence suggests {cat}"],
                                         "suggested_checks": []}})
        if schema == "branch":
            m = re.search(r"Hypothesis to test: (\w+)", first)
            h = m.group(1) if m else cat
            d = self._diag(h, obs, cites)
            d["score"] = 0.8 if h == cat else 0.2
            return json.dumps({"final": d})
        return json.dumps({"final": self._diag(cat, obs, cites)})

    @staticmethod
    def _diag(cat, obs, cites):
        prev = re.search(r"previous_version=([\w\.\-]+)", obs)
        svc = re.search(r"DEPLOYMENTS (\S+)", obs)
        act = {"type": "none", "service": None, "version": None, "note": ""}
        if cat in ("bad_deployment", "config_change") and prev and svc:
            act = {"type": "rollback", "service": svc.group(1), "version": prev.group(1), "note": "revert"}
        line = next((l for l in obs.splitlines() if "ERROR" in l or "WARN" in l), obs[:100])
        return {"category": cat, "root_cause": f"{cat}: {line[:160]}", "evidence": [line[:160]],
                "citations": cites, "recommended_action": act}
