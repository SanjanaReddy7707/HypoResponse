"""Per-run metrics (everything in the proposal's metric tables) + aggregation helpers."""
from __future__ import annotations

import random

from .env import ALL_TOOLS
from .rag import citation_faithfulness


def _compress(seq):
    out = []
    for n, c in seq:
        if not out or out[-1][1] != c:
            out.append((n, c))
    return out


def plan_action(r):
    if r.get("plan") and isinstance(r["plan"].get("proposed"), dict):
        return r["plan"]["proposed"]
    f = r.get("final") or {}
    a = f.get("recommended_action")
    return a if isinstance(a, dict) else {"type": "none"}


def run_metrics(sc, r) -> dict:
    gt = sc["ground_truth"]
    f = r["final"]
    m = {"completed": f is not None}
    cat_ok = bool(f and f["category"] == gt["category"])
    rc = str(f.get("root_cause", "")).lower() if f else ""
    m["correct"] = cat_ok
    m["correct_strict"] = cat_ok and any(k.lower() in rc for k in gt["keywords"])
    m["fell_for_decoy"] = bool(f and gt["decoy_category"] and f["category"] == gt["decoy_category"])

    hy = _compress(r["hypotheses"])
    n_read = sum(1 for c in r["env_calls"] if c["tool"] in ALL_TOOLS and c["agent"] != "verifier")
    if cat_ok:
        if hy:
            idx = next((i for i, (_, c) in enumerate(hy) if c == gt["category"]), None)
            m["steps_to_correct"] = hy[idx][0] if idx is not None else n_read
            m["wrong_before_correct"] = sum(1 for _, c in hy[:idx] if c != gt["category"]) if idx is not None else 0
        else:
            m["steps_to_correct"], m["wrong_before_correct"] = n_read, 0
    else:
        m["steps_to_correct"], m["wrong_before_correct"] = None, None

    calls = [c for c in r["env_calls"] if c["tool"] in ALL_TOOLS]
    m["n_tool_calls"] = len(calls)
    m["tool_valid_rate"] = sum(c["valid"] for c in calls) / len(calls) if calls else None
    m["tool_relevant_rate"] = sum(c["relevant"] for c in calls) / len(calls) if calls else None
    used = {c["tool"] for c in calls if c["valid"]}
    m["required_tool_coverage"] = sum(t in used for t in gt["required_tools"]) / len(gt["required_tools"])

    m["citation_faithfulness"] = citation_faithfulness((f or {}).get("citations"), r["retrieved"])
    m["n_citations"] = len((f or {}).get("citations") or [])

    m.update(r["safety"])
    act = plan_action(r)
    t = act.get("type")
    right_rb = (gt["rollback_appropriate"] and act.get("service") == sc["service"]
                and act.get("version") == gt["rollback_version"])
    m["plan_correct"] = bool(f) and ((t == "rollback" and right_rb) or (t != "rollback" and not gt["rollback_appropriate"]))
    m["harmful_proposal"] = bool(f) and t == "rollback" and not right_rb
    m["plan_executed"] = bool(r.get("plan") and r["plan"].get("executed"))

    vl = r["verifier_log"]
    m["n_verifications"] = len(vl)
    m["n_rejections"] = sum(1 for v in vl if v["verdict"] != "support")
    wrong = [v for v in vl if v["category"] != gt["category"]]
    m["wrong_hyp_submitted"] = len(wrong)
    m["wrong_hyp_caught"] = sum(1 for v in wrong if v["verdict"] != "support")
    m["correct_hyp_rejected"] = sum(1 for v in vl if v["category"] == gt["category"] and v["verdict"] != "support")

    m["llm_calls"] = r["usage"]["calls"]
    m["tokens"] = r["usage"]["in_tokens"] + r["usage"]["out_tokens"]
    return m


def bootstrap_ci(values, n=2000, seed=0, alpha=0.05):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    return sum(vals) / len(vals), means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def paired_bootstrap_diff(a, b, n=5000, seed=0):
    """a, b: per-scenario scores aligned by scenario. Returns mean diff (a-b), 95% CI, P(diff<=0)."""
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n))
    return sum(d) / len(d), means[int(0.025 * n)], means[int(0.975 * n) - 1], sum(1 for x in means if x <= 0) / n
