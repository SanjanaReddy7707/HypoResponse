#!/usr/bin/env python
"""Aggregate results/raw.jsonl into tables (markdown) and plots.

  python analyze.py --raw results/raw.jsonl --out results/summary
Scenario-level scores are first averaged over runs; CIs are bootstrap over scenarios.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hypo.metrics import bootstrap_ci, paired_bootstrap_diff

BOOL_METRICS = ["correct", "correct_strict", "completed", "fell_for_decoy", "plan_correct", "harmful_proposal",
                "unsafe_attempt", "unapproved_state_change", "harmful_rollback"]
NUM_METRICS = ["steps_to_correct", "wrong_before_correct", "n_tool_calls", "tool_valid_rate", "tool_relevant_rate",
               "required_tool_coverage", "citation_faithfulness", "llm_calls", "tokens"]


def load(path, injected=False):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [r for r in rows if r["injected_prior"] == injected]


def per_scenario(rows, system, metric, diff=None):
    """scenario_id -> mean over runs (None values skipped)."""
    acc = defaultdict(list)
    for r in rows:
        if r["system"] == system and (diff is None or r["difficulty"] == diff):
            v = r["metrics"].get(metric)
            if v is not None:
                acc[r["scenario_id"]].append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def fmt(ci, pct=False):
    m, lo, hi = ci
    if m is None:
        return "n/a"
    return f"{100*m:.1f}% [{100*lo:.0f}-{100*hi:.0f}]" if pct else f"{m:.2f} [{lo:.2f}-{hi:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.jsonl")
    ap.add_argument("--out", default="results/summary")
    ap.add_argument("--injected", action="store_true", help="analyse the wrong-prior-injection run instead")
    ap.add_argument("--ref", default="hypo_response")
    a = ap.parse_args()
    rows = load(a.raw, a.injected)
    systems = sorted({r["system"] for r in rows}, key=lambda s: [r["system"] for r in rows].index(s))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = []

    md.append("## Main results (mean over runs, 95% bootstrap CI over scenarios)\n")
    md.append("| System | Root-cause acc. | Strict acc. | Completion | Fell for decoy | Steps to correct | Wrong hyps before correct | Plan correct | Tokens |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s in systems:
        g = lambda m, pct=True: fmt(bootstrap_ci(list(per_scenario(rows, s, m).values())), pct)
        md.append(f"| {s} | {g('correct')} | {g('correct_strict')} | {g('completed')} | {g('fell_for_decoy')} | "
                  f"{g('steps_to_correct', False)} | {g('wrong_before_correct', False)} | {g('plan_correct')} | {g('tokens', False)} |")

    md.append("\n## Safety (lower is better; rates over scenarios)\n")
    md.append("| System | Unsafe attempt (no approval flag) | Unapproved state change | Harmful rollback executed | Harmful rollback proposed |")
    md.append("|---|---|---|---|---|")
    for s in systems:
        g = lambda m: fmt(bootstrap_ci(list(per_scenario(rows, s, m).values())), True)
        md.append(f"| {s} | {g('unsafe_attempt')} | {g('unapproved_state_change')} | {g('harmful_rollback')} | {g('harmful_proposal')} |")

    md.append("\n## Capability-specific metrics\n")
    md.append("| System | Citation faithfulness | Tool-call validity | Tool-call relevance | Required-tool coverage |")
    md.append("|---|---|---|---|---|")
    for s in systems:
        g = lambda m: fmt(bootstrap_ci(list(per_scenario(rows, s, m).values())), True)
        md.append(f"| {s} | {g('citation_faithfulness')} | {g('tool_valid_rate')} | {g('tool_relevant_rate')} | {g('required_tool_coverage')} |")

    md.append("\n## Root-cause accuracy by difficulty\n")
    diffs = ["easy", "ambiguous", "adversarial"]
    md.append("| System | " + " | ".join(diffs) + " |")
    md.append("|---|" + "---|" * len(diffs))
    for s in systems:
        cells = [fmt(bootstrap_ci(list(per_scenario(rows, s, "correct", d).values())), True) for d in diffs]
        md.append(f"| {s} | " + " | ".join(cells) + " |")

    md.append("\n## Verifier behaviour (systems with a verifier)\n")
    md.append("| System | Hypothesis rejection rate | Wrong hypotheses caught | Correct hypotheses wrongly rejected |")
    md.append("|---|---|---|---|")
    for s in systems:
        rs = [r["metrics"] for r in rows if r["system"] == s]
        nv, nr = sum(m["n_verifications"] for m in rs), sum(m["n_rejections"] for m in rs)
        if not nv:
            continue
        ww, wc = sum(m["wrong_hyp_submitted"] for m in rs), sum(m["wrong_hyp_caught"] for m in rs)
        cr = sum(m["correct_hyp_rejected"] for m in rs)
        md.append(f"| {s} | {100*nr/nv:.1f}% ({nr}/{nv}) | {('%.1f%% (%d/%d)' % (100*wc/ww, wc, ww)) if ww else 'n/a'} | {cr} |")

    if a.ref in systems:
        md.append(f"\n## Paired comparison vs `{a.ref}` (accuracy diff = ref - other; 95% CI; P(diff<=0))\n")
        md.append("| Other system | Δ accuracy | 95% CI | P(Δ≤0) |")
        md.append("|---|---|---|---|")
        ref = per_scenario(rows, a.ref, "correct")
        for s in systems:
            if s == a.ref:
                continue
            o = per_scenario(rows, s, "correct")
            ids = sorted(set(ref) & set(o))
            if ids:
                d, lo, hi, p = paired_bootstrap_diff([ref[i] for i in ids], [o[i] for i in ids])
                md.append(f"| {s} | {100*d:+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] | {p:.3f} |")

    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    accs = [bootstrap_ci(list(per_scenario(rows, s, "correct").values())) for s in systems]
    ax[0].bar(range(len(systems)), [x[0] for x in accs],
              yerr=[[x[0] - x[1] for x in accs], [x[2] - x[0] for x in accs]], capsize=3)
    ax[0].set_xticks(range(len(systems)), systems, rotation=60, ha="right")
    ax[0].set_ylabel("root-cause accuracy")
    ax[0].set_ylim(0, 1)
    uns = [bootstrap_ci(list(per_scenario(rows, s, "unapproved_state_change").values()))[0] or 0 for s in systems]
    harm = [bootstrap_ci(list(per_scenario(rows, s, "harmful_proposal").values()))[0] or 0 for s in systems]
    w = 0.4
    ax[1].bar([i - w / 2 for i in range(len(systems))], uns, w, label="unapproved state change")
    ax[1].bar([i + w / 2 for i in range(len(systems))], harm, w, label="harmful rollback proposed")
    ax[1].set_xticks(range(len(systems)), systems, rotation=60, ha="right")
    ax[1].legend()
    ax[1].set_ylabel("rate")
    plt.tight_layout()
    plt.savefig(out / "overview.png", dpi=150)
    print("wrote", out / "summary.md", out / "overview.png")


if __name__ == "__main__":
    main()
