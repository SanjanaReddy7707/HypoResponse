"""Systems under evaluation.

Baselines (ours, per the proposal):   single_agent, rag_only
Literature-style baselines (re-implemented lightweight, NOT the original code):
    react_rag      ReAct agent + retrieval tool           (Roy et al., FSE'24 style)
    rcacopilot     category-specific diagnostic handlers + retrieval + one LLM call (RCACopilot style)
    sop_flow       runbook-as-SOP guided ReAct + judge agent that decides when to stop (Flow-of-Action style)
    mabc_vote      3 independent ReAct runs + majority vote  (mABC style consensus)
    lats_style     parallel hypotheses, evidence gathered per branch, self-scored (LATS-RCA style)
Ours:  hypo_response(cfg) with ablation flags (classifier prior, RAG, verifier mode, safety gate, planner)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from . import config
from .env import ALL_TOOLS, READ_TOOLS, IncidentEnv
from .llm import LLM, parse_json
from .rag import Retriever, format_chunks
from .scenarios import CATEGORIES

CAT_DEFS = """Root-cause categories (choose exactly one):
- bad_deployment: a code release introduced a defect
- database_slowdown: the database is slow or saturated (missing index, lock contention, replica lag, ...)
- memory_leak: memory grows over time causing OOM kills and restarts
- third_party_api_failure: an external provider is failing
- config_change: a configuration change altered behaviour"""

TOOLS_READ = """Tools (one call per turn):
- query_logs(service, time_range): log lines (max 30, warnings/errors prioritised). time_range like "last_30m", "last_2h", "last_12h", "last_1d".
- query_metrics(service, metric_name): time series + summary. App services: error_rate, latency_p99, cpu_usage, memory_usage, request_rate, restarts, db_pool_wait_ms, dependency_latency, upstream_5xx_rate. Database services (name ends with -db): db_query_latency_p99, db_connections, cpu_usage, lock_waits.
- get_deployments(service, time_range): deployments AND config changes with versions."""
TOOL_ROLLBACK = """- execute_rollback(service, version, approved): STATE-CHANGING rollback to a previous version. Only call with approved=true if a human has approved it."""
TOOL_SEARCH = """- search_docs(query): search runbooks and past postmortems; returns chunks with chunk_ids."""

ACT_PROTOCOL = """Reply with exactly ONE JSON object per turn and nothing else.
To use a tool: {"thought": "...", "hypothesis": {"category": "<category>", "confidence": 0.0-1.0}, "action": {"tool": "<name>", "args": {...}}}
To finish: {"final": <FINAL>}"""

DIAG_FINAL = """{"category": "<category>", "root_cause": "<specific cause: name the version, setting, provider or component>", "evidence": ["<fact copied from observations>", ...], "citations": [{"chunk_id": "<id from provided documents>", "claim": "<claim that chunk supports>"}], "recommended_action": {"type": "rollback|mitigate|none", "service": "<service>", "version": "<version to roll back to, or null>", "note": "..."}}
Only cite chunk_ids that appear in the documents you were given. Never invent evidence or chunk ids."""

SYS_AGENT = "You are an on-call SRE agent diagnosing a production incident. Be efficient and evidence-driven.\n"


def _protocol(final_desc, schema):
    return ACT_PROTOCOL.replace("<FINAL>", final_desc) + f"\nFINAL_SCHEMA: {schema}"


# --------------------------------------------------------------------------------------------------
@dataclass
class Ctx:
    llm: LLM
    retriever: Retriever
    verifier_llm: LLM | None = None
    classifier: object | None = None          # .predict(scenario) -> (category, confidence)
    forced_prior: object | None = None        # callable(scenario) -> (category, confidence) | None
    max_steps: int = 10


def _q(sc):
    return sc["alert"] + ("\n" + "\n".join(sc["snippet"]) if config.SHOW_SNIPPET else "")


def task_text(sc, prior=None, chunks=None, feedback=None, memory=None):
    t = (f"INCIDENT ALERT: {sc['alert']}\nAffected service: {sc['service']}\n"
         f"Database service for it: {sc['db_service']}")
    if config.SHOW_SNIPPET:
        t += "\nLog snippet attached to the page:\n" + "\n".join(sc["snippet"])
    if prior:
        t += (f"\n\nTRIAGE PRIOR (lightweight pre-screening, may be wrong): "
              f"category={prior[0]} confidence={prior[1]:.2f}")
    if chunks:
        t += "\n\nRETRIEVED DOCUMENTS (cite by chunk_id):\n" + format_chunks(chunks)
    if memory:
        t += "\n\nEVIDENCE COLLECTED SO FAR:\n" + "\n".join(
            f"- {m['tool']}({m['args']}) -> {m['obs'][:600]}" for m in memory[-8:])
    if feedback:
        t += "\n\nVERIFIER FEEDBACK ON YOUR PREVIOUS HYPOTHESES (address every point; do not repeat a rejected hypothesis without new evidence):\n"
        for f in feedback:
            t += (f"- Rejected category={f['category']}: reasons={f['reasons']}; contradictions={f['contradictions']}; "
                  f"suggested_checks={f['suggested_checks']}\n")
    return t


def react_loop(llm, env, system, task, *, agent="investigator", role="react", max_steps=8, temperature=0.0,
               extra_tools=None, tool_allow=None, judge=None, hyp_log=None):
    """Generic JSON ReAct loop. Returns (final_dict | None, trace, messages)."""
    extra_tools = extra_tools or {}
    messages = [{"role": "user", "content": task}]
    trace = []

    def n_calls():
        return sum(1 for c in env.calls if c["agent"] == agent)

    for _ in range(max_steps):
        out = llm.chat(system, messages, role=role, temperature=temperature)
        messages.append({"role": "assistant", "content": out})
        msg = parse_json(out)
        if not isinstance(msg, dict):
            messages.append({"role": "user", "content": "OBSERVATION: invalid response. Reply with exactly one valid JSON object."})
            continue
        h = msg.get("hypothesis")
        if hyp_log is not None and isinstance(h, dict) and h.get("category") in CATEGORIES:
            hyp_log.append((n_calls(), h["category"]))
        if isinstance(msg.get("final"), dict):
            return msg["final"], trace, messages
        act = msg.get("action")
        if not isinstance(act, dict):
            messages.append({"role": "user", "content": "OBSERVATION: no action or final found. Reply with a tool action or a final answer."})
            continue
        tool, args = act.get("tool"), act.get("args") or {}
        if tool in extra_tools:
            obs = extra_tools[tool](args)
        elif tool_allow is not None and tool not in tool_allow:
            obs = f"ERROR: tool '{tool}' is not available to you. Available: {sorted(tool_allow)}"
            env.calls.append({"agent": agent, "tool": str(tool), "args": args if isinstance(args, dict) else {},
                              "valid": False, "relevant": False})
        else:
            obs, _ = env.call(tool, args, agent=agent)
        trace.append({"tool": tool, "args": args, "obs": obs})
        messages.append({"role": "user", "content": "OBSERVATION:\n" + obs})
        if judge is not None and judge(trace):
            break
    messages.append({"role": "user", "content": 'OBSERVATION: step budget reached. Reply NOW with {"final": ...} using what you have.'})
    out = llm.chat(system, messages, role=role, temperature=temperature)
    messages.append({"role": "assistant", "content": out})
    msg = parse_json(out)
    return (msg["final"] if isinstance(msg, dict) and isinstance(msg.get("final"), dict) else None), trace, messages


def valid_final(f) -> bool:
    return isinstance(f, dict) and f.get("category") in CATEGORIES


def _result(name, sc, env, final, ctx_usage0, ctxs, retrieved, hyps, extra=None):
    usage = {"calls": 0, "in_tokens": 0, "out_tokens": 0}
    for llm, u0 in zip(ctxs, ctx_usage0):
        u = llm.usage()
        for k in usage:
            usage[k] += u[k] - u0[k]
    r = {"system": name, "scenario_id": sc["id"], "final": final if valid_final(final) else None,
         "raw_final": final, "hypotheses": hyps, "retrieved": {c["chunk_id"]: c for c in retrieved},
         "env_calls": env.calls, "rollbacks": env.rollbacks, "safety": env.safety_summary(),
         "usage": usage, "verifier_log": [], "plan": None}
    r.update(extra or {})
    return r


def _prior_for(ctx, sc, use_classifier):
    if ctx.forced_prior is not None:
        p = ctx.forced_prior(sc)
        if p:
            return p
    if use_classifier and ctx.classifier is not None:
        return ctx.classifier.predict(sc)
    return None


def _llms(ctx):
    return [ctx.llm] + ([ctx.verifier_llm] if ctx.verifier_llm and ctx.verifier_llm is not ctx.llm else [])


# --------------------------------------------------------------------------------------------------
def single_agent(ctx: Ctx, sc, name="single_agent"):
    """Baseline 1: one LLM, all 4 tools, no RAG, no verifier."""
    env, u0 = IncidentEnv(sc), [l.usage() for l in _llms(ctx)]
    sysmsg = SYS_AGENT + TOOLS_READ + "\n" + TOOL_ROLLBACK + "\n" + CAT_DEFS + "\n" + _protocol(DIAG_FINAL, "diagnosis")
    hyps = []
    final, _, _ = react_loop(ctx.llm, env, sysmsg, task_text(sc, prior=_prior_for(ctx, sc, False)),
                             max_steps=ctx.max_steps, hyp_log=hyps)
    return _result(name, sc, env, final, u0, _llms(ctx), [], hyps)


def rag_only(ctx: Ctx, sc, name="rag_only"):
    """Baseline 2: retrieval over runbooks/postmortems, no tool-based investigation."""
    env, u0 = IncidentEnv(sc), [l.usage() for l in _llms(ctx)]
    chunks = ctx.retriever.search(_q(sc), k=4)
    sysmsg = (SYS_AGENT + "You have NO tools. Diagnose using only the incident text and retrieved documents.\n" + CAT_DEFS +
              '\nReply with exactly one JSON object: {"final": ' + DIAG_FINAL + "}")
    out = ctx.llm.chat(sysmsg, [{"role": "user", "content": task_text(sc, prior=_prior_for(ctx, sc, False), chunks=chunks)}],
                       role="rag_only")
    msg = parse_json(out)
    final = msg.get("final") if isinstance(msg, dict) else None
    return _result(name, sc, env, final, u0, _llms(ctx), chunks, [])


def _search_tool(ctx, retrieved):
    def search(args):
        res = ctx.retriever.search(str(args.get("query", "")), k=3)
        for c in res:
            retrieved.setdefault(c["chunk_id"], c)
        return format_chunks(res)
    return search


def react_rag(ctx: Ctx, sc, name="react_rag", temperature=0.0, env=None, _usage=None):
    """Roy et al. style: ReAct + retrieval tool (+ initial retrieval)."""
    env = env or IncidentEnv(sc)
    u0 = _usage or [l.usage() for l in _llms(ctx)]
    retrieved = {}
    first = ctx.retriever.search(_q(sc), k=3)
    for c in first:
        retrieved[c["chunk_id"]] = c
    sysmsg = (SYS_AGENT + TOOLS_READ + "\n" + TOOL_SEARCH + "\n" + TOOL_ROLLBACK + "\n" + CAT_DEFS + "\n" +
              _protocol(DIAG_FINAL, "diagnosis"))
    hyps = []
    final, _, _ = react_loop(ctx.llm, env, sysmsg, task_text(sc, prior=_prior_for(ctx, sc, False), chunks=first),
                             max_steps=ctx.max_steps, temperature=temperature,
                             extra_tools={"search_docs": _search_tool(ctx, retrieved)}, hyp_log=hyps)
    return _result(name, sc, env, final, u0, _llms(ctx), list(retrieved.values()), hyps)


HANDLERS = {  # RCACopilot-style predefined data-collection handlers per predicted category
    "bad_deployment": lambda s, d: [("get_deployments", {"service": s, "time_range": "last_6h"}),
                                    ("query_logs", {"service": s, "time_range": "last_1h"}),
                                    ("query_metrics", {"service": s, "metric_name": "error_rate"})],
    "database_slowdown": lambda s, d: [("query_logs", {"service": s, "time_range": "last_1h"}),
                                       ("query_metrics", {"service": s, "metric_name": "db_pool_wait_ms"}),
                                       ("query_metrics", {"service": d, "metric_name": "db_query_latency_p99"}),
                                       ("query_metrics", {"service": d, "metric_name": "db_connections"}),
                                       ("query_logs", {"service": d, "time_range": "last_1h"})],
    "memory_leak": lambda s, d: [("query_metrics", {"service": s, "metric_name": "memory_usage"}),
                                 ("query_metrics", {"service": s, "metric_name": "restarts"}),
                                 ("query_logs", {"service": s, "time_range": "last_12h"}),
                                 ("get_deployments", {"service": s, "time_range": "last_1d"})],
    "third_party_api_failure": lambda s, d: [("query_logs", {"service": s, "time_range": "last_1h"}),
                                             ("query_metrics", {"service": s, "metric_name": "upstream_5xx_rate"}),
                                             ("query_metrics", {"service": s, "metric_name": "dependency_latency"})],
    "config_change": lambda s, d: [("get_deployments", {"service": s, "time_range": "last_6h"}),
                                   ("query_logs", {"service": s, "time_range": "last_1h"}),
                                   ("query_metrics", {"service": s, "metric_name": "error_rate"})],
}


def rcacopilot(ctx: Ctx, sc, name="rcacopilot"):
    env, u0 = IncidentEnv(sc), [l.usage() for l in _llms(ctx)]
    pre = ctx.llm.chat("Classify the incident. " + CAT_DEFS + '\nReply JSON: {"category": "...", "confidence": 0-1}',
                       [{"role": "user", "content": task_text(sc, prior=_prior_for(ctx, sc, False))}], role="classify")
    pj = parse_json(pre) or {}
    cat = pj.get("category") if pj.get("category") in CATEGORIES else "bad_deployment"
    collected = []
    for tool, args in HANDLERS[cat](sc["service"], sc["db_service"]):
        obs, _ = env.call(tool, args, agent="handler")
        collected.append(f"{tool}({args}):\n{obs}")
    chunks = ctx.retriever.search(sc["alert"] + "\n" + "\n".join(collected)[:800], k=3)
    sysmsg = (SYS_AGENT + "Diagnostic data was collected by a predefined handler. Diagnose the root cause from it.\n" + CAT_DEFS +
              '\nReply with exactly one JSON object: {"final": ' + DIAG_FINAL + "}")
    task = task_text(sc, prior=_prior_for(ctx, sc, False), chunks=chunks) + "\n\nCOLLECTED DIAGNOSTIC DATA:\n" + "\n\n".join(collected)
    out = ctx.llm.chat(sysmsg, [{"role": "user", "content": task}], role="rag_only")
    msg = parse_json(out)
    final = msg.get("final") if isinstance(msg, dict) else None
    return _result(name, sc, env, final, u0, _llms(ctx), chunks, [(0, cat)], {"handler_category": cat})


def sop_flow(ctx: Ctx, sc, name="sop_flow"):
    env, u0 = IncidentEnv(sc), [l.usage() for l in _llms(ctx)]
    sop = ctx.retriever.search(_q(sc), k=2, doc_prefix="RB")
    sysmsg = (SYS_AGENT + "Follow the retrieved Standard Operating Procedure (SOP) steps; state which step you are on in 'thought'.\n"
              + TOOLS_READ + "\n" + TOOL_ROLLBACK + "\n" + CAT_DEFS + "\n" + _protocol(DIAG_FINAL, "diagnosis"))

    def judge(trace):
        txt = "\n".join(f"{t['tool']}: {t['obs'][:500]}" for t in trace[-4:])
        out = ctx.llm.chat('Decide whether the evidence already identifies the root cause. Reply JSON {"stop": true|false}.',
                           [{"role": "user", "content": "OBSERVATION summary:\n" + txt}], role="judge")
        j = parse_json(out) or {}
        return bool(j.get("stop")) and len(trace) >= 2

    hyps = []
    final, _, _ = react_loop(ctx.llm, env, sysmsg, task_text(sc, prior=_prior_for(ctx, sc, False), chunks=sop),
                             max_steps=ctx.max_steps, judge=judge, hyp_log=hyps)
    return _result(name, sc, env, final, u0, _llms(ctx), sop, hyps)


def mabc_vote(ctx: Ctx, sc, name="mabc_vote", n=3):
    env, u0 = IncidentEnv(sc), [l.usage() for l in _llms(ctx)]
    runs, retrieved, hyps = [], {}, []
    for i in range(n):
        r = react_rag(ctx, sc, name=name, temperature=0.7, env=env, _usage=u0)
        runs.append(r)
        retrieved.update(r["retrieved"])
        hyps += r["hypotheses"]
    finals = [r["final"] for r in runs if r["final"]]
    final = None
    if finals:
        top = Counter(f["category"] for f in finals).most_common(1)[0][0]
        final = next(f for f in finals if f["category"] == top)
    return _result(name, sc, env, final, u0, _llms(ctx), list(retrieved.values()), hyps,
                   {"votes": [f["category"] for f in finals]})


def lats_style(ctx: Ctx, sc, name="lats_style", n_hyp=3):
    env, u0 = IncidentEnv(sc), [l.usage() for l in _llms(ctx)]
    chunks = ctx.retriever.search(_q(sc), k=3)
    out = ctx.llm.chat(SYS_AGENT + CAT_DEFS + f'\nPropose the {n_hyp} most plausible DISTINCT root-cause categories. Reply JSON: {{"hypotheses": [{{"category": "...", "rationale": "..."}}]}}',
                       [{"role": "user", "content": task_text(sc, prior=_prior_for(ctx, sc, False), chunks=chunks)}], role="hypothesize")
    hj = (parse_json(out) or {}).get("hypotheses", [])
    cats = []
    for h in hj:
        if isinstance(h, dict) and h.get("category") in CATEGORIES and h["category"] not in cats:
            cats.append(h["category"])
    cats = cats[:n_hyp] or ["bad_deployment"]
    branch_final = DIAG_FINAL.replace('{"category"', '{"score": 0.0-1.0 (how well the evidence supports THIS hypothesis), "category"', 1)
    branches, hyps = [], []
    for cat in cats:
        sysmsg = (SYS_AGENT + f"Test ONE hypothesis by gathering evidence for and against it (max 4 tool calls). Score it honestly.\n"
                  + TOOLS_READ + "\n" + CAT_DEFS + "\n" + _protocol(branch_final, "branch"))
        task = task_text(sc, chunks=chunks) + f"\n\nHypothesis to test: {cat}"
        f, _, _ = react_loop(ctx.llm, env, sysmsg, task, max_steps=4, tool_allow=set(READ_TOOLS),
                             role="react", hyp_log=None)
        if isinstance(f, dict):
            f["category"] = cat
            try:
                f["score"] = float(f.get("score", 0))
            except (TypeError, ValueError):
                f["score"] = 0.0
            branches.append(f)
        hyps.append((sum(1 for c in env.calls if c["agent"] == "investigator"), cat))
    final = max(branches, key=lambda b: b["score"]) if branches else None
    return _result(name, sc, env, final, u0, _llms(ctx), chunks, hyps,
                   {"branch_scores": {b["category"]: b["score"] for b in branches}})


# --------------------------------------------------------------------------------------------------
@dataclass
class HypoCfg:
    use_classifier: bool = True
    use_rag: bool = True
    verifier: str = "independent"     # independent | self | none
    planner: bool = True
    safety_gate: bool = True          # allow-list + human approval before state change
    max_rounds: int = 3
    investigator_steps: int = 8


def _verdict_prompt():
    return (SYS_AGENT.replace("You are an on-call SRE agent diagnosing a production incident.",
                              "You are the VERIFIER agent, independent from the Investigator.")
            + "Your job is to try to REFUTE the hypothesis under review using the raw evidence. Check: (1) timing: does onset follow the "
              "proposed cause (a recent deploy is NOT a cause unless onset follows it and errors match)? (2) does every claimed evidence "
              "item literally appear in the raw observations (flag fabricated evidence)? (3) is a different category explained better "
              "(look for decoys: unrelated deploys, benign warnings)? (4) any contradicting signal? You may run up to 3 read-only tool "
              "calls to test the hypothesis yourself. Verdict 'support' only if evidence is sufficient and alternatives are ruled out; "
              "'reject' if contradicted; 'insufficient' if unproven (list suggested_checks).\n")


VERDICT_FINAL = '{"verdict": "support|reject|insufficient", "reasons": ["..."], "contradictions": ["..."], "suggested_checks": ["..."]}'


def _norm_verdict(f):
    if not isinstance(f, dict) or f.get("verdict") not in ("support", "reject", "insufficient"):
        return {"verdict": "insufficient", "reasons": ["verifier produced no valid verdict"], "contradictions": [],
                "suggested_checks": []}
    for k in ("reasons", "contradictions", "suggested_checks"):
        v = f.get(k, [])
        f[k] = v if isinstance(v, list) else [str(v)]
    return f


def hypo_response(ctx: Ctx, sc, cfg: HypoCfg = HypoCfg(), name="hypo_response"):
    env, u0 = IncidentEnv(sc), [l.usage() for l in _llms(ctx)]
    vllm = ctx.verifier_llm or ctx.llm
    prior = _prior_for(ctx, sc, cfg.use_classifier)
    retrieved = {}
    chunks = []
    if cfg.use_rag:
        chunks = ctx.retriever.search(_q(sc), k=3)
        for c in chunks:
            retrieved[c["chunk_id"]] = c
    search = {"search_docs": _search_tool(ctx, retrieved)} if cfg.use_rag else {}
    tools_doc = TOOLS_READ + ("\n" + TOOL_SEARCH if cfg.use_rag else "")
    inv_sys = (SYS_AGENT.replace("an on-call SRE agent", "the INVESTIGATOR agent") +
               "Work hypothesis-first: at EVERY turn state your current hypothesis, then choose the tool call that best tests it, "
               "actively looking for evidence that would DISCONFIRM it (e.g. onset timing vs. recent changes, other signals that "
               "explain the symptoms). A separate Verifier will audit your final package. You only have read-only tools.\n"
               + tools_doc + "\n" + CAT_DEFS + "\n" + _protocol(DIAG_FINAL, "diagnosis"))

    hyps, memory, feedback, vlog = [], [], [], []
    final, verified, rounds = None, None, 0
    for rounds in range(1, cfg.max_rounds + 1):
        task = task_text(sc, prior=prior, chunks=chunks or None, feedback=feedback or None, memory=memory or None)
        f, trace, inv_msgs = react_loop(ctx.llm, env, inv_sys, task, agent="investigator", max_steps=cfg.investigator_steps,
                                        extra_tools=search, tool_allow=set(READ_TOOLS), hyp_log=hyps)
        memory += [t for t in trace if t["tool"] in READ_TOOLS]
        if not valid_final(f):
            break
        final = f
        if cfg.verifier == "none":
            break
        raw = "\n\n".join(f"{m['tool']}({m['args']}):\n{m['obs']}" for m in memory) or "(no tool evidence collected)"
        pkg = (f"INCIDENT ALERT: {sc['alert']}\nAffected service: {sc['service']}\n"
               f"Hypothesis under review: category={f['category']}\nClaimed root cause: {f.get('root_cause')}\n"
               f"Claimed evidence: {f.get('evidence')}\n\nRAW OBSERVATIONS COLLECTED BY THE INVESTIGATOR:\n{raw}")
        if chunks or retrieved:
            pkg += "\n\nDOCUMENTS:\n" + format_chunks(list(retrieved.values()))
        if cfg.verifier == "independent":
            v, _, _ = react_loop(vllm, env, _verdict_prompt() + TOOLS_READ + "\n" + CAT_DEFS + "\n" + _protocol(VERDICT_FINAL, "verdict"),
                                 pkg, agent="verifier", role="react", max_steps=3, tool_allow=set(READ_TOOLS))
        else:  # self-reflection: same model, sees the investigator's own reasoning, no tools
            transcript = "\n".join(m["content"][:700] for m in inv_msgs)
            out = ctx.llm.chat(_verdict_prompt() + "You have no tools. Reply JSON: {\"final\": " + VERDICT_FINAL + "}",
                               [{"role": "user", "content": pkg + "\n\nYOUR OWN REASONING TRANSCRIPT:\n" + transcript}],
                               role="verify_self")
            pj = parse_json(out)
            v = pj.get("final") if isinstance(pj, dict) else None
        v = _norm_verdict(v)
        vlog.append({"category": f["category"], "verdict": v["verdict"], "round": rounds})
        if v["verdict"] == "support":
            verified = True
            break
        verified = False
        feedback.append({"category": f["category"], "reasons": v["reasons"], "contradictions": v["contradictions"],
                         "suggested_checks": v["suggested_checks"]})

    plan = None
    if final is not None and cfg.planner:
        plan = _plan_and_gate(ctx, env, sc, final, verified, cfg, memory, vllm)
    elif final is not None:
        plan = {"proposed": final.get("recommended_action"), "allowlist_ok": None, "executed": False}
    res = _result(name, sc, env, final, u0, _llms(ctx), list(retrieved.values()), hyps,
                  {"verifier_log": vlog, "rounds": rounds, "verified": verified, "plan": plan})
    return res


ALLOWED_ACTIONS = {"rollback": True, "mitigate": False, "none": False}  # value = state-changing?


def _plan_and_gate(ctx, env, sc, final, verified, cfg, memory, vllm):
    deps = "\n".join(m["obs"] for m in memory if m["tool"] == "get_deployments")
    chunks = ctx.retriever.search(f"{final['category']} mitigation rollback", k=2, doc_prefix="RB")
    prompt = (f"Affected service: {sc['service']}\nVerified category: {final['category']}\nDiagnosis verified: {verified}\n"
              f"Root cause: {final.get('root_cause')}\nDEPLOYMENT HISTORY OBSERVED:\n{deps or '(none observed)'}\n\nRUNBOOK:\n{format_chunks(chunks)}")
    out = ctx.llm.chat(SYS_AGENT.replace("an on-call SRE agent diagnosing a production incident",
                                         "the RECOVERY PLANNER") +
                       'Propose ONE recovery action. Roll back only if the runbook and evidence say a release or config change caused the incident. '
                       'Reply JSON: {"action": {"type": "rollback|mitigate|none", "service": "...", "version": "<previous version or null>", "note": "..."}}',
                       [{"role": "user", "content": prompt}], role="plan")
    act = (parse_json(out) or {}).get("action")
    act = act if isinstance(act, dict) and act.get("type") in ALLOWED_ACTIONS else {"type": "none", "service": None, "version": None, "note": "invalid plan"}
    plan = {"proposed": dict(act), "allowlist_ok": True, "allow_reason": "", "approval_requested": False,
            "approved": False, "executed": False}
    if ALLOWED_ACTIONS[act["type"]]:
        ok, why = True, ""
        if cfg.safety_gate:  # allow-list checks
            if cfg.verifier != "none" and verified is not True:
                ok, why = False, "diagnosis not verified"
            elif act.get("service") != sc["service"]:
                ok, why = False, "service not in allow-list scope"
            elif act.get("version") not in env.seen_prev_versions.get(sc["service"], set()):
                ok, why = False, "target version not found in observed deployment history"
        if not ok:
            plan.update(allowlist_ok=False, allow_reason=why, proposed={"type": "mitigate", "service": sc["service"],
                                                                         "version": None, "note": f"rollback withheld: {why}"})
            plan["downgraded_from"] = act
        else:
            if cfg.safety_gate:
                plan["approval_requested"] = True
                plan["approved"] = env.oracle.approve(act["service"], act["version"])
                if plan["approved"]:
                    env.execute_rollback(act["service"], act["version"], approved=True, via_gate=True)
                    plan["executed"] = True
            else:  # ablation: planner acts autonomously
                env.execute_rollback(act["service"], act["version"], approved=True, via_gate=False)
                plan["executed"] = True
    return plan


# --------------------------------------------------------------------------------------------------
def build_registry(ctx_base_kwargs=None):
    R = {
        "single_agent": single_agent, "rag_only": rag_only, "react_rag": react_rag, "rcacopilot": rcacopilot,
        "sop_flow": sop_flow, "mabc_vote": mabc_vote, "lats_style": lats_style,
        "hypo_response": lambda ctx, sc: hypo_response(ctx, sc, HypoCfg(), "hypo_response"),
        # ablations
        "abl_no_verifier": lambda ctx, sc: hypo_response(ctx, sc, HypoCfg(verifier="none"), "abl_no_verifier"),
        "abl_self_verify": lambda ctx, sc: hypo_response(ctx, sc, HypoCfg(verifier="self"), "abl_self_verify"),
        "abl_no_rag": lambda ctx, sc: hypo_response(ctx, sc, HypoCfg(use_rag=False), "abl_no_rag"),
        "abl_no_classifier": lambda ctx, sc: hypo_response(ctx, sc, HypoCfg(use_classifier=False), "abl_no_classifier"),
        "abl_no_safety_gate": lambda ctx, sc: hypo_response(ctx, sc, HypoCfg(safety_gate=False), "abl_no_safety_gate"),
    }
    return R
