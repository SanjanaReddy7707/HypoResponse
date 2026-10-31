import json
from hypo import scenarios as S
from hypo.env import IncidentEnv, parse_range
from hypo.llm import LLM, StubLLM, parse_json
from hypo.rag import Retriever, citation_faithfulness
from hypo.systems import Ctx, HypoCfg, hypo_response, single_agent
from hypo.metrics import run_metrics

SC = S.generate_test_set()


def sc_of(cat, difficulty=None):
    return next(s for s in SC if s["category"] == cat and (difficulty is None or s["difficulty"] == difficulty))


def test_dataset_shape():
    assert len(SC) == 30
    for c in S.CATEGORIES:
        assert sum(s["category"] == c for s in SC) == 6
    assert sum(s["difficulty"] == "easy" for s in SC) == 15
    assert all(s["ground_truth"]["required_tools"] for s in SC)


def test_no_train_test_phrase_leakage():
    train = S.generate_train_set(10)
    test_alerts = {s["alert"].split(".")[0].split(" ", 3)[-1][:25] for s in SC}
    import re
    openings_test = {re.sub(r"[\d\.]+", "N", s["alert"])[:30] for s in SC}
    openings_train = {re.sub(r"[\d\.]+", "N", s["alert"])[:30] for s in train}
    assert not (openings_test & openings_train)


def test_tools_validity():
    env = IncidentEnv(sc_of("bad_deployment"))
    s = env.sc["service"]
    assert env.call("query_logs", {"service": s, "time_range": "last_1h"})[1]
    assert not env.call("query_logs", {"service": "nope", "time_range": "last_1h"})[1]
    assert not env.call("query_logs", {"service": s, "time_range": "yesterday"})[1]
    assert not env.call("query_metrics", {"service": s, "metric_name": "bogus"})[1]
    assert parse_range("last_2h") == 120


def test_rollback_requires_flag_and_gate():
    sc = sc_of("bad_deployment")
    env = IncidentEnv(sc)
    v = sc["ground_truth"]["rollback_version"]
    env.call("execute_rollback", {"service": sc["service"], "version": v})  # no flag
    assert env.safety_summary()["unsafe_attempt"] and not env.rollbacks[0]["executed"]
    env.call("execute_rollback", {"service": sc["service"], "version": v, "approved": True})  # self-approved
    s = env.safety_summary()
    assert s["unapproved_state_change"] and not s["harmful_rollback"]
    # inappropriate scenario: rollback is harmful even if flagged
    sc2 = sc_of("third_party_api_failure")
    env2 = IncidentEnv(sc2)
    env2.call("execute_rollback", {"service": sc2["service"], "version": "v1.0.0", "approved": True})
    assert env2.safety_summary()["harmful_rollback"]


def test_oracle_only_approves_correct_rollback():
    sc = sc_of("config_change")
    env = IncidentEnv(sc)
    gt = sc["ground_truth"]
    assert env.oracle.approve(sc["service"], gt["rollback_version"])
    assert not env.oracle.approve(sc["service"], "cfg-0")
    sc2 = sc_of("database_slowdown")
    assert not IncidentEnv(sc2).oracle.approve(sc2["service"], "v1.0.0")


def test_json_parse_and_retrieval():
    assert parse_json('blah ```json\n{"a": {"b": 1}}\n``` x') == {"a": {"b": 1}}
    r = Retriever("data/corpus")
    top = r.search("memory usage rising OOMKilled restarts heap", k=2)
    assert top[0]["doc_id"] in ("RB-003", "PM-103")
    by = {c["chunk_id"]: c for c in r.chunks}
    cid = top[0]["chunk_id"]
    assert citation_faithfulness([{"chunk_id": cid, "claim": "memory usage rises steadily then OOMKilled restarts"}], by) == 1.0
    assert citation_faithfulness([{"chunk_id": "FAKE#1", "claim": "anything"}], by) == 0.0


class Scripted(LLM):
    """Investigator first claims the decoy category; verifier rejects it; second round is correct."""
    def __init__(self, decoy, truth, svc):
        super().__init__("scripted"); self.decoy, self.truth, self.svc = decoy, truth, svc; self.round = 0
    def chat(self, system, messages, role="agent", temperature=0.0, max_tokens=900):
        self.calls += 1
        first = messages[0]["content"]
        if "FINAL_SCHEMA: verdict" in system:
            claimed = first.split("category=")[1].split()[0] if "category=" in first else ""
            if claimed == self.decoy:
                return json.dumps({"final": {"verdict": "reject", "reasons": ["timing mismatch"], "contradictions": ["onset precedes deploy"], "suggested_checks": ["check db metrics"]}})
            return json.dumps({"final": {"verdict": "support", "reasons": ["ok"], "contradictions": [], "suggested_checks": []}})
        if role == "plan":
            return json.dumps({"action": {"type": "none", "service": None, "version": None, "note": ""}})
        if "VERIFIER FEEDBACK" in first:
            cat = self.truth
        else:
            cat = self.decoy
        return json.dumps({"final": {"category": cat, "root_cause": cat, "evidence": ["x"], "citations": [], "recommended_action": {"type": "none"}}})


def test_verifier_rejects_decoy_then_investigator_recovers():
    sc = sc_of("database_slowdown", "ambiguous")
    gt = sc["ground_truth"]
    llm = Scripted(gt["decoy_category"], gt["category"], sc["service"])
    ctx = Ctx(llm=llm, retriever=Retriever("data/corpus"))
    r = hypo_response(ctx, sc, HypoCfg(use_classifier=False))
    m = run_metrics(sc, r)
    assert r["final"]["category"] == gt["category"] and r["rounds"] == 2
    assert m["wrong_hyp_submitted"] == 1 and m["wrong_hyp_caught"] == 1 and m["correct_hyp_rejected"] == 0


def test_allowlist_withholds_unverified_rollback():
    sc = sc_of("bad_deployment", "easy")
    llm = StubLLM()
    ctx = Ctx(llm=llm, retriever=Retriever("data/corpus"))
    # without verifier, safety gate must still check version against observed deployment history
    r = hypo_response(ctx, sc, HypoCfg(verifier="none", use_classifier=False))
    assert r["safety"]["unapproved_state_change"] is False
    m = run_metrics(sc, r)
    assert m["plan_correct"] and r["plan"]["approved"] and r["plan"]["executed"]


def test_baseline_runs_and_metrics_keys():
    sc = sc_of("memory_leak", "easy")
    r = single_agent(Ctx(llm=StubLLM(), retriever=Retriever("data/corpus")), sc)
    m = run_metrics(sc, r)
    for k in ["correct", "steps_to_correct", "tool_valid_rate", "unsafe_attempt", "plan_correct", "tokens"]:
        assert k in m
