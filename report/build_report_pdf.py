"""Build report/HypoResponse_Report.pdf from results/raw.jsonl and results/summary/summary.md.

    python3 report/build_report_pdf.py

Headline tables are parsed from summary.md (produced by analyze.py, including bootstrap CIs); the completion /
conditional-accuracy analysis is computed here directly from raw.jsonl.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results/raw.jsonl"
SUMMARY = ROOT / "results/summary/summary.md"
CLF = ROOT / "results/classifier_preds_tfidf.json"
SCEN = ROOT / "data/scenarios.json"
OUT = ROOT / "report/HypoResponse_Report.pdf"

SYSTEMS = ["single_agent", "rag_only", "react_rag", "hypo_response", "abl_no_verifier", "abl_self_verify"]
LABEL = {"single_agent": "Single agent (B1)", "rag_only": "RAG only (B2)", "react_rag": "ReAct + RAG",
         "hypo_response": "HypoResponse (ours)", "abl_no_verifier": "Abl: no verifier",
         "abl_self_verify": "Abl: self-verify"}
CATS = ["bad_deployment", "database_slowdown", "memory_leak", "third_party_api_failure", "config_change"]
CAT_LABEL = {"bad_deployment": "Bad deploy", "database_slowdown": "DB slowdown", "memory_leak": "Memory leak",
             "third_party_api_failure": "3rd-party API", "config_change": "Config change"}

# palette: muted, colour-blind-safe
C_BLUE, C_ORANGE, C_GREEN, C_GREY = (colors.HexColor("#3B6EA8"), colors.HexColor("#E08A3C"),
                                     colors.HexColor("#4E9A6B"), colors.HexColor("#8C8C8C"))
C_HEAD = colors.HexColor("#E8EDF4")
C_OURS = colors.HexColor("#FFF6E0")

# ---------------------------------------------------------------------------------------------- data
rows = [json.loads(l) for l in RAW.open()]
scen = {s["id"]: s for s in json.load(SCEN.open())}
clf = json.load(CLF.open())
MODEL = sorted({r["model"] for r in rows})
N_RUNS = len({r["run"] for r in rows})
N_SCEN = len({r["scenario_id"] for r in rows})


def by(system):
    return [r for r in rows if r["system"] == system]


def parse_md_tables(text):
    """Return {section heading: [header, *rows]} for every pipe table in summary.md."""
    tables, head, cur = {}, None, None
    for line in text.splitlines():
        if line.startswith("## "):
            head, cur = line[3:].strip(), None
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if cur is None:
                cur = tables.setdefault(head, [])
            cur.append(cells)
        else:
            cur = None
    return tables


T = parse_md_tables(SUMMARY.read_text())


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


derived = {}
for s in SYSTEMS:
    rs = by(s)
    comp = [r for r in rs if r["metrics"]["completed"]]
    inc = [r for r in rs if not r["metrics"]["completed"]]
    last_ok = sum(1 for r in inc if r["hypotheses"] and r["hypotheses"][-1][1] == r["category"])
    any_ok = sum(1 for r in inc if any(h[1] == r["category"] for h in r["hypotheses"]))
    derived[s] = dict(
        n=len(rs), n_comp=len(comp), n_corr=sum(r["metrics"]["correct"] for r in rs),
        corr_given_comp=sum(r["metrics"]["correct"] for r in comp), n_inc=len(inc), inc_last_ok=last_ok,
        inc_any_ok=any_ok,
        calls=sum(r["metrics"]["llm_calls"] for r in rs) / len(rs),
        tokens=sum(r["metrics"]["tokens"] for r in rs) / len(rs),
        comp_by_cat={c: (sum(r["metrics"]["completed"] for r in rs if r["category"] == c),
                         sum(1 for r in rs if r["category"] == c)) for c in CATS},
        rollbacks_exec=sum(1 for r in rs if r["plan"] and r["plan"].get("executed")),
        downgraded=sum(1 for r in rs if r["plan"] and r["plan"].get("downgraded_from")),
        harmful_exec=sum(r["metrics"]["harmful_rollback"] for r in rs),
        unsafe=sum(r["metrics"]["unsafe_attempt"] for r in rs),
    )
verdicts = {s: collections.Counter(v["verdict"] for r in by(s) for v in r["verifier_log"])
            for s in ("hypo_response", "abl_self_verify")}
wrong_submitted = {s: sum(r["metrics"]["wrong_hyp_submitted"] for r in by(s)) for s in SYSTEMS}
decoy_any = sum(r["metrics"]["fell_for_decoy"] for r in rows)
clf_acc = sum(1 for k, (c, _) in clf.items() if scen[k]["category"] == c)
diff_counts = collections.Counter(s["difficulty"] for s in scen.values())

# ---------------------------------------------------------------------------------------------- styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1F3A5F"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1F3A5F"))
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=13.5, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.5, leading=11)
CELL = ParagraphStyle("C", parent=BODY, fontSize=8.3, leading=10, spaceAfter=0)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="Helvetica-Bold")
CAP = ParagraphStyle("Cap", parent=SMALL, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceBefore=3, spaceAfter=10)
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2)
CODE = ParagraphStyle("Code", parent=BODY, fontName="Courier", fontSize=8.5, leading=11)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=20, leading=25, textColor=colors.HexColor("#1F3A5F"))
SUB = ParagraphStyle("Sub", parent=BODY, alignment=TA_CENTER, fontSize=11, leading=15)


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def table(data, widths, ours_rows=(), bold_first_col=False, font=CELL):
    conv = []
    for i, row in enumerate(data):
        conv.append([Paragraph(str(c), CELLB if i == 0 or (bold_first_col and j == 0) else font)
                     for j, c in enumerate(row)])
    t = Table(conv, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), C_HEAD), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    for r in ours_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), C_OURS))
    t.setStyle(TableStyle(style))
    return t


def md_table(section, widths, rename=True, drop_cols=(), header=None):
    data = [[c for j, c in enumerate(r) if j not in drop_cols] for r in T[section]]
    if header:
        data[0] = header
    if rename:
        for r in data[1:]:
            r[0] = LABEL.get(r[0], r[0])
    ours = [i for i, r in enumerate(data) if r[0] == LABEL["hypo_response"]]
    return table(data, widths, ours_rows=ours)


# ---------------------------------------------------------------------------------------------- figures
def bar_chart(series, names, cats, title_y, width=16 * cm, height=6.8 * cm, colours=(C_BLUE, C_ORANGE, C_GREEN)):
    d = Drawing(width, height)
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 1.4 * cm, 0.9 * cm, width - 2.0 * cm, height - 2.2 * cm
    bc.data = series
    bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep = 0, 100, 20
    bc.valueAxis.labels.fontSize = 7
    bc.valueAxis.visibleGrid = True
    bc.valueAxis.gridStrokeColor = colors.HexColor("#E3E3E3")
    bc.categoryAxis.categoryNames = cats
    bc.categoryAxis.labels.fontSize = 7
    bc.categoryAxis.labels.angle = 0
    bc.categoryAxis.labels.dy = -2
    bc.groupSpacing, bc.barSpacing = 6, 1
    for i, c in enumerate(colours[:len(series)]):
        bc.bars[i].fillColor = c
        bc.bars[i].strokeColor = None
    bc.barLabelFormat = "%.0f"
    bc.barLabels.fontSize = 6
    bc.barLabels.nudge = 5
    d.add(bc)
    d.add(String(0.25 * cm, bc.y + bc.height / 2, title_y, fontSize=7, angle=90, textAnchor="middle"))
    lg = Legend()
    lg.x, lg.y = bc.x + 4, height - 0.2 * cm
    lg.alignment = "right"
    lg.columnMaximum = 1
    lg.fontSize = 7
    lg.dx = lg.dy = 7
    lg.deltax = 120
    lg.colorNamePairs = list(zip(colours[:len(series)], names))
    d.add(lg)
    return d


def architecture():
    w, h = 16 * cm, 11.4 * cm
    d = Drawing(w, h)
    boxes = [
        ("Incident alert (page text only)", None, colors.HexColor("#EEEEEE")),
        ("Pre-screen classifier", "TF-IDF + LR prior on category (LoRA Mistral-7B planned)", colors.HexColor("#E6EEF8")),
        ("Investigator agent", "RAG over runbooks + postmortems  •  hypothesis every turn  •  read-only logs / metrics / deploys tools", colors.HexColor("#E6EEF8")),
        ("Verifier agent (independent)", "sees claim + RAW observations only  •  up to 3 own tool calls  •  "
                                         "support / reject / insufficient + feedback", colors.HexColor("#FDEBD8")),
        ("Recovery planner + safety gate", "allow-list: verified diagnosis, service in scope, version seen in history  "
                                           "•  human approval  •  execute_rollback", colors.HexColor("#E3F2E8")),
        ("Output: root cause + evidence trail + citations + recovery plan", None, colors.HexColor("#EEEEEE")),
    ]
    bw, bh, gap = 12.4 * cm, 1.35 * cm, 0.52 * cm
    x0 = (w - bw) / 2 - 0.9 * cm
    ys = []
    for i, (t, sub, fill) in enumerate(boxes):
        y = h - 0.2 * cm - (i + 1) * bh - i * gap
        ys.append(y)
        d.add(Rect(x0, y, bw, bh, rx=5, ry=5, fillColor=fill, strokeColor=colors.HexColor("#7A8799"), strokeWidth=0.7))
        d.add(String(x0 + bw / 2, y + bh - (0.5 if sub else 0.78) * cm, t, fontName="Helvetica-Bold", fontSize=9, textAnchor="middle"))
        if sub:
            d.add(String(x0 + bw / 2, y + 0.33 * cm, sub, fontSize=6.8, textAnchor="middle", fillColor=colors.HexColor("#333333")))
        if i:
            xa, y_top = x0 + bw / 2, ys[i - 1]
            d.add(Line(xa, y_top, xa, y + bh + 2, strokeColor=colors.HexColor("#555555"), strokeWidth=1))
            d.add(Polygon([xa - 3, y + bh + 5, xa + 3, y + bh + 5, xa, y + bh], fillColor=colors.HexColor("#555555"), strokeColor=None))
    # feedback loop verifier -> investigator
    xr = x0 + bw
    yv, yi = ys[3] + bh / 2, ys[2] + bh / 2
    col = colors.HexColor("#C0392B")
    for a, b in [((xr, yv), (xr + 0.9 * cm, yv)), ((xr + 0.9 * cm, yv), (xr + 0.9 * cm, yi)), ((xr + 0.9 * cm, yi), (xr + 3, yi))]:
        d.add(Line(*a, *b, strokeColor=col, strokeWidth=1))
    d.add(Polygon([xr + 5, yi - 3, xr + 5, yi + 3, xr, yi], fillColor=col, strokeColor=None))
    d.add(String(xr + 1.0 * cm, (yv + yi) / 2 + 4, "reject /", fontSize=7, fillColor=col))
    d.add(String(xr + 1.0 * cm, (yv + yi) / 2 - 5, "insufficient", fontSize=7, fillColor=col))
    d.add(String(xr + 1.0 * cm, (yv + yi) / 2 - 14, "(<= 3 rounds)", fontSize=7, fillColor=col))
    return d


# ---------------------------------------------------------------------------------------------- content
story = []
H, D = derived["hypo_response"], derived
S1 = derived["single_agent"]

story += [Spacer(1, 2.2 * cm),
          P("HypoResponse: A Hypothesis-Driven Multi-Agent System for Automated Incident Root Cause Analysis", TITLE),
          Spacer(1, 0.4 * cm),
          P("Project Report", SUB), Spacer(1, 0.8 * cm),
          P("<b>Team</b><br/>2025901010 – Mullapudi Sri Sai Tejaswini<br/>2025901016 – Kusumasri Belide<br/>"
            "2025901023 – P Sanjana Reddy", SUB),
          Spacer(1, 0.8 * cm),
          P("Domain: Software Engineering / Site Reliability Engineering<br/>Target user: on-call engineers investigating production incidents", SUB),
          Spacer(1, 1.6 * cm)]

abstract = (
    f"On-call engineers diagnose production incidents under time pressure, and a wrong diagnosis can lead to a harmful "
    f"recovery action. We built <b>HypoResponse</b>, a multi-agent system that combines a pre-screen incident classifier, an "
    f"Investigator agent that forms explicit hypotheses grounded in retrieved runbooks and postmortems, an independent "
    f"Verifier agent that audits each hypothesis against raw tool evidence, and a recovery planner whose state-changing "
    f"actions must pass an allow-list and human approval. We evaluated it on a synthetic benchmark of {N_SCEN} incidents "
    f"(5 categories × 3 difficulty levels) with mock observability tools, against a single tool-using agent, a "
    f"retrieval-only baseline, a ReAct+RAG agent and two ablations, using {', '.join(MODEL)} with {N_RUNS} runs per system. "
    f"Measured end to end, HypoResponse reached {pct(H['n_corr'], H['n'])} root-cause accuracy versus "
    f"{pct(S1['n_corr'], S1['n'])} for the single agent. The gap is not caused by wrong diagnoses: whenever HypoResponse "
    f"returned a valid final answer it was correct ({H['corr_given_comp']}/{H['n_comp']}), and in {H['inc_last_ok']} of its "
    f"{H['n_inc']} failed runs the last hypothesis was correct but no parseable final answer was produced. The same failure "
    f"affects every system that receives retrieved documents. Because no system ever submitted a wrong hypothesis and none "
    f"fell for a decoy, the benchmark gave the Verifier nothing to catch, so the research question is "
    f"<b>not yet answered</b> by these experiments. The safety gate behaved as designed: it executed no unapproved or "
    f"harmful rollbacks and withheld two rollbacks for unverified diagnoses. We describe the fixes and harder scenarios "
    f"needed for a decisive experiment.")
story += [P("Abstract", H2), P(abstract), PageBreak()]

# 1 Introduction
story += [P("1. Introduction and Problem Statement", H1),
          P("Production incidents such as outages, error-rate spikes and latency regressions must be diagnosed quickly to limit "
            "customer impact. Today an on-call engineer sifts through logs, metrics, deployment history and runbooks by hand. "
            "Observability tools (Datadog, Grafana, PagerDuty) show <i>what</i> happened but not <i>why</i>. General-purpose LLM "
            "agents given the same tools tend to commit early to a plausible cause without checking it against evidence, which "
            "can lead to wrong fixes and unsafe actions such as unverified rollbacks."),
          P("<b>Research question.</b> Does a multi-agent incident response system that explicitly forms hypotheses and verifies "
            "them against retrieved evidence identify the correct root cause more accurately, and execute fewer unsafe recovery "
            "actions, than a single tool-using agent given the same tools and information?"),
          P("<b>Why it matters.</b>"),
          *bullets(["Every minute of diagnosis costs customer impact and revenue.",
                    "Engineers are often paged at night and must diagnose under pressure with little context.",
                    "Wrong diagnoses lead to wrong fixes, which can make incidents worse.",
                    "Unsafe recovery actions (unverified rollbacks, config changes) can cascade into larger outages.",
                    "Automating even the first minutes of triage, narrowing down the root cause, has large practical value."]),
          P("1.1 Gap in existing approaches", H2),
          table([["Existing approach", "Gap"],
                 ["Observability tools (Datadog, Grafana)", "Surface data; do not reason about it"],
                 ["Runbook automation", "Follows fixed scripts; cannot adapt to novel failures"],
                 ["Single LLM agent with tools", "Jumps to conclusions; no verification step; can take unsafe actions"],
                 ["RAG-only systems", "Retrieves relevant documents but does not investigate or reason"]],
                [6.5 * cm, 9.5 * cm]),
          Spacer(1, 6),
          P("The missing piece is a single investigation loop that combines retrieval, structured hypothesis formation, "
            "independent verification and safety controls on recovery actions."),
          P("1.2 Related work and positioning", H2),
          P("Agentic root-cause analysis is an active area, so we do not claim hypothesis verification is new in itself. "
            "ReAct-style agents with retrieval have been studied for cloud-incident RCA (Roy et al., FSE'24); RCACopilot "
            "(Chen et al., EuroSys'24) combines predefined diagnostic handlers with retrieval; Flow-of-Action (Pei et al., WWW'25) "
            "injects SOPs and a judge agent; mABC uses multi-agent voting; LATS-RCA searches over competing hypotheses; and "
            "Hypothesize-Then-Verify (ICSE-NIER'26) trains a small verifier of candidate root causes. Analyses of LLM RCA failures "
            "(Riddell et al., FORGE'26) highlight fabricated evidence, evidential insufficiency and anchoring, which motivates an "
            "explicit evidence-sufficiency check. Benchmarks include OpenRCA, AIOpsLab, ITBench and SREGym."),
          P("Our positioning: (i) we isolate the <i>independent</i> verifier as a controlled variable (vs. no verifier and "
            "self-verification); (ii) we treat recovery safety as a co-primary outcome rather than stopping at diagnosis; and "
            "(iii) we report grounding (citation faithfulness, evidence trail) alongside accuracy.")]

# 2 System
story += [PageBreak(), P("2. System Design", H1),
          architecture(),
          P("Figure 1. HypoResponse architecture. The red path is the hypothesis-verification loop.", CAP),
          P("<b>Pre-screen classifier.</b> Maps the alert text to one of five categories and passes it to the Investigator as a "
            "prior. The proposal planned a LoRA-tuned Mistral-7B; the evaluated system uses a TF-IDF + logistic-regression "
            "classifier trained on 600 synthetic incidents whose phrasing is disjoint from the test set."),
          P("<b>Investigator.</b> A JSON ReAct loop (≤ 8 steps per round) that must state a category hypothesis on every turn and "
            "choose the tool call that best <i>tests</i> it, including looking for disconfirming evidence (onset timing vs. "
            "recent changes). It has read-only tools plus <font face='Courier'>search_docs</font> over the corpus, and returns a "
            "package: category, specific root cause, evidence copied from observations, citations to chunk IDs, and a "
            "recommended action. Evidence memory and verifier feedback persist across rounds."),
          P("<b>Verifier.</b> Never sees the Investigator's reasoning, only the claim, the raw observations and the documents, "
            "which limits conformity. It is instructed to try to refute the hypothesis by checking timing, whether each evidence "
            "item literally appears in the observations, whether another category explains the symptoms better (decoys), and "
            "contradicting signals. It may run up to 3 read-only tool calls of its own. Non-support verdicts return reasons, "
            "contradictions and suggested checks to the Investigator, for at most 3 rounds."),
          P("<b>Recovery planner and safety gate.</b> The planner proposes one action (rollback / mitigate / none) using runbook "
            "chunks. A rollback must pass an allow-list (diagnosis verified, service in scope, target version actually observed "
            "in deployment history), then a simulated human approval oracle; only then is "
            "<font face='Courier'>execute_rollback</font> called. Otherwise the rollback is downgraded to a manual mitigation."),
          KeepTogether([P("2.1 Mapping to the five capabilities", H2),
          table([["Capability", "Where it appears", "Measured by"],
                 ["RAG", "TF-IDF retrieval over 5 runbooks + 5 postmortems (section chunks); search_docs tool", "Citation faithfulness"],
                 ["Reasoning", "Hypothesis-first ReAct; verifier feedback loop", "Steps / wrong hypotheses before correct; rejection rate"],
                 ["Agentic", "Investigator + Verifier + Planner with persistent memory", "Task completion rate"],
                 ["Tool calling", "query_logs, query_metrics, get_deployments, execute_rollback", "Tool validity, relevance, required-tool coverage; unsafe-action rate"],
                 ["Fine-tuning", "Pre-screen classifier", "Classification accuracy vs. zero-shot"]],
                [2.6 * cm, 7.4 * cm, 6.0 * cm])])]

# 3 Setup
story += [P("3. Experimental Setup", H1),
          P("3.1 Benchmark", H2),
          P(f"{N_SCEN} synthetic incidents, 6 per category (bad deployment, database slowdown, memory leak, third-party API "
            f"failure, config change). Within each category, {diff_counts['easy'] // 5} are <i>easy</i>, "
            f"{diff_counts['ambiguous'] // 5} <i>ambiguous</i> (a plausible decoy such as an unrelated recent deploy, or a config "
            f"change that mimics a DB slowdown) and {diff_counts['adversarial'] // 5} <i>adversarial</i> (decoy plus a misleading "
            f"log line). Each scenario provides logs, metrics and deployment/config history behind four mock tools, and ground "
            f"truth: category, specific-cause keywords, required tools, whether rollback is appropriate and to which version."),
          P("<b>Snippet leak (found while building).</b> Our first page format included the three latest error lines. With them, "
            "the TF-IDF classifier scored 100%; from the alert text alone it scores 40%. The snippet gave away the category, "
            "which would make tools, RAG and the verifier unnecessary, so all reported runs use the alert text only."),
          P("3.2 Systems evaluated", H2),
          table([["System", "Description"],
                 [LABEL["single_agent"], "One ReAct LLM with all 4 tools including execute_rollback; no RAG, no verifier (proposal baseline 1)"],
                 [LABEL["rag_only"], "Retrieves runbooks/postmortems; one LLM call; no tools (proposal baseline 2)"],
                 [LABEL["react_rag"], "ReAct agent with read-only tools plus a retrieval tool (Roy et al.-style)"],
                 [LABEL["hypo_response"], "Full system: classifier prior → Investigator ↔ independent Verifier → planner → gate"],
                 [LABEL["abl_no_verifier"], "HypoResponse without the Verifier (planner still gated by allow-list + approval)"],
                 [LABEL["abl_self_verify"], "Verifier replaced by self-reflection: same model, sees its own reasoning, no tools"]],
                [3.6 * cm, 12.4 * cm], ours_rows=[4]),
          Spacer(1, 6),
          P("3.3 Protocol and metrics", H2),
          P(f"Model: <b>{', '.join(MODEL)}</b> for all agents (the Verifier used the same model), temperature 0. {N_RUNS} runs per "
            f"system × {N_SCEN} scenarios = {len(rows)} episodes. Scenario scores are averaged over runs; 95% bootstrap CIs are over "
            f"scenarios, and a paired bootstrap is used against HypoResponse. <b>Root-cause accuracy</b> = correct category "
            f"(strict: also names the specific cause). <b>Completion</b> = a final answer with a valid category was produced. "
            f"<b>Steps to correct</b> = read-tool calls before the first correct hypothesis. Safety: <b>unsafe attempt</b> "
            f"(rollback call without approval flag), <b>unapproved state change</b> (rollback executed without genuine approval), "
            f"<b>harmful rollback</b> executed or proposed (wrong rollback). Citation faithfulness: the cited chunk was retrieved "
            f"and shares ≥ 50% of the claim's content words (lexical heuristic).")]

# 4 Results
story += [PageBreak(), P("4. Results", H1),
          P("4.1 Main comparison", H2),
          md_table("Main results (mean over runs, 95% bootstrap CI over scenarios)",
                   [2.6 * cm, 2.0 * cm, 2.0 * cm, 2.1 * cm, 1.5 * cm, 1.9 * cm, 1.9 * cm, 2.0 * cm], drop_cols=(8,),
                   header=["System", "Accuracy", "Strict acc.", "Completion", "Fell for decoy", "Steps to correct", "Wrong hyps before correct", "Plan correct"]),
          P("Table 1. End-to-end results. Incorrect or missing final answers both count as wrong. Token counts in Table 10.", CAP),
          P(f"Measured end to end, the single agent is clearly best ({pct(S1['n_corr'], S1['n'])}) and HypoResponse "
            f"({pct(H['n_corr'], H['n'])}) does not beat it. The paired bootstrap gives HypoResponse minus single agent = "
            f"{T['Paired comparison vs `hypo_response` (accuracy diff = ref - other; 95% CI; P(diff<=0))'][1][1]} "
            f"(CI {T['Paired comparison vs `hypo_response` (accuracy diff = ref - other; 95% CI; P(diff<=0))'][1][2]}). "
            f"HypoResponse is indistinguishable from its no-verifier and self-verify ablations (differences within ±4 pp; CIs span zero). "
            f"Note that the <i>completion</i> column almost equals the <i>accuracy</i> column for every system except the single agent; "
            f"Section 4.2 explains why."),
          P("4.2 Key finding: failures are invalid final answers, not wrong diagnoses", H2)]

comp_rows = [["System", "Valid final answer", "Correct | valid final", "End-to-end correct",
              "Failed runs whose last hypothesis was correct", "LLM calls / episode"]]
for s in SYSTEMS:
    d = D[s]
    comp_rows.append([LABEL[s], f"{d['n_comp']}/{d['n']} ({pct(d['n_comp'], d['n'])})",
                      f"{d['corr_given_comp']}/{d['n_comp']} ({pct(d['corr_given_comp'], d['n_comp'])})",
                      f"{d['n_corr']}/{d['n']}",
                      f"{d['inc_last_ok']}/{d['n_inc']}" if d["n_inc"] else "–", f"{d['calls']:.1f}"])
story += [table(comp_rows, [3.2 * cm, 2.6 * cm, 2.6 * cm, 2.2 * cm, 3.4 * cm, 2.0 * cm], ours_rows=[4]),
          P("Table 2. Episode counts (all runs pooled). Computed from results/raw.jsonl.", CAP),
          KeepTogether([bar_chart([[100 * D[s]["n_corr"] / D[s]["n"] for s in SYSTEMS],
                     [100 * D[s]["n_comp"] / D[s]["n"] for s in SYSTEMS],
                     [100 * D[s]["corr_given_comp"] / D[s]["n_comp"] for s in SYSTEMS]],
                    ["End-to-end accuracy", "Valid final answer rate", "Accuracy given a valid final answer"],
                    ["Single", "RAG-only", "ReAct+RAG", "HypoResp.", "No-verif.", "Self-verif."], "% of episodes"),
          P("Figure 2. Accuracy gaps come from missing final answers, not wrong ones.", CAP)]),
          P(f"Every system that receives retrieved documents in its prompt fails to produce a valid final answer in a large "
            f"share of episodes: RAG-only {pct(D['rag_only']['n_inc'], 60)}, ReAct+RAG {pct(D['react_rag']['n_inc'], 60)}, "
            f"HypoResponse {pct(H['n_inc'], 60)}, while the single agent (no documents) always finishes. When a valid final answer "
            f"<i>is</i> produced, accuracy is {pct(H['corr_given_comp'], H['n_comp'])} for HypoResponse and "
            f"{pct(S1['corr_given_comp'], S1['n_comp'])} for the single agent. In {H['inc_last_ok']} of HypoResponse's "
            f"{H['n_inc']} failed episodes the Investigator's last stated hypothesis was the correct category; the system knew the "
            f"answer but did not return it in the required JSON format. The failure therefore lies in the answer-format layer "
            f"of the harness interacting with the RAG prompt, not in hypothesis-driven reasoning."),
          P("<b>Limitation of this diagnosis.</b> The runner did not save the raw final output (<font face='Courier'>raw_final</font> "
            "is dropped when results are written), so we cannot yet tell whether the output was malformed JSON (e.g., unescaped "
            "quotes copied from log lines or documents), a schema slip such as a category string outside the five labels, or "
            "an exhausted step budget. Output truncation is unlikely: failed rag_only episodes use only ~80 more tokens than successful ones "
            "(1.79k vs. 1.71k), whereas hitting the 900-token output cap would show a much larger gap. Failures are also uneven by category (Table 4), with bad "
            "deployment hit hardest for the RAG systems.")]

cat_rows = [["System"] + [CAT_LABEL[c] for c in CATS]]
for s in SYSTEMS:
    cat_rows.append([LABEL[s]] + [f"{a}/{b}" for a, b in (D[s]["comp_by_cat"][c] for c in CATS)])
story += [KeepTogether([table(cat_rows, [3.6 * cm] + [2.48 * cm] * 5, ours_rows=[4]),
                        P("Table 4. Valid-final-answer rate by incident category (episodes).", CAP)])]

diff = T["Root-cause accuracy by difficulty"]
story += [P("4.3 Difficulty breakdown", H2),
          md_table("Root-cause accuracy by difficulty", [4.0 * cm, 4.0 * cm, 4.0 * cm, 4.0 * cm]),
          P("Table 5. End-to-end root-cause accuracy by scenario difficulty.", CAP),
          P(f"We expected verification to help most on ambiguous and adversarial incidents. It did not, but the decoys did not "
            f"work as designed either: <b>no system fell for a decoy in any of the {len(rows)} episodes</b> (fell-for-decoy = "
            f"{decoy_any}), and the single agent scored 100% on adversarial incidents. The decoys were too easy to rule out for "
            f"this model, so the difficulty levels did not separate the systems.")]

hv, sv = verdicts["hypo_response"], verdicts["abl_self_verify"]
story += [P("4.4 Verifier behaviour", H2),
          table([["System", "Verdicts issued", "support", "insufficient", "reject", "Wrong hypotheses submitted",
                  "Correct hypotheses not supported"],
                 [LABEL["hypo_response"], sum(hv.values()), hv["support"], hv["insufficient"], hv["reject"],
                  wrong_submitted["hypo_response"], hv["insufficient"] + hv["reject"]],
                 [LABEL["abl_self_verify"], sum(sv.values()), sv["support"], sv["insufficient"], sv["reject"],
                  wrong_submitted["abl_self_verify"], sv["insufficient"] + sv["reject"]]],
                [3.2 * cm, 2.0 * cm, 1.6 * cm, 2.0 * cm, 1.5 * cm, 2.8 * cm, 2.9 * cm], ours_rows=[1]),
          P("Table 6. Verifier verdicts (all rounds, all runs).", CAP),
          P(f"Every hypothesis submitted to a verifier was correct, so the Verifier had <b>no wrong hypotheses to catch</b>, and the "
            f"core claim (that independent verification catches wrong diagnoses) could not be tested. What we can measure is its "
            f"cost. The independent Verifier supported {hv['support']}/{sum(hv.values())} correct packages and returned a "
            f"non-support verdict on {hv['insufficient'] + hv['reject']} correct ones (false rejections), adding extra rounds and "
            f"about 50% more tokens than the no-verifier ablation. The self-verification ablation returned "
            f"<i>insufficient</i> on all {sum(sv.values())} verdicts. That pattern matches the code path that maps an unparseable "
            f"verdict to <i>insufficient</i>, so it may be the same output-format problem as in 4.2 rather than real "
            f"self-doubt. Because self-verify never supported a diagnosis, the gate withheld all of its rollbacks."),
          P("4.5 Safety", H2),
          md_table("Safety (lower is better; rates over scenarios)", [3.6 * cm, 3.4 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm]),
          P("Table 7. Safety metrics.", CAP)]

story += [table([["System", "Rollbacks executed", "Rollbacks withheld by gate", "Harmful rollbacks executed",
                  "Rollback calls without approval"]] +
                [[LABEL[s], D[s]["rollbacks_exec"], D[s]["downgraded"], D[s]["harmful_exec"], D[s]["unsafe"]]
                 for s in ("single_agent", "hypo_response", "abl_no_verifier", "abl_self_verify")],
                [3.6 * cm, 3.0 * cm, 3.2 * cm, 3.2 * cm, 3.0 * cm], ours_rows=[2]),
          P("Table 8. Recovery actions (episode counts).", CAP),
          P(f"The gated systems never executed an unapproved or harmful rollback. That is partly <b>by construction</b>: the gate "
            f"makes unapproved execution impossible, so the empirical question is whether it blocks the right things. "
            f"HypoResponse executed {H['rollbacks_exec']} approved, correct rollbacks and withheld 2 whose diagnosis was not verified. "
            f"The single agent, which can call execute_rollback directly, made {S1['unsafe']} rollback call without the approval "
            f"flag (1.7% of scenarios), which the mock environment blocked. Overall the base model was cautious, so harmful actions "
            f"were too rare to separate the systems statistically."),
          P("4.6 Grounding and tool use", H2),
          md_table("Capability-specific metrics", [3.6 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm]),
          P("Table 9. Capability metrics (citation faithfulness only applies to systems that cite documents).", CAP),
          P("Tool calls were almost always valid (≥ 99.6%). HypoResponse had the highest required-tool coverage among the full "
            "systems (98.3%), consistent with its hypothesis-testing prompt, at slightly lower relevance (93.8%, i.e. more "
            "exploratory calls). Citation faithfulness was about 60–66% for all RAG systems. The heuristic is lexical, so this "
            "is a lower bound on true faithfulness, not an audit."),
          P("4.7 Pre-screen classifier (fine-tuning capability)", H2),
          P(f"The TF-IDF + logistic-regression classifier reaches <b>{pct(clf_acc, len(clf))}</b> ({clf_acc}/{len(clf)}) from the "
            f"alert text alone (chance 20%), and 100% when the leaky error snippet is attached. So most category information "
            f"is in the logs, not the alert, which is why tool-based investigation is needed. The zero-shot LLM classifier and the "
            f"LoRA Mistral-7B classifier from the proposal were <b>not run</b> (no GPU was available; the Colab scripts are written "
            f"but untested), so the fine-tuned vs. zero-shot comparison is still open. Since a 40%-accurate prior is wrong more "
            f"often than right, we checked whether wrong priors explain HypoResponse's failures. They do not: completion was "
            f"similar whether the prior was right or wrong."),
          P("4.8 Cost", H2),
          md_table("Main results (mean over runs, 95% bootstrap CI over scenarios)", [5 * cm, 5 * cm],
                   drop_cols=(1, 2, 3, 4, 5, 6, 7)),
          P(f"Table 10. Mean tokens per episode. HypoResponse used ~{H['tokens'] / S1['tokens']:.1f}× the tokens of the single "
            f"agent ({H['calls']:.1f} vs. {S1['calls']:.1f} LLM calls per episode).", CAP)]

# 5 Discussion
story += [PageBreak(), P("5. Discussion", H1),
          P("5.1 Answer to the research question", H2),
          P("<b>Not yet answered.</b> End to end, HypoResponse did not outperform the single-agent baseline in accuracy, and "
            "harmful actions were too rare to show a safety difference. Two confounds prevent reading this as evidence "
            "against the design:"),
          *bullets([f"<b>Answer-format failures dominate the accuracy metric.</b> Every RAG-fed system loses 35–90% of episodes to "
                    f"invalid final answers. Among valid answers HypoResponse is {pct(H['corr_given_comp'], H['n_comp'])} correct.",
                    "<b>The benchmark never produced the situation the Verifier is for.</b> No wrong hypothesis was submitted "
                    "and no system fell for a decoy, so verification could only cost (false rejections, tokens), never help. "
                    "This is the proposal's 'all systems perform similarly on easy incidents' risk, and it occurred."]),
          P("5.2 What did work", H2),
          *bullets(["The full pipeline (classifier, RAG, multi-round Investigator–Verifier loop, planner, allow-list, approval) "
                    "runs end to end on a real LLM with resumable, logged experiments and paired statistics.",
                    "The safety gate behaved as specified: zero unapproved or harmful state changes; rollbacks were executed "
                    "only for verified diagnoses with versions actually seen in deployment history.",
                    "Hypothesis-first prompting gave the highest required-tool coverage.",
                    "Building the benchmark revealed a real evaluation pitfall: a 3-line log snippet in the page leaks the "
                    "category (40% → 100% for a bag-of-words classifier). Agentic-RCA benchmarks should check for this."]),
          P("5.3 Deviations from the proposal", H2),
          table([["Planned", "Done", "Status"],
                 ["3 runs per system", f"{N_RUNS} runs", "Reduced (budget)"],
                 ["Baselines 1 & 2 + our system", "All three run, plus ReAct+RAG", "Done"],
                 ["Ablations", "No-verifier, self-verify run; no-RAG, no-classifier, no-gate implemented but not run", "Partial"],
                 ["Literature-style baselines (RCACopilot, Flow-of-Action, mABC, LATS)", "Implemented, not run", "Not run"],
                 ["LoRA Mistral-7B classifier vs. zero-shot", "TF-IDF classifier only", "Not run"],
                 ["Wrong-prior injection; harder ambiguous incidents", "Implemented, not run", "Not run"],
                 ["Different model family for Verifier", "Same model used", "Not run"]],
                [6.0 * cm, 7.2 * cm, 2.8 * cm]),
          Spacer(1, 6),
          P("5.4 Threats to validity", H2),
          *bullets(["<b>Synthetic data.</b> Template-generated single-cause incidents; production failures are concurrent and noisier.",
                    f"<b>Small n.</b> {N_SCEN} scenarios × {N_RUNS} runs; wide CIs. HypoResponse's completion agreed across the two "
                    "runs for only 19/30 scenarios, so run-to-run variance is substantial.",
                    "<b>Corpus–benchmark overlap.</b> The same authors wrote both, so retrieval benefit is optimistic.",
                    "<b>Correlated errors.</b> Investigator and Verifier share a model and may share blind spots.",
                    "<b>Simulated approval and lexical citation metric</b>, rather than a human study and an audited judge.",
                    "<b>Re-implemented baselines</b> may under- or over-state the original systems."]),
          P("5.5 Next steps toward a decisive experiment", H2),
          *bullets(["Persist the raw LLM output for every final answer and verdict, and diagnose the parse failures.",
                    "Make the harness robust: on an invalid final, re-prompt once with the exact validation error; validate "
                    "the category against the label set with a repair step. Apply identically to all systems.",
                    "Report accuracy both end to end and conditional on a valid answer, so format robustness and diagnostic "
                    "skill are measured separately.",
                    "Make the benchmark hard enough to produce wrong first hypotheses: stronger decoys, concurrent changes, missing "
                    "telemetry, and the wrong-prior injection condition, which directly tests whether the Verifier rescues a "
                    "confidently wrong belief.",
                    "Run the remaining ablations and literature-style baselines, the zero-shot and LoRA classifiers, a Verifier "
                    "from a different model family, and 3 runs per system; then add an external benchmark (e.g., ITBench or SREGym)."]),
          P("6. Conclusion", H1),
          P(f"We designed and implemented HypoResponse, a hypothesis-driven Investigator–Verifier system with gated recovery for "
            f"incident root-cause analysis, together with a 30-incident benchmark, mock tools, a RAG corpus, baselines, ablations "
            f"and a full metric suite. In our first real-model evaluation, HypoResponse is correct whenever it returns an answer "
            f"({H['corr_given_comp']}/{H['n_comp']}) and its safety gate prevented every unapproved or harmful state change. "
            f"However, answer-format failures in RAG-fed prompts and a benchmark that never produced wrong hypotheses mean the "
            f"value of independent verification is still untested. Fixing the output layer and adding harder, decoy-driven and "
            f"wrong-prior scenarios are the next steps for answering the research question."),
          P("References", H2),
          P("Roy et al., <i>Exploring LLM-based Agents for Root Cause Analysis</i>, FSE'24 Companion. · Chen et al., <i>Automatic "
            "Root Cause Analysis via Large Language Models for Cloud Incidents</i> (RCACopilot), EuroSys'24. · Pei et al., "
            "<i>Flow-of-Action</i>, WWW'25 Companion. · Zhang et al., <i>mABC</i>, 2024. · Wang et al., <i>RCAgent</i>, CIKM'24. · "
            "<i>LATS-RCA</i>, arXiv 2605.03505. · <i>Hypothesize-Then-Verify</i>, ICSE-NIER'26. · Riddell et al., <i>Stalled, "
            "Biased, and Confused</i>, FORGE'26. · Xu et al., <i>OpenRCA</i>. · Chen et al., <i>AIOpsLab</i>, MLSys'25. · Jha et "
            "al., <i>ITBench</i>, ICML'25. · Clark et al., <i>SREGym</i>, arXiv 2605.07161.", SMALL)]


def on_page(canv, doc):
    if doc.page == 1:
        return
    canv.saveState()
    canv.setFont("Helvetica", 8)
    canv.setFillColor(colors.HexColor("#777777"))
    canv.drawString(2 * cm, 1.2 * cm, "HypoResponse — Project Report")
    canv.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canv.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="HypoResponse Project Report",
                        author="Tejaswini, Kusumasri Belide, P Sanjana Reddy")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
