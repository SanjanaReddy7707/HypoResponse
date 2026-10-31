# HypoResponse: Hypothesis-Driven Multi-Agent Incident Root Cause Analysis with Gated Recovery

*Draft. Sections marked **[FILL]** must be completed from `results/summary/summary.md` after the LLM experiments are run.
No LLM experiment results exist in this draft; the only measured number is the offline TF-IDF classifier (Section 5.1).*

## 1. Introduction
On-call engineers must diagnose production incidents under time pressure, and a wrong diagnosis can lead to a harmful recovery action.
Observability tools expose signals but do not reason over them; single LLM agents with tools tend to commit early to a plausible cause.
We study whether an independent **Verifier** agent that audits an Investigator's hypothesis against raw evidence, combined with a
**gated recovery planner** (allow-list + human approval for state changes), improves root-cause accuracy and reduces unsafe or harmful recovery actions.

**Research question.** Does an independent evidence-verification loop, with gated recovery actions, give more accurate diagnoses and fewer
harmful recovery proposals than single-agent, retrieval-only, and published agentic-RCA designs given the same tools and information?

## 2. Related work and positioning
Agentic RCA is an active area, so we do **not** claim that hypothesis verification is new in itself.
- *Tool-using RCA agents.* Roy et al. (FSE'24) study ReAct agents with retrieval for cloud-incident RCA; RCACopilot (Chen et al., EuroSys'24) uses predefined diagnostic handlers plus retrieval but predicts root-cause categories and needs hand-built handlers.
- *Structured / multi-agent RCA.* Flow-of-Action (Pei et al., WWW'25) injects SOPs and auxiliary agents including a judge, reporting a rise from 35.50% (ReAct) to 64.01% on its benchmark. mABC uses multi-agent voting; MA-RCA uses retrieval, analysis, validation and reporting agents.
- *Hypothesis-centric RCA.* LATS-RCA frames diagnosis as reflection-guided tree search over competing hypotheses (91.3% on LO2 but 60–70% in a production deployment, at ~75 API calls per incident); "Hypothesize-Then-Verify" (ICSE-NIER'26) distils a teacher into a small verifier of candidate root causes. VerifyMAS applies hypothesis verification to failure attribution in multi-agent systems.
- *Why verification matters.* Riddell et al. (FORGE'26) catalogue 16 reasoning failure types in LLM RCA (fabricated evidence, evidential insufficiency, anchoring) and recommend evidence-sufficiency checks; Cloud-OpsBench reports 0.76 root-cause accuracy but only 0.38 evidence closure; R2Act reports that 39.5–62.0% of correctly diagnosed incidents still receive an invalid recovery action.
- *Benchmarks.* OpenRCA (335 incidents), AIOpsLab, ITBench, SREGym, ITBench-AA.

**Our positioning.** (i) We isolate the *independent verifier* as a controlled variable against self-reflection, voting and no verification (ablations), rather than bundling it with other changes. (ii) Most RCA work stops at diagnosis; we make recovery safety a co-primary outcome and measure harmful proposals and unapproved state changes. (iii) We report grounding (citation faithfulness, evidence trail) alongside accuracy. Our literature-style baselines are lightweight re-implementations of those ideas on our environment, not the original systems.

## 3. System
```
alert → [pre-screen classifier prior] → Investigator (RAG + read-only tools, hypothesis every turn)
      ↔ Verifier (raw evidence only; own read-only tool calls; support / reject / insufficient + feedback)  (≤3 rounds)
      → Recovery Planner → allow-list (service in scope, version seen in deployment history, diagnosis verified)
      → simulated human approval → execute_rollback
```
Design choices: the Verifier never sees the Investigator's reasoning, only the claim and raw tool observations, to limit conformity; it can run up to three tool calls of its own. Rejection feedback (reasons, contradictions, suggested checks) is returned to the Investigator, which keeps an evidence memory across rounds. If no hypothesis is verified within three rounds, rollback is withheld.

## 4. Experimental setup
**Benchmark.** 30 synthetic incidents: 5 categories (bad deployment, database slowdown, memory leak, third-party API failure, config change) × 6, of which per category 3 are *easy*, 2 *ambiguous* (a plausible decoy such as an unrelated recent deploy or a config change that mimics a DB slowdown) and 1 *adversarial* (decoy plus a misleading log line). Each scenario has logs, metrics (down-sampled) and deployment/config history served by four mock tools, and ground truth: category, specific cause keywords, required tools, whether rollback is appropriate and to which version. **RAG corpus:** 5 runbooks + 5 postmortems.
**Systems.** Two proposal baselines, five literature-style baselines, the full system, and five ablations (see the Systems table in the README).
**Protocol.** 3 runs per system; scenario scores averaged over runs; 95% bootstrap CIs over scenarios; paired bootstrap against the full system. Additional conditions: **wrong-prior injection** (every system receives a confident wrong triage prior on scenarios that have a decoy) and the **easy setting** (error-line snippet in the page). Model: **[FILL: model, verifier model, temperature]**.
**Metrics.** See README; safety metrics are *unsafe attempt*, *unapproved state change*, *harmful rollback executed* and *harmful rollback proposed*.

## 5. Results
### 5.1 Pre-screen classifier (fine-tuning capability)
Measured offline. A TF-IDF + logistic-regression classifier trained on 600 synthetic incidents (disjoint phrasing from the test set) reaches **40% accuracy** (12/30; chance 20%) from the alert text alone. With the 3-line error snippet attached it reaches **100%**, because the snippet reveals the failing component; we therefore exclude the snippet by default and report the leak as a benchmark-design finding. **[FILL]** zero-shot LLM accuracy and LoRA Mistral-7B accuracy (`classifier/eval_classifiers.py`), and whether a tuned model beats zero-shot. A null result is reportable.

### 5.2 Main comparison **[FILL]**
Table from `results/summary/summary.md`: accuracy, strict accuracy, steps to correct, wrong hypotheses before correct, completion, tokens, with CIs, per system.
### 5.3 Difficulty breakdown **[FILL]** — expected to be where verification matters (ambiguous/adversarial).
### 5.4 Verifier analysis **[FILL]** — rejection rate, wrong hypotheses caught, correct hypotheses wrongly rejected; independent vs self-verification.
### 5.5 Safety **[FILL]** — baselines with direct `execute_rollback` access vs gated system; no-gate ablation. Note: the gated system's unapproved-state-change rate is zero by construction; the substantive comparison is harmful-proposal rate and plan correctness.
### 5.6 Grounding and tool use **[FILL]** — citation faithfulness, tool validity/relevance/coverage.
### 5.7 Wrong-prior injection **[FILL]** — does the system recover from a confidently wrong first belief?
### 5.8 Ablations **[FILL]** — contribution of verifier, independence, RAG, classifier prior, safety gate.

## 6. Threats to validity
- **Synthetic data.** Template-generated, single-cause incidents; production failures are concurrent, intermittent and noisier. Published results show large benchmark-to-production drops (e.g. LATS-RCA).
- **Small n.** 30 scenarios; report intervals and paired differences, not significance claims.
- **Corpus–benchmark overlap.** Same authors wrote both, so retrieval benefit is optimistic.
- **Leaky signals.** We found and removed one (snippet); others may remain (e.g. category-specific metric names). Mitigation: red-herring and decoy scenarios, and an external benchmark run.
- **Correlated errors.** Investigator and Verifier on the same model may share blind spots; run with `--verifier-model` from a different family.
- **Simulated approval.** Deterministic oracle, not humans. Citation faithfulness is a lexical heuristic.
- **Re-implemented baselines** may under- or over-state the originals.
- **Recommended before submission:** evaluate on an external benchmark (e.g. ITBench-AA offline snapshots or SREGym-Lite) and audit a sample of traces by hand.

## 7. Conclusion **[FILL after results]**

## References (to verify and format)
Roy et al., *Exploring LLM-based Agents for Root Cause Analysis*, FSE'24 Companion · Chen et al., *Automatic Root Cause Analysis via LLMs for Cloud Incidents* (RCACopilot), EuroSys'24 · Pei et al., *Flow-of-Action*, WWW'25 Companion · Zhang et al., *mABC*, 2024 · Wang et al., *RCAgent*, CIKM'24 · *LATS-RCA* (arXiv 2605.03505) · *Hypothesize-Then-Verify* (ICSE-NIER'26) · *VerifyMAS* (arXiv 2605.17467) · Riddell et al., *Stalled, Biased, and Confused*, FORGE'26 · Xu et al., *OpenRCA* · Chen et al., *AIOpsLab*, MLSys'25 · Jha et al., *ITBench*, ICML'25 · Clark et al., *SREGym* (arXiv 2605.07161) · *Incident-Arena* (arXiv 2610.00648), citing R2Act and Cloud-OpsBench.
Details above were gathered from abstracts and search snippets, not full-text reading; verify each figure against the source.
