## Main results (mean over runs, 95% bootstrap CI over scenarios)

| System | Root-cause acc. | Strict acc. | Completion | Fell for decoy | Steps to correct | Wrong hyps before correct | Plan correct | Tokens |
|---|---|---|---|---|---|---|---|---|
| single_agent | 93.3% [83-100] | 90.0% [77-100] | 100.0% [100-100] | 0.0% [0-0] | 0.32 [0.14-0.52] | 0.20 [0.09-0.32] | 93.3% [85-100] | 14207.38 [11482.32-17261.27] |
| rag_only | 10.0% [3-18] | 8.3% [2-15] | 10.0% [3-18] | 0.0% [0-0] | 0.00 [0.00-0.00] | 0.00 [0.00-0.00] | 0.0% [0-0] | 1784.63 [1765.98-1805.75] |
| react_rag | 63.3% [48-78] | 58.3% [43-73] | 65.0% [48-80] | 0.0% [0-0] | 0.14 [0.00-0.36] | 0.14 [0.00-0.36] | 60.0% [45-75] | 32769.82 [26210.65-38793.97] |
| hypo_response | 51.7% [37-67] | 50.0% [35-65] | 51.7% [37-67] | 0.0% [0-0] | 0.36 [0.14-0.62] | 0.36 [0.14-0.62] | 45.0% [32-60] | 48869.82 [43327.22-55671.65] |
| abl_no_verifier | 55.0% [40-70] | 53.3% [38-68] | 58.3% [43-73] | 0.0% [0-0] | 0.24 [0.05-0.48] | 0.24 [0.05-0.48] | 55.0% [40-70] | 31837.03 [29444.00-34269.85] |
| abl_self_verify | 51.7% [38-65] | 51.7% [38-65] | 51.7% [38-65] | 0.0% [0-0] | 0.22 [0.04-0.39] | 0.22 [0.04-0.39] | 28.3% [15-42] | 70089.32 [61740.53-78257.00] |

## Safety (lower is better; rates over scenarios)

| System | Unsafe attempt (no approval flag) | Unapproved state change | Harmful rollback executed | Harmful rollback proposed |
|---|---|---|---|---|
| single_agent | 1.7% [0-5] | 0.0% [0-0] | 0.0% [0-0] | 0.0% [0-0] |
| rag_only | 0.0% [0-0] | 0.0% [0-0] | 0.0% [0-0] | 3.3% [0-8] |
| react_rag | 0.0% [0-0] | 0.0% [0-0] | 0.0% [0-0] | 0.0% [0-0] |
| hypo_response | 0.0% [0-0] | 0.0% [0-0] | 0.0% [0-0] | 0.0% [0-0] |
| abl_no_verifier | 0.0% [0-0] | 0.0% [0-0] | 0.0% [0-0] | 0.0% [0-0] |
| abl_self_verify | 0.0% [0-0] | 0.0% [0-0] | 0.0% [0-0] | 0.0% [0-0] |

## Capability-specific metrics

| System | Citation faithfulness | Tool-call validity | Tool-call relevance | Required-tool coverage |
|---|---|---|---|---|
| single_agent | n/a | 99.6% [99-100] | 98.6% [96-100] | 90.8% [84-96] |
| rag_only | 63.9% [44-83] | n/a | n/a | 0.0% [0-0] |
| react_rag | 60.4% [42-78] | 100.0% [100-100] | 99.4% [98-100] | 95.8% [91-99] |
| hypo_response | 59.7% [42-77] | 100.0% [100-100] | 93.8% [90-97] | 98.3% [95-100] |
| abl_no_verifier | 62.5% [43-82] | 100.0% [100-100] | 96.4% [94-99] | 95.8% [92-99] |
| abl_self_verify | 66.3% [50-82] | 100.0% [100-100] | 91.5% [88-95] | 99.2% [98-100] |

## Root-cause accuracy by difficulty

| System | easy | ambiguous | adversarial |
|---|---|---|---|
| single_agent | 93.3% [80-100] | 90.0% [70-100] | 100.0% [100-100] |
| rag_only | 10.0% [0-20] | 10.0% [0-25] | 10.0% [0-30] |
| react_rag | 56.7% [33-80] | 75.0% [55-95] | 60.0% [30-90] |
| hypo_response | 60.0% [40-80] | 35.0% [15-55] | 60.0% [30-90] |
| abl_no_verifier | 63.3% [43-80] | 45.0% [20-70] | 50.0% [10-90] |
| abl_self_verify | 60.0% [40-77] | 50.0% [35-65] | 30.0% [0-70] |

## Verifier behaviour (systems with a verifier)

| System | Hypothesis rejection rate | Wrong hypotheses caught | Correct hypotheses wrongly rejected |
|---|---|---|---|
| hypo_response | 32.4% (11/34) | n/a | 11 |
| abl_self_verify | 100.0% (61/61) | n/a | 61 |

## Paired comparison vs `hypo_response` (accuracy diff = ref - other; 95% CI; P(diff<=0))

| Other system | Δ accuracy | 95% CI | P(Δ≤0) |
|---|---|---|---|
| single_agent | -41.7 pp | [-55.0, -28.3] | 1.000 |
| rag_only | +41.7 pp | [+28.3, +55.0] | 0.000 |
| react_rag | -11.7 pp | [-28.3, +5.0] | 0.925 |
| abl_no_verifier | -3.3 pp | [-21.7, +13.3] | 0.677 |
| abl_self_verify | +0.0 pp | [-15.0, +13.3] | 0.530 |