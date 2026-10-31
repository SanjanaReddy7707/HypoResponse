# RB-001 Runbook: Bad deployment (code regression)

## Symptoms
Error rate steps up abruptly (not gradually) on a single service within minutes of a release. All pods fail the same way. Latency may rise modestly; CPU and memory stay normal. Traffic volume is unchanged.

## Diagnosis
1. Call get_deployments for the affected service and compare the deploy timestamp with the onset of errors. A deploy within a few minutes before onset is a strong signal; a deploy tens of minutes earlier is weak.
2. Query error logs. New stack traces, KeyError, NullPointerException or schema errors tagged with the new build version point to a code regression.
3. Confirm the errors are absent before the deploy time. Do not blame a deploy only because it is the most recent change; check that error onset follows it.
4. Rule out dependencies: database latency and upstream error rates should look normal.

## Mitigation
If the deploy correlates with onset and errors reference the new build, roll back to the previous version. Rollback is a state-changing action and requires human approval. Verify error rate recovers within 5 minutes after rollback.

## Do not
Do not roll back when evidence points to a database, dependency or traffic cause; rollback will not help and delays the real fix.
