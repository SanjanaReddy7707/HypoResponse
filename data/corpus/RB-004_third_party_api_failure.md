# RB-004 Runbook: Third-party API failure

## Symptoms
Errors concentrate on features that call an external provider. Logs show upstream 5xx responses, timeouts calling the provider host, or an open circuit breaker. Internal resources (CPU, memory, database) look healthy. Onset is a step.

## Diagnosis
1. Query service logs for upstream errors and name the provider host involved.
2. Check upstream_5xx_rate and dependency_latency; a jump to the timeout value on the dependency with healthy internal metrics points to the provider.
3. Check the provider status page if referenced in logs.
4. Check recent deployments, but a benign release made shortly before onset does not explain upstream errors if the errors name the provider.

## Mitigation
Enable fallback or degrade the feature, tune the circuit breaker, and notify the provider. Do not roll back our own releases; the failure is outside our system.

## Do not
Do not roll back or change configuration to compensate for an external outage without approval.
