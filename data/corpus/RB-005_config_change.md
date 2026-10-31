# RB-005 Runbook: Configuration change

## Symptoms
Behaviour changes abruptly with no code release and no traffic change. Host-level metrics look normal. Errors often name a limit or setting: pool exhausted, timeout exceeded, rate limited, or a feature path failing.

## Diagnosis
1. Call get_deployments and look for entries of type config near the onset. Config changes appear alongside deploys.
2. Read logs for config reload messages and for errors that quote a setting name and value.
3. Distinguish from a database slowdown: a config-induced pool problem shows high pool wait in the application while database latency itself is normal.
4. Compare old and new values in the change record.

## Mitigation
Revert the configuration to the previous version. Config revert uses the same rollback tool and requires human approval. Verify recovery after the revert.

## Do not
Do not scale up the database or restart services before checking recent config changes.
