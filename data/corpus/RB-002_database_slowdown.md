# RB-002 Runbook: Database slowdown

## Symptoms
Request latency climbs progressively and timeouts rise. Application logs show timeouts acquiring connections from the pool or query timeouts. Application CPU and memory are normal. Errors grow with latency rather than appearing as a step.

## Diagnosis
1. Query metrics on the database service: db_query_latency_p99, db_connections, lock_waits. Rising latency with connections near the limit indicates saturation.
2. Check the application metric db_pool_wait_ms; a rising value means requests are queueing for the database.
3. Read database logs for slow queries (sequential scans, missing indexes), lock waits blocked by long transactions, or replica lag.
4. Check recent deployments but do not assume they are the cause; a database that was already degrading before a deploy is not caused by it.

## Mitigation
Kill the blocking transaction, add the missing index, or fail reads over to a healthy replica. Rolling back an application release does not fix a database problem. Escalate to the database on-call.

## Do not
Do not restart the database without approval; do not roll back the application release unless the deploy is shown to cause the load.
