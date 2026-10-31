# RB-003 Runbook: Memory leak

## Symptoms
Memory usage rises steadily over hours, followed by OOMKilled restarts, long garbage-collection pauses, and finally latency and errors. Degradation is gradual and looks like a slow ramp, not a step.

## Diagnosis
1. Query memory_usage over a long window (last_12h). A sustained upward ramp that resets after restarts indicates a leak.
2. Check restarts and logs for OutOfMemoryError, OOMKilled and full GC messages.
3. Look for the leaking component in warning logs: unbounded caches, growing handle or descriptor counts, accumulating listeners.
4. Check get_deployments over a long window. If memory growth began right after a release, that release likely introduced the leak. If growth predates recent releases, it is a long-standing leak.
5. Rule out a traffic increase: request_rate should not explain the memory growth.

## Mitigation
Restart pods to restore service temporarily. If a recent release introduced the leak, roll back with human approval. Otherwise open a defect for the leaking component; rollback will not help.

## Do not
Do not conclude capacity shortage from high memory alone; check the shape of the curve and the request rate.
