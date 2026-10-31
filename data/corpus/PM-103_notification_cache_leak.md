# PM-103 Postmortem: Unbounded cache in notification-worker

## Summary
Pods restarted repeatedly with OOMKilled for two hours; latency degraded over the previous day.

## Timeline
Memory usage rose from 45% to 98% over about nine hours, resetting at each restart. Request rate was flat. A deploy had introduced a lookup cache eight hours before the first restart. Warning logs reported cache size growing with no evictions.

## Root cause
A lookup cache added without a size bound or eviction policy.

## Resolution
Rollback of the release was approved and memory stabilised. A bounded cache was shipped later.

## Lessons
Check the shape of the memory curve over a long window. A leak that begins after a release is a rollback candidate; one that predates recent releases is not.
