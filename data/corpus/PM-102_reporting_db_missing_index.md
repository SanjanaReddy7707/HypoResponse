# PM-102 Postmortem: Reporting-db latency from a missing index

## Summary
Latency on the reporting service climbed over 40 minutes and then requests timed out. A release had gone out 55 minutes before the page, so the first hypothesis was a bad deploy.

## Timeline
The responder compared the deploy time with onset: errors started 50 minutes after the deploy and ramped gradually, unlike a step. Database metrics showed db_query_latency_p99 and connections rising together. Database logs showed a sequential scan on a table that had grown tenfold.

## Root cause
A missing index on a column added to a frequent query; table growth pushed the plan into a full scan.

## Resolution
Index created online. Rollback of the release was considered and rejected because the evidence did not link the release to the latency ramp.

## Lessons
A recent deploy is not a cause until the timing and error signature match. Gradual ramps point away from deploys.
