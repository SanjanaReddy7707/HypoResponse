# PM-101 Postmortem: Invoice-service schema regression

## Summary
Invoice-service returned HTTP 500 for 38 minutes after release v5.2.0. Error rate went from under 1% to 27% immediately after the deploy.

## Timeline
Release v5.2.0 completed at 14:02. Errors began at 14:04. The on-call first suspected the database because of a few slow query warnings, but database latency was flat. Logs showed a KeyError for a removed field tagged with the new build.

## Root cause
The release removed a field that downstream callers still sent. The code path had no tests for the old schema.

## Resolution
Rollback to v5.1.3 approved at 14:36; error rate recovered in 3 minutes.

## Lessons
Slow query warnings that were present for days were a red herring. Always compare onset time with metrics before blaming the database. Add contract tests for request schemas.
