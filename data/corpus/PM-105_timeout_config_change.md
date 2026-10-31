# PM-105 Postmortem: Client timeout config reduced by mistake

## Summary
Search requests failed with outbound timeouts after a configuration update, with no code release.

## Timeline
Error rate stepped up two minutes after a config change. get_deployments listed a config entry reducing a client timeout from 3000ms to 30ms. Logs quoted the new timeout in each error. Host metrics and database latency were normal; an initial theory of database slowness was dropped.

## Root cause
A typo in a configuration value (30 instead of 3000).

## Resolution
Config reverted to the previous version after approval; errors stopped immediately.

## Lessons
Config changes appear in the deployment history; check them first when there is no release. Use validation on configuration ranges.
