# PM-104 Postmortem: Address-verification provider outage

## Summary
Signup failures rose to 30% for 25 minutes. Internal dashboards were healthy.

## Timeline
Logs showed timeouts calling the verification provider host and an open circuit breaker. CPU, memory and database metrics were normal. A small release shipped 12 minutes before onset and was initially blamed, but it contained only a logging change and errors named the provider.

## Root cause
Outage at the external address-verification provider.

## Resolution
The feature was degraded to skip verification until the provider recovered. No rollback was performed.

## Lessons
Read the error text: when it names an upstream host, the cause is outside the service. A rollback would have wasted time.
