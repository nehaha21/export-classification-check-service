# Incident Log

## Incident: Rate limiter interfered with cache load test

**Date:** 2026-10-06  
**Component:** `/classify` API  
**Severity:** Low  
**Status:** Resolved

### What happened

A Locust cache load test was run with 20 concurrent users for 30 seconds while the API rate limiter was configured for 10 requests per client per 60 seconds.

The test produced 390 requests, with 380 requests failing with HTTP 429 responses. The resulting failure rate was 97.44%, so the measurement could not be used to evaluate cache performance.

### Cause

The per-client rate limiter was correctly enforcing its configured limit, but this limit interfered with the cache load-test measurement.

### Resolution

The rate limiter was temporarily raised to 1000 requests per minute for the cache measurement so that rate limiting would not interfere with the test.

After the measurement, the production configuration was restored to 10 requests per minute.

A new cache load test completed with 395 requests, 0% failures, and a p95 latency of 39 ms.

### Lesson

Load tests for individual performance features must isolate other intentional limits that could affect the measurement.