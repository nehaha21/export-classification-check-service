## D6 Load Testing

The declared p95 latency SLO is 30 seconds, configured in `config/config.yaml`.

Load testing was performed with Locust against the `/classify` endpoint using the deterministic retrieval, classification-model, and verification-model stubs.

| Concurrent users | Requests | Failure rate | RPS | p50 | p95 | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 93 | 0% | 3.23 | 16 ms | 26 ms | 30 ms |
| 10 | 181 | 0% | 6.26 | 15 ms | 26 ms | 97 ms |
| 20 | 353 | 0% | 12.15 | 15 ms | 50 ms | 210 ms |

All three tests remained within the declared p95 SLO with zero request failures.

As concurrency increased from 5 to 20 users, throughput increased from 3.23 RPS to 12.15 RPS. At 20 users, p95 latency increased to 50 ms, indicating increased latency under higher concurrency, while the service continued to complete requests successfully.

These measurements use deterministic LLM stubs and therefore measure the service/load-test path rather than real LLM latency.




## D7 Caching and Rate Limiting

The API uses an in-memory response cache with a 60-second TTL and a per-client fixed-window rate limiter.

### Cache verification

Repeated identical requests were tested against `/classify`.

- First request: `X-Cache: MISS`
- Second identical request: `X-Cache: HIT`
- Both requests returned HTTP 200.

### Cache load-test measurement

Locust was run with 20 concurrent users for 30 seconds using repeated identical requests.

| Metric | Cached run |
|---|---:|
| Requests | 395 |
| Failure rate | 0% |
| RPS | 13.65 |
| p50 | 3 ms |
| p95 | 39 ms |
| Max | 59 ms |

The earlier D6 20-user baseline had a p95 of 50 ms and throughput of 12.15 RPS. The cached run reduced p95 by 11 ms (22%) and increased throughput to 13.65 RPS.

The load-test rate limit was temporarily raised to 1000 requests/minute so that rate limiting did not interfere with the cache measurement. It was restored to the production configuration of 10 requests/minute afterward.

### Rate-limit verification

The per-client rate limiter was tested with repeated requests from the same local client.

- Requests within the configured limit returned HTTP 200.
- Requests after the limit returned HTTP 429.