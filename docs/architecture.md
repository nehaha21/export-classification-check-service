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