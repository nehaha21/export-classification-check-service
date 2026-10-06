from src.monitoring.metrics import MetricsRecorder


def test_quality_summary():
    metrics = MetricsRecorder()

    metrics.start("run-1")
    metrics.finish(
        correlation_id="run-1",
        status="clear-for-filing",
        attempts=1,
        tool_calls=2,
    )

    metrics.start("run-2")
    metrics.finish(
        correlation_id="run-2",
        status="needs-review",
        attempts=2,
        tool_calls=3,
        retry_count=1,
    )

    summary = metrics.quality_summary()

    assert summary["success_rate"] == 0.5
    assert summary["retry_rate"] == 0.5
    assert summary["average_latency_seconds"] >= 0.0