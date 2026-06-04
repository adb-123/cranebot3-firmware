from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://127.0.0.1:3000").rstrip("/")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:9090").rstrip("/")
LOKI_URL = os.environ.get("LOKI_URL", "http://127.0.0.1:3100").rstrip("/")
TEMPO_URL = os.environ.get("TEMPO_URL", "http://127.0.0.1:3200").rstrip("/")
OTLP_TRACES_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
APP_METRICS_URL = os.environ.get("APP_METRICS_URL", "http://127.0.0.1:9464/metrics")
SMOKE_SERVICE = os.environ.get("NF_OBSERVABILITY_SMOKE_SERVICE", "nf-robot-observability-smoke")
STACK_JOBS = ("prometheus", "grafana", "loki", "tempo", "promtail", "otel-collector")


def get_text(url: str, timeout: float = 10.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {body}") from exc


def get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    return json.loads(get_text(url, timeout=timeout))


def query_params(base_url: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    return get_json(f"{base_url}{path}?{query}")


def wait_until(name: str, fn, timeout: float = 45.0, interval: float = 2.0):
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            value = fn()
            if value:
                return value
        except Exception as exc:  # noqa: BLE001 - test helper reports final failure.
            last_error = exc
        time.sleep(interval)
    if last_error is not None:
        raise RuntimeError(f"{name} did not become ready: {last_error}") from last_error
    raise RuntimeError(f"{name} did not become ready")


def assert_grafana_datasource(name: str) -> dict[str, Any]:
    data = get_json(f"{GRAFANA_URL}/api/datasources/name/{urllib.parse.quote(name)}")
    if data.get("name") != name:
        raise AssertionError(f"Grafana datasource {name!r} not provisioned: {data}")
    return data


def assert_basic_stack_health() -> None:
    grafana = get_json(f"{GRAFANA_URL}/api/health")
    if grafana.get("database") != "ok":
        raise AssertionError(f"Grafana health not ok: {grafana}")
    get_text(f"{PROMETHEUS_URL}/-/ready")
    get_text(f"{LOKI_URL}/ready")

    def tempo_ready() -> bool:
        text = get_text(f"{TEMPO_URL}/ready")
        return "ready" in text.lower()

    wait_until("Tempo ready", tempo_ready, timeout=75.0)
    for datasource in ("Prometheus", "Loki", "Tempo"):
        assert_grafana_datasource(datasource)


def assert_prometheus_stack_targets() -> None:
    result = query_params(PROMETHEUS_URL, "/api/v1/query", {"query": "up"})
    values = result.get("data", {}).get("result", [])
    by_job = {item.get("metric", {}).get("job"): item.get("value", [None, "0"])[1] for item in values}
    missing = [job for job in STACK_JOBS if by_job.get(job) != "1"]
    if missing:
        raise AssertionError(f"Prometheus stack targets are not all up: missing={missing} values={by_job}")


def emit_smoke_log(test_id: str) -> None:
    log_path = ROOT / "logs" / "nf_robot-observability.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "level": "INFO",
        "logger": "nf_robot.observability.smoke",
        "service": "stringman-headless",
        "host": "observability-smoke",
        "message": f"observability smoke log {test_id}",
        "trace_id": test_id.replace("-", "")[:32].ljust(32, "0"),
    }
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        fp.write("\n")


def emit_smoke_trace(test_id: str) -> str:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError("Install nf_robot[observability] before running trace smoke tests") from exc

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": SMOKE_SERVICE,
                "service.version": "smoke",
                "deployment.environment": "local",
            }
        )
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_TRACES_ENDPOINT)))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(SMOKE_SERVICE)
    with tracer.start_as_current_span("observability.smoke_trace") as span:
        span.set_attribute("smoke.test_id", test_id)
        span.set_attribute("component", "observability-test")
        trace_id = f"{span.get_span_context().trace_id:032x}"
    provider.force_flush(timeout_millis=5000)
    provider.shutdown()
    return trace_id


def assert_app_metrics() -> None:
    text = get_text(APP_METRICS_URL)
    if "nf_robot_up" not in text:
        raise AssertionError(f"App metrics endpoint is reachable but nf_robot_up is absent: {APP_METRICS_URL}")


def assert_prometheus_app_target() -> None:
    result = query_params(PROMETHEUS_URL, "/api/v1/query", {"query": 'up{job="stringman-headless"}'})
    values = result.get("data", {}).get("result", [])
    if not values:
        raise AssertionError("Prometheus has no stringman-headless target. Is the app running on :9464?")
    if not any(item.get("value", [None, "0"])[1] == "1" for item in values):
        raise AssertionError(f"Prometheus stringman-headless target is not up: {values}")


def assert_loki_log(test_id: str) -> dict[str, Any]:
    def check():
        now = time.time()
        data = query_params(
            LOKI_URL,
            "/loki/api/v1/query_range",
            {
                "query": f'{{app="nf_robot"}} |= "observability smoke log {test_id}"',
                "start": int((now - 300) * 1_000_000_000),
                "end": int((now + 30) * 1_000_000_000),
                "limit": 20,
            },
        )
        streams = data.get("data", {}).get("result", [])
        return data if streams else None

    return wait_until("Loki smoke log", check, timeout=60.0)


def assert_dashboard_renders_without_no_data_limit(page_text: str, *, max_no_data: int = 2) -> None:
    if "failed to load its application files" in page_text:
        raise AssertionError("Grafana frontend did not render; application files fallback is visible")
    no_data_count = page_text.count("No data")
    if no_data_count > max_no_data:
        raise AssertionError(f"Dashboard still has too many No data panels: {no_data_count}")


def assert_tempo_trace(test_id: str, trace_id: str | None = None) -> dict[str, Any]:
    def attr_value(attr: dict[str, Any]) -> str | None:
        value = attr.get("value", {})
        for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if key in value:
                return str(value[key])
        return None

    def trace_has_test_id(trace: dict[str, Any]) -> bool:
        for batch in trace.get("batches", []):
            for scope in batch.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    for attr in span.get("attributes", []):
                        if attr.get("key") == "smoke.test_id" and attr_value(attr) == test_id:
                            return True
        return False

    def check_trace_id():
        if not trace_id:
            return None
        try:
            trace = get_json(f"{TEMPO_URL}/api/traces/{trace_id}")
        except Exception:
            return None
        return trace if trace_has_test_id(trace) else None

    if trace_id:
        return wait_until("Tempo smoke trace", check_trace_id, timeout=90.0)

    def check():
        data = query_params(
            TEMPO_URL,
            "/api/search",
            {"limit": 50},
        )
        traces = data.get("traces", []) or data.get("data", {}).get("traces", [])
        for trace_summary in traces:
            if trace_summary.get("rootServiceName") != SMOKE_SERVICE:
                continue
            trace_id = trace_summary.get("traceID")
            if not trace_id:
                continue
            trace = get_json(f"{TEMPO_URL}/api/traces/{trace_id}")
            if trace_has_test_id(trace):
                return trace
        return None

    return wait_until("Tempo smoke trace", check, timeout=90.0)


def smoke_id(prefix: str) -> str:
    return f"{prefix}-{int(time.time() * 1000)}"
