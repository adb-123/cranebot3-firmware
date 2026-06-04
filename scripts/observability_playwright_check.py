#!/usr/bin/env python3
from __future__ import annotations

import json
import os

from playwright.sync_api import sync_playwright

from observability_smoke_common import (
    GRAFANA_URL,
    assert_dashboard_renders_without_no_data_limit,
    assert_app_metrics,
    assert_basic_stack_health,
    assert_loki_log,
    assert_prometheus_app_target,
    assert_prometheus_stack_targets,
    assert_tempo_trace,
    emit_smoke_log,
    emit_smoke_trace,
    smoke_id,
)


def page_json(page) -> dict:
    return json.loads(page.text_content("body") or "{}")


def main() -> None:
    test_id = smoke_id("playwright")
    require_app = os.environ.get("NF_OBSERVABILITY_REQUIRE_APP", "").lower() in {"1", "true", "yes"}
    emit_smoke_log(test_id)
    trace_id = emit_smoke_trace(test_id)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(locale="en-US")
        page = context.new_page()
        page.goto(f"{GRAFANA_URL}/api/health", wait_until="networkidle")
        grafana_health = page_json(page)
        if grafana_health.get("database") != "ok":
            raise AssertionError(f"Grafana health failed: {grafana_health}")
        for datasource in ("Prometheus", "Loki", "Tempo"):
            page.goto(f"{GRAFANA_URL}/api/datasources/name/{datasource}", wait_until="networkidle")
            datasource_payload = page_json(page)
            if datasource_payload.get("name") != datasource:
                raise AssertionError(f"Grafana datasource {datasource} failed: {datasource_payload}")
        page.goto(f"{GRAFANA_URL}/api/dashboards/uid/nf-robot-observability", wait_until="networkidle")
        dashboard_payload = page_json(page)
        if dashboard_payload.get("dashboard", {}).get("title") != "NF Robot Observability":
            raise AssertionError(f"Grafana dashboard provisioning failed: {dashboard_payload}")
        page.goto(f"{GRAFANA_URL}/d/nf-robot-observability/nf-robot-observability", wait_until="domcontentloaded")
        page.wait_for_timeout(6000)
        assert_dashboard_renders_without_no_data_limit(page.text_content("body") or "")
        browser.close()

    assert_basic_stack_health()
    assert_prometheus_stack_targets()
    if require_app:
        assert_app_metrics()
        assert_prometheus_app_target()
    assert_loki_log(test_id)
    assert_tempo_trace(test_id, trace_id)
    print(f"observability playwright smoke passed test_id={test_id}")


if __name__ == "__main__":
    main()
