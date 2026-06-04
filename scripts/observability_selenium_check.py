#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService

from observability_smoke_common import (
    GRAFANA_URL,
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


def browser_fetch_json(driver, url: str) -> dict:
    payload = driver.execute_async_script(
        """
        const url = arguments[0];
        const done = arguments[arguments.length - 1];
        fetch(url)
            .then(async response => {
                const text = await response.text();
                try {
                    done(JSON.parse(text));
                } catch (error) {
                    done({ "__error": String(error), "text": text });
                }
            })
            .catch(error => done({ "__error": String(error) }));
        """,
        url,
    )
    if payload.get("__error"):
        raise AssertionError(f"Browser fetch failed for {url}: {payload}")
    return payload


def first_existing(*paths: str | None) -> str | None:
    for path in paths:
        if path and Path(path).exists():
            return path
    return None


def make_driver():
    default_browser = "chrome"
    if platform.machine().lower() in {"aarch64", "arm64"} and shutil.which("geckodriver"):
        default_browser = "firefox"
    browser = os.environ.get("SELENIUM_BROWSER", default_browser).lower()
    if browser == "firefox":
        options = FirefoxOptions()
        options.add_argument("-headless")
        firefox_binary = first_existing(
            os.environ.get("FIREFOX_BINARY"),
            "/snap/firefox/current/usr/lib/firefox/firefox",
            shutil.which("firefox"),
        )
        if firefox_binary:
            options.binary_location = firefox_binary
        geckodriver = os.environ.get("GECKODRIVER") or shutil.which("geckodriver")
        if geckodriver:
            return webdriver.Firefox(options=options, service=FirefoxService(executable_path=geckodriver))
        return webdriver.Firefox(options=options)
    options = ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    chrome_binary = (
        os.environ.get("CHROME_BINARY")
        or shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if chrome_binary:
        options.binary_location = chrome_binary
    chromedriver = os.environ.get("CHROMEDRIVER") or shutil.which("chromedriver")
    if chromedriver:
        return webdriver.Chrome(options=options, service=ChromeService(executable_path=chromedriver))
    return webdriver.Chrome(options=options)


def main() -> None:
    test_id = smoke_id("selenium")
    require_app = os.environ.get("NF_OBSERVABILITY_REQUIRE_APP", "").lower() in {"1", "true", "yes"}
    emit_smoke_log(test_id)
    trace_id = emit_smoke_trace(test_id)

    driver = make_driver()
    try:
        driver.get(GRAFANA_URL)
        grafana_health = browser_fetch_json(driver, f"{GRAFANA_URL}/api/health")
        if grafana_health.get("database") != "ok":
            raise AssertionError(f"Grafana health failed: {grafana_health}")
        for datasource in ("Prometheus", "Loki", "Tempo"):
            datasource_payload = browser_fetch_json(driver, f"{GRAFANA_URL}/api/datasources/name/{datasource}")
            if datasource_payload.get("name") != datasource:
                raise AssertionError(f"Grafana datasource {datasource} failed: {datasource_payload}")
        dashboard_payload = browser_fetch_json(driver, f"{GRAFANA_URL}/api/dashboards/uid/nf-robot-observability")
        if dashboard_payload.get("dashboard", {}).get("title") != "NF Robot Observability":
            raise AssertionError(f"Grafana dashboard provisioning failed: {dashboard_payload}")
        driver.get(f"{GRAFANA_URL}/d/nf-robot-observability/nf-robot-observability")
    finally:
        driver.quit()

    assert_basic_stack_health()
    assert_prometheus_stack_targets()
    if require_app:
        assert_app_metrics()
        assert_prometheus_app_target()
    assert_loki_log(test_id)
    assert_tempo_trace(test_id, trace_id)
    print(f"observability selenium smoke passed test_id={test_id}")


if __name__ == "__main__":
    main()
