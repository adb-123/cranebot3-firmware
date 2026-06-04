# Local Observability

This directory runs a local observability stack for `stringman-headless`:

- Grafana: dashboards and data source explorer at `http://localhost:3000`
- Prometheus: metrics at `http://localhost:9090`
- Loki: logs at `http://localhost:3100`
- Tempo: traces at `http://localhost:3200`
- OpenTelemetry Collector: OTLP ingest on `localhost:4317` and `localhost:4318`
- Promtail: tails `../logs/*.jsonl` and `../logs/*.log`

## Install App Dependencies

```bash
python -m pip install -e ".[host,observability]"
```

Install the browser smoke-test dependencies only when you want to run the
Grafana/Loki/Tempo validation scripts:

```bash
python -m pip install -e ".[observability-test]"
python -m playwright install chromium
```

Selenium uses a local `chromedriver` or `geckodriver` when one is available, and
defaults to Firefox on Linux ARM hosts with `geckodriver` installed. If browser
driver resolution fails, install Chrome/Chromium plus `chromedriver`, or run with
`SELENIUM_BROWSER=firefox`.

## Start The Stack

```bash
docker compose -f observability/docker-compose.yml up -d
```

Open Grafana:

```text
http://localhost:3000/d/nf-robot-observability/nf-robot-observability
```

Anonymous admin access is enabled for local development only.

The stack emits its own health metrics even when `stringman-headless` is not
running. Prometheus scrapes `prometheus`, `grafana`, `loki`, `tempo`,
`promtail`, and `otel-collector` inside the Compose network. Use Prometheus'
Targets page to check stack health:

```text
http://localhost:9090/targets
```

If the app is stopped, `up{job="stringman-headless"}` is `0` and robot-specific
panels show explicit `waiting` or `missing` rows. That means the host metrics
endpoint on `host.docker.internal:9464` is unavailable or the app has not emitted
that metric family yet; it does not mean Grafana, Prometheus, Loki, Tempo,
Promtail, or the OTel collector are down.

## Start The App With Observability

`stringman-headless` exposes Prometheus metrics on `0.0.0.0:9464`, writes
JSON logs to `logs/nf_robot-observability.jsonl`, and exports OTel traces to the
collector on `http://127.0.0.1:4318/v1/traces`.

```bash
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:4318/v1/traces \
NF_PROMETHEUS_HOST=0.0.0.0 \
NF_PROMETHEUS_PORT=9464 \
NF_OBSERVABILITY_LOG=logs/nf_robot-observability.jsonl \
stringman-headless --config bedroom.conf --no_ai --no_ortho --observability-debug
```

Useful overrides:

```bash
NF_OBSERVABILITY_DEBUG=1 stringman-headless --config bedroom.conf --no_ai --no_ortho
stringman-headless --metrics-port 9465 --observability-log logs/custom-observability.jsonl
stringman-headless --no_observability
```

Use `--observability-debug` when the dashboard needs payload-level evidence.
It records bounded telemetry payload summaries at DEBUG level in the JSONL log
without dumping raw frame bytes.

## Validate Data Sources

Run both browser smoke scripts while the stack is running:

```bash
python scripts/observability_playwright_check.py
python scripts/observability_selenium_check.py
```

Each script verifies:

- Grafana health, data source provisioning, dashboard provisioning, and rendered dashboard health.
- Prometheus can scrape the stack targets: Prometheus, Grafana, Loki, Tempo, Promtail, and OTel Collector.
- Loki can ingest and query a smoke log written to `logs/nf_robot-observability.jsonl`.
- Tempo can ingest and find a synthetic OTel trace with a unique smoke test ID.

When `stringman-headless` is running and you want the smoke scripts to require
the app metrics endpoint too, set:

```bash
NF_OBSERVABILITY_REQUIRE_APP=1 python scripts/observability_playwright_check.py
NF_OBSERVABILITY_REQUIRE_APP=1 python scripts/observability_selenium_check.py
```

You can also validate stack scrape health directly in Prometheus with:

```promql
up{job=~"prometheus|grafana|loki|tempo|promtail|otel-collector"}
```

## Ports

| Port | Service | Purpose |
|---:|---|---|
| 3000 | Grafana | Dashboard/UI |
| 9090 | Prometheus | Metrics query/storage |
| 3100 | Loki | Log query/ingest |
| 3200 | Tempo | Trace query |
| 4317 | OTel Collector | OTLP gRPC ingest |
| 4318 | OTel Collector | OTLP HTTP ingest |
| 9464 | stringman-headless | Prometheus metrics endpoint |

## Metrics Added

- `nf_robot_anchor_poses`
- `nf_robot_app_exporter_ready`
- `nf_robot_app_info`
- `nf_robot_calibration_anchor_sees_origin`
- `nf_robot_calibration_origin_visible_anchors`
- `nf_robot_command_duration_seconds`
- `nf_robot_commanded_grip`
- `nf_robot_commanded_grip_last_seen_timestamp_seconds`
- `nf_robot_commanded_grip_present`
- `nf_robot_commanded_velocity_last_seen_timestamp_seconds`
- `nf_robot_commanded_velocity_meters_per_second`
- `nf_robot_commanded_velocity_norm_meters_per_second`
- `nf_robot_commands_total`
- `nf_robot_component_connected`
- `nf_robot_component_error`
- `nf_robot_component_expected`
- `nf_robot_component_last_seen_age_seconds`
- `nf_robot_component_last_seen_timestamp_seconds`
- `nf_robot_component_motor_enabled`
- `nf_robot_component_temperature_celsius`
- `nf_robot_component_video_connected`
- `nf_robot_component_video_status`
- `nf_robot_component_waiting`
- `nf_robot_component_websocket_connections_total`
- `nf_robot_component_websocket_status`
- `nf_robot_event_loop_lag_seconds`
- `nf_robot_gantry_sightings`
- `nf_robot_grip_cam_prediction`
- `nf_robot_gripper_sensor`
- `nf_robot_gripper_sensor_present`
- `nf_robot_gripper_sensors_available`
- `nf_robot_gripper_sensors_last_seen_timestamp_seconds`
- `nf_robot_lerobot_episode_status`
- `nf_robot_line_slack`
- `nf_robot_line_tension_newtons`
- `nf_robot_max_safe_tension_newtons`
- `nf_robot_named_position_meters`
- `nf_robot_named_position_visible`
- `nf_robot_observability_debug_enabled`
- `nf_robot_operation_active`
- `nf_robot_operation_last_seen_timestamp_seconds`
- `nf_robot_operation_progress_percent`
- `nf_robot_position_estimate_age_seconds`
- `nf_robot_position_estimate_available`
- `nf_robot_position_estimate_last_seen_timestamp_seconds`
- `nf_robot_position_factor_value`
- `nf_robot_position_meters`
- `nf_robot_safety_tension_stops_total`
- `nf_robot_swing_cancellation_enabled`
- `nf_robot_swing_cancellation_last_seen_timestamp_seconds`
- `nf_robot_swing_cancellation_present`
- `nf_robot_target_list_last_seen_timestamp_seconds`
- `nf_robot_targets_by_status`
- `nf_robot_targets_known`
- `nf_robot_telemetry_buffer_items`
- `nf_robot_telemetry_flush_bytes_total`
- `nf_robot_telemetry_flush_duration_seconds`
- `nf_robot_telemetry_flushes_total`
- `nf_robot_telemetry_items_total`
- `nf_robot_telemetry_last_seen_timestamp_seconds`
- `nf_robot_telemetry_payload_expected`
- `nf_robot_telemetry_payload_seen`
- `nf_robot_tension_available`
- `nf_robot_ui_clients`
- `nf_robot_up`
- `nf_robot_uplink_last_seen_timestamp_seconds`
- `nf_robot_uplink_online`
- `nf_robot_uptime_seconds`
- `nf_robot_velocity_meters_per_second`
- `nf_robot_velocity_norm_meters_per_second`
- `nf_robot_video_bytes_total`
- `nf_robot_video_detection_rate_per_second`
- `nf_robot_video_feed_ready`
- `nf_robot_video_feed_ready_last_seen_timestamp_seconds`
- `nf_robot_video_framerate`
- `nf_robot_video_frames_total`
- `nf_robot_video_jpeg_encode_duration_seconds`
- `nf_robot_video_latency_seconds`
- `nf_robot_video_stats_available`
- `nf_robot_video_stats_last_seen_timestamp_seconds`
- `nf_robot_video_stream_clients`
