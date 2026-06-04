from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover - Python 3.8 fallback only.
    from importlib_metadata import version  # type: ignore


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

EXPECTED_TELEMETRY_PAYLOADS = (
    "component_conn_status",
    "vid_stats",
    "pos_estimate",
    "pos_factors_debug",
    "gantry_sightings",
    "new_anchor_poses",
    "named_position",
    "last_commanded_vel",
    "raw_commanded_vel",
    "pop_message",
    "grip_sensors",
    "grip_cam_preditions",
    "target_list",
    "video_ready",
    "uplink_status",
    "episode_control",
    "operation_progress",
    "last_commanded_grip",
    "swing_cancellation_state",
    "visibility_states",
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _label(value: Any, fallback: str = "unknown") -> str:
    text = str(value if value is not None else fallback)
    return text[:128] or fallback


def _enum_number(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        enum_value = getattr(value, "value", None)
        try:
            return int(enum_value)
        except Exception:
            return 0


def _vec_value(vec: Any, axis: str) -> float:
    try:
        return float(getattr(vec, axis))
    except Exception:
        return 0.0


def _vec_norm(vec: Any) -> float:
    return math.sqrt(sum(_vec_value(vec, axis) ** 2 for axis in ("x", "y", "z")))


def _vec_dict(vec: Any) -> dict[str, float] | None:
    if vec is None:
        return None
    return {axis: _vec_value(vec, axis) for axis in ("x", "y", "z")}


def _enum_name(value: Any) -> str:
    return _label(getattr(value, "name", None) or value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if math.isfinite(result):
        return result
    return default


def _as_debug_json(value: Any, *, depth: int = 0, max_items: int = 12) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if depth >= 4:
        return _label(type(value).__name__)
    if isinstance(value, dict):
        return {
            _label(key): _as_debug_json(val, depth=depth + 1, max_items=max_items)
            for key, val in list(value.items())[:max_items]
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        summary = [_as_debug_json(item, depth=depth + 1, max_items=max_items) for item in items[:max_items]]
        if len(items) > max_items:
            summary.append({"truncated_count": len(items) - max_items})
        return summary
    if is_dataclass(value):
        return {
            field.name: _as_debug_json(getattr(value, field.name), depth=depth + 1, max_items=max_items)
            for field in fields(value)[:max_items]
        }
    if hasattr(value, "to_dict"):
        try:
            return _as_debug_json(value.to_dict(), depth=depth + 1, max_items=max_items)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _as_debug_json(
            {
                key: val
                for key, val in list(vars(value).items())[:max_items]
                if not key.startswith("_") and not callable(val)
            },
            depth=depth + 1,
            max_items=max_items,
        )
    return _label(value)


class JsonTraceLogHandler(logging.Handler):
    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "ts": time.time(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "service": OBS.service_name,
                "host": socket.gethostname(),
            }
            trace_id, span_id = OBS.current_trace_ids()
            if trace_id:
                payload["trace_id"] = trace_id
            if span_id:
                payload["span_id"] = span_id
            event = getattr(record, "observability_event", None)
            if event:
                payload["event"] = event
            event_payload = getattr(record, "observability_payload", None)
            if event_payload is not None:
                payload["payload"] = event_payload
            if record.exc_info:
                payload["exception"] = self._formatter.formatException(record.exc_info)
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))
                fp.write("\n")
        except Exception:
            self.handleError(record)


class Observability:
    def __init__(self) -> None:
        self.service_name = "nf_robot"
        self.service_version = "unknown"
        self.enabled = False
        self.metrics_enabled = False
        self.tracing_enabled = False
        self._metrics_started = False
        self._log_handler: JsonTraceLogHandler | None = None

        self._trace = None
        self._tracer = None
        self._counters: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}
        self._info: dict[str, Any] = {}
        self._operation_labels: set[str] = set()
        self._target_status_labels: set[tuple[str, str]] = set()
        self._expected_components: dict[tuple[str, str], float] = {}
        self._component_seen_at: dict[tuple[str, str], float] = {}
        self.debug_enabled = False

    def configure(
        self,
        *,
        service_name: str,
        robot_id: str | None = None,
        metrics_port: int | None = None,
        metrics_host: str | None = None,
        log_path: str | Path | None = None,
        log_level: str | int | None = None,
    ) -> None:
        self.service_name = service_name
        try:
            self.service_version = version("nf_robot")
        except Exception:
            self.service_version = "unknown"

        self.enabled = _env_bool("NF_OBSERVABILITY_ENABLED", True)
        if not self.enabled:
            return

        metrics_host = metrics_host or os.environ.get("NF_PROMETHEUS_HOST", "0.0.0.0")
        metrics_port = metrics_port or int(os.environ.get("NF_PROMETHEUS_PORT", "9464"))
        log_path = log_path or os.environ.get("NF_OBSERVABILITY_LOG", "logs/nf_robot-observability.jsonl")

        self._configure_logging(log_path, log_level)
        self._configure_metrics(metrics_host, metrics_port, robot_id)
        self._configure_tracing()

    def _configure_logging(self, log_path: str | Path, log_level: str | int | None = None) -> None:
        if self._log_handler is not None:
            if log_level is not None:
                resolved = self._resolve_log_level(log_level)
                self._log_handler.setLevel(resolved)
                self.debug_enabled = resolved <= logging.DEBUG
            return
        self._log_handler = JsonTraceLogHandler(log_path)
        resolved_level = self._resolve_log_level(log_level)
        self.debug_enabled = resolved_level <= logging.DEBUG
        self._log_handler.setLevel(resolved_level)
        nf_logger = logging.getLogger("nf_robot")
        if nf_logger.level in (logging.NOTSET, logging.WARNING, logging.ERROR, logging.CRITICAL):
            nf_logger.setLevel(min(logging.INFO, resolved_level))
        nf_logger.addHandler(self._log_handler)

    def _resolve_log_level(self, log_level: str | int | None) -> int:
        if log_level is None:
            if _env_bool("NF_OBSERVABILITY_DEBUG", False):
                return logging.DEBUG
            log_level = os.environ.get("NF_OBSERVABILITY_LOG_LEVEL", "INFO")
        if isinstance(log_level, int):
            return log_level
        resolved = logging.getLevelName(str(log_level).upper())
        if isinstance(resolved, int):
            return resolved
        return logging.INFO

    def _configure_metrics(self, host: str, port: int, robot_id: str | None) -> None:
        try:
            from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
        except ImportError:
            logger.warning("prometheus_client unavailable; Prometheus metrics are disabled")
            return

        if self.metrics_enabled:
            return

        if not self._metrics_started:
            start_http_server(port, addr=host)
            self._metrics_started = True
            logger.info("Prometheus metrics listening on %s:%s", host, port)

        self.metrics_enabled = True
        self._info["app"] = Info("nf_robot_app", "Stringman app metadata")
        self._info["app"].info(
            {
                "service_name": self.service_name,
                "service_version": self.service_version,
                "robot_id": robot_id or "unknown",
                "host": socket.gethostname(),
            }
        )
        self._gauges["up"] = Gauge("nf_robot_up", "Whether the nf_robot process is running")
        self._gauges["exporter_ready"] = Gauge(
            "nf_robot_app_exporter_ready", "Whether the observability exporter initialized successfully"
        )
        self._gauges["debug_enabled"] = Gauge(
            "nf_robot_observability_debug_enabled", "Whether debug-level observability recording is enabled"
        )
        self._gauges["uptime"] = Gauge("nf_robot_uptime_seconds", "Process uptime in seconds")
        self._gauges["ui_clients"] = Gauge("nf_robot_ui_clients", "Connected local UI websocket clients")
        self._gauges["telemetry_buffer"] = Gauge("nf_robot_telemetry_buffer_items", "Buffered telemetry items")
        self._gauges["video_clients"] = Gauge(
            "nf_robot_video_stream_clients", "Connected local MJPEG stream clients", ["port"]
        )
        self._gauges["event_loop_lag"] = Gauge("nf_robot_event_loop_lag_seconds", "Last event loop lag sample")
        self._gauges["line_tension"] = Gauge(
            "nf_robot_line_tension_newtons", "Passive safety EMA line tension in newtons", ["line"]
        )
        self._gauges["max_safe_tension"] = Gauge(
            "nf_robot_max_safe_tension_newtons", "Passive safety configured max safe line tension"
        )
        self._gauges["component_connected"] = Gauge(
            "nf_robot_component_connected",
            "Current component websocket connection state",
            ["component", "kind"],
        )
        self._gauges["component_video_connected"] = Gauge(
            "nf_robot_component_video_connected",
            "Current component video connection state",
            ["component", "kind"],
        )
        self._gauges["component_websocket_status"] = Gauge(
            "nf_robot_component_websocket_status",
            "Current component websocket status, one status label is 1 for each component",
            ["component", "kind", "status"],
        )
        self._gauges["component_video_status"] = Gauge(
            "nf_robot_component_video_status",
            "Current component video status, one status label is 1 for each component",
            ["component", "kind", "status"],
        )
        self._gauges["component_temperature"] = Gauge(
            "nf_robot_component_temperature_celsius",
            "Component Raspberry Pi SOC temperature in degrees Celsius",
            ["component", "kind"],
        )
        self._gauges["component_motor_enabled"] = Gauge(
            "nf_robot_component_motor_enabled",
            "Whether the component motor reports torque enabled",
            ["component", "kind"],
        )
        self._gauges["component_error"] = Gauge(
            "nf_robot_component_error",
            "Whether the latest component status carried an error message",
            ["component", "kind"],
        )
        self._gauges["component_last_seen"] = Gauge(
            "nf_robot_component_last_seen_timestamp_seconds",
            "Unix timestamp of the latest component status telemetry",
            ["component", "kind"],
        )
        self._gauges["component_expected"] = Gauge(
            "nf_robot_component_expected",
            "Whether the app expects this component to exist from config",
            ["component", "kind"],
        )
        self._gauges["component_waiting"] = Gauge(
            "nf_robot_component_waiting",
            "Whether the app is still waiting for this expected component to report connected",
            ["component", "kind"],
        )
        self._gauges["component_last_seen_age"] = Gauge(
            "nf_robot_component_last_seen_age_seconds",
            "Age of the latest component status telemetry; zero means never seen",
            ["component", "kind"],
        )
        self._gauges["telemetry_last_seen"] = Gauge(
            "nf_robot_telemetry_last_seen_timestamp_seconds",
            "Unix timestamp of the latest telemetry payload by item type",
            ["item_type"],
        )
        self._gauges["telemetry_payload_expected"] = Gauge(
            "nf_robot_telemetry_payload_expected",
            "Whether this telemetry payload type is expected by the observer contract",
            ["item_type"],
        )
        self._gauges["telemetry_payload_seen"] = Gauge(
            "nf_robot_telemetry_payload_seen",
            "Whether this telemetry payload type has been seen since process start",
            ["item_type"],
        )
        self._gauges["video_detection_rate"] = Gauge(
            "nf_robot_video_detection_rate_per_second",
            "AprilTag detections per second summed across cameras",
        )
        self._gauges["video_latency"] = Gauge(
            "nf_robot_video_latency_seconds",
            "Frame capture to observer processing latency",
        )
        self._gauges["video_framerate"] = Gauge(
            "nf_robot_video_framerate",
            "Average camera ingest frame rate across cameras",
        )
        self._gauges["video_stats_last_seen"] = Gauge(
            "nf_robot_video_stats_last_seen_timestamp_seconds",
            "Unix timestamp of the latest video statistics telemetry",
        )
        self._gauges["video_stats_available"] = Gauge(
            "nf_robot_video_stats_available",
            "Whether video statistics telemetry has been observed",
        )
        self._gauges["video_feed_ready"] = Gauge(
            "nf_robot_video_feed_ready",
            "Whether a video feed is ready by feed, component, and transport",
            ["feed", "component", "transport"],
        )
        self._gauges["video_feed_ready_last_seen"] = Gauge(
            "nf_robot_video_feed_ready_last_seen_timestamp_seconds",
            "Unix timestamp of the latest video-ready telemetry by feed and component",
            ["feed", "component"],
        )
        self._gauges["position_factor"] = Gauge(
            "nf_robot_position_factor_value",
            "Position estimator diagnostic value by factor and axis",
            ["factor", "axis"],
        )
        self._gauges["gantry_sightings"] = Gauge(
            "nf_robot_gantry_sightings", "Number of gantry sightings in the latest telemetry item"
        )
        self._gauges["anchor_poses"] = Gauge(
            "nf_robot_anchor_poses", "Number of anchor poses in the latest setup telemetry item"
        )
        self._gauges["named_position_visible"] = Gauge(
            "nf_robot_named_position_visible", "Whether a named position marker is visible", ["name"]
        )
        self._gauges["named_position"] = Gauge(
            "nf_robot_named_position_meters", "Named position marker coordinates", ["name", "axis"]
        )
        self._gauges["robot_position"] = Gauge(
            "nf_robot_position_meters",
            "Robot position estimate by part and axis",
            ["part", "axis"],
        )
        self._gauges["robot_velocity"] = Gauge(
            "nf_robot_velocity_meters_per_second",
            "Gantry velocity estimate by axis",
            ["axis"],
        )
        self._gauges["robot_velocity_norm"] = Gauge(
            "nf_robot_velocity_norm_meters_per_second",
            "Gantry velocity vector norm",
        )
        self._gauges["position_estimate_age"] = Gauge(
            "nf_robot_position_estimate_age_seconds",
            "Age of the latest position estimate",
        )
        self._gauges["position_estimate_available"] = Gauge(
            "nf_robot_position_estimate_available",
            "Whether position estimate telemetry has been observed",
        )
        self._gauges["position_estimate_last_seen"] = Gauge(
            "nf_robot_position_estimate_last_seen_timestamp_seconds",
            "Unix timestamp of the latest position estimate telemetry",
        )
        self._gauges["line_slack"] = Gauge(
            "nf_robot_line_slack",
            "Whether each line is currently estimated slack",
            ["line"],
        )
        self._gauges["tension_available"] = Gauge(
            "nf_robot_tension_available",
            "Whether position telemetry is currently carrying line tension values",
        )
        self._gauges["gripper_sensor"] = Gauge(
            "nf_robot_gripper_sensor",
            "Latest gripper sensor values",
            ["sensor"],
        )
        self._gauges["gripper_sensor_present"] = Gauge(
            "nf_robot_gripper_sensor_present",
            "Whether the latest gripper sensor telemetry carried this sensor",
            ["sensor"],
        )
        self._gauges["gripper_sensors_available"] = Gauge(
            "nf_robot_gripper_sensors_available",
            "Whether gripper sensor telemetry has been observed",
        )
        self._gauges["gripper_sensors_last_seen"] = Gauge(
            "nf_robot_gripper_sensors_last_seen_timestamp_seconds",
            "Unix timestamp of the latest gripper sensor telemetry",
        )
        self._gauges["operation_progress"] = Gauge(
            "nf_robot_operation_progress_percent",
            "Progress percent for the latest named operation",
            ["operation"],
        )
        self._gauges["operation_active"] = Gauge(
            "nf_robot_operation_active",
            "Whether a named operation is currently active",
            ["operation"],
        )
        self._gauges["operation_last_seen"] = Gauge(
            "nf_robot_operation_last_seen_timestamp_seconds",
            "Unix timestamp of the latest operation progress telemetry",
            ["operation"],
        )
        self._gauges["commanded_velocity"] = Gauge(
            "nf_robot_commanded_velocity_meters_per_second",
            "Latest commanded velocity by command kind and axis",
            ["kind", "axis"],
        )
        self._gauges["commanded_velocity_norm"] = Gauge(
            "nf_robot_commanded_velocity_norm_meters_per_second",
            "Latest commanded velocity norm by command kind",
            ["kind"],
        )
        self._gauges["commanded_velocity_last_seen"] = Gauge(
            "nf_robot_commanded_velocity_last_seen_timestamp_seconds",
            "Unix timestamp of the latest commanded velocity telemetry by command kind",
            ["kind"],
        )
        self._gauges["commanded_grip"] = Gauge(
            "nf_robot_commanded_grip",
            "Latest commanded gripper values",
            ["command"],
        )
        self._gauges["commanded_grip_present"] = Gauge(
            "nf_robot_commanded_grip_present",
            "Whether the latest commanded gripper telemetry carried this command",
            ["command"],
        )
        self._gauges["commanded_grip_last_seen"] = Gauge(
            "nf_robot_commanded_grip_last_seen_timestamp_seconds",
            "Unix timestamp of the latest commanded gripper telemetry",
        )
        self._gauges["swing_cancellation_enabled"] = Gauge(
            "nf_robot_swing_cancellation_enabled",
            "Whether swing cancellation is currently enabled",
        )
        self._gauges["swing_cancellation_present"] = Gauge(
            "nf_robot_swing_cancellation_present",
            "Whether swing cancellation state telemetry has been observed",
        )
        self._gauges["swing_cancellation_last_seen"] = Gauge(
            "nf_robot_swing_cancellation_last_seen_timestamp_seconds",
            "Unix timestamp of the latest swing cancellation telemetry",
        )
        self._gauges["targets_known"] = Gauge(
            "nf_robot_targets_known",
            "Number of targets in the latest target queue snapshot",
        )
        self._gauges["targets_by_status"] = Gauge(
            "nf_robot_targets_by_status",
            "Number of targets grouped by status and source",
            ["status", "source"],
        )
        self._gauges["target_list_last_seen"] = Gauge(
            "nf_robot_target_list_last_seen_timestamp_seconds",
            "Unix timestamp of the latest target queue snapshot",
        )
        self._gauges["grip_cam_prediction"] = Gauge(
            "nf_robot_grip_cam_prediction",
            "Latest gripper camera model prediction values",
            ["signal"],
        )
        self._gauges["uplink_online"] = Gauge(
            "nf_robot_uplink_online", "Whether the control-plane uplink reports online"
        )
        self._gauges["uplink_last_seen"] = Gauge(
            "nf_robot_uplink_last_seen_timestamp_seconds", "Unix timestamp of the latest uplink status telemetry"
        )
        self._gauges["episode_status"] = Gauge(
            "nf_robot_lerobot_episode_status", "Latest Lerobot episode control status enum value"
        )
        self._gauges["calibration_origin_visible_anchors"] = Gauge(
            "nf_robot_calibration_origin_visible_anchors",
            "Number of anchors seeing the origin calibration card",
        )
        self._gauges["calibration_anchor_sees_origin"] = Gauge(
            "nf_robot_calibration_anchor_sees_origin",
            "Whether an anchor sees the origin calibration card",
            ["anchor"],
        )
        self._counters["commands"] = Counter("nf_robot_commands_total", "Control commands handled", ["command"])
        self._counters["telemetry"] = Counter(
            "nf_robot_telemetry_items_total", "Telemetry items queued for UIs", ["item_type"]
        )
        self._counters["flushes"] = Counter("nf_robot_telemetry_flushes_total", "Telemetry flush batches")
        self._counters["flush_bytes"] = Counter(
            "nf_robot_telemetry_flush_bytes_total", "Telemetry bytes sent to UI sockets"
        )
        self._counters["component_ws"] = Counter(
            "nf_robot_component_websocket_connections_total",
            "Component websocket connection attempts",
            ["component", "outcome"],
        )
        self._counters["video_frames"] = Counter(
            "nf_robot_video_frames_total", "Video frames accepted by streamers", ["streamer"]
        )
        self._counters["video_bytes"] = Counter(
            "nf_robot_video_bytes_total", "Video bytes encoded by local MJPEG streamers", ["port"]
        )
        self._counters["safety_stops"] = Counter(
            "nf_robot_safety_tension_stops_total", "Passive safety stops due to high line tension"
        )
        self._histograms["command_duration"] = Histogram(
            "nf_robot_command_duration_seconds", "Control command handling duration", ["command"]
        )
        self._histograms["flush_duration"] = Histogram(
            "nf_robot_telemetry_flush_duration_seconds", "Telemetry flush duration"
        )
        self._histograms["jpeg_encode"] = Histogram(
            "nf_robot_video_jpeg_encode_duration_seconds", "MJPEG encode duration", ["port"]
        )
        self._gauges["exporter_ready"].set(1)
        self._gauges["debug_enabled"].set(1 if self.debug_enabled else 0)
        self._gauges["up"].set(1)
        for gauge_name in (
            "video_stats_available",
            "position_estimate_available",
            "tension_available",
            "gripper_sensors_available",
            "swing_cancellation_enabled",
            "swing_cancellation_present",
            "targets_known",
            "uplink_online",
            "episode_status",
            "calibration_origin_visible_anchors",
            "gantry_sightings",
            "anchor_poses",
        ):
            gauge = self._gauges.get(gauge_name)
            if gauge is not None:
                gauge.set(0)
        for item_type in EXPECTED_TELEMETRY_PAYLOADS:
            expected = self._gauges.get("telemetry_payload_expected")
            if expected is not None:
                expected.labels(item_type).set(1)
            seen = self._gauges.get("telemetry_payload_seen")
            if seen is not None:
                seen.labels(item_type).set(0)
            last_seen = self._gauges.get("telemetry_last_seen")
            if last_seen is not None:
                last_seen.labels(item_type).set(0)
        for line in range(4):
            slack = self._gauges.get("line_slack")
            if slack is not None:
                slack.labels(str(line)).set(0)
            tension = self._gauges.get("line_tension")
            if tension is not None:
                tension.labels(str(line)).set(0)
            origin = self._gauges.get("calibration_anchor_sees_origin")
            if origin is not None:
                origin.labels(str(line)).set(0)
        for kind in ("total", "raw"):
            velocity = self._gauges.get("commanded_velocity")
            if velocity is not None:
                for axis in ("x", "y", "z"):
                    velocity.labels(kind, axis).set(0)
            velocity_norm = self._gauges.get("commanded_velocity_norm")
            if velocity_norm is not None:
                velocity_norm.labels(kind).set(0)
        for command in ("finger_speed", "wrist_speed"):
            grip = self._gauges.get("commanded_grip")
            if grip is not None:
                grip.labels(command).set(0)
            grip_present = self._gauges.get("commanded_grip_present")
            if grip_present is not None:
                grip_present.labels(command).set(0)
        for sensor in ("range", "angle", "pressure", "wrist", "target_force"):
            sensor_present = self._gauges.get("gripper_sensor_present")
            if sensor_present is not None:
                sensor_present.labels(sensor).set(0)
        for signal in ("move_x", "move_y", "prob_target_in_view", "prob_holding", "grip_angle"):
            prediction = self._gauges.get("grip_cam_prediction")
            if prediction is not None:
                prediction.labels(signal).set(0)
        self._debug_state("metrics_initialized", state="waiting_for_telemetry")

    def _configure_tracing(self) -> None:
        if self.tracing_enabled:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            logger.warning("OpenTelemetry packages unavailable; tracing is disabled")
            return

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318/v1/traces"
        )
        resource = Resource.create(
            {
                "service.name": self.service_name,
                "service.version": self.service_version,
                "deployment.environment": os.environ.get("NF_ENVIRONMENT", "local"),
                "host.name": socket.gethostname(),
            }
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        self._trace = trace
        self._tracer = trace.get_tracer(self.service_name, self.service_version)
        self.tracing_enabled = True
        logger.info("OpenTelemetry tracing exporting to %s", endpoint)

    def current_trace_ids(self) -> tuple[str | None, str | None]:
        if self._trace is None:
            return None, None
        span = self._trace.get_current_span()
        context = span.get_span_context()
        if not context or not context.is_valid:
            return None, None
        return f"{context.trace_id:032x}", f"{context.span_id:016x}"

    def _debug_state(self, event: str, **payload: Any) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "observability.%s",
            event,
            extra={
                "observability_event": event,
                "observability_payload": _as_debug_json(payload),
            },
        )

    @contextmanager
    def span(self, name: str, **attrs: Any):
        if not self.tracing_enabled or self._tracer is None:
            yield None
            return
        with self._tracer.start_as_current_span(name) as span:
            for key, value in attrs.items():
                span.set_attribute(key, _label(value))
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_attribute("error", True)
                raise

    def trace_async(self, name: str, **attrs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(name, **attrs):
                    return await func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def observe_async_task(self, name: str, coro: Any, **attrs: Any) -> Any:
        async def wrapper() -> Any:
            with self.span(name, **attrs):
                return await coro

        return wrapper()

    def set_uptime(self, started_at: float) -> None:
        gauge = self._gauges.get("uptime")
        if gauge is not None:
            gauge.set(max(0.0, time.time() - started_at))

    def set_ui_clients(self, count: int) -> None:
        gauge = self._gauges.get("ui_clients")
        if gauge is not None:
            gauge.set(count)

    def set_telemetry_buffer(self, count: int) -> None:
        gauge = self._gauges.get("telemetry_buffer")
        if gauge is not None:
            gauge.set(count)

    def record_command(self, command: str, duration: float | None = None) -> None:
        counter = self._counters.get("commands")
        if counter is not None:
            counter.labels(_label(command)).inc()
        hist = self._histograms.get("command_duration")
        if hist is not None and duration is not None:
            hist.labels(_label(command)).observe(duration)

    def record_telemetry_item(self, item_type: str) -> None:
        counter = self._counters.get("telemetry")
        if counter is not None:
            counter.labels(_label(item_type)).inc()

    def record_telemetry_payload(self, item_type: str, payload: Any) -> None:
        try:
            self.record_telemetry_last_seen(item_type)
            self._debug_state("telemetry_payload", item_type=item_type, payload=payload)
            if item_type == "component_conn_status":
                self.record_component_status(payload)
            elif item_type == "vid_stats":
                self.record_video_stats(payload)
            elif item_type == "video_ready":
                self.record_video_ready(payload)
            elif item_type in {"pos_estimate", "position_estimate"}:
                self.record_position_estimate(payload)
            elif item_type == "pos_factors_debug":
                self.record_position_factors(payload)
            elif item_type == "gantry_sightings":
                self.record_gantry_sightings(payload)
            elif item_type == "new_anchor_poses":
                self.record_anchor_poses(payload)
            elif item_type == "named_position":
                self.record_named_position(payload)
            elif item_type == "grip_sensors":
                self.record_gripper_sensors(payload)
            elif item_type == "grip_cam_preditions":
                self.record_grip_cam_predictions(payload)
            elif item_type == "operation_progress":
                self.record_operation_progress(payload)
            elif item_type == "last_commanded_vel":
                self.record_commanded_velocity(payload, kind="total")
            elif item_type == "raw_commanded_vel":
                self.record_commanded_velocity(payload, kind="raw")
            elif item_type == "last_commanded_grip":
                self.record_commanded_grip(payload)
            elif item_type == "swing_cancellation_state":
                self.record_swing_cancellation(payload)
            elif item_type == "target_list":
                self.record_target_list(payload)
            elif item_type == "uplink_status":
                self.record_uplink_status(payload)
            elif item_type == "episode_control":
                self.record_episode_control(payload)
            elif item_type == "visibility_states":
                self.record_visibility_states(payload)
        except Exception:
            logger.exception("Failed to record observability payload for %s", item_type)

    def record_telemetry_last_seen(self, item_type: str) -> None:
        item_type = _label(item_type)
        seen = self._gauges.get("telemetry_payload_seen")
        if seen is not None:
            seen.labels(item_type).set(1)
        gauge = self._gauges.get("telemetry_last_seen")
        if gauge is not None:
            gauge.labels(item_type).set(time.time())

    def record_telemetry_flush(self, *, bytes_sent: int, recipients: int, duration: float) -> None:
        flushes = self._counters.get("flushes")
        if flushes is not None:
            flushes.inc()
        byte_counter = self._counters.get("flush_bytes")
        if byte_counter is not None:
            byte_counter.inc(max(0, bytes_sent * max(0, recipients)))
        hist = self._histograms.get("flush_duration")
        if hist is not None:
            hist.observe(duration)

    def record_component_ws(self, component: str, outcome: str) -> None:
        counter = self._counters.get("component_ws")
        if counter is not None:
            counter.labels(_label(component), _label(outcome)).inc()

    def set_expected_components(self, *, anchors: Any = None, gripper: bool = True) -> None:
        if gripper:
            self.record_component_absent("gripper", "gripper")
        for anchor in anchors or []:
            anchor_num = getattr(anchor, "num", anchor)
            self.record_component_absent(f"anchor_{anchor_num}", "anchor")
        self._debug_state(
            "expected_components",
            components=[{"component": component, "kind": kind} for component, kind in self._expected_components],
        )

    def record_component_absent(self, component: str, kind: str) -> None:
        component = _label(component)
        kind = _label(kind)
        key = (component, kind)
        self._expected_components[key] = 1.0
        expected = self._gauges.get("component_expected")
        if expected is not None:
            expected.labels(component, kind).set(1)
        waiting = self._gauges.get("component_waiting")
        if waiting is not None:
            waiting.labels(component, kind).set(1)
        connected = self._gauges.get("component_connected")
        if connected is not None:
            connected.labels(component, kind).set(0)
        video_connected = self._gauges.get("component_video_connected")
        if video_connected is not None:
            video_connected.labels(component, kind).set(0)
        ws_status = self._gauges.get("component_websocket_status")
        video_status = self._gauges.get("component_video_status")
        for gauge in (ws_status, video_status):
            if gauge is not None:
                gauge.labels(component, kind, "not_detected").set(1)
                gauge.labels(component, kind, "connecting").set(0)
                gauge.labels(component, kind, "connected").set(0)
        age = self._gauges.get("component_last_seen_age")
        if age is not None:
            age.labels(component, kind).set(0)

    def record_component_status(self, status: Any) -> None:
        kind = "gripper" if getattr(status, "is_gripper", False) else "anchor"
        component = "gripper" if kind == "gripper" else f"anchor_{getattr(status, 'anchor_num', 0)}"
        ws_value = _enum_number(getattr(status, "websocket_status", 0))
        video_value = _enum_number(getattr(status, "video_status", 0))
        status_names = {0: "not_detected", 1: "connecting", 2: "connected"}
        now = time.time()
        key = (component, kind)
        self._expected_components[key] = 1.0
        self._component_seen_at[key] = now

        connected = self._gauges.get("component_connected")
        if connected is not None:
            connected.labels(component, kind).set(1 if ws_value == 2 else 0)
        video_connected = self._gauges.get("component_video_connected")
        if video_connected is not None:
            video_connected.labels(component, kind).set(1 if video_value == 2 else 0)
        expected = self._gauges.get("component_expected")
        if expected is not None:
            expected.labels(component, kind).set(1)
        waiting = self._gauges.get("component_waiting")
        if waiting is not None:
            waiting.labels(component, kind).set(1 if ws_value != 2 else 0)

        ws_status = self._gauges.get("component_websocket_status")
        if ws_status is not None:
            for value, name in status_names.items():
                ws_status.labels(component, kind, name).set(1 if ws_value == value else 0)
        video_status = self._gauges.get("component_video_status")
        if video_status is not None:
            for value, name in status_names.items():
                video_status.labels(component, kind, name).set(1 if video_value == value else 0)

        temp = getattr(status, "temp", None)
        temp_gauge = self._gauges.get("component_temperature")
        if temp_gauge is not None and temp is not None:
            temp_gauge.labels(component, kind).set(float(temp))

        motor = self._gauges.get("component_motor_enabled")
        if motor is not None:
            motor.labels(component, kind).set(1 if _enum_number(getattr(status, "motor_enabled", 0)) == 1 else 0)

        error = self._gauges.get("component_error")
        if error is not None:
            error.labels(component, kind).set(1 if getattr(status, "error_message", None) else 0)
        last_seen = self._gauges.get("component_last_seen")
        if last_seen is not None:
            last_seen.labels(component, kind).set(now)
        age = self._gauges.get("component_last_seen_age")
        if age is not None:
            age.labels(component, kind).set(0)

    def record_video_stats(self, stats: Any) -> None:
        now = time.time()
        detection = self._gauges.get("video_detection_rate")
        if detection is not None:
            detection.set(_safe_float(getattr(stats, "detection_rate", 0.0)))
        latency = self._gauges.get("video_latency")
        if latency is not None:
            latency.set(_safe_float(getattr(stats, "video_latency", 0.0)))
        framerate = self._gauges.get("video_framerate")
        if framerate is not None:
            framerate.set(_safe_float(getattr(stats, "video_framerate", 0.0)))
        last_seen = self._gauges.get("video_stats_last_seen")
        if last_seen is not None:
            last_seen.set(now)
        available = self._gauges.get("video_stats_available")
        if available is not None:
            available.set(1)

    def record_video_ready(self, ready: Any) -> None:
        feed = _label(getattr(ready, "feed_number", None), "unknown")
        anchor_num = getattr(ready, "anchor_num", None)
        if getattr(ready, "is_gripper", False):
            component = "gripper"
        elif anchor_num is not None:
            component = f"anchor_{anchor_num}"
        else:
            component = "aux"
        gauge = self._gauges.get("video_feed_ready")
        if gauge is not None:
            has_local = bool(getattr(ready, "local_uri", None))
            has_remote = bool(getattr(ready, "stream_path", None))
            gauge.labels(feed, component, "local").set(1 if has_local else 0)
            gauge.labels(feed, component, "remote").set(1 if has_remote else 0)
            gauge.labels(feed, component, "waiting").set(0 if has_local or has_remote else 1)
        last_seen = self._gauges.get("video_feed_ready_last_seen")
        if last_seen is not None:
            last_seen.labels(feed, component).set(time.time())

    def record_position_factors(self, factors: Any) -> None:
        gauge = self._gauges.get("position_factor")
        if gauge is None:
            return
        for name in ("visual_pos", "visual_vel", "hanging_pos", "hanging_vel"):
            vec = getattr(factors, name, None)
            if vec is None:
                continue
            for axis in ("x", "y", "z"):
                gauge.labels(name, axis).set(_vec_value(vec, axis))
        gauge.labels("spin", "z").set(_safe_float(getattr(factors, "spin", 0.0)))

    def record_gantry_sightings(self, sightings: Any) -> None:
        gauge = self._gauges.get("gantry_sightings")
        if gauge is not None:
            gauge.set(len(getattr(sightings, "sightings", []) or []))

    def record_anchor_poses(self, poses: Any) -> None:
        gauge = self._gauges.get("anchor_poses")
        if gauge is not None:
            gauge.set(len(getattr(poses, "poses", []) or []))

    def record_named_position(self, named_position: Any) -> None:
        name = _label(getattr(named_position, "name", None), "unnamed")
        position = getattr(named_position, "position", None)
        visible = self._gauges.get("named_position_visible")
        if visible is not None:
            visible.labels(name).set(1 if position is not None else 0)
        gauge = self._gauges.get("named_position")
        if gauge is not None and position is not None:
            for axis in ("x", "y", "z"):
                gauge.labels(name, axis).set(_vec_value(position, axis))

    def record_position_estimate(self, estimate: Any) -> None:
        now = time.time()
        available = self._gauges.get("position_estimate_available")
        if available is not None:
            available.set(1)
        last_seen = self._gauges.get("position_estimate_last_seen")
        if last_seen is not None:
            last_seen.set(now)
        position = self._gauges.get("robot_position")
        gantry_position = getattr(estimate, "gantry_position", None)
        if position is not None and gantry_position is not None:
            for axis in ("x", "y", "z"):
                position.labels("gantry", axis).set(_vec_value(gantry_position, axis))

        gripper_pose = getattr(estimate, "gripper_pose", None)
        gripper_position = getattr(gripper_pose, "position", None)
        if position is not None and gripper_position is not None:
            for axis in ("x", "y", "z"):
                position.labels("gripper", axis).set(_vec_value(gripper_position, axis))

        velocity = getattr(estimate, "gantry_velocity", None)
        velocity_gauge = self._gauges.get("robot_velocity")
        if velocity_gauge is not None and velocity is not None:
            for axis in ("x", "y", "z"):
                velocity_gauge.labels(axis).set(_vec_value(velocity, axis))
        velocity_norm = self._gauges.get("robot_velocity_norm")
        if velocity_norm is not None and velocity is not None:
            velocity_norm.set(_vec_norm(velocity))

        data_ts = float(getattr(estimate, "data_ts", 0.0) or 0.0)
        age = self._gauges.get("position_estimate_age")
        if age is not None and data_ts > 0:
            age.set(max(0.0, now - data_ts))

        slack = self._gauges.get("line_slack")
        if slack is not None:
            for index, value in enumerate(getattr(estimate, "slack", []) or []):
                slack.labels(str(index)).set(1 if value else 0)

        tension_values = list(getattr(estimate, "tension", []) or [])
        tension_available = self._gauges.get("tension_available")
        if tension_available is not None:
            tension_available.set(1 if tension_values else 0)
        line_gauge = self._gauges.get("line_tension")
        if line_gauge is not None:
            for index, value in enumerate(tension_values):
                line_gauge.labels(str(index)).set(float(value))

    def record_gripper_sensors(self, sensors: Any) -> None:
        now = time.time()
        gauge = self._gauges.get("gripper_sensor")
        available = self._gauges.get("gripper_sensors_available")
        if available is not None:
            available.set(1)
        last_seen = self._gauges.get("gripper_sensors_last_seen")
        if last_seen is not None:
            last_seen.set(now)
        present = self._gauges.get("gripper_sensor_present")
        for sensor in ("range", "angle", "pressure", "wrist", "target_force"):
            value = getattr(sensors, sensor, None)
            if present is not None:
                present.labels(sensor).set(1 if value is not None else 0)
            if value is not None:
                if gauge is not None:
                    gauge.labels(sensor).set(_safe_float(value))

    def record_operation_progress(self, progress: Any) -> None:
        operation = _label(getattr(progress, "name", None), "unnamed")
        percent = _safe_float(getattr(progress, "percent_complete", 0.0))
        progress_gauge = self._gauges.get("operation_progress")
        if progress_gauge is not None:
            progress_gauge.labels(operation).set(percent)
        active = self._gauges.get("operation_active")
        if active is not None:
            self._operation_labels.add(operation)
            for known_operation in self._operation_labels:
                active.labels(known_operation).set(1 if known_operation == operation and percent < 100.0 else 0)
        last_seen = self._gauges.get("operation_last_seen")
        if last_seen is not None:
            last_seen.labels(operation).set(time.time())

    def record_commanded_velocity(self, command: Any, *, kind: str) -> None:
        velocity = getattr(command, "velocity", None)
        if velocity is None:
            self._debug_state("commanded_velocity_absent", kind=kind)
            return
        velocity_gauge = self._gauges.get("commanded_velocity")
        if velocity_gauge is not None:
            for axis in ("x", "y", "z"):
                velocity_gauge.labels(kind, axis).set(_vec_value(velocity, axis))
        norm = self._gauges.get("commanded_velocity_norm")
        if norm is not None:
            norm.labels(kind).set(_vec_norm(velocity))
        last_seen = self._gauges.get("commanded_velocity_last_seen")
        if last_seen is not None:
            last_seen.labels(kind).set(time.time())

    def record_commanded_grip(self, grip: Any) -> None:
        now = time.time()
        gauge = self._gauges.get("commanded_grip")
        present = self._gauges.get("commanded_grip_present")
        for command in ("finger_speed", "wrist_speed"):
            value = getattr(grip, command, None)
            if present is not None:
                present.labels(command).set(1 if value is not None else 0)
            if value is not None and gauge is not None:
                gauge.labels(command).set(_safe_float(value))
        last_seen = self._gauges.get("commanded_grip_last_seen")
        if last_seen is not None:
            last_seen.set(now)

    def record_grip_cam_predictions(self, prediction: Any) -> None:
        gauge = self._gauges.get("grip_cam_prediction")
        if gauge is None:
            return
        for signal in ("move_x", "move_y", "prob_target_in_view", "prob_holding", "grip_angle"):
            gauge.labels(signal).set(_safe_float(getattr(prediction, signal, 0.0)))

    def record_swing_cancellation(self, state: Any) -> None:
        enabled = self._gauges.get("swing_cancellation_enabled")
        if enabled is not None:
            enabled.set(1 if getattr(state, "enabled", False) else 0)
        present = self._gauges.get("swing_cancellation_present")
        if present is not None:
            present.set(1)
        last_seen = self._gauges.get("swing_cancellation_last_seen")
        if last_seen is not None:
            last_seen.set(time.time())

    def record_target_list(self, target_list: Any) -> None:
        targets = list(getattr(target_list, "targets", []) or [])
        total = self._gauges.get("targets_known")
        if total is not None:
            total.set(len(targets))
        counts: dict[tuple[str, str], int] = {}
        for target in targets:
            status = _enum_name(getattr(target, "status", None)).lower()
            source = _label(getattr(target, "source", None), "unknown")
            key = (status, source)
            counts[key] = counts.get(key, 0) + 1
        status_gauge = self._gauges.get("targets_by_status")
        if status_gauge is not None:
            known_labels = self._target_status_labels | set(counts)
            for status, source in known_labels:
                status_gauge.labels(status, source).set(counts.get((status, source), 0))
            self._target_status_labels = known_labels
        last_seen = self._gauges.get("target_list_last_seen")
        if last_seen is not None:
            last_seen.set(time.time())

    def record_uplink_status(self, status: Any) -> None:
        gauge = self._gauges.get("uplink_online")
        if gauge is not None:
            gauge.set(1 if getattr(status, "online", False) else 0)
        last_seen = self._gauges.get("uplink_last_seen")
        if last_seen is not None:
            last_seen.set(time.time())

    def record_episode_control(self, episode_control: Any) -> None:
        status = getattr(episode_control, "status", None)
        gauge = self._gauges.get("episode_status")
        if gauge is not None and status is not None:
            gauge.set(_enum_number(status))

    def record_visibility_states(self, states: Any) -> None:
        anchors = list(getattr(states, "anchors_seeing_origin_card", []) or [])
        count = self._gauges.get("calibration_origin_visible_anchors")
        if count is not None:
            count.set(len(anchors))
        per_anchor = self._gauges.get("calibration_anchor_sees_origin")
        if per_anchor is not None:
            seen = {int(anchor) for anchor in anchors}
            for anchor in range(4):
                per_anchor.labels(str(anchor)).set(1 if anchor in seen else 0)

    def record_runtime_state(
        self,
        *,
        connected_components: dict[tuple[str, str], bool] | None = None,
        gripper_present: bool | None = None,
        anchor_count: int | None = None,
        active_velocity_keys: Any = None,
        telemetry_buffer_size: int | None = None,
        target_count: int | None = None,
    ) -> None:
        now = time.time()
        connected_components = connected_components or {}
        waiting = self._gauges.get("component_waiting")
        connected = self._gauges.get("component_connected")
        age = self._gauges.get("component_last_seen_age")
        for component, kind in self._expected_components:
            is_connected = bool(connected_components.get((component, kind), False))
            if waiting is not None:
                waiting.labels(component, kind).set(0 if is_connected else 1)
            if connected is not None:
                connected.labels(component, kind).set(1 if is_connected else 0)
            if age is not None:
                seen_at = self._component_seen_at.get((component, kind), 0.0)
                age.labels(component, kind).set(max(0.0, now - seen_at) if seen_at else 0)
        if target_count is not None:
            targets = self._gauges.get("targets_known")
            if targets is not None:
                targets.set(max(0, target_count))
        self._debug_state(
            "runtime_state",
            connected_components=[
                {"component": component, "kind": kind, "connected": connected}
                for (component, kind), connected in connected_components.items()
            ],
            gripper_present=gripper_present,
            anchor_count=anchor_count,
            active_velocity_keys=sorted(_label(key) for key in (active_velocity_keys or [])),
            telemetry_buffer_size=telemetry_buffer_size,
            target_count=target_count,
        )

    def set_video_clients(self, port: int, count: int) -> None:
        gauge = self._gauges.get("video_clients")
        if gauge is not None:
            gauge.labels(str(port)).set(count)

    def record_video_frame(self, streamer: str = "nf_video_streamer") -> None:
        counter = self._counters.get("video_frames")
        if counter is not None:
            counter.labels(_label(streamer)).inc()

    def record_video_encode(self, port: int, byte_count: int, duration: float) -> None:
        byte_counter = self._counters.get("video_bytes")
        if byte_counter is not None:
            byte_counter.labels(str(port)).inc(max(0, byte_count))
        hist = self._histograms.get("jpeg_encode")
        if hist is not None:
            hist.labels(str(port)).observe(max(0.0, duration))

    def record_event_loop_lag(self, lag: float) -> None:
        gauge = self._gauges.get("event_loop_lag")
        if gauge is not None:
            gauge.set(max(0.0, lag))

    def record_tension(self, values: Any, max_safe_tension: float) -> None:
        values = list(values) if values is not None else []
        max_gauge = self._gauges.get("max_safe_tension")
        if max_gauge is not None:
            max_gauge.set(max_safe_tension)
        available = self._gauges.get("tension_available")
        if available is not None:
            available.set(1 if values else 0)
        line_gauge = self._gauges.get("line_tension")
        if line_gauge is not None:
            for index, value in enumerate(values):
                line_gauge.labels(str(index)).set(float(value))
        self._debug_state("tension_sample", values=values, max_safe_tension=max_safe_tension)

    def record_safety_stop(self) -> None:
        counter = self._counters.get("safety_stops")
        if counter is not None:
            counter.inc()
        self._debug_state("safety_stop", reason="tension_limit")


OBS = Observability()


def init_observability(**kwargs: Any) -> Observability:
    OBS.configure(**kwargs)
    return OBS


def create_task(coro: Any, name: str, **attrs: Any) -> asyncio.Task:
    return asyncio.create_task(OBS.observe_async_task(name, coro, **attrs))
