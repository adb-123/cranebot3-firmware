from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


DEFAULT_CALIBRATION_ARTIFACT_DIR = Path("logs/calibration")
SCHEMA_VERSION = 1


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_session_id(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", session_id.strip())
    safe = safe.strip(".-")
    return safe or "calibration-session"


def _json_safe(value: Any, warnings: list[dict[str, Any]], path: str) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value), warnings, path)

    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, int) and not isinstance(value, bool):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        warnings.append(
            {
                "kind": "non_finite_value",
                "path": path,
                "replacement": None,
            }
        )
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat().replace("+00:00", "Z")

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            converted = value.tolist()
        except TypeError:
            converted = None
        if converted is not None and converted is not value:
            return _json_safe(converted, warnings, path)

    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        try:
            converted = value.item()
        except (TypeError, ValueError):
            converted = None
        if converted is not None and converted is not value:
            return _json_safe(converted, warnings, path)

    if isinstance(value, Mapping):
        safe_dict = {}
        for key, item in value.items():
            safe_key = str(key)
            safe_dict[safe_key] = _json_safe(item, warnings, f"{path}.{safe_key}")
        return safe_dict

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item, warnings, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    return str(value)


def _encode_json(payload: Mapping[str, Any]) -> str:
    warnings: list[dict[str, Any]] = []
    safe_payload = _json_safe(payload, warnings, "$")
    if warnings:
        safe_payload.setdefault("warnings", [])
        safe_payload["warnings"].extend(warnings)
    return json.dumps(
        safe_payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON by replacing the destination only after a full fsync'd temp file."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encode_json(payload)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=final_path.parent,
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
        try:
            dir_fd = os.open(final_path.parent, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                try:
                    os.fsync(dir_fd)
                except OSError:
                    pass
            finally:
                os.close(dir_fd)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return final_path


class CalibrationArtifactSession:
    """Collects one calibration attempt and persists it as a durable JSON artifact."""

    def __init__(
        self,
        session_id: str | None = None,
        artifact_dir: str | Path = DEFAULT_CALIBRATION_ARTIFACT_DIR,
        metadata: Mapping[str, Any] | None = None,
        now: Callable[[], str] = _utc_timestamp,
    ):
        self.session_id = session_id or uuid.uuid4().hex
        self.artifact_dir = Path(artifact_dir)
        self._now = now

        created_at = self._now()
        self.created_at = created_at
        self.updated_at = created_at
        self.phase = "created"
        self.status = "running"
        self.metadata = dict(metadata or {})
        self.observations: list[dict[str, Any]] = []
        self.line_health_samples: list[dict[str, Any]] = []
        self.optimizer_reports: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.status_history: list[dict[str, Any]] = [
            {
                "timestamp": created_at,
                "phase": self.phase,
                "status": self.status,
            }
        ]

    @property
    def path(self) -> Path:
        return self.artifact_dir / f"{_safe_session_id(self.session_id)}.json"

    def set_phase(self, phase: str, **fields: Any) -> None:
        self.phase = phase
        self._touch()
        self.status_history.append(
            self._event(phase=phase, status=self.status, **fields)
        )

    def set_status(self, status: str, message: str | None = None, **fields: Any) -> None:
        self.status = status
        self._touch()
        event = self._event(status=status, phase=self.phase, **fields)
        if message is not None:
            event["message"] = message
        self.status_history.append(event)

    def record_observation(self, **fields: Any) -> None:
        self.observations.append(self._event(**fields))

    def record_line_health(self, **fields: Any) -> None:
        self.line_health_samples.append(self._event(**fields))

    def record_optimizer_report(self, **fields: Any) -> None:
        self.optimizer_reports.append(self._event(**fields))

    def warn(self, message: str, **fields: Any) -> None:
        self.warnings.append(self._event(message=message, **fields))

    def fail(self, message: str, **fields: Any) -> None:
        failure = self._event(message=message, phase=fields.pop("phase", self.phase), **fields)
        self.failures.append(failure)
        self.set_status("failed", message=message)

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "phase": self.phase,
            "status": self.status,
            "metadata": self.metadata,
            "status_history": self.status_history,
            "observations": self.observations,
            "line_health_samples": self.line_health_samples,
            "optimizer_reports": self.optimizer_reports,
            "warnings": self.warnings,
            "failures": self.failures,
        }
        warnings: list[dict[str, Any]] = []
        safe_payload = _json_safe(payload, warnings, "$")
        if warnings:
            safe_payload.setdefault("warnings", [])
            safe_payload["warnings"].extend(warnings)
        return safe_payload

    def write(self, status: str | None = None, message: str | None = None) -> Path:
        if status is not None:
            self.set_status(status, message=message)
        return write_json_atomic(self.path, self.snapshot())

    def _touch(self) -> None:
        self.updated_at = self._now()

    def _event(self, **fields: Any) -> dict[str, Any]:
        event = {
            "timestamp": self._now(),
            "phase": self.phase,
        }
        event.update(fields)
        return event
