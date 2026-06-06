from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from nf_robot.common.cv_common import locate_markers
from nf_robot.common.pose_functions import average_pose, compose_poses, invert_pose
from nf_robot.host.floor_view import generate_orthographic_floor_maps

logger = logging.getLogger(__name__)

EXTERNAL_CAMERA_KEYS = ("externalRoomCameras", "external_room_cameras")
DEFAULT_MAP_SIZE_PX = 600
DEFAULT_MAP_EXTENT_M = 8.0


def _now() -> float:
    return time.monotonic()


def _wall_time() -> float:
    return time.time()


def _mapping_value(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return default


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _float_list(value: Any, *, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must be a list of {length} numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


@dataclass(frozen=True)
class CameraPose:
    """Camera-to-world or marker-to-world pose using Stringman's rvec/tvec convention."""

    rotation: list[float]
    position: list[float]

    @classmethod
    def from_json(cls, value: Any, *, label: str) -> "CameraPose":
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return cls(
                rotation=_float_list(value[0], length=3, label=f"{label}.rotation"),
                position=_float_list(value[1], length=3, label=f"{label}.position"),
            )
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a pose object")
        rotation = _mapping_value(value, "rotation", "rvec", "rot")
        position = _mapping_value(value, "position", "tvec", "pos")
        return cls(
            rotation=_float_list(rotation, length=3, label=f"{label}.rotation"),
            position=_float_list(position, length=3, label=f"{label}.position"),
        )

    @classmethod
    def from_tuple(cls, pose: tuple[np.ndarray, np.ndarray]) -> "CameraPose":
        return cls(
            rotation=[float(item) for item in np.asarray(pose[0], dtype=float).reshape(3)],
            position=[float(item) for item in np.asarray(pose[1], dtype=float).reshape(3)],
        )

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.rotation, dtype=float).reshape(3),
            np.asarray(self.position, dtype=float).reshape(3),
        )

    def to_json(self) -> dict[str, list[float]]:
        return {"rotation": list(self.rotation), "position": list(self.position)}


@dataclass(frozen=True)
class KnownMarker:
    name: str
    pose: CameraPose
    required: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pose": self.pose.to_json(),
            "required": self.required,
        }


@dataclass(frozen=True)
class ExternalRoomCameraSpec:
    name: str
    enabled: bool
    source_type: str
    image_topic: str
    camera_info_topic: str
    ros_domain_id: int | None = None
    rmw: str | None = None
    frame_id: str | None = None
    aliases: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    min_known_markers: int = 1
    promote_after_frames: int = 5
    stable_position_std_m: float = 0.08
    stale_after_s: float = 2.0

    @classmethod
    def from_json(cls, value: Any) -> "ExternalRoomCameraSpec":
        if not isinstance(value, dict):
            raise ValueError("external camera entries must be objects")
        name = str(_mapping_value(value, "name", "id") or "").strip()
        if not name:
            raise ValueError("external camera entry is missing name")
        image_topic = str(_mapping_value(value, "imageTopic", "image_topic", "topic") or "").strip()
        camera_info_topic = str(_mapping_value(value, "cameraInfoTopic", "camera_info_topic") or "").strip()
        if not image_topic:
            raise ValueError(f"external camera {name!r} is missing imageTopic")
        if not camera_info_topic:
            raise ValueError(f"external camera {name!r} is missing cameraInfoTopic")

        aliases = _mapping_value(value, "aliases", default=())
        labels = _mapping_value(value, "labels", "roles", default=())
        return cls(
            name=name,
            enabled=_bool_value(_mapping_value(value, "enabled"), True),
            source_type=str(_mapping_value(value, "sourceType", "source_type", default="ros2_compressed_image")),
            image_topic=image_topic,
            camera_info_topic=camera_info_topic,
            ros_domain_id=(
                None
                if _mapping_value(value, "rosDomainId", "ros_domain_id") is None
                else int(_mapping_value(value, "rosDomainId", "ros_domain_id"))
            ),
            rmw=_mapping_value(value, "rmw", "rmwImplementation", "rmw_implementation"),
            frame_id=_mapping_value(value, "frameId", "frame_id"),
            aliases=tuple(str(item) for item in aliases) if isinstance(aliases, list) else (),
            labels=tuple(str(item) for item in labels) if isinstance(labels, list) else (),
            min_known_markers=max(1, int(_mapping_value(value, "minKnownMarkers", "min_known_markers", default=1))),
            promote_after_frames=max(1, int(_mapping_value(value, "promoteAfterFrames", "promote_after_frames", default=5))),
            stable_position_std_m=float(
                _mapping_value(value, "stablePositionStdM", "stable_position_std_m", default=0.08)
            ),
            stale_after_s=float(_mapping_value(value, "staleAfterS", "stale_after_s", default=2.0)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "sourceType": self.source_type,
            "imageTopic": self.image_topic,
            "cameraInfoTopic": self.camera_info_topic,
            "rosDomainId": self.ros_domain_id,
            "rmw": self.rmw,
            "frameId": self.frame_id,
            "aliases": list(self.aliases),
            "labels": list(self.labels),
            "minKnownMarkers": self.min_known_markers,
            "promoteAfterFrames": self.promote_after_frames,
            "stablePositionStdM": self.stable_position_std_m,
            "staleAfterS": self.stale_after_s,
        }


@dataclass(frozen=True)
class ExternalRoomCameraRegistry:
    enabled: bool = False
    cameras: tuple[ExternalRoomCameraSpec, ...] = ()
    known_markers: dict[str, KnownMarker] = field(default_factory=dict)
    known_marker_pose_files: tuple[str, ...] = ()
    fusion: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_robot_config(cls, config: dict[str, Any]) -> "ExternalRoomCameraRegistry":
        raw = None
        for key in EXTERNAL_CAMERA_KEYS:
            if key in config:
                raw = config[key]
                break
        return cls.from_json(raw)

    @classmethod
    def from_json(cls, value: Any) -> "ExternalRoomCameraRegistry":
        if value is None:
            return cls()
        if isinstance(value, list):
            value = {"enabled": True, "cameras": value}
        if not isinstance(value, dict):
            raise ValueError("externalRoomCameras must be an object or list")

        raw_cameras = _mapping_value(value, "cameras", "cameraSources", default=[])
        if not isinstance(raw_cameras, list):
            raise ValueError("externalRoomCameras.cameras must be a list")

        raw_markers = _mapping_value(value, "knownMarkers", "known_markers", default={})
        known_markers: dict[str, KnownMarker] = {}
        if isinstance(raw_markers, dict):
            iterable = raw_markers.items()
        elif isinstance(raw_markers, list):
            iterable = ((str(item.get("name")), item) for item in raw_markers if isinstance(item, dict))
        else:
            raise ValueError("externalRoomCameras.knownMarkers must be an object or list")
        for marker_name, marker_value in iterable:
            if not marker_name or not isinstance(marker_value, dict):
                continue
            pose_value = _mapping_value(marker_value, "pose", default=marker_value)
            known_markers[str(marker_name)] = KnownMarker(
                name=str(marker_name),
                pose=CameraPose.from_json(pose_value, label=f"knownMarkers.{marker_name}"),
                required=_bool_value(_mapping_value(marker_value, "required"), False),
            )

        raw_pose_files = _mapping_value(value, "knownMarkerPoseFiles", "known_marker_pose_files", default=[]) or []
        if isinstance(raw_pose_files, str):
            pose_files = (raw_pose_files,)
        elif isinstance(raw_pose_files, list):
            pose_files = tuple(str(item) for item in raw_pose_files)
        else:
            raise ValueError("externalRoomCameras.knownMarkerPoseFiles must be a string or list")

        return cls(
            enabled=_bool_value(_mapping_value(value, "enabled"), bool(raw_cameras)),
            cameras=tuple(ExternalRoomCameraSpec.from_json(item) for item in raw_cameras),
            known_markers=known_markers,
            known_marker_pose_files=pose_files,
            fusion=dict(_mapping_value(value, "fusion", default={}) or {}),
        )

    def enabled_cameras(self) -> list[ExternalRoomCameraSpec]:
        if not self.enabled:
            return []
        return [camera for camera in self.cameras if camera.enabled]

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "knownMarkers": {name: marker.to_json() for name, marker in sorted(self.known_markers.items())},
            "knownMarkerPoseFiles": list(self.known_marker_pose_files),
            "cameras": [camera.to_json() for camera in self.cameras],
            "fusion": dict(self.fusion),
        }


@dataclass
class CameraCalibrationState:
    state: str = "unseen"
    pose: CameraPose | None = None
    confidence: float = 0.0
    known_marker_count: int = 0
    detected_marker_count: int = 0
    stable_frame_count: int = 0
    residual_position_m: float | None = None
    viewing_direction: list[float] | None = None
    message: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "pose": None if self.pose is None else self.pose.to_json(),
            "confidence": self.confidence,
            "knownMarkerCount": self.known_marker_count,
            "detectedMarkerCount": self.detected_marker_count,
            "stableFrameCount": self.stable_frame_count,
            "residualPositionM": self.residual_position_m,
            "viewingDirection": self.viewing_direction,
            "message": self.message,
        }


def load_robot_config_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_external_room_camera_registry(path: Path) -> ExternalRoomCameraRegistry:
    base_path = Path(path).resolve().parent
    registry = ExternalRoomCameraRegistry.from_robot_config(load_robot_config_json(path))
    if not registry.known_marker_pose_files:
        return registry

    known_markers = dict(registry.known_markers)
    for marker_file in registry.known_marker_pose_files:
        marker_path = Path(marker_file)
        if not marker_path.is_absolute():
            marker_path = base_path / marker_path
        known_markers.update(load_known_markers_from_file(marker_path))
    return ExternalRoomCameraRegistry(
        enabled=registry.enabled,
        cameras=registry.cameras,
        known_markers=known_markers,
        known_marker_pose_files=registry.known_marker_pose_files,
        fusion=registry.fusion,
    )


def load_known_markers_from_file(path: Path) -> dict[str, KnownMarker]:
    value = load_robot_config_json(Path(path))
    raw_markers = _mapping_value(value, "knownMarkers", "known_markers", "markers", default=value)
    registry = ExternalRoomCameraRegistry.from_json({"enabled": True, "knownMarkers": raw_markers, "cameras": []})
    return registry.known_markers


def camera_calibration_from_camera_info(info: Any) -> SimpleNamespace:
    if isinstance(info, dict):
        k = _mapping_value(info, "k", "K")
        d = _mapping_value(info, "d", "D", default=[])
        width = _mapping_value(info, "width")
        height = _mapping_value(info, "height")
    else:
        k = getattr(info, "k")
        d = getattr(info, "d", [])
        width = getattr(info, "width")
        height = getattr(info, "height")
    return SimpleNamespace(
        intrinsic_matrix=[float(item) for item in k],
        distortion_coeff=[float(item) for item in d],
        resolution=SimpleNamespace(width=int(width), height=int(height)),
    )


def camera_calibration_to_json(camera_cal: Any | None) -> dict[str, Any] | None:
    if camera_cal is None:
        return None
    return {
        "resolution": {
            "width": int(camera_cal.resolution.width),
            "height": int(camera_cal.resolution.height),
        },
        "intrinsicMatrix": [float(item) for item in camera_cal.intrinsic_matrix],
        "distortionCoeff": [float(item) for item in camera_cal.distortion_coeff],
    }


def decode_compressed_image(data: bytes | bytearray | memoryview) -> np.ndarray | None:
    raw = np.frombuffer(bytes(data), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to encode JPEG frame")
    return bytes(encoded)


def marker_detections_to_json(detections: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for detection in detections:
        pose = detection.get("p")
        payload.append(
            {
                "name": str(detection.get("n")),
                "centerPx": [float(item) for item in detection.get("center", [])],
                "poseCamera": None
                if pose is None
                else {
                    "rotation": [float(item) for item in np.asarray(pose[0], dtype=float).reshape(3)],
                    "position": [float(item) for item in np.asarray(pose[1], dtype=float).reshape(3)],
                },
            }
        )
    return payload


def _viewing_direction(pose: CameraPose) -> list[float]:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(pose.rotation, dtype=float).reshape(3))
    direction = rotation_matrix @ np.asarray([0.0, 0.0, 1.0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return [0.0, 0.0, 0.0]
    return [float(item) for item in direction / norm]


def _pose_position_residual_m(poses: list[tuple[np.ndarray, np.ndarray]], average: tuple[np.ndarray, np.ndarray]) -> float:
    if len(poses) <= 1:
        return 0.0
    avg_pos = np.asarray(average[1], dtype=float).reshape(3)
    distances = [float(np.linalg.norm(np.asarray(pose[1], dtype=float).reshape(3) - avg_pos)) for pose in poses]
    return float(np.mean(distances))


def estimate_camera_pose_from_detections(
    detections: list[dict[str, Any]],
    known_markers: dict[str, KnownMarker],
    *,
    min_known_markers: int = 1,
) -> CameraCalibrationState:
    if not detections:
        return CameraCalibrationState(state="unseen", message="no RGB markers detected")

    camera_pose_candidates: list[tuple[np.ndarray, np.ndarray]] = []
    known_names: list[str] = []
    for detection in detections:
        name = str(detection.get("n"))
        known_marker = known_markers.get(name)
        marker_pose_camera = detection.get("p")
        if known_marker is None or marker_pose_camera is None:
            continue
        camera_pose_candidates.append(
            compose_poses([known_marker.pose.as_tuple(), invert_pose(marker_pose_camera)])
        )
        known_names.append(name)

    if not camera_pose_candidates:
        return CameraCalibrationState(
            state="visible",
            detected_marker_count=len(detections),
            message="markers visible, but none have configured room poses",
        )

    average = camera_pose_candidates[0] if len(camera_pose_candidates) == 1 else average_pose(camera_pose_candidates)
    pose = CameraPose.from_tuple(average)
    residual = _pose_position_residual_m(camera_pose_candidates, average)
    enough_markers = len(set(known_names)) >= min_known_markers
    confidence = min(1.0, 0.35 + 0.25 * len(set(known_names)))
    if residual > 0.0:
        confidence *= max(0.1, min(1.0, 1.0 - residual))

    return CameraCalibrationState(
        state="solving" if enough_markers else "visible",
        pose=pose,
        confidence=float(confidence if enough_markers else min(confidence, 0.35)),
        known_marker_count=len(set(known_names)),
        detected_marker_count=len(detections),
        residual_position_m=residual,
        viewing_direction=_viewing_direction(pose),
        message=(
            f"known markers: {', '.join(sorted(set(known_names)))}"
            if enough_markers
            else f"need {min_known_markers} known markers, saw {len(set(known_names))}"
        ),
    )


def _annotate_frame(frame_bgr: np.ndarray, detections: list[dict[str, Any]], calibration: CameraCalibrationState) -> np.ndarray:
    annotated = frame_bgr.copy()
    for detection in detections:
        center = tuple(int(round(v)) for v in detection.get("center", (0, 0)))
        name = str(detection.get("n"))
        color = (0, 255, 0) if calibration.state == "calibrated" else (0, 190, 255)
        cv2.circle(annotated, center, 10, color, 2)
        cv2.putText(
            annotated,
            name,
            (center[0] + 12, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        f"{calibration.state} conf={calibration.confidence:.2f}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


class ExternalCameraRuntime:
    def __init__(self, spec: ExternalRoomCameraSpec, registry: ExternalRoomCameraRegistry) -> None:
        self.spec = spec
        self.registry = registry
        self.camera_cal: Any | None = None
        self.frame_bgr: np.ndarray | None = None
        self.frame_jpeg: bytes | None = None
        self.overlay_jpeg: bytes | None = None
        self.detections: list[dict[str, Any]] = []
        self.calibration = CameraCalibrationState()
        self.last_frame_at = 0.0
        self.last_frame_wall_at = 0.0
        self.frames_seen = 0
        self._fps_window: list[float] = []
        self._pose_history: list[CameraPose] = []
        self._lock = threading.RLock()

    def set_camera_info(self, info: Any) -> None:
        with self._lock:
            self.camera_cal = camera_calibration_from_camera_info(info)
            if self.calibration.state == "unseen":
                self.calibration = CameraCalibrationState(state="visible", message="camera_info received; waiting for frame")

    def handle_compressed_image(self, data: bytes | bytearray | memoryview) -> None:
        frame = decode_compressed_image(data)
        if frame is not None:
            self.handle_frame_bgr(frame)

    def handle_frame_bgr(self, frame_bgr: np.ndarray) -> None:
        now = _now()
        with self._lock:
            self.frame_bgr = frame_bgr
            self.frame_jpeg = encode_jpeg(frame_bgr)
            self.last_frame_at = now
            self.last_frame_wall_at = _wall_time()
            self.frames_seen += 1
            self._fps_window.append(now)
            self._fps_window = [item for item in self._fps_window if now - item <= 2.0]

            if self.camera_cal is None:
                self.detections = []
                self.calibration = CameraCalibrationState(state="visible", message="waiting for camera_info")
                self.overlay_jpeg = self.frame_jpeg
                return

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = list(locate_markers(rgb, self.camera_cal) or [])
            estimate = estimate_camera_pose_from_detections(
                detections,
                self.registry.known_markers,
                min_known_markers=self.spec.min_known_markers,
            )
            self.detections = detections
            self.calibration = self._promote_if_stable(estimate)
            self.overlay_jpeg = encode_jpeg(_annotate_frame(frame_bgr, detections, self.calibration))

    def _promote_if_stable(self, estimate: CameraCalibrationState) -> CameraCalibrationState:
        if estimate.pose is None or estimate.state not in {"solving", "calibrated"}:
            self._pose_history = []
            return estimate

        self._pose_history.append(estimate.pose)
        self._pose_history = self._pose_history[-max(self.spec.promote_after_frames, 1):]
        estimate.stable_frame_count = len(self._pose_history)

        if len(self._pose_history) < self.spec.promote_after_frames:
            estimate.state = "solving"
            estimate.message = f"{estimate.message}; waiting for stable frames"
            return estimate

        positions = np.asarray([pose.position for pose in self._pose_history], dtype=float)
        std_m = float(np.max(np.std(positions, axis=0)))
        if std_m > self.spec.stable_position_std_m:
            estimate.state = "degraded"
            estimate.confidence = min(estimate.confidence, 0.45)
            estimate.message = f"camera pose unstable: position std {std_m:.3f} m"
            return estimate

        estimate.state = "calibrated"
        estimate.confidence = max(estimate.confidence, 0.75)
        estimate.message = f"{estimate.message}; stable position std {std_m:.3f} m"
        return estimate

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            age = None if self.last_frame_at <= 0.0 else max(0.0, _now() - self.last_frame_at)
            stale = age is None or age > self.spec.stale_after_s
            return {
                "name": self.spec.name,
                "enabled": self.spec.enabled,
                "sourceType": self.spec.source_type,
                "imageTopic": self.spec.image_topic,
                "cameraInfoTopic": self.spec.camera_info_topic,
                "frameId": self.spec.frame_id,
                "labels": list(self.spec.labels),
                "aliases": list(self.spec.aliases),
                "rosDomainId": self.spec.ros_domain_id,
                "framesSeen": self.frames_seen,
                "fps": self.fps(),
                "stale": stale,
                "ageS": age,
                "lastFrameAt": self.last_frame_wall_at or None,
                "cameraInfo": camera_calibration_to_json(self.camera_cal),
                "detections": marker_detections_to_json(self.detections),
                "calibration": self.calibration.to_json(),
            }

    def fps(self) -> float:
        now = _now()
        self._fps_window = [item for item in self._fps_window if now - item <= 2.0]
        if len(self._fps_window) < 2:
            return 0.0
        elapsed = max(1e-6, self._fps_window[-1] - self._fps_window[0])
        return float((len(self._fps_window) - 1) / elapsed)

    def floor_projection_client(self) -> SimpleNamespace | None:
        with self._lock:
            if (
                self.frame_bgr is None
                or self.camera_cal is None
                or self.calibration.pose is None
                or self.calibration.state != "calibrated"
            ):
                return None
            return SimpleNamespace(
                name=self.spec.name,
                last_frame_resized=self.frame_bgr.copy(),
                camera_pose=self.calibration.pose.as_tuple(),
                camera_cal=self.camera_cal,
            )


@dataclass
class FusedMapSnapshot:
    status: str = "unavailable"
    summary: dict[str, Any] = field(default_factory=dict)
    floor_jpeg: bytes | None = None
    obstacle_jpeg: bytes | None = None
    gaussian_jpeg: bytes | None = None
    disagreement_jpeg: bytes | None = None
    updated_at: float | None = None

    def to_json(self) -> dict[str, Any]:
        payload = dict(self.summary)
        payload.update({"status": self.status, "updatedAt": self.updated_at})
        return payload


def _room_mask(map_size_px: int, map_extent_m: float, bounds: list[float] | None) -> np.ndarray:
    mask = np.zeros((map_size_px, map_size_px), dtype=np.uint8)
    if not bounds:
        mask[:, :] = 255
        return mask
    min_x, min_y, max_x, max_y = [float(item) for item in bounds]
    pixels_per_m = map_size_px / map_extent_m

    def to_px(x: float, y: float) -> tuple[int, int]:
        return (
            int(round(map_size_px * 0.5 + x * pixels_per_m)),
            int(round(map_size_px * 0.5 - y * pixels_per_m)),
        )

    p1 = to_px(min_x, max_y)
    p2 = to_px(max_x, min_y)
    x1, x2 = sorted((max(0, p1[0]), min(map_size_px, p2[0])))
    y1, y2 = sorted((max(0, p1[1]), min(map_size_px, p2[1])))
    mask[y1:y2, x1:x2] = 255
    return mask


def build_fused_room_maps(
    clients: list[SimpleNamespace],
    *,
    map_size_px: int = DEFAULT_MAP_SIZE_PX,
    map_extent_m: float = DEFAULT_MAP_EXTENT_M,
    room_bounds: list[float] | None = None,
    include_unknown: bool = True,
) -> FusedMapSnapshot:
    if not clients:
        return FusedMapSnapshot(
            status="unavailable",
            summary={"calibratedCameraCount": 0, "message": "no calibrated external cameras"},
            updated_at=_wall_time(),
        )

    heatmaps = [np.zeros(client.last_frame_resized.shape[:2], dtype=np.float32) for client in clients]
    _combined_heatmap, floor_bgr = generate_orthographic_floor_maps(
        clients,
        heatmaps,
        camera_cal=None,
        map_size_px=map_size_px,
        map_extent_meters=map_extent_m,
    )
    room = _room_mask(map_size_px, map_extent_m, room_bounds)
    touched = (floor_bgr.sum(axis=2) > 24).astype(np.uint8) * 255
    touched_room = cv2.bitwise_and(touched, room)
    covered_area_px = int(np.count_nonzero(touched_room))
    room_area_px = max(1, int(np.count_nonzero(room)))

    gray = cv2.cvtColor(floor_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    obstacle_mask = cv2.dilate(edges, kernel)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
    obstacle_mask = cv2.bitwise_and(obstacle_mask, touched_room)
    if include_unknown:
        unknown = cv2.bitwise_and(room, cv2.bitwise_not(touched))
        obstacle_mask = cv2.bitwise_or(obstacle_mask, unknown)

    gaussian = cv2.GaussianBlur(touched_room.astype(np.float32) / 255.0, (0, 0), sigmaX=9, sigmaY=9)
    gaussian_u8 = np.clip(gaussian * 255.0, 0, 255).astype(np.uint8)
    gaussian_bgr = cv2.applyColorMap(gaussian_u8, cv2.COLORMAP_VIRIDIS)

    per_camera_gray: list[np.ndarray] = []
    per_camera_masks: list[np.ndarray] = []
    for client in clients:
        single_heatmap = [np.zeros(client.last_frame_resized.shape[:2], dtype=np.float32)]
        _hm, single_bgr = generate_orthographic_floor_maps(
            [client],
            single_heatmap,
            camera_cal=None,
            map_size_px=map_size_px,
            map_extent_meters=map_extent_m,
        )
        per_camera_gray.append(cv2.cvtColor(single_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32))
        per_camera_masks.append((single_bgr.sum(axis=2) > 24).astype(np.uint8))

    disagreement_u8 = np.zeros((map_size_px, map_size_px), dtype=np.uint8)
    if len(per_camera_gray) > 1:
        stack = np.stack(per_camera_gray, axis=0)
        coverage = np.stack(per_camera_masks, axis=0).sum(axis=0)
        std = np.std(stack, axis=0)
        disagreement_u8 = np.where(coverage >= 2, np.clip(std * 2.0, 0, 255), 0).astype(np.uint8)
    disagreement_bgr = cv2.applyColorMap(disagreement_u8, cv2.COLORMAP_MAGMA)

    obstacle_bgr = cv2.cvtColor(obstacle_mask, cv2.COLOR_GRAY2BGR)
    obstacle_bgr[:, :, 1] = np.maximum(obstacle_bgr[:, :, 1], floor_bgr[:, :, 1] // 3)
    obstacle_area_px = int(np.count_nonzero(cv2.bitwise_and(obstacle_mask, room)))
    disagreement_px = int(np.count_nonzero(disagreement_u8))

    return FusedMapSnapshot(
        status="ready",
        summary={
            "calibratedCameraCount": len(clients),
            "cameras": [str(getattr(client, "name", index)) for index, client in enumerate(clients)],
            "mapSizePx": map_size_px,
            "mapExtentM": map_extent_m,
            "coverageRatio": covered_area_px / room_area_px,
            "obstacleCandidateRatio": obstacle_area_px / room_area_px,
            "disagreementRatio": disagreement_px / room_area_px,
            "unknownAreaTreatedAsObstacle": bool(include_unknown),
        },
        floor_jpeg=encode_jpeg(floor_bgr),
        obstacle_jpeg=encode_jpeg(obstacle_bgr),
        gaussian_jpeg=encode_jpeg(gaussian_bgr),
        disagreement_jpeg=encode_jpeg(disagreement_bgr),
        updated_at=_wall_time(),
    )


class ExternalCameraBridgeState:
    def __init__(self, registry: ExternalRoomCameraRegistry) -> None:
        self.registry = registry
        self.cameras = {spec.name: ExternalCameraRuntime(spec, registry) for spec in registry.enabled_cameras()}
        self.discovered_candidates: list[dict[str, Any]] = []
        self.fused_maps = FusedMapSnapshot()
        self._lock = threading.RLock()

    def set_camera_info(self, camera_name: str, info: Any) -> None:
        runtime = self.cameras.get(camera_name)
        if runtime is not None:
            runtime.set_camera_info(info)

    def handle_compressed_image(self, camera_name: str, data: bytes | bytearray | memoryview) -> None:
        runtime = self.cameras.get(camera_name)
        if runtime is not None:
            runtime.handle_compressed_image(data)

    def update_discovery(self, topics_and_types: Iterable[tuple[str, list[str]]]) -> None:
        configured_image_topics = {camera.image_topic for camera in self.registry.enabled_cameras()}
        configured_info_topics = {camera.camera_info_topic for camera in self.registry.enabled_cameras()}
        candidates = discover_ros_camera_candidates(topics_and_types)
        for candidate in candidates:
            candidate["configured"] = (
                candidate.get("imageTopic") in configured_image_topics
                or candidate.get("cameraInfoTopic") in configured_info_topics
            )
        with self._lock:
            self.discovered_candidates = candidates

    def rebuild_fused_maps(self) -> None:
        fusion = self.registry.fusion
        map_size_px = int(_mapping_value(fusion, "mapSizePx", "map_size_px", default=DEFAULT_MAP_SIZE_PX))
        map_extent_m = float(_mapping_value(fusion, "mapExtentM", "map_extent_m", default=DEFAULT_MAP_EXTENT_M))
        room_bounds = _mapping_value(fusion, "roomBounds", "room_bounds")
        include_unknown = _bool_value(_mapping_value(fusion, "includeUnknown", "include_unknown"), True)
        clients = [runtime.floor_projection_client() for runtime in self.cameras.values()]
        self.fused_maps = build_fused_room_maps(
            [client for client in clients if client is not None],
            map_size_px=map_size_px,
            map_extent_m=map_extent_m,
            room_bounds=room_bounds,
            include_unknown=include_unknown,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "ok": bool(self.cameras),
            "registry": self.registry.to_json(),
            "cameras": [runtime.snapshot() for runtime in self.cameras.values()],
            "discoveredCandidates": self.discovered_candidates,
            "fusedMaps": self.fused_maps.to_json(),
        }


def discover_ros_camera_candidates(topics_and_types: Iterable[tuple[str, list[str]]]) -> list[dict[str, Any]]:
    topics = {name: list(types) for name, types in topics_and_types}
    candidates = []
    for topic_name, topic_types in sorted(topics.items()):
        if not topic_name.endswith("/camera_info"):
            continue
        base = topic_name[: -len("/camera_info")]
        compressed = f"{base}/image_raw/compressed"
        raw = f"{base}/image_raw"
        compressed_depth = f"{base}/image_raw/compressedDepth"
        if compressed in topics:
            image_topic = compressed
            source_type = "ros2_compressed_image"
        elif raw in topics:
            image_topic = raw
            source_type = "ros2_raw_image"
        elif compressed_depth in topics:
            image_topic = compressed_depth
            source_type = "ros2_compressed_depth"
        else:
            image_topic = None
            source_type = None
        candidates.append(
            {
                "name": base.strip("/").replace("/", "_") or "camera",
                "baseTopic": base,
                "cameraInfoTopic": topic_name,
                "cameraInfoTypes": topic_types,
                "imageTopic": image_topic,
                "sourceType": source_type,
                "configured": False,
            }
        )
    return candidates


def save_artifact_snapshot(state: ExternalCameraBridgeState, artifact_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = state.snapshot()
    (artifact_dir / "external_room_cameras.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for name, runtime in state.cameras.items():
        if runtime.frame_jpeg:
            (artifact_dir / f"{name}_latest.jpg").write_bytes(runtime.frame_jpeg)
        if runtime.overlay_jpeg:
            (artifact_dir / f"{name}_overlay.jpg").write_bytes(runtime.overlay_jpeg)
    maps = state.fused_maps
    if maps.floor_jpeg:
        (artifact_dir / "fused_floor.jpg").write_bytes(maps.floor_jpeg)
    if maps.obstacle_jpeg:
        (artifact_dir / "fused_obstacles.jpg").write_bytes(maps.obstacle_jpeg)
    if maps.gaussian_jpeg:
        (artifact_dir / "fused_floor_gaussian.jpg").write_bytes(maps.gaussian_jpeg)
    if maps.disagreement_jpeg:
        (artifact_dir / "fused_disagreement.jpg").write_bytes(maps.disagreement_jpeg)
    return payload


class ExternalCameraHttpHandler(BaseHTTPRequestHandler):
    server: "ExternalCameraHttpServer"

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("external camera http: " + format, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path in {"/", "/healthz", "/cameras"}:
                self._send_json(self.server.bridge_state.snapshot())
                return
            if path == "/maps/latest.json":
                self._send_json(self.server.bridge_state.fused_maps.to_json())
                return
            if path == "/maps/floor.jpg":
                self._send_image(self.server.bridge_state.fused_maps.floor_jpeg)
                return
            if path == "/maps/obstacles.jpg":
                self._send_image(self.server.bridge_state.fused_maps.obstacle_jpeg)
                return
            if path == "/maps/gaussian.jpg":
                self._send_image(self.server.bridge_state.fused_maps.gaussian_jpeg)
                return
            if path == "/maps/disagreement.jpg":
                self._send_image(self.server.bridge_state.fused_maps.disagreement_jpeg)
                return
            if path.startswith("/cameras/"):
                self._handle_camera_path(path)
                return
            self._send_text(HTTPStatus.NOT_FOUND, "not found\n")
        except Exception as exc:
            logger.exception("external camera HTTP request failed")
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_camera_path(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) < 4:
            self._send_text(HTTPStatus.NOT_FOUND, "not found\n")
            return
        name = unquote(parts[2])
        action = parts[3]
        runtime = self.server.bridge_state.cameras.get(name)
        if runtime is None:
            self._send_text(HTTPStatus.NOT_FOUND, "camera not found\n")
            return
        if action == "snapshot.jpg":
            self._send_image(runtime.frame_jpeg)
            return
        if action == "overlay.jpg":
            self._send_image(runtime.overlay_jpeg)
            return
        if action == "markers":
            self._send_json({"camera": name, "detections": marker_detections_to_json(runtime.detections)})
            return
        if action == "status":
            self._send_json(runtime.snapshot())
            return
        self._send_text(HTTPStatus.NOT_FOUND, "not found\n")

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(_json_safe(payload), allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_image(self, body: bytes | None) -> None:
        if not body:
            self._send_text(HTTPStatus.NOT_FOUND, "image unavailable\n")
            return
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: HTTPStatus, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class ExternalCameraHttpServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], bridge_state: ExternalCameraBridgeState) -> None:
        super().__init__(server_address, ExternalCameraHttpHandler)
        self.bridge_state = bridge_state


def build_ros_node_class():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, CompressedImage

    class ExternalCameraRosNode(Node):
        def __init__(
            self,
            registry: ExternalRoomCameraRegistry,
            state: ExternalCameraBridgeState,
            *,
            active_domain_id: int | None,
            artifact_dir: Path | None = None,
            artifact_interval_s: float = 0.0,
        ) -> None:
            super().__init__("stringman_external_camera_bridge")
            self.registry = registry
            self.state = state
            self.active_domain_id = active_domain_id
            self.artifact_dir = artifact_dir
            self.artifact_interval_s = max(0.0, float(artifact_interval_s))
            self.last_artifact_at = 0.0

            for spec in registry.enabled_cameras():
                if spec.source_type != "ros2_compressed_image":
                    self.get_logger().warning(f"Skipping unsupported external camera source {spec.to_json()}")
                    continue
                if active_domain_id is not None and spec.ros_domain_id not in (None, active_domain_id):
                    self.get_logger().info(
                        f"Skipping camera {spec.name} for ROS domain {active_domain_id}; "
                        f"camera is configured for {spec.ros_domain_id}"
                    )
                    continue
                self.create_subscription(
                    CameraInfo,
                    spec.camera_info_topic,
                    lambda msg, name=spec.name: self.state.set_camera_info(name, msg),
                    10,
                )
                self.create_subscription(
                    CompressedImage,
                    spec.image_topic,
                    lambda msg, name=spec.name: self.state.handle_compressed_image(name, msg.data),
                    qos_profile_sensor_data,
                )
                self.get_logger().info(f"Subscribed external camera {spec.name}: {spec.image_topic}")

            self.create_timer(1.0, self._housekeeping)

        def _housekeeping(self) -> None:
            self.state.update_discovery(self.get_topic_names_and_types())
            self.state.rebuild_fused_maps()
            if self.artifact_dir is not None and self.artifact_interval_s > 0.0:
                now = _now()
                if now - self.last_artifact_at >= self.artifact_interval_s:
                    save_artifact_snapshot(self.state, self.artifact_dir)
                    self.last_artifact_at = now

    return rclpy, ExternalCameraRosNode


def _first_enabled_domain(registry: ExternalRoomCameraRegistry) -> int | None:
    for camera in registry.enabled_cameras():
        if camera.ros_domain_id is not None:
            return camera.ros_domain_id
    return None


def _first_enabled_rmw(registry: ExternalRoomCameraRegistry) -> str | None:
    for camera in registry.enabled_cameras():
        if camera.rmw:
            return camera.rmw
    return None


def start_http_server(state: ExternalCameraBridgeState, host: str, port: int) -> ExternalCameraHttpServer:
    server = ExternalCameraHttpServer((host, port), state)
    thread = threading.Thread(target=server.serve_forever, name="external-camera-http", daemon=True)
    thread.start()
    return server


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge configured external RGB cameras into Stringman room mapping.")
    parser.add_argument("--config", default="bedroom.conf", help="Robot config containing externalRoomCameras")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", default=8091, type=int, help="HTTP status/snapshot port")
    parser.add_argument("--ros-domain-id", type=int, default=None, help="ROS domain to use; defaults to first camera")
    parser.add_argument("--rmw", default=None, help="RMW implementation; defaults to first camera")
    parser.add_argument("--artifact-dir", default=None, help="Directory for periodic external camera artifacts")
    parser.add_argument("--artifact-interval-s", default=10.0, type=float, help="Artifact write interval")
    parser.add_argument("--no-ros", action="store_true", help="Start only the HTTP/API state without ROS subscriptions")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    registry = load_external_room_camera_registry(Path(args.config))
    domain_id = args.ros_domain_id if args.ros_domain_id is not None else _first_enabled_domain(registry)
    rmw = args.rmw or _first_enabled_rmw(registry)
    if domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
    if rmw:
        os.environ["RMW_IMPLEMENTATION"] = rmw

    state = ExternalCameraBridgeState(registry)
    server = start_http_server(state, args.host, args.port)
    print("Stringman external camera bridge")
    print(f"  Config: {Path(args.config).resolve()}")
    print(f"  HTTP: http://{args.host}:{args.port}")
    print(f"  ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', '<default>')}")
    print(f"  RMW_IMPLEMENTATION: {os.environ.get('RMW_IMPLEMENTATION', '<default>')}")
    print(f"  Cameras: {', '.join(state.cameras) or 'none'}")

    if args.no_ros:
        try:
            while True:
                state.rebuild_fused_maps()
                time.sleep(1.0)
        except KeyboardInterrupt:
            return 0
        finally:
            server.shutdown()

    rclpy, node_class = build_ros_node_class()
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    rclpy.init()
    node = node_class(
        registry,
        state,
        active_domain_id=domain_id,
        artifact_dir=artifact_dir,
        artifact_interval_s=args.artifact_interval_s,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        return 0
    finally:
        if artifact_dir is not None:
            save_artifact_snapshot(state, artifact_dir)
        try:
            node.destroy_node()
        except Exception:
            logger.debug("ROS node was already destroyed", exc_info=True)
        try:
            rclpy.shutdown()
        except Exception:
            logger.debug("ROS context was already shut down", exc_info=True)
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
