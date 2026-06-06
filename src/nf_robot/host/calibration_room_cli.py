"""Plan calibration safety blocks from room dimensions and catch-risk objects."""

from __future__ import annotations

import argparse
import cv2
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import nf_robot.common.definitions as model_constants
from nf_robot.common.pose_functions import compose_poses
from nf_robot.host.calibration_safety_apply_cli import derive_line_endpoints
from nf_robot.host.floor_view import generate_orthographic_floor_maps


MODE_CHOICES = ("full", "constrained", "manual_assisted")
PLAN_QUALITY_LEVELS = ("marginal", "usable", "strong")
PLAN_QUALITY_RANK = {level: index for index, level in enumerate(PLAN_QUALITY_LEVELS)}


class CalibrationRoomPlanError(ValueError):
    """Raised when no safe room plan can be generated."""


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass(frozen=True)
class LineEndpoint:
    name: str
    point: Point

    def to_config(self) -> List[float]:
        return [_round(self.point.x), _round(self.point.y)]


@dataclass(frozen=True)
class NoGoZone:
    name: str
    zone_type: str
    margin_m: float = 0.0
    center: Optional[Point] = None
    radius_m: Optional[float] = None
    rect: Optional[Rect] = None
    points: Optional[List[Point]] = None

    def to_config(self) -> Dict[str, object]:
        if self.zone_type == "circle":
            if self.center is None or self.radius_m is None:
                raise CalibrationRoomPlanError("circle no-go zone is missing center or radius")
            return {
                "name": self.name,
                "type": "circle",
                "center": [_round(self.center.x), _round(self.center.y)],
                "radiusM": _round(self.radius_m),
                "marginM": _round(self.margin_m),
            }

        if self.zone_type == "rect":
            if self.rect is None:
                raise CalibrationRoomPlanError("rect no-go zone is missing bounds")
            return {
                "name": self.name,
                "type": "rect",
                "rect": [
                    [_round(self.rect.min_x), _round(self.rect.min_y)],
                    [_round(self.rect.max_x), _round(self.rect.max_y)],
                ],
                "marginM": _round(self.margin_m),
            }

        if self.zone_type == "polygon":
            if self.points is None or len(self.points) < 3:
                raise CalibrationRoomPlanError("polygon no-go zone is missing points")
            return {
                "name": self.name,
                "type": "polygon",
                "polygon": [[_round(point.x), _round(point.y)] for point in self.points],
                "marginM": _round(self.margin_m),
            }

        raise CalibrationRoomPlanError(f"unknown no-go zone type: {self.zone_type}")


def _round(value: float) -> float:
    return round(float(value), 4)


def _finite_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationRoomPlanError(f"{label} must be a number") from exc
    if not math.isfinite(result):
        raise CalibrationRoomPlanError(f"{label} must be finite")
    return result


def _positive(value: float, label: str) -> float:
    if value <= 0.0:
        raise CalibrationRoomPlanError(f"{label} must be greater than zero")
    return value


def _non_negative(value: float, label: str) -> float:
    if value < 0.0:
        raise CalibrationRoomPlanError(f"{label} must be zero or greater")
    return value


def parse_line_endpoint(spec: str, index: int = 0) -> LineEndpoint:
    """Parse NAME,X,Y or X,Y into a line endpoint."""

    parts = [part.strip() for part in spec.split(",")]
    if len(parts) == 2:
        name = f"line_endpoint_{index + 1}"
        x_text, y_text = parts
    elif len(parts) == 3:
        name, x_text, y_text = parts
        if not name:
            raise CalibrationRoomPlanError("line endpoint name cannot be empty")
    else:
        raise CalibrationRoomPlanError("line endpoint must be NAME,X,Y or X,Y")

    return LineEndpoint(
        name=name,
        point=Point(
            _finite_float(x_text, "line endpoint x"),
            _finite_float(y_text, "line endpoint y"),
        ),
    )


def parse_circle_no_go(spec: str) -> NoGoZone:
    """Parse NAME,X,Y,RADIUS[,MARGIN] into a circle no-go zone."""

    parts = [part.strip() for part in spec.split(",")]
    if len(parts) not in (4, 5):
        raise CalibrationRoomPlanError("circle no-go must be NAME,X,Y,RADIUS[,MARGIN]")

    name = parts[0]
    if not name:
        raise CalibrationRoomPlanError("circle no-go name cannot be empty")

    radius_m = _non_negative(_finite_float(parts[3], "circle radius"), "circle radius")
    margin_m = 0.0
    if len(parts) == 5:
        margin_m = _non_negative(_finite_float(parts[4], "circle margin"), "circle margin")

    return NoGoZone(
        name=name,
        zone_type="circle",
        center=Point(_finite_float(parts[1], "circle x"), _finite_float(parts[2], "circle y")),
        radius_m=radius_m,
        margin_m=margin_m,
    )


def parse_rect_no_go(spec: str) -> NoGoZone:
    """Parse NAME,X1,Y1,X2,Y2[,MARGIN] into a rectangular no-go zone."""

    parts = [part.strip() for part in spec.split(",")]
    if len(parts) not in (5, 6):
        raise CalibrationRoomPlanError("rect no-go must be NAME,X1,Y1,X2,Y2[,MARGIN]")

    name = parts[0]
    if not name:
        raise CalibrationRoomPlanError("rect no-go name cannot be empty")

    x1 = _finite_float(parts[1], "rect x1")
    y1 = _finite_float(parts[2], "rect y1")
    x2 = _finite_float(parts[3], "rect x2")
    y2 = _finite_float(parts[4], "rect y2")
    margin_m = 0.0
    if len(parts) == 6:
        margin_m = _non_negative(_finite_float(parts[5], "rect margin"), "rect margin")

    return NoGoZone(
        name=name,
        zone_type="rect",
        rect=Rect(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
        margin_m=margin_m,
    )


def parse_polygon_no_go(spec: str) -> NoGoZone:
    """Parse NAME,X1,Y1,X2,Y2,X3,Y3...[,MARGIN] into a polygon no-go zone."""

    parts = [part.strip() for part in spec.split(",")]
    if len(parts) < 7:
        raise CalibrationRoomPlanError("polygon no-go must be NAME,X1,Y1,X2,Y2,X3,Y3...[,MARGIN]")

    name = parts[0]
    if not name:
        raise CalibrationRoomPlanError("polygon no-go name cannot be empty")

    numeric_parts = parts[1:]
    margin_m = 0.0
    if len(numeric_parts) % 2 == 1:
        margin_m = _non_negative(_finite_float(numeric_parts[-1], "polygon margin"), "polygon margin")
        numeric_parts = numeric_parts[:-1]
    if len(numeric_parts) < 6 or len(numeric_parts) % 2 != 0:
        raise CalibrationRoomPlanError("polygon no-go must include at least three x,y point pairs")

    points = [
        Point(
            _finite_float(numeric_parts[index], "polygon x"),
            _finite_float(numeric_parts[index + 1], "polygon y"),
        )
        for index in range(0, len(numeric_parts), 2)
    ]
    return NoGoZone(name=name, zone_type="polygon", points=points, margin_m=margin_m)


def parse_hazard_no_go(spec: str, default_radius_m: float) -> NoGoZone:
    """Parse NAME,X,Y[,RADIUS[,MARGIN]] into a temporary hazard avoidance zone."""

    parts = [part.strip() for part in spec.split(",")]
    if len(parts) not in (3, 4, 5):
        raise CalibrationRoomPlanError("hazard must be NAME,X,Y[,RADIUS[,MARGIN]]")
    name = parts[0]
    if not name:
        raise CalibrationRoomPlanError("hazard name cannot be empty")
    radius_m = default_radius_m
    margin_m = 0.0
    if len(parts) >= 4:
        radius_m = _non_negative(_finite_float(parts[3], "hazard radius"), "hazard radius")
    if len(parts) == 5:
        margin_m = _non_negative(_finite_float(parts[4], "hazard margin"), "hazard margin")
    return NoGoZone(
        name=f"hazard:{name}",
        zone_type="circle",
        center=Point(_finite_float(parts[1], "hazard x"), _finite_float(parts[2], "hazard y")),
        radius_m=radius_m,
        margin_m=margin_m,
    )


def _room_mapping_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _room_value(room_spec: dict[str, Any], *keys: str) -> Any:
    value = _room_mapping_value(room_spec, *keys)
    if value is not None:
        return value
    nested_room = room_spec.get("room")
    if isinstance(nested_room, dict):
        return _room_mapping_value(nested_room, *keys)
    return None


def _room_float(room_spec: dict[str, Any], keys: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    value = _room_value(room_spec, *keys)
    if value is None:
        return default
    return _finite_float(str(value), keys[0])


def _room_bool(room_spec: dict[str, Any], keys: Sequence[str], default: bool = False) -> bool:
    value = _room_value(room_spec, *keys)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
    raise CalibrationRoomPlanError(f"{keys[0]} must be boolean")


def _room_string(room_spec: dict[str, Any], keys: Sequence[str], default: str) -> str:
    value = _room_value(room_spec, *keys)
    if value is None:
        return default
    return str(value)


def _room_axis_from_point(room_spec: dict[str, Any], point_key: str, axis: str) -> Optional[float]:
    value = _room_value(room_spec, point_key)
    if value is None:
        return None
    if isinstance(value, dict):
        axis_value = value.get(axis)
        if axis_value is None:
            return None
        return _finite_float(str(axis_value), f"{point_key}.{axis}")
    if isinstance(value, list) and len(value) >= 2:
        index = 0 if axis == "x" else 1
        return _finite_float(str(value[index]), f"{point_key}.{axis}")
    raise CalibrationRoomPlanError(f"{point_key} must be an object with x/y or [x,y]")


def _room_origin_axis(room_spec: dict[str, Any], axis: str, aliases: Sequence[str], default: float) -> float:
    value = _room_float(room_spec, aliases, None)
    if value is not None:
        return value
    point_value = _room_axis_from_point(room_spec, "origin", axis)
    if point_value is not None:
        return point_value
    return default


def _load_room_file(path: Path) -> dict[str, Any]:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise CalibrationRoomPlanError("room file must be a JSON object")
    return raw


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _point_from_json(value: Any, label: str) -> Point:
    if isinstance(value, dict):
        if "point" in value:
            return _point_from_json(value["point"], label)
        if "position" in value:
            return _point_from_json(value["position"], label)
        if "center" in value:
            return _point_from_json(value["center"], label)
        if "x" in value and "y" in value:
            return Point(
                _finite_float(str(value["x"]), f"{label} x"),
                _finite_float(str(value["y"]), f"{label} y"),
            )
    if isinstance(value, list) and len(value) >= 2:
        return Point(
            _finite_float(str(value[0]), f"{label} x"),
            _finite_float(str(value[1]), f"{label} y"),
        )
    raise CalibrationRoomPlanError(f"{label} must be [x,y] or an object with x/y")


def _json_margin(value: dict[str, Any]) -> float:
    margin_value = _room_mapping_value(value, "marginM", "margin_m", "margin")
    if margin_value is None:
        return 0.0
    return _non_negative(_finite_float(str(margin_value), "no-go margin"), "no-go margin")


def parse_line_endpoint_json(value: Any, index: int = 0) -> LineEndpoint:
    if isinstance(value, str):
        return parse_line_endpoint(value, index)
    name = f"line_endpoint_{index + 1}"
    if isinstance(value, dict):
        raw_name = value.get("name")
        if raw_name:
            name = str(raw_name)
        point = _point_from_json(value, f"line endpoint {name}")
    else:
        point = _point_from_json(value, f"line endpoint {name}")
    return LineEndpoint(name=name, point=point)


def parse_hazard_json(value: Any, default_radius_m: float, index: int = 0) -> NoGoZone:
    if isinstance(value, str):
        return parse_hazard_no_go(value, default_radius_m)
    if not isinstance(value, dict):
        raise CalibrationRoomPlanError("hazard entries must be objects or NAME,X,Y[,RADIUS[,MARGIN]] strings")
    name = str(value.get("name") or f"hazard_{index + 1}")
    point = _point_from_json(value, f"hazard {name}")
    radius_value = _room_mapping_value(value, "radiusM", "radius_m", "radius")
    radius_m = default_radius_m
    if radius_value is not None:
        radius_m = _non_negative(_finite_float(str(radius_value), "hazard radius"), "hazard radius")
    return NoGoZone(
        name=f"hazard:{name}",
        zone_type="circle",
        center=point,
        radius_m=radius_m,
        margin_m=_json_margin(value),
    )


def _artifact_hazard_point(value: dict[str, Any], label: str) -> Optional[Point]:
    for key in (
        "point",
        "position",
        "center",
        "gantryPosition",
        "gantry_position",
        "safeProbeCenter",
        "safe_probe_center",
    ):
        if key in value:
            try:
                return _point_from_json(value[key], f"{label} {key}")
            except CalibrationRoomPlanError:
                pass
    for key in ("xy", "pointXY", "point_xy", "gantryXY", "gantry_xy", "positionXY", "position_xy"):
        if key in value:
            try:
                return _point_from_json(value[key], f"{label} {key}")
            except CalibrationRoomPlanError:
                pass
    if "x" in value and "y" in value:
        try:
            return _point_from_json(value, label)
        except CalibrationRoomPlanError:
            pass
    return None


def _artifact_hazard_name(value: dict[str, Any], fallback: str) -> str:
    for key in ("name", "kind", "type", "reason"):
        item = value.get(key)
        if item:
            return str(item).replace(" ", "_")
    return fallback


def _mapping_is_hazard(value: dict[str, Any]) -> bool:
    marker_parts = [
        str(value.get(key, "")).lower()
        for key in ("kind", "type", "name", "reason", "message")
    ]
    marker = " ".join(marker_parts)
    return any(word in marker for word in ("hazard", "catch", "tension", "snag", "collision"))


def hazard_zones_from_artifact(artifact: Any, default_radius_m: float) -> List[NoGoZone]:
    zones: List[NoGoZone] = []
    seen = set()

    def add_zone(source: dict[str, Any], fallback: str) -> None:
        point = _artifact_hazard_point(source, fallback)
        if point is None:
            return
        radius_value = _room_mapping_value(source, "radiusM", "radius_m", "radius", "avoidRadiusM", "avoid_radius_m")
        radius_m = default_radius_m
        if radius_value is not None:
            radius_m = _non_negative(_finite_float(str(radius_value), "artifact hazard radius"), "artifact hazard radius")
        margin_m = _json_margin(source)
        key = (round(point.x, 6), round(point.y, 6), round(radius_m, 6), round(margin_m, 6))
        if key in seen:
            return
        seen.add(key)
        zones.append(
            NoGoZone(
                name=f"hazard:{_artifact_hazard_name(source, fallback)}",
                zone_type="circle",
                center=point,
                radius_m=radius_m,
                margin_m=margin_m,
            )
        )

    def visit(value: Any, fallback: str = "artifact_hazard") -> None:
        if isinstance(value, dict):
            nested_hazard = value.get("hazard")
            if isinstance(nested_hazard, dict):
                add_zone(nested_hazard, _artifact_hazard_name(value, fallback))
                visit(nested_hazard, _artifact_hazard_name(value, fallback))
            if _mapping_is_hazard(value):
                add_zone(value, fallback)
            for child_key, child in value.items():
                if child_key == "hazard" and isinstance(child, dict):
                    continue
                visit(child, _artifact_hazard_name(value, fallback))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{fallback}_{index + 1}")

    visit(artifact)
    return zones


def latest_artifact_paths_from_dir(directory: Path, limit: int = 1) -> List[Path]:
    if limit <= 0:
        raise CalibrationRoomPlanError("hazard artifact limit must be greater than zero")
    if not directory.exists() or not directory.is_dir():
        raise CalibrationRoomPlanError(f"artifact directory does not exist: {directory}")
    candidates = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".json"]
    if not candidates:
        raise CalibrationRoomPlanError(f"artifact directory contains no JSON artifacts: {directory}")
    candidates.sort(key=lambda path: path.name)
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.stat().st_ctime_ns), reverse=True)
    return candidates[:limit]


def line_endpoints_from_config(config: dict[str, Any]) -> List[LineEndpoint]:
    return [
        LineEndpoint(name=f"derived_line_endpoint_{index + 1}", point=Point(point[0], point[1]))
        for index, point in enumerate(derive_line_endpoints(config))
    ]


def _vec3_from_mapping(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise CalibrationRoomPlanError(f"{label} must be an object with x/y/z")
    try:
        return np.array([float(value["x"]), float(value["y"]), float(value["z"])], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationRoomPlanError(f"{label} must include numeric x/y/z") from exc


def _pose_from_config(value: Any, label: str) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, dict):
        raise CalibrationRoomPlanError(f"{label} must be a pose object")
    rotation = _room_mapping_value(value, "rotation", "rot")
    position = _room_mapping_value(value, "position", "pos")
    return _vec3_from_mapping(rotation, f"{label}.rotation"), _vec3_from_mapping(position, f"{label}.position")


def _anchor_xy_points_from_config(config: dict[str, Any]) -> List[Point]:
    points = [endpoint.point for endpoint in line_endpoints_from_config(config)]
    anchors = config.get("anchors")
    if isinstance(anchors, list):
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, dict):
                continue
            pose = anchor.get("pose")
            if isinstance(pose, dict):
                try:
                    position = _vec3_from_mapping(_room_mapping_value(pose, "position", "pos"), f"anchor {index + 1} position")
                    points.append(Point(float(position[0]), float(position[1])))
                except CalibrationRoomPlanError:
                    pass
    return points


def derive_room_bounds_from_config(config: dict[str, Any], padding_m: float = 0.6) -> Tuple[float, float, float, float]:
    points = _anchor_xy_points_from_config(config)
    if len(points) < 2:
        raise CalibrationRoomPlanError("cannot derive room bounds from config without at least two anchor/eyelet points")
    padding_m = _non_negative(float(padding_m), "room derivation padding")
    min_x = min(point.x for point in points) - padding_m
    max_x = max(point.x for point in points) + padding_m
    min_y = min(point.y for point in points) - padding_m
    max_y = max(point.y for point in points) + padding_m
    return min_x, min_y, max_x - min_x, max_y - min_y


def _camera_calibration_from_config(config: dict[str, Any]) -> Any:
    raw = _room_mapping_value(config, "cameraCal", "camera_cal")
    if not isinstance(raw, dict):
        raise CalibrationRoomPlanError("camera calibration config is required for camera-derived planning")
    resolution = _room_mapping_value(raw, "resolution", "res")
    if not isinstance(resolution, dict):
        raise CalibrationRoomPlanError("camera calibration resolution is required for camera-derived planning")
    intrinsic = _room_mapping_value(raw, "intrinsicMatrix", "intrinsic_matrix")
    distortion = _room_mapping_value(raw, "distortionCoeff", "distortion_coeff")
    if not isinstance(intrinsic, list) or len(intrinsic) != 9:
        raise CalibrationRoomPlanError("camera calibration intrinsicMatrix must contain 9 numbers")
    if not isinstance(distortion, list) or len(distortion) < 4:
        raise CalibrationRoomPlanError("camera calibration distortionCoeff must contain distortion coefficients")
    return SimpleNamespace(
        intrinsic_matrix=[float(item) for item in intrinsic],
        distortion_coeff=[float(item) for item in distortion],
        resolution=SimpleNamespace(
            width=int(_room_mapping_value(resolution, "width", "w")),
            height=int(_room_mapping_value(resolution, "height", "h")),
        ),
    )


def _anchor_camera_pose_from_config(config: dict[str, Any], anchor_num: int) -> np.ndarray:
    anchors = config.get("anchors")
    if not isinstance(anchors, list) or anchor_num >= len(anchors):
        raise CalibrationRoomPlanError(f"anchor {anchor_num} is missing from config")
    anchor = anchors[anchor_num]
    if not isinstance(anchor, dict):
        raise CalibrationRoomPlanError(f"anchor {anchor_num} config must be an object")
    anchor_pose = _pose_from_config(anchor.get("pose"), f"anchor {anchor_num} pose")
    anchor_type = str(_room_mapping_value(config, "anchorType", "anchor_type") or "").upper()
    if "ARPEGGIO" in anchor_type:
        indirect_line = _room_mapping_value(anchor, "indirectLine", "indirect_line") or {}
        cam_tilt = _room_mapping_value(indirect_line, "camTilt", "cam_tilt") if isinstance(indirect_line, dict) else None
        cam_tilt = 22.0 if cam_tilt is None else float(cam_tilt)
        extra_tilt = (22.0 - cam_tilt) / 180.0 * math.pi
        pose = compose_poses([
            anchor_pose,
            model_constants.arp_anchor_camera,
            (np.array([extra_tilt, 0.0, 0.0], dtype=float), np.zeros(3, dtype=float)),
        ])
    else:
        pose = compose_poses([
            anchor_pose,
            model_constants.anchor_camera,
            (np.zeros(3, dtype=float), np.zeros(3, dtype=float)),
        ])
    return np.array(pose, dtype=float)


def _preferred_camera_nums(config: dict[str, Any]) -> List[int]:
    preferred = _room_mapping_value(config, "preferredCameras", "preferred_cameras")
    if isinstance(preferred, list) and preferred:
        return [int(item) for item in preferred]
    anchors = config.get("anchors")
    if isinstance(anchors, list):
        return list(range(len(anchors)))
    return []


def _camera_stream_specs(config: dict[str, Any], explicit_urls: Sequence[str]) -> List[Tuple[int, str]]:
    preferred = _preferred_camera_nums(config)
    if explicit_urls:
        specs: List[Tuple[int, str]] = []
        for index, spec in enumerate(explicit_urls):
            if "," in spec:
                camera_text, url = spec.split(",", 1)
                specs.append((int(camera_text.strip()), url.strip()))
            else:
                if index >= len(preferred):
                    raise CalibrationRoomPlanError("camera stream URL without anchor number has no matching preferred camera")
                specs.append((preferred[index], spec.strip()))
        return specs
    return [(camera_num, f"http://127.0.0.1:{4247 + camera_num}/stream.mjpeg") for camera_num in preferred]


def _capture_camera_frame(url: str, warmup_frames: int = 2) -> np.ndarray:
    cap = cv2.VideoCapture(url)
    try:
        frame = None
        ok = False
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
        if not ok or frame is None:
            raise CalibrationRoomPlanError(f"camera stream did not produce a frame: {url}")
        return frame
    finally:
        cap.release()


def _world_to_map_px(point: Point, map_size_px: int, map_extent_m: float) -> Tuple[int, int]:
    pixels_per_m = map_size_px / map_extent_m
    return (
        int(round(map_size_px * 0.5 + point.x * pixels_per_m)),
        int(round(map_size_px * 0.5 - point.y * pixels_per_m)),
    )


def _rect_from_component(
    component_slice: Tuple[slice, slice],
    *,
    origin_x_m: float,
    origin_y_m: float,
    room_width_m: float,
    room_depth_m: float,
    map_size_px: int,
    map_extent_m: float,
    name: str,
) -> NoGoZone:
    y_slice, x_slice = component_slice
    pixels_per_m = map_size_px / map_extent_m
    min_x = (x_slice.start - map_size_px * 0.5) / pixels_per_m
    max_x = (x_slice.stop - map_size_px * 0.5) / pixels_per_m
    max_y = (map_size_px * 0.5 - y_slice.start) / pixels_per_m
    min_y = (map_size_px * 0.5 - y_slice.stop) / pixels_per_m
    room_min_x = origin_x_m
    room_max_x = origin_x_m + room_width_m
    room_min_y = origin_y_m
    room_max_y = origin_y_m + room_depth_m
    return NoGoZone(
        name=name,
        zone_type="rect",
        rect=Rect(
            float(max(room_min_x, min_x)),
            float(max(room_min_y, min_y)),
            float(min(room_max_x, max_x)),
            float(min(room_max_y, max_y)),
        ),
        margin_m=0.0,
    )


def camera_no_go_zones_from_streams(
    config: dict[str, Any],
    *,
    origin_x_m: float,
    origin_y_m: float,
    room_width_m: float,
    room_depth_m: float,
    stream_urls: Sequence[str] = (),
    map_size_px: int = 1000,
    map_extent_m: Optional[float] = None,
    min_component_area_m2: float = 0.04,
    max_component_area_ratio: float = 0.18,
    dilate_m: float = 0.08,
    max_zones: int = 24,
    include_unknown: bool = True,
) -> Tuple[List[NoGoZone], Dict[str, Any]]:
    map_size_px = int(_positive(float(map_size_px), "camera map size"))
    room_min_x = float(origin_x_m)
    room_max_x = float(origin_x_m + room_width_m)
    room_min_y = float(origin_y_m)
    room_max_y = float(origin_y_m + room_depth_m)
    if map_extent_m is None:
        max_abs = max(abs(room_min_x), abs(room_max_x), abs(room_min_y), abs(room_max_y), 1.0)
        map_extent_m = max_abs * 2.2
    map_extent_m = _positive(float(map_extent_m), "camera map extent")
    min_component_area_m2 = _positive(float(min_component_area_m2), "minimum camera component area")
    max_component_area_ratio = _positive(float(max_component_area_ratio), "maximum camera component area ratio")
    dilate_m = _non_negative(float(dilate_m), "camera clutter dilation")
    max_zones = int(_positive(float(max_zones), "maximum camera no-go zones"))

    specs = _camera_stream_specs(config, stream_urls)
    if not specs:
        raise CalibrationRoomPlanError("no preferred cameras available for camera-derived planning")
    frames = [(anchor_num, url, _capture_camera_frame(url)) for anchor_num, url in specs]
    clients = [
        SimpleNamespace(
            anchor_num=anchor_num,
            last_frame_resized=frame,
            camera_pose=_anchor_camera_pose_from_config(config, anchor_num),
        )
        for anchor_num, _url, frame in frames
    ]
    heatmaps = np.zeros((len(clients),) + clients[0].last_frame_resized.shape[:2], dtype=np.float32)
    _ortho_heatmap, ortho_bgr = generate_orthographic_floor_maps(
        clients,
        heatmaps,
        _camera_calibration_from_config(config),
        map_size_px=map_size_px,
        map_extent_meters=map_extent_m,
    )

    room_mask = np.zeros((map_size_px, map_size_px), dtype=np.uint8)
    min_px = _world_to_map_px(Point(room_min_x, room_max_y), map_size_px, map_extent_m)
    max_px = _world_to_map_px(Point(room_max_x, room_min_y), map_size_px, map_extent_m)
    x1, x2 = sorted((max(0, min_px[0]), min(map_size_px, max_px[0])))
    y1, y2 = sorted((max(0, min_px[1]), min(map_size_px, max_px[1])))
    room_mask[y1:y2, x1:x2] = 255

    touched = (ortho_bgr.sum(axis=2) > 24).astype(np.uint8) * 255
    gray = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 120)
    pixels_per_m = map_size_px / map_extent_m
    kernel_px = max(1, int(round(dilate_m * pixels_per_m)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px * 2 + 1, kernel_px * 2 + 1))
    clutter = cv2.dilate(edges, kernel)
    clutter = cv2.morphologyEx(clutter, cv2.MORPH_CLOSE, kernel)
    clutter = cv2.bitwise_and(clutter, room_mask)
    clutter = cv2.bitwise_and(clutter, touched)
    unknown = cv2.bitwise_and(room_mask, cv2.bitwise_not(touched)) if include_unknown else np.zeros_like(room_mask)
    obstacle_mask = cv2.bitwise_or(clutter, unknown)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats((obstacle_mask > 0).astype(np.uint8), 8)
    min_area_px = min_component_area_m2 * pixels_per_m * pixels_per_m
    room_area_px = max(1, int(np.count_nonzero(room_mask)))
    max_area_px = room_area_px * max_component_area_ratio
    zones: List[NoGoZone] = []
    component_summaries = []
    skipped_large_components = 0
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < min_area_px:
            continue
        if area > max_area_px:
            skipped_large_components += 1
            continue
        area_m2 = float(area) / (pixels_per_m * pixels_per_m)
        component_summaries.append((area_m2, x, y, w, h))
    component_summaries.sort(reverse=True)
    for zone_index, (area_m2, x, y, w, h) in enumerate(component_summaries[:max_zones]):
        zones.append(
            _rect_from_component(
                (slice(y, y + h), slice(x, x + w)),
                origin_x_m=origin_x_m,
                origin_y_m=origin_y_m,
                room_width_m=room_width_m,
                room_depth_m=room_depth_m,
                map_size_px=map_size_px,
                map_extent_m=map_extent_m,
                name=f"camera_clutter_{zone_index + 1}",
            )
        )

    covered_area_px = int(np.count_nonzero(cv2.bitwise_and(room_mask, touched)))
    summary = {
        "cameraCount": len(frames),
        "cameraStreams": [{"anchor": anchor_num, "url": url} for anchor_num, url, _frame in frames],
        "mapSizePx": map_size_px,
        "mapExtentM": _round(map_extent_m),
        "coverageRatio": _round(covered_area_px / room_area_px),
        "detectedNoGoZoneCount": len(zones),
        "candidateComponentCount": len(component_summaries),
        "skippedLargeComponentCount": skipped_large_components,
        "maxComponentAreaRatio": _round(max_component_area_ratio),
        "unknownAreaTreatedAsNoGo": bool(include_unknown),
    }
    return zones, summary


def _dedupe_line_endpoints(endpoints: Sequence[LineEndpoint]) -> List[LineEndpoint]:
    deduped: List[LineEndpoint] = []
    seen = set()
    for endpoint in endpoints:
        key = (round(endpoint.point.x, 6), round(endpoint.point.y, 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(endpoint)
    return deduped


def parse_no_go_zone_json(value: Any, index: int = 0) -> NoGoZone:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.startswith("circle:"):
            return parse_circle_no_go(normalized[len("circle:"):])
        if normalized.startswith("rect:"):
            return parse_rect_no_go(normalized[len("rect:"):])
        if normalized.startswith("polygon:"):
            return parse_polygon_no_go(normalized[len("polygon:"):])
        raise CalibrationRoomPlanError("string no-go zones in room files must start with circle:, rect:, or polygon:")
    if not isinstance(value, dict):
        raise CalibrationRoomPlanError("no-go zone entries must be objects")

    name = str(value.get("name") or f"zone_{index + 1}")
    kind = str(_room_mapping_value(value, "type", "kind", "shape") or "").strip().lower()
    margin_m = _json_margin(value)
    radius_value = _room_mapping_value(value, "radiusM", "radius_m", "radius")
    if kind == "circle" or "center" in value or radius_value is not None:
        if radius_value is None:
            raise CalibrationRoomPlanError(f"circle no-go {name!r} is missing radius")
        return NoGoZone(
            name=name,
            zone_type="circle",
            center=_point_from_json(value.get("center"), f"circle no-go {name} center"),
            radius_m=_non_negative(_finite_float(str(radius_value), "circle radius"), "circle radius"),
            margin_m=margin_m,
        )

    rect_value = _room_mapping_value(value, "rect", "rectangle", "bounds")
    if kind in ("rect", "rectangle") or rect_value is not None or ("min" in value and "max" in value):
        if rect_value is not None:
            if not isinstance(rect_value, list) or len(rect_value) != 2:
                raise CalibrationRoomPlanError(f"rect no-go {name!r} bounds must be two points")
            first = _point_from_json(rect_value[0], f"rect no-go {name} min")
            second = _point_from_json(rect_value[1], f"rect no-go {name} max")
        else:
            first = _point_from_json(value.get("min"), f"rect no-go {name} min")
            second = _point_from_json(value.get("max"), f"rect no-go {name} max")
        return NoGoZone(
            name=name,
            zone_type="rect",
            rect=Rect(min(first.x, second.x), min(first.y, second.y), max(first.x, second.x), max(first.y, second.y)),
            margin_m=margin_m,
        )

    polygon_value = _room_mapping_value(value, "polygon", "points")
    if kind == "polygon" or polygon_value is not None:
        if not isinstance(polygon_value, list) or len(polygon_value) < 3:
            raise CalibrationRoomPlanError(f"polygon no-go {name!r} must include at least three points")
        return NoGoZone(
            name=name,
            zone_type="polygon",
            points=[
                _point_from_json(point, f"polygon no-go {name} point {point_index + 1}")
                for point_index, point in enumerate(polygon_value)
            ],
            margin_m=margin_m,
        )
    raise CalibrationRoomPlanError(f"no-go zone {name!r} must define circle or rect geometry")


def _room_list(room_spec: dict[str, Any], *keys: str) -> list[Any]:
    value = _room_value(room_spec, *keys)
    if value is None:
        return []
    if not isinstance(value, list):
        raise CalibrationRoomPlanError(f"{keys[0]} must be a list")
    return value


def _point_in_rect(point: Point, rect: Rect) -> bool:
    return rect.min_x <= point.x <= rect.max_x and rect.min_y <= point.y <= rect.max_y


def _expanded_rect(rect: Rect, margin_m: float) -> Rect:
    return Rect(
        rect.min_x - margin_m,
        rect.min_y - margin_m,
        rect.max_x + margin_m,
        rect.max_y + margin_m,
    )


def _obstacle_margin(zone: NoGoZone, global_margin_m: float) -> float:
    return zone.margin_m + global_margin_m


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _distance_point_to_segment(point: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return _distance(point, start)

    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest = Point(start.x + t * dx, start.y + t * dy)
    return _distance(point, closest)


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a.x, c.x) - 1e-9 <= b.x <= max(a.x, c.x) + 1e-9
        and min(a.y, c.y) - 1e-9 <= b.y <= max(a.y, c.y) + 1e-9
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = _cross(a, b, c)
    o2 = _cross(a, b, d)
    o3 = _cross(c, d, a)
    o4 = _cross(c, d, b)

    if abs(o1) <= 1e-9 and _on_segment(a, c, b):
        return True
    if abs(o2) <= 1e-9 and _on_segment(a, d, b):
        return True
    if abs(o3) <= 1e-9 and _on_segment(c, a, d):
        return True
    if abs(o4) <= 1e-9 and _on_segment(c, b, d):
        return True

    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def _segment_intersects_rect(start: Point, end: Point, rect: Rect) -> bool:
    if _point_in_rect(start, rect) or _point_in_rect(end, rect):
        return True

    corners = [
        Point(rect.min_x, rect.min_y),
        Point(rect.max_x, rect.min_y),
        Point(rect.max_x, rect.max_y),
        Point(rect.min_x, rect.max_y),
    ]
    edges = zip(corners, corners[1:] + corners[:1])
    return any(_segments_intersect(start, end, edge_start, edge_end) for edge_start, edge_end in edges)


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current.y > point.y) != (previous.y > point.y):
            crossing_x = (previous.x - current.x) * (point.y - current.y) / (
                (previous.y - current.y) or 1e-12
            ) + current.x
            if point.x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _polygon_edges(points: Sequence[Point]) -> Iterable[Tuple[Point, Point]]:
    return zip(points, list(points[1:]) + [points[0]])


def _distance_segment_to_segment(a: Point, b: Point, c: Point, d: Point) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _distance_point_to_segment(a, c, d),
        _distance_point_to_segment(b, c, d),
        _distance_point_to_segment(c, a, b),
        _distance_point_to_segment(d, a, b),
    )


def _distance_point_to_polygon_edges(point: Point, polygon: Sequence[Point]) -> float:
    return min(
        _distance_point_to_segment(point, edge_start, edge_end)
        for edge_start, edge_end in _polygon_edges(polygon)
    )


def _segment_intersects_polygon(start: Point, end: Point, polygon: Sequence[Point], margin_m: float) -> bool:
    if _point_in_polygon(start, polygon) or _point_in_polygon(end, polygon):
        return True
    for edge_start, edge_end in _polygon_edges(polygon):
        if _distance_segment_to_segment(start, end, edge_start, edge_end) <= margin_m:
            return True
    return False


def _point_inside_no_go(point: Point, zone: NoGoZone, global_margin_m: float) -> bool:
    margin_m = _obstacle_margin(zone, global_margin_m)
    if zone.zone_type == "circle":
        if zone.center is None or zone.radius_m is None:
            raise CalibrationRoomPlanError(f"circle no-go {zone.name!r} is incomplete")
        return _distance(point, zone.center) <= zone.radius_m + margin_m

    if zone.zone_type == "rect":
        if zone.rect is None:
            raise CalibrationRoomPlanError(f"rect no-go {zone.name!r} is incomplete")
        return _point_in_rect(point, _expanded_rect(zone.rect, margin_m))

    if zone.zone_type == "polygon":
        if zone.points is None or len(zone.points) < 3:
            raise CalibrationRoomPlanError(f"polygon no-go {zone.name!r} is incomplete")
        return _point_in_polygon(point, zone.points) or (
            _distance_point_to_polygon_edges(point, zone.points) <= margin_m
        )

    raise CalibrationRoomPlanError(f"unknown no-go zone type: {zone.zone_type}")


def _segment_intersects_no_go(start: Point, end: Point, zone: NoGoZone, global_margin_m: float) -> bool:
    margin_m = _obstacle_margin(zone, global_margin_m)
    if zone.zone_type == "circle":
        if zone.center is None or zone.radius_m is None:
            raise CalibrationRoomPlanError(f"circle no-go {zone.name!r} is incomplete")
        return _distance_point_to_segment(zone.center, start, end) <= zone.radius_m + margin_m

    if zone.zone_type == "rect":
        if zone.rect is None:
            raise CalibrationRoomPlanError(f"rect no-go {zone.name!r} is incomplete")
        return _segment_intersects_rect(start, end, _expanded_rect(zone.rect, margin_m))

    if zone.zone_type == "polygon":
        if zone.points is None or len(zone.points) < 3:
            raise CalibrationRoomPlanError(f"polygon no-go {zone.name!r} is incomplete")
        return _segment_intersects_polygon(start, end, zone.points, margin_m)

    raise CalibrationRoomPlanError(f"unknown no-go zone type: {zone.zone_type}")


def _no_go_clearance(point: Point, zone: NoGoZone, global_margin_m: float) -> float:
    margin_m = _obstacle_margin(zone, global_margin_m)
    if zone.zone_type == "circle":
        if zone.center is None or zone.radius_m is None:
            raise CalibrationRoomPlanError(f"circle no-go {zone.name!r} is incomplete")
        return _distance(point, zone.center) - (zone.radius_m + margin_m)

    if zone.zone_type == "rect":
        if zone.rect is None:
            raise CalibrationRoomPlanError(f"rect no-go {zone.name!r} is incomplete")
        rect = _expanded_rect(zone.rect, margin_m)
        if _point_in_rect(point, rect):
            return -min(
                point.x - rect.min_x,
                rect.max_x - point.x,
                point.y - rect.min_y,
                rect.max_y - point.y,
            )
        dx = max(rect.min_x - point.x, 0.0, point.x - rect.max_x)
        dy = max(rect.min_y - point.y, 0.0, point.y - rect.max_y)
        return math.hypot(dx, dy)

    if zone.zone_type == "polygon":
        if zone.points is None or len(zone.points) < 3:
            raise CalibrationRoomPlanError(f"polygon no-go {zone.name!r} is incomplete")
        margin_m = _obstacle_margin(zone, global_margin_m)
        edge_distance = _distance_point_to_polygon_edges(point, zone.points)
        if _point_in_polygon(point, zone.points):
            return -edge_distance
        return edge_distance - margin_m

    raise CalibrationRoomPlanError(f"unknown no-go zone type: {zone.zone_type}")


def _point_hits_any_no_go(point: Point, zones: Sequence[NoGoZone], global_margin_m: float) -> bool:
    return any(_point_inside_no_go(point, zone, global_margin_m) for zone in zones)


def _segment_hits_any_no_go(start: Point, end: Point, zones: Sequence[NoGoZone], global_margin_m: float) -> bool:
    return any(_segment_intersects_no_go(start, end, zone, global_margin_m) for zone in zones)


def _grid_values(start: float, end: float, step: float) -> List[float]:
    if start > end:
        return []
    count = int(math.floor((end - start) / step))
    values = [start + index * step for index in range(count + 1)]
    if not values or values[-1] < end - step * 0.25:
        values.append(end)
    return values


def _candidate_points(zone: Rect, grid_step_m: float, max_candidates: int) -> Iterable[Point]:
    xs = _grid_values(zone.min_x, zone.max_x, grid_step_m)
    ys = _grid_values(zone.min_y, zone.max_y, grid_step_m)
    candidate_count = len(xs) * len(ys)
    if candidate_count > max_candidates:
        raise CalibrationRoomPlanError(
            f"room grid would evaluate {candidate_count} candidates; increase --grid-step-m"
        )

    for y in ys:
        for x in xs:
            yield Point(x, y)


def _diamond_points(center: Point, half_width_m: float, half_height_m: float) -> List[Point]:
    return [
        Point(center.x - half_width_m, center.y),
        Point(center.x, center.y + half_height_m),
        Point(center.x + half_width_m, center.y),
        Point(center.x, center.y - half_height_m),
    ]


def _diamond_safe(
    center: Point,
    half_width_m: float,
    half_height_m: float,
    zone: Rect,
    no_go_zones: Sequence[NoGoZone],
    line_endpoints: Sequence[LineEndpoint],
    obstacle_margin_m: float,
) -> bool:
    points = _diamond_points(center, half_width_m, half_height_m)
    if any(not _point_in_rect(point, zone) for point in points):
        return False
    if any(_point_hits_any_no_go(point, no_go_zones, obstacle_margin_m) for point in points):
        return False

    transitions = zip(points, points[1:] + points[:1])
    for start, end in transitions:
        if _segment_hits_any_no_go(start, end, no_go_zones, obstacle_margin_m):
            return False

    for endpoint in line_endpoints:
        for point in [center] + points:
            if _segment_hits_any_no_go(endpoint.point, point, no_go_zones, obstacle_margin_m):
                return False

    return True


def _fit_probe_half_extents(
    center: Point,
    zone: Rect,
    no_go_zones: Sequence[NoGoZone],
    line_endpoints: Sequence[LineEndpoint],
    obstacle_margin_m: float,
    min_half_width_m: float,
    min_half_height_m: float,
    max_half_width_cap_m: float,
    max_half_height_cap_m: float,
) -> Tuple[float, float]:
    half_width_m = min(max_half_width_cap_m, center.x - zone.min_x, zone.max_x - center.x)
    half_height_m = min(max_half_height_cap_m, center.y - zone.min_y, zone.max_y - center.y)
    if half_width_m < min_half_width_m or half_height_m < min_half_height_m:
        raise CalibrationRoomPlanError("candidate cannot fit minimum probe envelope")

    while half_width_m >= min_half_width_m and half_height_m >= min_half_height_m:
        if _diamond_safe(
            center,
            half_width_m,
            half_height_m,
            zone,
            no_go_zones,
            line_endpoints,
            obstacle_margin_m,
        ):
            return half_width_m, half_height_m
        half_width_m *= 0.8
        half_height_m *= 0.8

    raise CalibrationRoomPlanError("candidate cannot fit a clear probe envelope")


def _candidate_score(
    point: Point,
    zone: Rect,
    no_go_zones: Sequence[NoGoZone],
    obstacle_margin_m: float,
    room_center: Point,
    half_width_m: float,
    half_height_m: float,
) -> Tuple[float, float, float]:
    boundary_clearance = min(
        point.x - zone.min_x,
        zone.max_x - point.x,
        point.y - zone.min_y,
        zone.max_y - point.y,
    )
    obstacle_clearance = min(
        [_no_go_clearance(point, zone_item, obstacle_margin_m) for zone_item in no_go_zones],
        default=boundary_clearance,
    )
    clearance = min(boundary_clearance, obstacle_clearance)
    probe_area = half_width_m * half_height_m
    center_penalty = _distance(point, room_center)
    return clearance, probe_area, -center_penalty


def _zone_polygon(zone: Rect) -> List[List[float]]:
    return [
        [_round(zone.min_x), _round(zone.min_y)],
        [_round(zone.max_x), _round(zone.min_y)],
        [_round(zone.max_x), _round(zone.max_y)],
        [_round(zone.min_x), _round(zone.max_y)],
    ]


def build_room_plan_recommended_actions(summary: dict[str, Any]) -> List[str]:
    actions: List[str] = []
    candidate_counts = summary.get("candidateCounts", {})
    reasons = candidate_counts.get("reasons", {}) if isinstance(candidate_counts, dict) else {}
    searched = int(candidate_counts.get("searched", 0) or 0) if isinstance(candidate_counts, dict) else 0
    accepted = int(candidate_counts.get("accepted", 0) or 0) if isinstance(candidate_counts, dict) else 0
    line_endpoint_count = int(summary.get("lineEndpointCount", 0) or 0)
    obstacle_count = int(summary.get("obstacleCount", 0) or 0)
    hazard_count = int(summary.get("hazardAvoidanceCount", 0) or 0)
    clearance_score = float(summary.get("clearanceScore", 0.0) or 0.0)
    selected_probe = summary.get("selectedProbe", {})
    max_half_width_m = float(selected_probe.get("maxHalfWidthM", 0.0) or 0.0) if isinstance(selected_probe, dict) else 0.0
    max_half_height_m = float(selected_probe.get("maxHalfHeightM", 0.0) or 0.0) if isinstance(selected_probe, dict) else 0.0

    if line_endpoint_count == 0:
        actions.append(
            "Add lineEndpoints or use --derive-line-endpoints-from-config so cable sweeps are checked before calibration."
        )
    if obstacle_count == 0:
        actions.append(
            "Add noGoZones for furniture, tripods, plants, monitors, and other cable catch risks before calibrating in a cluttered room."
        )
    if hazard_count > 0:
        actions.append(
            "Recent hazard positions were included as temporary no-go zones; inspect them in the SVG before calibrating again."
        )
    if int(reasons.get("cable_sweep_intersects_no_go", 0) or 0) > 0:
        actions.append(
            "Review cable sweeps in the SVG preview; move endpoints, move catch-risk objects, or choose a safer room area."
        )
    if int(reasons.get("inside_no_go", 0) or 0) > 0:
        actions.append(
            "Some candidate probe centers landed inside no-go zones; confirm object dimensions and margins are not over-broad."
        )
    if int(reasons.get("probe_envelope_too_small", 0) or 0) > 0:
        actions.append(
            "Some candidates could not fit the minimum probe envelope; reduce margins, increase usable room area, or lower minimum probe bounds."
        )
    if searched > 0 and accepted > 0 and accepted / searched < 0.05:
        actions.append(
            "Fewer than 5 percent of candidates were accepted; inspect the room preview before running calibration."
        )
    if clearance_score < 0.08:
        actions.append(
            "Selected center has low clearance; prefer manual_assisted mode and inspect the SVG before applying this plan."
        )
    if max_half_width_m < 0.1 or max_half_height_m < 0.05:
        actions.append(
            "Selected probe diamond is near minimum size; calibration confidence may be lower unless the room can be cleared."
        )
    return actions


def build_room_plan_quality(summary: dict[str, Any]) -> dict[str, Any]:
    candidate_counts = summary.get("candidateCounts", {})
    reasons = candidate_counts.get("reasons", {}) if isinstance(candidate_counts, dict) else {}
    searched = int(candidate_counts.get("searched", 0) or 0) if isinstance(candidate_counts, dict) else 0
    accepted = int(candidate_counts.get("accepted", 0) or 0) if isinstance(candidate_counts, dict) else 0
    accepted_ratio = accepted / searched if searched > 0 else 0.0
    line_endpoint_count = int(summary.get("lineEndpointCount", 0) or 0)
    obstacle_count = int(summary.get("obstacleCount", 0) or 0)
    clearance_score = float(summary.get("clearanceScore", 0.0) or 0.0)
    selected_probe = summary.get("selectedProbe", {})
    max_half_width_m = float(selected_probe.get("maxHalfWidthM", 0.0) or 0.0) if isinstance(selected_probe, dict) else 0.0
    max_half_height_m = float(selected_probe.get("maxHalfHeightM", 0.0) or 0.0) if isinstance(selected_probe, dict) else 0.0

    marginal_reasons: List[str] = []
    usable_reasons: List[str] = []
    if line_endpoint_count == 0:
        marginal_reasons.append("line endpoints missing")
    if obstacle_count == 0:
        marginal_reasons.append("no-go zones missing")
    if accepted_ratio < 0.05:
        marginal_reasons.append("accepted candidate ratio below 5 percent")
    elif accepted_ratio < 0.2:
        usable_reasons.append("accepted candidate ratio below 20 percent")
    if clearance_score < 0.08:
        marginal_reasons.append("selected center clearance below 0.08m")
    elif clearance_score < 0.15:
        usable_reasons.append("selected center clearance below 0.15m")
    if max_half_width_m < 0.1 or max_half_height_m < 0.05:
        marginal_reasons.append("selected probe diamond near minimum size")
    elif max_half_width_m < 0.15 or max_half_height_m < 0.08:
        usable_reasons.append("selected probe diamond is small")
    if int(reasons.get("cable_sweep_intersects_no_go", 0) or 0) > 0:
        usable_reasons.append("some cable-sweep candidates crossed no-go zones")

    if marginal_reasons:
        level = "marginal"
        reasons_out = marginal_reasons + usable_reasons
    elif usable_reasons:
        level = "usable"
        reasons_out = usable_reasons
    else:
        level = "strong"
        reasons_out = []

    return {
        "level": level,
        "acceptedCandidateRatio": _round(accepted_ratio),
        "reasons": reasons_out,
    }


def build_room_plan(
    room_width_m: float,
    room_depth_m: float,
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
    calibration_zone_inset_m: float = 0.15,
    no_go_zones: Optional[Sequence[NoGoZone]] = None,
    hazard_zones: Optional[Sequence[NoGoZone]] = None,
    line_endpoints: Optional[Sequence[LineEndpoint]] = None,
    obstacle_margin_m: float = 0.12,
    hazard_avoid_radius_m: float = 0.2,
    grid_step_m: float = 0.05,
    mode: str = "manual_assisted",
    min_probe_half_width_m: float = 0.05,
    min_probe_half_height_m: float = 0.04,
    max_probe_half_width_cap_m: float = 0.35,
    max_probe_half_height_cap_m: float = 0.25,
    validation_distance_m: float = 0.025,
    validation_speed_mps: float = 0.015,
    manual_assist_timeout_s: float = 60.0,
    allow_degraded_reference: bool = False,
    skip_safe_motion_validation: bool = False,
    max_candidates: int = 250000,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Build a calibrationSafety block and a planning summary."""

    room_width_m = _positive(float(room_width_m), "room width")
    room_depth_m = _positive(float(room_depth_m), "room depth")
    calibration_zone_inset_m = _non_negative(float(calibration_zone_inset_m), "calibration zone inset")
    obstacle_margin_m = _non_negative(float(obstacle_margin_m), "obstacle margin")
    hazard_avoid_radius_m = _non_negative(float(hazard_avoid_radius_m), "hazard avoid radius")
    grid_step_m = _positive(float(grid_step_m), "grid step")
    min_probe_half_width_m = _positive(float(min_probe_half_width_m), "minimum probe half width")
    min_probe_half_height_m = _positive(float(min_probe_half_height_m), "minimum probe half height")
    max_probe_half_width_cap_m = _positive(float(max_probe_half_width_cap_m), "maximum probe half width cap")
    max_probe_half_height_cap_m = _positive(float(max_probe_half_height_cap_m), "maximum probe half height cap")
    validation_distance_m = _positive(float(validation_distance_m), "validation distance")
    validation_speed_mps = _positive(float(validation_speed_mps), "validation speed")
    manual_assist_timeout_s = _positive(float(manual_assist_timeout_s), "manual assist timeout")

    if mode not in MODE_CHOICES:
        raise CalibrationRoomPlanError(f"mode must be one of: {', '.join(MODE_CHOICES)}")
    if calibration_zone_inset_m * 2.0 >= room_width_m or calibration_zone_inset_m * 2.0 >= room_depth_m:
        raise CalibrationRoomPlanError("calibration zone inset leaves no usable room area")
    if min_probe_half_width_m > max_probe_half_width_cap_m:
        raise CalibrationRoomPlanError("minimum probe half width exceeds maximum cap")
    if min_probe_half_height_m > max_probe_half_height_cap_m:
        raise CalibrationRoomPlanError("minimum probe half height exceeds maximum cap")

    hazard_zone_list = list(hazard_zones or [])
    no_go_zone_list = list(no_go_zones or []) + hazard_zone_list
    line_endpoint_list = list(line_endpoints or [])
    zone = Rect(
        origin_x_m + calibration_zone_inset_m,
        origin_y_m + calibration_zone_inset_m,
        origin_x_m + room_width_m - calibration_zone_inset_m,
        origin_y_m + room_depth_m - calibration_zone_inset_m,
    )
    room_center = Point(origin_x_m + room_width_m * 0.5, origin_y_m + room_depth_m * 0.5)

    rejections = {
        "inside_no_go": 0,
        "cable_sweep_intersects_no_go": 0,
        "probe_envelope_too_small": 0,
    }
    searched = 0
    accepted = 0
    selected_point: Optional[Point] = None
    selected_half_width_m = 0.0
    selected_half_height_m = 0.0
    selected_score: Optional[Tuple[float, float, float]] = None

    for candidate in _candidate_points(zone, grid_step_m, max_candidates):
        searched += 1
        if _point_hits_any_no_go(candidate, no_go_zone_list, obstacle_margin_m):
            rejections["inside_no_go"] += 1
            continue

        if any(
            _segment_hits_any_no_go(endpoint.point, candidate, no_go_zone_list, obstacle_margin_m)
            for endpoint in line_endpoint_list
        ):
            rejections["cable_sweep_intersects_no_go"] += 1
            continue

        try:
            half_width_m, half_height_m = _fit_probe_half_extents(
                candidate,
                zone,
                no_go_zone_list,
                line_endpoint_list,
                obstacle_margin_m,
                min_probe_half_width_m,
                min_probe_half_height_m,
                max_probe_half_width_cap_m,
                max_probe_half_height_cap_m,
            )
        except CalibrationRoomPlanError:
            rejections["probe_envelope_too_small"] += 1
            continue

        accepted += 1
        score = _candidate_score(
            candidate,
            zone,
            no_go_zone_list,
            obstacle_margin_m,
            room_center,
            half_width_m,
            half_height_m,
        )
        if selected_score is None or score > selected_score:
            selected_point = candidate
            selected_half_width_m = half_width_m
            selected_half_height_m = half_height_m
            selected_score = score

    if selected_point is None or selected_score is None:
        raise CalibrationRoomPlanError(
            "no safe calibration probe center found; reduce no-go margins, move catch-risk objects, "
            "or use manual_assisted mode with a different safe room area"
        )

    safety = {
        "mode": mode,
        "safeProbeCenter": [_round(selected_point.x), _round(selected_point.y)],
        "calibrationZone": _zone_polygon(zone),
        "noGoZones": [zone_item.to_config() for zone_item in no_go_zone_list],
        "lineEndpoints": [endpoint.to_config() for endpoint in line_endpoint_list],
        "obstacleMarginM": _round(obstacle_margin_m),
        "hazardAvoidRadiusM": _round(hazard_avoid_radius_m),
        "minProbeHalfWidthM": _round(min_probe_half_width_m),
        "minProbeHalfHeightM": _round(min_probe_half_height_m),
        "maxProbeHalfWidthM": _round(selected_half_width_m),
        "maxProbeHalfHeightM": _round(selected_half_height_m),
        "validationDistanceM": _round(validation_distance_m),
        "validationSpeedMps": _round(validation_speed_mps),
        "manualAssistTimeoutS": _round(manual_assist_timeout_s),
        "allowDegradedReference": bool(allow_degraded_reference),
        "skipSafeMotionValidation": bool(skip_safe_motion_validation),
    }
    summary = {
        "room": {
            "origin": {"x": _round(origin_x_m), "y": _round(origin_y_m)},
            "widthM": _round(room_width_m),
            "depthM": _round(room_depth_m),
            "calibrationZoneInsetM": _round(calibration_zone_inset_m),
        },
        "selectedCenter": {"x": _round(selected_point.x), "y": _round(selected_point.y)},
        "selectedProbe": {
            "maxHalfWidthM": safety["maxProbeHalfWidthM"],
            "maxHalfHeightM": safety["maxProbeHalfHeightM"],
        },
        "candidateCounts": {
            "searched": searched,
            "accepted": accepted,
            "rejected": searched - accepted,
            "reasons": rejections,
        },
        "obstacleCount": len(no_go_zone_list),
        "hazardAvoidanceCount": len(hazard_zone_list),
        "lineEndpointCount": len(line_endpoint_list),
        "clearanceScore": _round(selected_score[0]),
        "probeAreaScore": _round(selected_score[1]),
    }
    summary["recommendedActions"] = build_room_plan_recommended_actions(summary)
    summary["planQuality"] = build_room_plan_quality(summary)
    return safety, summary


def _svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_point(point: Sequence[float], min_x: float, max_y: float, scale: float, pad: float) -> Tuple[float, float]:
    return (
        pad + (float(point[0]) - min_x) * scale,
        pad + (max_y - float(point[1])) * scale,
    )


def _svg_points(points: Sequence[Sequence[float]], min_x: float, max_y: float, scale: float, pad: float) -> str:
    return " ".join(
        f"{_round(x)},{_round(y)}"
        for x, y in (_svg_point(point, min_x, max_y, scale, pad) for point in points)
    )


def build_room_plan_svg(safety: dict[str, Any], summary: dict[str, Any], width_px: int = 960) -> str:
    """Render a lightweight operator preview for the generated calibration room plan."""

    room = summary.get("room", {})
    origin = room.get("origin", {})
    min_x = float(origin.get("x", 0.0))
    min_y = float(origin.get("y", 0.0))
    room_width_m = float(room.get("widthM", 1.0))
    room_depth_m = float(room.get("depthM", 1.0))
    max_y = min_y + room_depth_m
    pad = 48.0
    drawing_width = max(320.0, float(width_px) - pad * 2.0)
    scale = drawing_width / max(room_width_m, 1e-9)
    height_px = int(max(320.0, room_depth_m * scale + pad * 2.0))
    room_height_px = room_depth_m * scale
    selected = safety.get("safeProbeCenter", [min_x + room_width_m * 0.5, min_y + room_depth_m * 0.5])
    center_x, center_y = _svg_point(selected, min_x, max_y, scale, pad)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width_px)}" height="{height_px}" viewBox="0 0 {int(width_px)} {height_px}">',
        "<style>",
        "text{font-family:monospace;font-size:13px;fill:#1f2933}",
        ".room{fill:#f8fafc;stroke:#243b53;stroke-width:2}",
        ".zone{fill:#dbeafe;stroke:#2563eb;stroke-width:2;fill-opacity:.28}",
        ".nogomargin{fill:#fee2e2;stroke:#f87171;stroke-width:1;stroke-dasharray:5 4;fill-opacity:.28}",
        ".nogo{fill:#fecaca;stroke:#b91c1c;stroke-width:2;fill-opacity:.72}",
        ".sweep{stroke:#64748b;stroke-width:1.4;stroke-dasharray:4 5}",
        ".diamond{fill:#bfdbfe;stroke:#1d4ed8;stroke-width:2;fill-opacity:.38}",
        ".center{fill:#1d4ed8;stroke:#eff6ff;stroke-width:3}",
        ".endpoint{fill:#f59e0b;stroke:#7c2d12;stroke-width:1.5}",
        "</style>",
        f'<rect class="room" x="{_round(pad)}" y="{_round(pad)}" width="{_round(room_width_m * scale)}" height="{_round(room_height_px)}"/>',
        f'<text x="{_round(pad)}" y="24">calibration room plan: {room_width_m}m x {room_depth_m}m</text>',
    ]

    calibration_zone = safety.get("calibrationZone")
    if isinstance(calibration_zone, list) and calibration_zone:
        lines.append(
            f'<polygon class="zone" points="{_svg_points(calibration_zone, min_x, max_y, scale, pad)}">'
            '<title>calibration zone</title></polygon>'
        )

    for endpoint_index, endpoint in enumerate(safety.get("lineEndpoints") or []):
        if not isinstance(endpoint, list) or len(endpoint) < 2:
            continue
        x, y = _svg_point(endpoint, min_x, max_y, scale, pad)
        lines.append(
            f'<line class="sweep" x1="{_round(x)}" y1="{_round(y)}" x2="{_round(center_x)}" y2="{_round(center_y)}">'
            f'<title>cable sweep endpoint {endpoint_index + 1} to selected probe center</title></line>'
        )
        lines.append(
            f'<circle class="endpoint" cx="{_round(x)}" cy="{_round(y)}" r="5">'
            f'<title>line endpoint {endpoint_index + 1}: {endpoint}</title></circle>'
        )

    for zone in safety.get("noGoZones") or []:
        if not isinstance(zone, dict):
            continue
        name = _svg_escape(zone.get("name", "no-go"))
        margin = float(zone.get("marginM", 0.0) or 0.0)
        if "center" in zone and "radiusM" in zone:
            center = zone["center"]
            if isinstance(center, list) and len(center) >= 2:
                x, y = _svg_point(center, min_x, max_y, scale, pad)
                radius = float(zone.get("radiusM", 0.0) or 0.0) * scale
                if margin > 0.0:
                    lines.append(
                        f'<circle class="nogomargin" cx="{_round(x)}" cy="{_round(y)}" r="{_round(radius + margin * scale)}">'
                        f'<title>no-go margin {name}</title></circle>'
                    )
                lines.append(
                    f'<circle class="nogo" cx="{_round(x)}" cy="{_round(y)}" r="{_round(radius)}">'
                    f'<title>no-go {name}</title></circle>'
                )
        elif "rect" in zone and isinstance(zone["rect"], list) and len(zone["rect"]) == 2:
            p1, p2 = zone["rect"]
            if isinstance(p1, list) and isinstance(p2, list) and len(p1) >= 2 and len(p2) >= 2:
                min_rect_x = min(float(p1[0]), float(p2[0]))
                max_rect_x = max(float(p1[0]), float(p2[0]))
                min_rect_y = min(float(p1[1]), float(p2[1]))
                max_rect_y = max(float(p1[1]), float(p2[1]))
                x, y = _svg_point([min_rect_x, max_rect_y], min_x, max_y, scale, pad)
                rect_w = (max_rect_x - min_rect_x) * scale
                rect_h = (max_rect_y - min_rect_y) * scale
                if margin > 0.0:
                    mx, my = _svg_point([min_rect_x - margin, max_rect_y + margin], min_x, max_y, scale, pad)
                    lines.append(
                        f'<rect class="nogomargin" x="{_round(mx)}" y="{_round(my)}" width="{_round(rect_w + margin * 2.0 * scale)}" height="{_round(rect_h + margin * 2.0 * scale)}">'
                        f'<title>no-go margin {name}</title></rect>'
                    )
                lines.append(
                    f'<rect class="nogo" x="{_round(x)}" y="{_round(y)}" width="{_round(rect_w)}" height="{_round(rect_h)}">'
                    f'<title>no-go {name}</title></rect>'
                )
        elif "polygon" in zone and isinstance(zone["polygon"], list):
            lines.append(
                f'<polygon class="nogo" points="{_svg_points(zone["polygon"], min_x, max_y, scale, pad)}">'
                f'<title>no-go {name}; margin {margin}m</title></polygon>'
            )

    half_w = float(safety.get("maxProbeHalfWidthM", 0.0) or 0.0)
    half_h = float(safety.get("maxProbeHalfHeightM", 0.0) or 0.0)
    probe_points = [
        [float(selected[0]) - half_w, float(selected[1])],
        [float(selected[0]), float(selected[1]) + half_h],
        [float(selected[0]) + half_w, float(selected[1])],
        [float(selected[0]), float(selected[1]) - half_h],
    ]
    lines.append(
        f'<polygon class="diamond" points="{_svg_points(probe_points, min_x, max_y, scale, pad)}">'
        '<title>adaptive probe diamond</title></polygon>'
    )
    lines.append(
        f'<circle class="center" cx="{_round(center_x)}" cy="{_round(center_y)}" r="7">'
        f'<title>selected probe center {selected}</title></circle>'
    )
    lines.append(
        f'<text x="{_round(center_x + 10)}" y="{_round(center_y - 10)}">selected probe center</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def format_room_plan_summary(summary: dict[str, Any]) -> str:
    selected = summary.get("selectedCenter", {})
    selected_x = selected.get("x", "?") if isinstance(selected, dict) else "?"
    selected_y = selected.get("y", "?") if isinstance(selected, dict) else "?"
    counts = summary.get("candidateCounts", {})
    searched = counts.get("searched", 0) if isinstance(counts, dict) else 0
    accepted = counts.get("accepted", 0) if isinstance(counts, dict) else 0
    rejected = counts.get("rejected", 0) if isinstance(counts, dict) else 0
    quality = summary.get("planQuality", {})
    quality_level = quality.get("level", "unknown") if isinstance(quality, dict) else "unknown"
    accepted_ratio = quality.get("acceptedCandidateRatio", 0.0) if isinstance(quality, dict) else 0.0
    selected_probe = summary.get("selectedProbe", {})
    half_width_m = selected_probe.get("maxHalfWidthM", "?") if isinstance(selected_probe, dict) else "?"
    half_height_m = selected_probe.get("maxHalfHeightM", "?") if isinstance(selected_probe, dict) else "?"
    actions = summary.get("recommendedActions") or []

    lines = [
        f"selected safeProbeCenter=({selected_x}, {selected_y})",
        f"planQuality={quality_level} acceptedCandidateRatio={accepted_ratio}",
        f"candidates searched={searched} accepted={accepted} rejected={rejected}",
        f"selectedProbe maxHalfWidthM={half_width_m} maxHalfHeightM={half_height_m}",
        f"clearanceScore={summary.get('clearanceScore', '?')} probeAreaScore={summary.get('probeAreaScore', '?')}",
    ]
    if actions:
        lines.append("recommendedActions:")
        lines.extend(f"  - {action}" for action in actions)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a calibrationSafety block from room dimensions and catch-risk objects.",
    )
    parser.add_argument("--room-file", type=Path, help="JSON room description with dimensions, endpoints, and no-go zones")
    parser.add_argument(
        "--derive-line-endpoints-from-config",
        type=Path,
        metavar="CONFIG",
        help="Derive lineEndpoints from anchor and eyelet positions in an existing robot config",
    )
    parser.add_argument(
        "--overwrite-line-endpoints",
        action="store_true",
        help="Use only config-derived and CLI endpoints when endpoint derivation is enabled",
    )
    parser.add_argument(
        "--allow-empty-derived-line-endpoints",
        action="store_true",
        help="Do not fail if endpoint derivation finds no anchors or eyelets",
    )
    parser.add_argument("--room-width-m", type=float)
    parser.add_argument("--room-depth-m", type=float)
    parser.add_argument("--origin-x-m", type=float)
    parser.add_argument("--origin-y-m", type=float)
    parser.add_argument(
        "--derive-room-from-config",
        action="store_true",
        help="Infer room bounds from anchor and eyelet coordinates in --derive-line-endpoints-from-config",
    )
    parser.add_argument(
        "--room-derive-padding-m",
        type=float,
        default=0.6,
        help="Padding around config-derived anchor/eyelet bounds when deriving room size",
    )
    parser.add_argument("--calibration-zone-inset-m", type=float)
    parser.add_argument("--grid-step-m", type=float)
    parser.add_argument("--max-candidates", type=int, default=250000)
    parser.add_argument("--mode", choices=MODE_CHOICES)
    parser.add_argument("--line-endpoint", action="append", default=[], metavar="NAME,X,Y")
    parser.add_argument("--add-circle-no-go", action="append", default=[], metavar="NAME,X,Y,RADIUS[,MARGIN]")
    parser.add_argument("--add-rect-no-go", action="append", default=[], metavar="NAME,X1,Y1,X2,Y2[,MARGIN]")
    parser.add_argument("--add-polygon-no-go", action="append", default=[], metavar="NAME,X1,Y1,X2,Y2,X3,Y3...[,MARGIN]")
    parser.add_argument("--add-hazard", action="append", default=[], metavar="NAME,X,Y[,RADIUS[,MARGIN]]")
    parser.add_argument(
        "--hazards-from-artifact",
        action="append",
        type=Path,
        default=[],
        metavar="ARTIFACT",
        help="Extract recent hazard positions from a prior calibration artifact JSON",
    )
    parser.add_argument(
        "--hazards-from-artifact-dir",
        type=Path,
        metavar="DIR",
        help="Extract hazard positions from the newest JSON artifacts in a calibration artifact directory",
    )
    parser.add_argument(
        "--hazard-artifact-limit",
        type=int,
        default=1,
        help="Number of newest artifacts to read with --hazards-from-artifact-dir",
    )
    parser.add_argument("--obstacle-margin-m", type=float)
    parser.add_argument("--hazard-avoid-radius-m", type=float)
    parser.add_argument("--min-probe-half-width-m", type=float)
    parser.add_argument("--min-probe-half-height-m", type=float)
    parser.add_argument("--max-probe-half-width-cap-m", type=float)
    parser.add_argument("--max-probe-half-height-cap-m", type=float)
    parser.add_argument("--validation-distance-m", type=float)
    parser.add_argument("--validation-speed-mps", type=float)
    parser.add_argument("--manual-assist-timeout-s", type=float)
    parser.add_argument(
        "--from-camera-streams",
        action="store_true",
        help="Use live anchor camera MJPEG streams to add conservative camera-derived no-go zones",
    )
    parser.add_argument(
        "--camera-stream-url",
        action="append",
        default=[],
        metavar="[ANCHOR,]URL",
        help="Camera stream URL. Defaults to local anchor streams from preferredCameras.",
    )
    parser.add_argument("--camera-map-size-px", type=int, default=1000)
    parser.add_argument("--camera-map-extent-m", type=float)
    parser.add_argument("--camera-min-component-area-m2", type=float, default=0.04)
    parser.add_argument("--camera-max-component-area-ratio", type=float, default=0.18)
    parser.add_argument("--camera-clutter-dilate-m", type=float, default=0.08)
    parser.add_argument("--camera-max-no-go-zones", type=int, default=24)
    parser.add_argument(
        "--camera-ignore-unknown-space",
        action="store_true",
        help="Do not mark camera-unobserved room areas as no-go zones",
    )
    degraded_group = parser.add_mutually_exclusive_group()
    degraded_group.add_argument("--allow-degraded-reference", action="store_true")
    degraded_group.add_argument("--no-allow-degraded-reference", action="store_true")
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument("--skip-safe-motion-validation", action="store_true")
    validation_group.add_argument("--no-skip-safe-motion-validation", action="store_true")
    parser.add_argument("--include-plan-summary", action="store_true")
    parser.add_argument(
        "--require-plan-quality",
        choices=PLAN_QUALITY_LEVELS,
        help="Fail before writing output unless generated plan quality meets this threshold",
    )
    parser.add_argument("--summary", action="store_true", help="print a concise human summary to stderr")
    parser.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")
    parser.add_argument("--svg-output", type=Path, help="write an SVG preview of the generated room plan")
    return parser


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        room_spec = _load_room_file(args.room_file) if args.room_file is not None else {}
        raw_endpoint_config = None
        if args.derive_line_endpoints_from_config is not None:
            raw_endpoint_config = _load_json(args.derive_line_endpoints_from_config)
            if not isinstance(raw_endpoint_config, dict):
                raise CalibrationRoomPlanError("derived endpoint config must be a JSON object")
        if args.from_camera_streams and raw_endpoint_config is None:
            raise CalibrationRoomPlanError("--from-camera-streams requires --derive-line-endpoints-from-config")

        room_width_m = args.room_width_m
        if room_width_m is None:
            room_width_m = _room_float(room_spec, ("roomWidthM", "room_width_m", "widthM", "width_m", "width"))
        room_depth_m = args.room_depth_m
        if room_depth_m is None:
            room_depth_m = _room_float(room_spec, ("roomDepthM", "room_depth_m", "depthM", "depth_m", "depth"))
        derived_room_bounds = None
        if (
            (args.derive_room_from_config or args.from_camera_streams)
            and raw_endpoint_config is not None
            and (room_width_m is None or room_depth_m is None or args.origin_x_m is None or args.origin_y_m is None)
        ):
            derived_room_bounds = derive_room_bounds_from_config(
                raw_endpoint_config,
                padding_m=args.room_derive_padding_m,
            )
            if room_width_m is None:
                room_width_m = derived_room_bounds[2]
            if room_depth_m is None:
                room_depth_m = derived_room_bounds[3]
        if room_width_m is None:
            raise CalibrationRoomPlanError("--room-width-m is required unless --room-file provides roomWidthM or --derive-room-from-config is used")
        if room_depth_m is None:
            raise CalibrationRoomPlanError("--room-depth-m is required unless --room-file provides roomDepthM or --derive-room-from-config is used")
        origin_x_m = args.origin_x_m
        if origin_x_m is None and derived_room_bounds is not None:
            origin_x_m = derived_room_bounds[0]
        if origin_x_m is None:
            origin_x_m = _room_origin_axis(
                room_spec,
                "x",
                ("originXM", "origin_x_m", "originX", "origin_x"),
                0.0,
            )
        origin_y_m = args.origin_y_m
        if origin_y_m is None and derived_room_bounds is not None:
            origin_y_m = derived_room_bounds[1]
        if origin_y_m is None:
            origin_y_m = _room_origin_axis(
                room_spec,
                "y",
                ("originYM", "origin_y_m", "originY", "origin_y"),
                0.0,
            )
        hazard_avoid_radius_m = args.hazard_avoid_radius_m if args.hazard_avoid_radius_m is not None else _room_float(
            room_spec,
            ("hazardAvoidRadiusM", "hazard_avoid_radius_m"),
            0.2,
        )

        file_no_go_zones = [
            parse_no_go_zone_json(zone, index=index)
            for index, zone in enumerate(_room_list(
                room_spec,
                "noGoZones",
                "no_go_zones",
                "catchRiskObjects",
                "catch_risk_objects",
                "obstacles",
            ))
        ]
        no_go_zones = file_no_go_zones + [parse_circle_no_go(spec) for spec in args.add_circle_no_go]
        no_go_zones.extend(parse_rect_no_go(spec) for spec in args.add_rect_no_go)
        no_go_zones.extend(parse_polygon_no_go(spec) for spec in args.add_polygon_no_go)
        hazard_zones = [
            parse_hazard_json(hazard, hazard_avoid_radius_m, index=index)
            for index, hazard in enumerate(_room_list(
                room_spec,
                "recentHazards",
                "recent_hazards",
                "hazardPositions",
                "hazard_positions",
                "calibrationHazards",
                "calibration_hazards",
            ))
        ]
        hazard_zones.extend(parse_hazard_no_go(spec, hazard_avoid_radius_m) for spec in args.add_hazard)
        hazard_artifact_sources = list(args.hazards_from_artifact)
        if args.hazards_from_artifact_dir is not None:
            hazard_artifact_sources.extend(
                latest_artifact_paths_from_dir(args.hazards_from_artifact_dir, args.hazard_artifact_limit)
            )
        for artifact_path in hazard_artifact_sources:
            hazard_zones.extend(hazard_zones_from_artifact(_load_json(artifact_path), hazard_avoid_radius_m))
        file_line_endpoints = [
            parse_line_endpoint_json(endpoint, index=index)
            for index, endpoint in enumerate(_room_list(
                room_spec,
                "lineEndpoints",
                "line_endpoints",
                "cableEndpoints",
                "cable_endpoints",
            ))
        ]
        derived_line_endpoints: List[LineEndpoint] = []
        if raw_endpoint_config is not None:
            derived_line_endpoints = line_endpoints_from_config(raw_endpoint_config)
            if not derived_line_endpoints and not args.allow_empty_derived_line_endpoints:
                raise CalibrationRoomPlanError(
                    "--derive-line-endpoints-from-config found no anchor or eyelet endpoints"
                )

        cli_line_endpoints = [
            parse_line_endpoint(spec, index=index) for index, spec in enumerate(args.line_endpoint)
        ]
        if args.overwrite_line_endpoints and args.derive_line_endpoints_from_config is not None:
            line_endpoints = _dedupe_line_endpoints(derived_line_endpoints + cli_line_endpoints)
        else:
            line_endpoints = _dedupe_line_endpoints(
                file_line_endpoints + derived_line_endpoints + cli_line_endpoints
            )
        camera_scan_summary = None
        if args.from_camera_streams:
            camera_no_go_zones, camera_scan_summary = camera_no_go_zones_from_streams(
                raw_endpoint_config,
                origin_x_m=origin_x_m,
                origin_y_m=origin_y_m,
                room_width_m=room_width_m,
                room_depth_m=room_depth_m,
                stream_urls=args.camera_stream_url,
                map_size_px=args.camera_map_size_px,
                map_extent_m=args.camera_map_extent_m,
                min_component_area_m2=args.camera_min_component_area_m2,
                max_component_area_ratio=args.camera_max_component_area_ratio,
                dilate_m=args.camera_clutter_dilate_m,
                max_zones=args.camera_max_no_go_zones,
                include_unknown=not args.camera_ignore_unknown_space,
            )
            no_go_zones.extend(camera_no_go_zones)
        allow_degraded_reference = _room_bool(
            room_spec,
            ("allowDegradedReference", "allow_degraded_reference"),
            False,
        )
        if args.allow_degraded_reference:
            allow_degraded_reference = True
        if args.no_allow_degraded_reference:
            allow_degraded_reference = False

        skip_safe_motion_validation = _room_bool(
            room_spec,
            ("skipSafeMotionValidation", "skip_safe_motion_validation"),
            False,
        )
        if args.skip_safe_motion_validation:
            skip_safe_motion_validation = True
        if args.no_skip_safe_motion_validation:
            skip_safe_motion_validation = False

        safety, summary = build_room_plan(
            room_width_m=room_width_m,
            room_depth_m=room_depth_m,
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
            calibration_zone_inset_m=args.calibration_zone_inset_m if args.calibration_zone_inset_m is not None else _room_float(
                room_spec,
                ("calibrationZoneInsetM", "calibration_zone_inset_m", "zoneInsetM", "zone_inset_m"),
                0.15,
            ),
            no_go_zones=no_go_zones,
            hazard_zones=hazard_zones,
            line_endpoints=line_endpoints,
            obstacle_margin_m=args.obstacle_margin_m if args.obstacle_margin_m is not None else _room_float(
                room_spec,
                ("obstacleMarginM", "obstacle_margin_m"),
                0.12,
            ),
            hazard_avoid_radius_m=hazard_avoid_radius_m,
            grid_step_m=args.grid_step_m if args.grid_step_m is not None else _room_float(
                room_spec,
                ("gridStepM", "grid_step_m"),
                0.05,
            ),
            mode=args.mode or _room_string(room_spec, ("mode", "calibrationMode", "calibration_mode"), "manual_assisted"),
            min_probe_half_width_m=args.min_probe_half_width_m if args.min_probe_half_width_m is not None else _room_float(
                room_spec,
                ("minProbeHalfWidthM", "min_probe_half_width_m"),
                0.05,
            ),
            min_probe_half_height_m=args.min_probe_half_height_m if args.min_probe_half_height_m is not None else _room_float(
                room_spec,
                ("minProbeHalfHeightM", "min_probe_half_height_m"),
                0.04,
            ),
            max_probe_half_width_cap_m=args.max_probe_half_width_cap_m if args.max_probe_half_width_cap_m is not None else _room_float(
                room_spec,
                ("maxProbeHalfWidthCapM", "max_probe_half_width_cap_m", "maxProbeHalfWidthM", "max_probe_half_width_m"),
                0.35,
            ),
            max_probe_half_height_cap_m=args.max_probe_half_height_cap_m if args.max_probe_half_height_cap_m is not None else _room_float(
                room_spec,
                ("maxProbeHalfHeightCapM", "max_probe_half_height_cap_m", "maxProbeHalfHeightM", "max_probe_half_height_m"),
                0.25,
            ),
            validation_distance_m=args.validation_distance_m if args.validation_distance_m is not None else _room_float(
                room_spec,
                ("validationDistanceM", "validation_distance_m"),
                0.025,
            ),
            validation_speed_mps=args.validation_speed_mps if args.validation_speed_mps is not None else _room_float(
                room_spec,
                ("validationSpeedMps", "validation_speed_mps"),
                0.015,
            ),
            manual_assist_timeout_s=args.manual_assist_timeout_s if args.manual_assist_timeout_s is not None else _room_float(
                room_spec,
                ("manualAssistTimeoutS", "manual_assist_timeout_s"),
                60.0,
            ),
            allow_degraded_reference=allow_degraded_reference,
            skip_safe_motion_validation=skip_safe_motion_validation,
            max_candidates=args.max_candidates,
        )
        if hazard_artifact_sources:
            summary["hazardArtifactSources"] = [str(path) for path in hazard_artifact_sources]
        if derived_room_bounds is not None:
            summary["room"]["derivedFromConfig"] = True
            summary["room"]["derivePaddingM"] = _round(args.room_derive_padding_m)
        if camera_scan_summary is not None:
            summary["cameraSafetyScan"] = camera_scan_summary
    except CalibrationRoomPlanError as exc:
        print(f"calibration room plan failed: {exc}", file=sys.stderr)
        return 2

    payload: Dict[str, object]
    if args.include_plan_summary:
        payload = {"calibrationSafety": safety, "roomPlan": summary}
    else:
        payload = safety

    if args.require_plan_quality:
        quality = summary.get("planQuality", {})
        level = str(quality.get("level", "marginal"))
        if PLAN_QUALITY_RANK.get(level, -1) < PLAN_QUALITY_RANK[args.require_plan_quality]:
            print(
                "calibration room plan quality gate failed: "
                f"{level} < {args.require_plan_quality}",
                file=sys.stderr,
            )
            for reason in quality.get("reasons", []):
                print(f"  - {reason}", file=sys.stderr)
            return 3

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    if args.svg_output:
        args.svg_output.write_text(build_room_plan_svg(safety, summary), encoding="utf-8")

    if args.summary:
        print(format_room_plan_summary(summary), file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
