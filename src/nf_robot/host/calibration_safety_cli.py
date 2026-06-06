from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


VALID_MODES = {"full", "constrained", "manual_assisted"}
SAFETY_KEYS = ("calibrationSafety", "calibration_safety", "roomSafety", "room_safety")
ZONE_LIST_KEYS = ("noGoZones", "no_go_zones", "catchRiskObjects", "catch_risk_objects", "obstacles")
ZONE_KEYS = ("calibrationZone", "calibration_zone", "safeProbeZone", "safe_probe_zone", "safeZone", "safe_zone")
POINT_KEYS = ("safeProbeCenter", "safe_probe_center", "calibrationCenter", "calibration_center")
LINE_ENDPOINT_KEYS = ("lineEndpoints", "line_endpoints", "cableEndpoints", "cable_endpoints")
NONNEGATIVE_KEYS = (
    "maxProbeHalfWidthM", "max_probe_half_width_m",
    "maxProbeHalfHeightM", "max_probe_half_height_m",
    "minProbeHalfWidthM", "min_probe_half_width_m",
    "minProbeHalfHeightM", "min_probe_half_height_m",
    "obstacleMarginM", "obstacle_margin_m",
    "hazardAvoidRadiusM", "hazard_avoid_radius_m",
    "validationDistanceM", "validation_distance_m",
    "validationSpeedMps", "validation_speed_mps",
    "validationSettleS", "validation_settle_s",
    "manualAssistTimeoutS", "manual_assist_timeout_s",
)
BOOLEAN_KEYS = (
    "allowDegradedReference", "allow_degraded_reference",
    "skipSafeMotionValidation", "skip_safe_motion_validation",
)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _point2(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_finite_number(item) for item in value)


def _polygon(value: Any, min_points: int = 3) -> bool:
    return isinstance(value, list) and len(value) >= min_points and all(_point2(item) for item in value)


def _rect(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_point2(item) for item in value)


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        xi, yi = current
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _point_in_zone(point: list[float], zone: dict[str, Any]) -> bool:
    kind = zone.get("kind")
    margin = zone.get("margin", 0.0) or 0.0
    if kind == "polygon":
        if _point_in_polygon(point, zone["points"]):
            return True
        # Approximate polygon margin by checking distance to vertices/edges.
        points = zone["points"]
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            vx = end[0] - start[0]
            vy = end[1] - start[1]
            wx = point[0] - start[0]
            wy = point[1] - start[1]
            denom = vx * vx + vy * vy
            t = 0.0 if denom == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
            closest = [start[0] + t * vx, start[1] + t * vy]
            if (point[0] - closest[0]) ** 2 + (point[1] - closest[1]) ** 2 <= margin ** 2:
                return True
        return False
    if kind == "rect":
        xs = [item[0] for item in zone["points"]]
        ys = [item[1] for item in zone["points"]]
        return min(xs) - margin <= point[0] <= max(xs) + margin and min(ys) - margin <= point[1] <= max(ys) + margin
    if kind == "circle":
        center = zone["center"]
        radius = zone["radius"]
        return (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2 <= (radius + margin) ** 2
    return False


def _segment_hits_zone(start: list[float], end: list[float], zone: dict[str, Any]) -> bool:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.sqrt(dx * dx + dy * dy)
    samples = max(8, min(200, int(distance / 0.05) + 2))
    for index in range(samples):
        t = 0.0 if samples == 1 else index / (samples - 1)
        point = [start[0] + dx * t, start[1] + dy * t]
        if _point_in_zone(point, zone):
            return True
    return False


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, Any]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None, None


def _first_number(mapping: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, float | None]:
    key, value = _first_present(mapping, keys)
    if key is None:
        return None, None
    if _finite_number(value):
        return key, float(value)
    return key, None


def _extract_safety(config: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    key, value = _first_present(config, SAFETY_KEYS)
    if value is None:
        return None, {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return key, value


def _validate_zone(zone: Any, index: int, errors: list[str]) -> dict[str, Any] | None:
    if _polygon(zone):
        return {"name": f"zone_{index}", "kind": "polygon", "points": zone, "margin": 0.0}
    if not isinstance(zone, dict):
        errors.append(f"no-go zone {index} must be a polygon array or object")
        return None

    name = str(zone.get("name") or f"zone_{index}")
    margin = 0.0
    for key in ("marginM", "margin_m", "margin"):
        if key in zone and (not _finite_number(zone[key]) or zone[key] < 0):
            errors.append(f"zone {name} {key} must be a non-negative finite number")
        elif key in zone:
            margin = zone[key]

    for key in ("polygon", "points"):
        if key in zone:
            if not _polygon(zone[key]):
                errors.append(f"zone {name} {key} must be a polygon with at least 3 [x,y] points")
                return None
            return {"name": name, "kind": "polygon", "points": zone[key], "margin": margin}

    for key in ("rect", "rectangle", "bounds"):
        if key in zone:
            if not _rect(zone[key]):
                errors.append(f"zone {name} {key} must be two [x,y] points")
                return None
            return {"name": name, "kind": "rect", "points": zone[key], "margin": margin}

    radius_key = next((key for key in ("radiusM", "radius_m", "radius") if key in zone), None)
    if "center" in zone or radius_key is not None:
        if not _point2(zone.get("center")):
            errors.append(f"zone {name} center must be [x,y]")
            return None
        if radius_key is None or not _finite_number(zone[radius_key]) or zone[radius_key] < 0:
            errors.append(f"zone {name} radius must be a non-negative finite number")
            return None
        return {"name": name, "kind": "circle", "center": zone["center"], "radius": zone[radius_key], "margin": margin}

    errors.append(f"zone {name} must define polygon, rect, bounds, or center plus radius")
    return None


def validate_safety(safety: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    mode = safety.get("mode") or safety.get("calibrationMode") or safety.get("calibration_mode") or "full"
    mode = str(mode).strip().lower().replace("-", "_")
    if mode not in VALID_MODES:
        errors.append(f"mode must be one of {sorted(VALID_MODES)}")

    for key in POINT_KEYS:
        if key in safety and not _point2(safety[key]):
            errors.append(f"{key} must be [x,y]")
    line_endpoint_key, line_endpoints = _first_present(safety, LINE_ENDPOINT_KEYS)
    if line_endpoints is not None:
        if not isinstance(line_endpoints, list) or not all(_point2(item) for item in line_endpoints):
            errors.append(f"{line_endpoint_key} must be a list of [x,y] endpoints")
    for key in ZONE_KEYS:
        if key in safety and not _polygon(safety[key]):
            errors.append(f"{key} must be a polygon with at least 3 [x,y] points")
    for key in NONNEGATIVE_KEYS:
        if key in safety and (not _finite_number(safety[key]) or safety[key] < 0):
            errors.append(f"{key} must be a non-negative finite number")
    for key in BOOLEAN_KEYS:
        if key in safety and not isinstance(safety[key], bool):
            errors.append(f"{key} must be boolean")

    min_w_key, min_w = _first_number(safety, ("minProbeHalfWidthM", "min_probe_half_width_m"))
    max_w_key, max_w = _first_number(safety, ("maxProbeHalfWidthM", "max_probe_half_width_m"))
    if min_w is not None and max_w is not None and min_w > max_w:
        errors.append(f"{min_w_key} must be <= {max_w_key}")

    min_h_key, min_h = _first_number(safety, ("minProbeHalfHeightM", "min_probe_half_height_m"))
    max_h_key, max_h = _first_number(safety, ("maxProbeHalfHeightM", "max_probe_half_height_m"))
    if min_h is not None and max_h is not None and min_h > max_h:
        errors.append(f"{min_h_key} must be <= {max_h_key}")

    zone_key, raw_zones = _first_present(safety, ZONE_LIST_KEYS)
    normalized_zones = []
    if raw_zones is not None:
        if not isinstance(raw_zones, list):
            errors.append(f"{zone_key} must be a list")
        else:
            for index, zone in enumerate(raw_zones):
                normalized_zone = _validate_zone(zone, index, errors)
                if normalized_zone is not None:
                    normalized_zones.append(normalized_zone)

    zone_name, calibration_zone = _first_present(safety, ZONE_KEYS)
    point_name, probe_center = _first_present(safety, POINT_KEYS)
    if _point2(probe_center) and _polygon(calibration_zone):
        if not _point_in_polygon(probe_center, calibration_zone):
            errors.append(f"{point_name} must be inside {zone_name}")
    if _point2(probe_center):
        blocked_zones = [
            zone["name"]
            for zone in normalized_zones
            if _point_in_zone(probe_center, zone)
        ]
        if blocked_zones:
            errors.append(f"{point_name} must not be inside no-go zones {blocked_zones}")
        if isinstance(line_endpoints, list) and all(_point2(item) for item in line_endpoints):
            blocked_sweeps = []
            for endpoint_index, endpoint in enumerate(line_endpoints):
                for zone in normalized_zones:
                    if _segment_hits_zone(endpoint, probe_center, zone):
                        blocked_sweeps.append({
                            "endpoint": endpoint_index,
                            "zone": zone["name"],
                        })
            if blocked_sweeps:
                errors.append(f"{line_endpoint_key} cable sweeps to {point_name} cross no-go zones {blocked_sweeps}")
    summary = {
        "mode": mode,
        "probe_center_key": point_name,
        "probe_center": probe_center,
        "calibration_zone_key": zone_name,
        "calibration_zone_points": len(calibration_zone) if isinstance(calibration_zone, list) else 0,
        "line_endpoint_key": line_endpoint_key,
        "line_endpoint_count": len(line_endpoints) if isinstance(line_endpoints, list) else 0,
        "no_go_zone_count": len(normalized_zones),
        "no_go_zones": normalized_zones,
        "allow_degraded_reference": bool(safety.get("allowDegradedReference") or safety.get("allow_degraded_reference")),
        "skip_safe_motion_validation": bool(safety.get("skipSafeMotionValidation") or safety.get("skip_safe_motion_validation")),
    }
    return errors, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize calibrationSafety settings.")
    parser.add_argument("config", type=Path, help="Robot config JSON, e.g. bedroom.conf")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("config must be a JSON object")
    safety_key, safety = _extract_safety(config)
    errors, summary = validate_safety(safety)
    result = {
        "config": str(args.config),
        "safety_key": safety_key,
        "has_calibration_safety": safety_key is not None,
        "valid": len(errors) == 0,
        "errors": errors,
        "summary": summary,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"config: {args.config}")
        print(f"calibration safety key: {safety_key or '(none)'}")
        print(f"valid: {result['valid']}")
        for error in errors:
            print(f"  - {error}")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
