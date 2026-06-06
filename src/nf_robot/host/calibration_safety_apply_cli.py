from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nf_robot.host.calibration_safety_cli import validate_safety


DEFAULT_SAFETY = {
    "mode": "manual_assisted",
    "safeProbeCenter": [0.0, 0.0],
    "calibrationZone": [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ],
    "maxProbeHalfWidthM": 0.65,
    "maxProbeHalfHeightM": 0.08,
    "minProbeHalfWidthM": 0.08,
    "minProbeHalfHeightM": 0.02,
    "obstacleMarginM": 0.08,
    "hazardAvoidRadiusM": 0.35,
    "validationDistanceM": 0.04,
    "validationSpeedMps": 0.03,
    "validationSettleS": 0.35,
    "manualAssistTimeoutS": 20,
    "allowDegradedReference": False,
    "skipSafeMotionValidation": False,
}

APPEND_LIST_KEYS = {
    "noGoZones",
    "no_go_zones",
    "catchRiskObjects",
    "catch_risk_objects",
    "obstacles",
    "lineEndpoints",
    "line_endpoints",
    "cableEndpoints",
    "cable_endpoints",
}


def _dict_get_any(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _vec_xy(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        try:
            x = float(value["x"])
            y = float(value["y"])
        except (KeyError, TypeError, ValueError):
            return None
        return [x, y]
    if isinstance(value, list) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
    return None


def derive_line_endpoints(config: dict[str, Any]) -> list[list[float]]:
    endpoints: list[list[float]] = []
    anchors = config.get("anchors")
    if not isinstance(anchors, list):
        return endpoints

    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        pose = anchor.get("pose")
        if isinstance(pose, dict):
            position = _dict_get_any(pose, "position", "pos")
            point = _vec_xy(position)
            if point is not None:
                endpoints.append(point)

        indirect_line = _dict_get_any(anchor, "indirectLine", "indirect_line")
        if isinstance(indirect_line, dict):
            eyelet_pos = _dict_get_any(indirect_line, "eyeletPos", "eyelet_pos")
            point = _vec_xy(eyelet_pos)
            if point is not None:
                endpoints.append(point)

    deduped: list[list[float]] = []
    seen = set()
    for point in endpoints:
        key = (round(point[0], 6), round(point[1], 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


def _split_spec(spec: str, expected: tuple[int, ...], label: str) -> list[str]:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) not in expected:
        expected_text = " or ".join(str(item) for item in expected)
        raise ValueError(f"{label} spec must have {expected_text} comma-separated fields")
    return parts


def _float_field(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number: {value!r}") from exc


def _nonnegative_float_field(value: str, name: str) -> float:
    parsed = _float_field(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative: {value!r}")
    return parsed


def parse_no_go_zone_specs(
    circle_specs: list[str] | None = None,
    rect_specs: list[str] | None = None,
) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for spec in circle_specs or []:
        parts = _split_spec(spec, (4, 5), "--add-circle-no-go")
        if not parts[0]:
            raise ValueError("circle no-go name must not be empty")
        zone = {
            "name": parts[0],
            "center": [_float_field(parts[1], "circle x"), _float_field(parts[2], "circle y")],
            "radiusM": _nonnegative_float_field(parts[3], "circle radiusM"),
        }
        if len(parts) == 5:
            zone["marginM"] = _nonnegative_float_field(parts[4], "circle marginM")
        zones.append(zone)

    for spec in rect_specs or []:
        parts = _split_spec(spec, (5, 6), "--add-rect-no-go")
        if not parts[0]:
            raise ValueError("rect no-go name must not be empty")
        zone = {
            "name": parts[0],
            "rect": [
                [_float_field(parts[1], "rect x1"), _float_field(parts[2], "rect y1")],
                [_float_field(parts[3], "rect x2"), _float_field(parts[4], "rect y2")],
            ],
        }
        if len(parts) == 6:
            zone["marginM"] = _nonnegative_float_field(parts[5], "rect marginM")
        zones.append(zone)
    return zones


def _dedupe_list(items: list[Any]) -> list[Any]:
    deduped = []
    seen = set()
    for item in items:
        try:
            marker = json.dumps(item, sort_keys=True, separators=(",", ":"))
        except TypeError:
            marker = repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def append_no_go_zones(safety: dict[str, Any], zones: list[dict[str, Any]]) -> dict[str, Any]:
    updated = deepcopy(safety)
    if not zones:
        return updated
    existing = updated.get("noGoZones")
    if not isinstance(existing, list):
        existing = []
    updated["noGoZones"] = _dedupe_list(deepcopy(existing) + deepcopy(zones))
    return updated


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_safety_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("safety JSON must be an object")
    for key in ("calibrationSafety", "calibration_safety", "roomSafety", "room_safety"):
        value = raw.get(key)
        if isinstance(value, dict):
            return deepcopy(value)
    return deepcopy(raw)


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        elif isinstance(value, list) and isinstance(merged.get(key), list) and key in APPEND_LIST_KEYS:
            if not value and merged[key]:
                # Merge mode is conservative: an empty list in defaults or an
                # imported safety file should not erase operator-defined no-go
                # zones/endpoints. Use --replace for intentional full replacement.
                continue
            if value:
                # Non-empty list patches append to existing operator lists in
                # merge mode. Use --replace for intentional full replacement.
                merged[key] = _dedupe_list(deepcopy(merged[key]) + deepcopy(value))
            else:
                merged[key] = deepcopy(value)
        elif isinstance(value, list):
            merged[key] = deepcopy(value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_updated_config(
    config: dict[str, Any],
    safety: dict[str, Any],
    key: str = "calibrationSafety",
    replace: bool = False,
) -> dict[str, Any]:
    updated = deepcopy(config)
    existing = updated.get(key)
    if replace or not isinstance(existing, dict):
        updated[key] = deepcopy(safety)
    else:
        updated[key] = _merge_dict(existing, safety)
    return updated


def apply_numeric_overrides(safety: dict[str, Any], overrides: dict[str, float | None]) -> dict[str, Any]:
    updated = deepcopy(safety)
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated


def apply_boolean_overrides(
    safety: dict[str, Any],
    *,
    allow_degraded_reference: bool = False,
    no_allow_degraded_reference: bool = False,
    skip_safe_motion_validation: bool = False,
    no_skip_safe_motion_validation: bool = False,
) -> dict[str, Any]:
    updated = deepcopy(safety)
    if allow_degraded_reference:
        updated["allowDegradedReference"] = True
    if no_allow_degraded_reference:
        updated["allowDegradedReference"] = False
    if skip_safe_motion_validation:
        updated["skipSafeMotionValidation"] = True
    if no_skip_safe_motion_validation:
        updated["skipSafeMotionValidation"] = False
    return updated


def _backup_path(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return path.with_suffix(path.suffix + f".{timestamp}.bak")


def _write_json_atomic(path: Path, payload: dict[str, Any], backup: bool = True) -> Path | None:
    backup_path = None
    if backup and path.exists():
        backup_path = _backup_path(path)
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or merge calibrationSafety settings into a robot config."
    )
    parser.add_argument("config", type=Path, help="Robot config JSON, e.g. bedroom.conf")
    parser.add_argument(
        "--safety",
        type=Path,
        help="JSON file containing a calibrationSafety block or a full config with one",
    )
    parser.add_argument(
        "--key",
        default="calibrationSafety",
        choices=("calibrationSafety", "calibration_safety", "roomSafety", "room_safety"),
        help="Config key to write",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "constrained", "manual_assisted"),
        help="Override safety mode after loading defaults/file",
    )
    parser.add_argument(
        "--max-probe-half-width-m",
        type=float,
        help="Override maxProbeHalfWidthM",
    )
    parser.add_argument(
        "--max-probe-half-height-m",
        type=float,
        help="Override maxProbeHalfHeightM",
    )
    parser.add_argument(
        "--min-probe-half-width-m",
        type=float,
        help="Override minProbeHalfWidthM",
    )
    parser.add_argument(
        "--min-probe-half-height-m",
        type=float,
        help="Override minProbeHalfHeightM",
    )
    parser.add_argument(
        "--validation-distance-m",
        type=float,
        help="Override validationDistanceM",
    )
    parser.add_argument(
        "--validation-speed-mps",
        type=float,
        help="Override validationSpeedMps",
    )
    parser.add_argument(
        "--manual-assist-timeout-s",
        type=float,
        help="Override manualAssistTimeoutS",
    )
    parser.add_argument(
        "--obstacle-margin-m",
        type=float,
        help="Override obstacleMarginM",
    )
    parser.add_argument(
        "--hazard-avoid-radius-m",
        type=float,
        help="Override hazardAvoidRadiusM",
    )
    degraded_group = parser.add_mutually_exclusive_group()
    degraded_group.add_argument(
        "--allow-degraded-reference",
        action="store_true",
        help="Set allowDegradedReference=true",
    )
    degraded_group.add_argument(
        "--no-allow-degraded-reference",
        action="store_true",
        help="Set allowDegradedReference=false",
    )
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument(
        "--skip-safe-motion-validation",
        action="store_true",
        help="Set skipSafeMotionValidation=true",
    )
    validation_group.add_argument(
        "--no-skip-safe-motion-validation",
        action="store_true",
        help="Set skipSafeMotionValidation=false",
    )
    parser.add_argument(
        "--derive-line-endpoints",
        action="store_true",
        help="Populate lineEndpoints from anchor/eyelet positions in the existing config when possible",
    )
    parser.add_argument(
        "--overwrite-line-endpoints",
        action="store_true",
        help="Replace existing lineEndpoints when used with --derive-line-endpoints",
    )
    parser.add_argument(
        "--allow-empty-derived-line-endpoints",
        action="store_true",
        help="Do not fail if --derive-line-endpoints finds no endpoints",
    )
    parser.add_argument(
        "--add-circle-no-go",
        action="append",
        metavar="NAME,X,Y,RADIUS[,MARGIN]",
        help="Append a circular no-go zone to calibrationSafety",
    )
    parser.add_argument(
        "--add-rect-no-go",
        action="append",
        metavar="NAME,X1,Y1,X2,Y2[,MARGIN]",
        help="Append a rectangular no-go zone to calibrationSafety",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing safety block instead of merging into it",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write config in-place. Without this, the merged config is printed to stdout.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write <config>.bak when using --write",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="When --write is used, print machine-readable operation summary",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print only the operation summary instead of the full dry-run config",
    )
    args = parser.parse_args()

    config = _load_json(args.config)
    if not isinstance(config, dict):
        raise SystemExit("config must be a JSON object")

    safety = deepcopy(DEFAULT_SAFETY)
    if args.safety is not None:
        safety = _merge_dict(safety, _extract_safety_payload(_load_json(args.safety)))
    if args.mode is not None:
        safety["mode"] = args.mode
    numeric_overrides = {
        "maxProbeHalfWidthM": args.max_probe_half_width_m,
        "maxProbeHalfHeightM": args.max_probe_half_height_m,
        "minProbeHalfWidthM": args.min_probe_half_width_m,
        "minProbeHalfHeightM": args.min_probe_half_height_m,
        "validationDistanceM": args.validation_distance_m,
        "validationSpeedMps": args.validation_speed_mps,
        "manualAssistTimeoutS": args.manual_assist_timeout_s,
        "obstacleMarginM": args.obstacle_margin_m,
        "hazardAvoidRadiusM": args.hazard_avoid_radius_m,
    }
    safety = apply_numeric_overrides(safety, numeric_overrides)
    safety = apply_boolean_overrides(
        safety,
        allow_degraded_reference=args.allow_degraded_reference,
        no_allow_degraded_reference=args.no_allow_degraded_reference,
        skip_safe_motion_validation=args.skip_safe_motion_validation,
        no_skip_safe_motion_validation=args.no_skip_safe_motion_validation,
    )
    try:
        added_no_go_zones = parse_no_go_zone_specs(args.add_circle_no_go, args.add_rect_no_go)
    except ValueError as exc:
        print(json.dumps({"valid": False, "errors": [str(exc)]}, indent=2))
        return 1
    safety = append_no_go_zones(safety, added_no_go_zones)
    derived_line_endpoints = []
    if args.derive_line_endpoints:
        derived_line_endpoints = derive_line_endpoints(config)
        if not derived_line_endpoints and not args.allow_empty_derived_line_endpoints:
            print(json.dumps({
                "valid": False,
                "errors": ["--derive-line-endpoints found no anchor or eyelet endpoints in config"],
                "derived_line_endpoint_count": 0,
            }, indent=2))
            return 1
        if derived_line_endpoints and (
            args.overwrite_line_endpoints
            or "lineEndpoints" not in safety
            and "line_endpoints" not in safety
            and "cableEndpoints" not in safety
            and "cable_endpoints" not in safety
        ):
            safety["lineEndpoints"] = derived_line_endpoints

    patch_errors, patch_summary = validate_safety(safety)
    if patch_errors:
        print(json.dumps({"valid": False, "errors": patch_errors, "summary": patch_summary}, indent=2))
        return 1

    updated = build_updated_config(
        config,
        safety,
        key=args.key,
        replace=args.replace,
    )
    updated_safety = updated.get(args.key)
    final_errors, final_summary = validate_safety(updated_safety if isinstance(updated_safety, dict) else {})
    if final_errors:
        print(json.dumps({"valid": False, "errors": final_errors, "summary": final_summary}, indent=2))
        return 1

    operation = {
        "config": str(args.config),
        "key": args.key,
        "write": bool(args.write),
        "backup": bool(args.write and not args.no_backup),
        "backup_path": None,
        "replace": bool(args.replace),
        "valid": True,
        "derived_line_endpoint_count": len(derived_line_endpoints),
        "added_no_go_zone_count": len(added_no_go_zones),
        "summary": final_summary,
    }

    if args.write:
        backup_path = _write_json_atomic(args.config, updated, backup=not args.no_backup)
        operation["backup_path"] = str(backup_path) if backup_path is not None else None
        print(json.dumps(operation, indent=2, sort_keys=True) if args.json else "calibrationSafety updated")
    elif args.summary:
        print(json.dumps(operation, indent=2, sort_keys=True))
    else:
        print(json.dumps(updated, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
