import json
import sys

from nf_robot.host.calibration_artifact_cli import build_summary
from nf_robot.host.calibration_artifacts import CalibrationArtifactSession
from pathlib import Path

from nf_robot.common.config_loader import create_default_config, load_config, save_config
from nf_robot.host.calibration_safety_apply_cli import (
    _backup_path,
    _extract_safety_payload,
    DEFAULT_SAFETY,
    append_no_go_zones,
    apply_boolean_overrides,
    apply_numeric_overrides,
    build_updated_config,
    derive_line_endpoints,
    main as apply_safety_main,
    parse_no_go_zone_specs,
)
from nf_robot.host.calibration_safety_cli import validate_safety


def test_validate_safety_accepts_cluttered_room_constraints():
    errors, summary = validate_safety(
        {
            "mode": "manual_assisted",
            "safeProbeCenter": [0.0, 0.0],
            "calibrationZone": [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            "noGoZones": [
                {
                    "name": "lamp",
                    "center": [0.5, -0.2],
                    "radiusM": 0.2,
                    "marginM": 0.1,
                },
                {
                    "name": "chair",
                    "polygon": [
                        [-0.8, 0.4],
                        [-0.2, 0.4],
                        [-0.2, 0.9],
                        [-0.8, 0.9],
                    ],
                },
            ],
            "validationDistanceM": 0.04,
            "skipSafeMotionValidation": False,
        }
    )

    assert errors == []
    assert summary["mode"] == "manual_assisted"
    assert summary["no_go_zone_count"] == 2
    assert summary["skip_safe_motion_validation"] is False


def test_validate_safety_rejects_invalid_zone_and_mode():
    errors, summary = validate_safety(
        {
            "mode": "unsafe",
            "calibrationZone": [[0, 0], [1, 0]],
            "noGoZones": [{"name": "bad", "center": [0.0], "radiusM": -1.0}],
            "allowDegradedReference": "yes",
        }
    )

    assert summary["mode"] == "unsafe"
    assert any("mode must be one of" in error for error in errors)
    assert any("calibrationZone must be a polygon" in error for error in errors)
    assert any("allowDegradedReference must be boolean" in error for error in errors)
    assert any("zone bad center must be [x,y]" in error for error in errors)


def test_validate_safety_rejects_probe_center_outside_safe_envelope():
    errors, summary = validate_safety(
        {
            "safeProbeCenter": [2.0, 0.0],
            "calibrationZone": [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            "noGoZones": [
                {
                    "name": "blocked_corner",
                    "rect": [[1.5, -0.5], [2.5, 0.5]],
                }
            ],
        }
    )

    assert summary["probe_center"] == [2.0, 0.0]
    assert any("safeProbeCenter must be inside calibrationZone" in error for error in errors)
    assert any("safeProbeCenter must not be inside no-go zones" in error for error in errors)


def test_validate_safety_applies_no_go_margins_to_probe_center():
    errors, summary = validate_safety(
        {
            "safeProbeCenter": [0.29, 0.0],
            "calibrationZone": [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            "noGoZones": [
                {
                    "name": "lamp",
                    "center": [0.0, 0.0],
                    "radiusM": 0.2,
                    "marginM": 0.1,
                }
            ],
        }
    )

    assert summary["probe_center"] == [0.29, 0.0]
    assert any("safeProbeCenter must not be inside no-go zones ['lamp']" in error for error in errors)


def test_validate_safety_reports_line_endpoint_count():
    errors, summary = validate_safety(
        {
            "safeProbeCenter": [0.0, 0.0],
            "lineEndpoints": [[-1.0, 0.0], [1.0, 0.0]],
            "calibrationZone": [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
        }
    )

    assert errors == []
    assert summary["line_endpoint_key"] == "lineEndpoints"
    assert summary["line_endpoint_count"] == 2


def test_validate_safety_rejects_cable_sweep_through_no_go_zone():
    errors, summary = validate_safety(
        {
            "safeProbeCenter": [1.0, 0.0],
            "lineEndpoints": [[-1.0, 0.0]],
            "calibrationZone": [
                [-2.0, -1.0],
                [2.0, -1.0],
                [2.0, 1.0],
                [-2.0, 1.0],
            ],
            "noGoZones": [
                {
                    "name": "center_table",
                    "rect": [[-0.2, -0.2], [0.2, 0.2]],
                    "marginM": 0.05,
                }
            ],
        }
    )

    assert summary["line_endpoint_count"] == 1
    assert any("lineEndpoints cable sweeps to safeProbeCenter cross no-go zones" in error for error in errors)


def test_validate_safety_rejects_inverted_probe_size_bounds():
    errors, summary = validate_safety(
        {
            "minProbeHalfWidthM": 0.5,
            "maxProbeHalfWidthM": 0.2,
            "minProbeHalfHeightM": 0.1,
            "maxProbeHalfHeightM": 0.05,
        }
    )

    assert summary["mode"] == "full"
    assert "minProbeHalfWidthM must be <= maxProbeHalfWidthM" in errors
    assert "minProbeHalfHeightM must be <= maxProbeHalfHeightM" in errors


def test_calibration_safety_apply_merges_without_dropping_existing_fields():
    config = {
        "robotId": "robot-1",
        "calibrationSafety": {
            "mode": "full",
            "noGoZones": [{"name": "lamp", "center": [0.0, 0.0], "radiusM": 0.2}],
            "operatorNote": "keep",
        },
    }
    safety_patch = {
        "mode": "manual_assisted",
        "validationDistanceM": 0.04,
    }

    updated = build_updated_config(config, safety_patch)

    assert updated["robotId"] == "robot-1"
    assert updated["calibrationSafety"]["mode"] == "manual_assisted"
    assert updated["calibrationSafety"]["validationDistanceM"] == 0.04
    assert updated["calibrationSafety"]["operatorNote"] == "keep"
    assert updated["calibrationSafety"]["noGoZones"][0]["name"] == "lamp"


def test_config_loader_preserves_calibration_safety_block(tmp_path):
    path = tmp_path / "robot.conf"
    payload = json.loads(create_default_config().to_json(indent=2))
    payload["calibrationSafety"] = {
        "mode": "manual_assisted",
        "allowDegradedReference": False,
        "skipSafeMotionValidation": False,
        "noGoZones": [{"name": "hazard:test", "center": [0.1, 0.2], "radiusM": 0.3}],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    loaded = load_config(path)
    save_config(loaded, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["calibrationSafety"] == payload["calibrationSafety"]


def test_calibration_safety_apply_default_merge_does_not_drop_existing_no_go_zones():
    config = {
        "calibrationSafety": {
            "noGoZones": [{"name": "lamp", "center": [0.0, 0.0], "radiusM": 0.2}],
        },
    }

    updated = build_updated_config(config, DEFAULT_SAFETY)

    assert updated["calibrationSafety"]["noGoZones"] == [
        {"name": "lamp", "center": [0.0, 0.0], "radiusM": 0.2}
    ]
    assert updated["calibrationSafety"]["mode"] == "manual_assisted"


def test_calibration_safety_apply_empty_list_patch_does_not_drop_existing_no_go_zones():
    config = {
        "calibrationSafety": {
            "noGoZones": [{"name": "lamp", "center": [0.0, 0.0], "radiusM": 0.2}],
            "lineEndpoints": [[1.0, 2.0]],
        },
    }
    safety_patch = {
        "noGoZones": [],
        "lineEndpoints": [],
    }

    updated = build_updated_config(config, safety_patch)

    assert updated["calibrationSafety"]["noGoZones"] == [
        {"name": "lamp", "center": [0.0, 0.0], "radiusM": 0.2}
    ]
    assert updated["calibrationSafety"]["lineEndpoints"] == [[1.0, 2.0]]


def test_calibration_safety_apply_non_empty_list_patch_appends_existing_lists():
    config = {
        "calibrationSafety": {
            "noGoZones": [{"name": "existing", "center": [0.0, 0.0], "radiusM": 0.2}],
            "lineEndpoints": [[1.0, 2.0]],
        },
    }
    safety_patch = {
        "noGoZones": [{"name": "new", "center": [1.0, 0.0], "radiusM": 0.1}],
        "lineEndpoints": [[3.0, 4.0]],
    }

    updated = build_updated_config(config, safety_patch)

    assert [zone["name"] for zone in updated["calibrationSafety"]["noGoZones"]] == ["existing", "new"]
    assert updated["calibrationSafety"]["lineEndpoints"] == [[1.0, 2.0], [3.0, 4.0]]


def test_calibration_safety_apply_summary_reports_merged_no_go_zones(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "robot.conf"
    patch_path = tmp_path / "safety.json"
    config_path.write_text(
        json.dumps(
            {
                "calibrationSafety": {
                    **DEFAULT_SAFETY,
                    "noGoZones": [
                        {"name": "camera_clutter_1", "center": [0.8, 0.8], "radiusM": 0.05}
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    patch_path.write_text(
        json.dumps(
            {
                "calibrationSafety": {
                    "noGoZones": [
                        {"name": "hazard:tension_limit", "center": [-0.8, 0.8], "radiusM": 0.05}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["calibration-safety-apply", str(config_path), "--safety", str(patch_path), "--summary"],
    )

    assert apply_safety_main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["summary"]["no_go_zone_count"] == 2
    assert [zone["name"] for zone in payload["summary"]["no_go_zones"]] == [
        "camera_clutter_1",
        "hazard:tension_limit",
    ]


def test_calibration_safety_apply_list_merge_deduplicates_repeated_entries():
    config = {
        "calibrationSafety": {
            "noGoZones": [{"name": "existing", "center": [0.0, 0.0], "radiusM": 0.2}],
            "lineEndpoints": [[1.0, 2.0]],
        },
    }
    safety_patch = {
        "noGoZones": [{"name": "existing", "center": [0.0, 0.0], "radiusM": 0.2}],
        "lineEndpoints": [[1.0, 2.0]],
    }

    updated = build_updated_config(config, safety_patch)

    assert updated["calibrationSafety"]["noGoZones"] == [
        {"name": "existing", "center": [0.0, 0.0], "radiusM": 0.2}
    ]
    assert updated["calibrationSafety"]["lineEndpoints"] == [[1.0, 2.0]]


def test_calibration_safety_apply_parses_and_appends_no_go_zone_specs():
    zones = parse_no_go_zone_specs(
        circle_specs=["lamp,0.5,-0.2,0.3,0.1"],
        rect_specs=["table,-1.0,-0.5,1.0,0.5,0.2"],
    )
    safety = append_no_go_zones(
        {"noGoZones": [{"name": "keep", "center": [0.0, 0.0], "radiusM": 0.1}]},
        zones,
    )

    assert zones == [
        {"name": "lamp", "center": [0.5, -0.2], "radiusM": 0.3, "marginM": 0.1},
        {"name": "table", "rect": [[-1.0, -0.5], [1.0, 0.5]], "marginM": 0.2},
    ]
    assert [zone["name"] for zone in safety["noGoZones"]] == ["keep", "lamp", "table"]


def test_calibration_safety_apply_rejects_invalid_no_go_zone_spec():
    try:
        parse_no_go_zone_specs(circle_specs=["lamp,0.5"])
    except ValueError as exc:
        assert "--add-circle-no-go spec must have" in str(exc)
    else:
        raise AssertionError("invalid no-go zone spec should raise ValueError")


def test_calibration_safety_apply_rejects_negative_no_go_radius_or_margin():
    for kwargs, expected in [
        ({"circle_specs": ["lamp,0.0,0.0,-0.1"]}, "circle radiusM must be non-negative"),
        ({"circle_specs": ["lamp,0.0,0.0,0.1,-0.1"]}, "circle marginM must be non-negative"),
        ({"rect_specs": ["table,0.0,0.0,1.0,1.0,-0.1"]}, "rect marginM must be non-negative"),
    ]:
        try:
            parse_no_go_zone_specs(**kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("negative no-go spec field should raise ValueError")


def test_calibration_safety_apply_rejects_empty_no_go_names():
    for kwargs, expected in [
        ({"circle_specs": [",0.0,0.0,0.1"]}, "circle no-go name must not be empty"),
        ({"rect_specs": [",0.0,0.0,1.0,1.0"]}, "rect no-go name must not be empty"),
    ]:
        try:
            parse_no_go_zone_specs(**kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("empty no-go name should raise ValueError")


def test_calibration_safety_apply_replace_drops_existing_safety_fields():
    config = {
        "robotId": "robot-1",
        "calibrationSafety": {
            "mode": "full",
            "operatorNote": "drop",
        },
    }
    safety_patch = {
        "mode": "constrained",
    }

    updated = build_updated_config(config, safety_patch, replace=True)

    assert updated["robotId"] == "robot-1"
    assert updated["calibrationSafety"] == {"mode": "constrained"}


def test_calibration_safety_apply_numeric_overrides_update_probe_and_validation_settings():
    safety = {
        "maxProbeHalfWidthM": 0.65,
        "maxProbeHalfHeightM": 0.08,
        "validationDistanceM": 0.04,
    }

    updated = apply_numeric_overrides(
        safety,
        {
            "maxProbeHalfWidthM": 0.3,
            "maxProbeHalfHeightM": 0.04,
            "validationDistanceM": 0.02,
            "obstacleMarginM": 0.12,
            "hazardAvoidRadiusM": 0.5,
            "manualAssistTimeoutS": None,
        },
    )

    assert updated["maxProbeHalfWidthM"] == 0.3
    assert updated["maxProbeHalfHeightM"] == 0.04
    assert updated["validationDistanceM"] == 0.02
    assert updated["obstacleMarginM"] == 0.12
    assert updated["hazardAvoidRadiusM"] == 0.5
    assert "manualAssistTimeoutS" not in updated


def test_calibration_safety_apply_boolean_overrides_update_fallback_flags():
    safety = {
        "allowDegradedReference": False,
        "skipSafeMotionValidation": False,
    }

    enabled = apply_boolean_overrides(
        safety,
        allow_degraded_reference=True,
        skip_safe_motion_validation=True,
    )
    disabled = apply_boolean_overrides(
        enabled,
        no_allow_degraded_reference=True,
        no_skip_safe_motion_validation=True,
    )

    assert enabled["allowDegradedReference"] is True
    assert enabled["skipSafeMotionValidation"] is True
    assert disabled["allowDegradedReference"] is False
    assert disabled["skipSafeMotionValidation"] is False


def test_calibration_safety_backup_path_is_timestamped():
    backup = _backup_path(Path("bedroom.conf"))

    assert backup.name.startswith("bedroom.conf.")
    assert backup.name.endswith("Z.bak")


def test_derive_line_endpoints_from_anchor_and_eyelet_config():
    endpoints = derive_line_endpoints(
        {
            "anchors": [
                {
                    "pose": {"position": {"x": 1.0, "y": 2.0, "z": 3.0}},
                    "indirectLine": {"eyeletPos": {"x": 4.0, "y": 5.0, "z": 6.0}},
                },
                {
                    "pose": {"position": {"x": -1.0, "y": -2.0, "z": 3.0}},
                    "indirect_line": {"eyelet_pos": {"x": -4.0, "y": -5.0, "z": 6.0}},
                },
            ]
        }
    )

    assert endpoints == [[1.0, 2.0], [4.0, 5.0], [-1.0, -2.0], [-4.0, -5.0]]


def test_derive_line_endpoints_returns_empty_when_config_has_no_geometry():
    assert derive_line_endpoints({"anchors": [{"pose": {}}]}) == []


def test_derived_line_endpoints_can_trigger_cable_sweep_validation_error():
    config = {
        "anchors": [
            {
                "pose": {"position": {"x": -1.0, "y": 0.0, "z": 2.0}},
            }
        ],
    }
    safety = {
        "safeProbeCenter": [1.0, 0.0],
        "calibrationZone": [
            [-2.0, -1.0],
            [2.0, -1.0],
            [2.0, 1.0],
            [-2.0, 1.0],
        ],
        "noGoZones": [
            {
                "name": "center_table",
                "rect": [[-0.2, -0.2], [0.2, 0.2]],
            }
        ],
    }
    safety["lineEndpoints"] = derive_line_endpoints(config)

    errors, summary = validate_safety(safety)

    assert summary["line_endpoint_count"] == 1
    assert any("cable sweeps" in error for error in errors)


def test_calibration_artifact_summary_counts_fatal_and_recoverable_hazards():
    artifact = CalibrationArtifactSession(
        session_id="unit-test",
        now=lambda: "2026-06-05T00:00:00Z",
    )
    artifact.record_observation(
        kind="calibration_hazard",
        hazard={
            "kind": "safe_motion_validation",
            "fatal": False,
            "lines": [1],
        },
    )
    artifact.fail(
        "fatal hazard",
        hazard={
            "kind": "tension_limit",
            "fatal": True,
            "lines": [3],
        },
    )

    snapshot = artifact.snapshot()

    assert snapshot["summary"]["hazard_count"] == 2
    assert snapshot["summary"]["fatal_hazard_count"] == 1
    assert snapshot["summary"]["recoverable_hazard_count"] == 1


def test_artifact_cli_summary_surfaces_adaptive_and_validation_evidence():
    artifact = {
        "session_id": "abc",
        "status": "failed",
        "phase": "safe_motion_validation",
        "created_at": "2026-06-05T00:00:00Z",
        "updated_at": "2026-06-05T00:01:00Z",
        "summary": {"fatal_hazard_count": 0, "recoverable_hazard_count": 1},
        "failures": [{"message": "health gate failed"}],
        "warnings": [],
        "optimizer_reports": [{"name": "eyelet", "success": True}],
        "line_health_samples": [
            {
                "kind": "safe_motion_after_probe",
                "high_tension_lines": [2],
            }
        ],
        "observations": [
            {
                "kind": "adaptive_diamond_plan",
                "search": {
                    "selected": {
                        "half_height_m": 0.05,
                        "half_width_m": 0.4,
                    },
                    "candidates": [
                        {
                            "half_height_m": 0.1,
                            "half_width_m": 1.0,
                            "safe": False,
                            "reason": "cable sweep crosses chair",
                        },
                        {
                            "half_height_m": 0.05,
                            "half_width_m": 0.4,
                            "safe": True,
                            "reason": "ok",
                        },
                    ],
                },
            },
            {
                "kind": "safe_motion_validation_plan",
                "safe_candidates": ["x_plus"],
                "rejected": [{"label": "x_minus", "reason": "outside zone"}],
            },
        ],
    }

    summary = build_summary(artifact, artifact_path="logs/calibration/failed.json")

    assert summary["latest_failure"]["message"] == "health gate failed"
    assert summary["latest_line_health"]["high_tension_lines"] == [2]
    assert summary["adaptive_diamond_selected"]["half_width_m"] == 0.4
    assert summary["adaptive_diamond_rejections"][0]["reason"] == "cable sweep crosses chair"
    assert summary["safe_validation_candidates"] == ["x_plus"]
    assert any("high-tension lines [2]" in action for action in summary["recommended_actions"])
    assert any("manual_assisted mode" in action for action in summary["recommended_actions"])


def test_artifact_cli_recommends_actions_from_health_gate_reasons():
    summary = build_summary(
        {
            "session_id": "health",
            "status": "failed",
            "phase": "arpeggio_eyelet_solve",
            "failures": [
                {
                    "message": "calibration health gate failed",
                    "health": {
                        "reasons": [
                            "missing measured line deltas: ['bot_to_rig']",
                            "origin visual coverage weak: [0, 12]",
                            "calibration probe coverage near minimum: half_h=0.02, half_w=0.05",
                        ]
                    },
                }
            ],
            "warnings": [],
            "optimizer_reports": [],
            "line_health_samples": [],
            "observations": [],
        }
    )

    assert any("line-length telemetry is fresh" in action for action in summary["recommended_actions"])
    assert any("marker visibility" in action for action in summary["recommended_actions"])
    assert any("widen probe limits" in action for action in summary["recommended_actions"])


def test_artifact_cli_recommends_actions_from_line_tension_profiles():
    summary = build_summary(
        {
            "session_id": "profiles",
            "status": "completed",
            "phase": "line_tension_profile_diagnostic",
            "failures": [],
            "warnings": [],
            "optimizer_reports": [],
            "line_health_samples": [
                {
                    "kind": "line_tension_profile_diagnostic",
                    "valid": True,
                    "high_tension_lines": [],
                    "line_tension_profiles": [
                        {"line": 1, "status": "high_friction_healthy"},
                        {"line": 2, "status": "low_tension_but_responsive"},
                    ],
                }
            ],
            "observations": [],
        }
    )

    assert any("read low tension but responded" in action for action in summary["recommended_actions"])
    assert any("higher stable friction/tension" in action for action in summary["recommended_actions"])


def test_artifact_cli_recommends_rerun_when_safe_motion_validation_was_skipped():
    summary = build_summary(
        {
            "session_id": "skipped",
            "status": "completed",
            "phase": "swing_cancellation_calibration",
            "failures": [],
            "warnings": [],
            "optimizer_reports": [],
            "line_health_samples": [],
            "observations": [
                {
                    "kind": "calibration_safety_constraints",
                    "safety": {
                        "safe_motion_validation_skipped": True,
                    },
                }
            ],
        }
    )

    assert any("Safe-motion validation was skipped" in action for action in summary["recommended_actions"])


def test_artifact_cli_recommends_fixing_stale_gantry_visual_anchors():
    summary = build_summary(
        {
            "session_id": "visuals",
            "status": "failed",
            "phase": "post_anchor_reference",
            "failures": [{"message": "reference length reset failed"}],
            "warnings": [],
            "optimizer_reports": [],
            "line_health_samples": [
                {
                    "kind": "reference_length_reset_failed",
                    "valid": True,
                    "high_tension_lines": [],
                    "gantry_visual_reference": {
                        "stale_or_missing_anchors": [1],
                    },
                }
            ],
            "observations": [],
        }
    )

    assert any("stale/missing anchor cameras [1]" in action for action in summary["recommended_actions"])


def test_artifact_cli_recommends_full_rerun_after_degraded_reference_reset():
    summary = build_summary(
        {
            "session_id": "degraded",
            "status": "completed",
            "phase": "swing_cancellation_calibration",
            "failures": [],
            "warnings": [],
            "optimizer_reports": [],
            "line_health_samples": [
                {
                    "kind": "reference_length_reset_ok",
                    "valid": True,
                    "high_tension_lines": [],
                    "degraded_reference": True,
                }
            ],
            "observations": [],
        }
    )

    assert any("degraded one-anchor visual evidence" in action for action in summary["recommended_actions"])
