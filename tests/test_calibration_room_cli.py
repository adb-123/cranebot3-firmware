import json
import sys

import pytest

from nf_robot.host.calibration_room_cli import (
    CalibrationRoomPlanError,
    build_room_plan,
    build_room_plan_quality,
    build_room_plan_recommended_actions,
    build_room_plan_svg,
    format_room_plan_summary,
    hazard_zones_from_artifact,
    latest_artifact_paths_from_dir,
    main,
    parse_circle_no_go,
    parse_hazard_no_go,
    parse_line_endpoint,
    parse_polygon_no_go,
    parse_rect_no_go,
)
from nf_robot.host.calibration_safety_apply_cli import _extract_safety_payload
from nf_robot.host.calibration_safety_cli import validate_safety


def test_room_plan_selects_center_clear_of_no_go_objects():
    safety, summary = build_room_plan(
        room_width_m=3.0,
        room_depth_m=2.0,
        grid_step_m=0.25,
        obstacle_margin_m=0.05,
        no_go_zones=[parse_circle_no_go("table,1.5,1.0,0.35,0.05")],
        line_endpoints=[parse_line_endpoint("anchor,0.0,0.0")],
    )

    center = safety["safeProbeCenter"]
    assert safety["mode"] == "manual_assisted"
    assert center[0] != pytest.approx(1.5)
    assert center[1] != pytest.approx(1.0)
    assert safety["maxProbeHalfWidthM"] >= safety["minProbeHalfWidthM"]
    assert safety["maxProbeHalfHeightM"] >= safety["minProbeHalfHeightM"]
    assert summary["candidateCounts"]["accepted"] > 0


def test_room_plan_recommends_endpoint_derivation_when_sweeps_are_unchecked():
    _safety, summary = build_room_plan(
        room_width_m=2.0,
        room_depth_m=2.0,
        grid_step_m=0.25,
    )

    assert any("--derive-line-endpoints-from-config" in action for action in summary["recommendedActions"])
    assert summary["planQuality"]["level"] == "marginal"


def test_room_plan_rejects_when_no_safe_probe_area_exists():
    with pytest.raises(CalibrationRoomPlanError):
        build_room_plan(
            room_width_m=1.0,
            room_depth_m=1.0,
            calibration_zone_inset_m=0.1,
            grid_step_m=0.2,
            obstacle_margin_m=0.0,
            no_go_zones=[parse_rect_no_go("covered,0,0,1,1")],
        )


def test_room_plan_avoids_cable_sweeps_through_no_go_objects():
    safety, summary = build_room_plan(
        room_width_m=4.0,
        room_depth_m=2.0,
        calibration_zone_inset_m=0.2,
        grid_step_m=0.2,
        obstacle_margin_m=0.0,
        no_go_zones=[parse_rect_no_go("screen,1.8,0.0,2.2,2.0")],
        line_endpoints=[parse_line_endpoint("left_anchor,0.2,1.0")],
    )

    assert safety["safeProbeCenter"][0] < 1.8
    assert summary["candidateCounts"]["reasons"]["cable_sweep_intersects_no_go"] > 0
    assert any("Review cable sweeps" in action for action in summary["recommendedActions"])


def test_room_plan_supports_polygon_no_go_objects():
    safety, summary = build_room_plan(
        room_width_m=3.0,
        room_depth_m=2.0,
        grid_step_m=0.25,
        obstacle_margin_m=0.05,
        no_go_zones=[parse_polygon_no_go("sofa,1.1,0.6,1.8,0.5,1.9,1.2,1.2,1.3,0.05")],
        line_endpoints=[parse_line_endpoint("anchor,0.0,0.0")],
    )
    errors, validation_summary = validate_safety(safety)

    assert safety["noGoZones"][0]["name"] == "sofa"
    assert "polygon" in safety["noGoZones"][0]
    assert errors == []
    assert validation_summary["no_go_zone_count"] == 1
    assert summary["candidateCounts"]["accepted"] > 0


def test_room_plan_treats_recent_hazards_as_temporary_no_go_zones():
    safety, summary = build_room_plan(
        room_width_m=3.0,
        room_depth_m=2.0,
        grid_step_m=0.25,
        obstacle_margin_m=0.05,
        hazard_zones=[parse_hazard_no_go("catch,1.5,1.0,0.25,0.05", 0.2)],
        line_endpoints=[parse_line_endpoint("anchor,0.0,0.0")],
    )
    errors, validation_summary = validate_safety(safety)

    assert summary["hazardAvoidanceCount"] == 1
    assert safety["noGoZones"][0]["name"] == "hazard:catch"
    assert errors == []
    assert validation_summary["no_go_zone_count"] == 1
    assert any("Recent hazard positions" in action for action in summary["recommendedActions"])


def test_room_plan_extracts_hazard_positions_from_artifact():
    zones = hazard_zones_from_artifact(
        {
            "observations": [
                {
                    "kind": "calibration_hazard",
                    "hazard": {
                        "kind": "tension_limit",
                        "position": {"x": 1.0, "y": 0.8},
                        "radiusM": 0.25,
                    },
                }
            ],
            "failures": [
                {
                    "message": "catch risk object snagged line",
                    "gantry_xy": [1.4, 0.9],
                }
            ],
        },
        default_radius_m=0.2,
    )

    assert [zone.name for zone in zones] == ["hazard:tension_limit", "hazard:artifact_hazard_1"]
    assert zones[0].radius_m == 0.25
    assert zones[1].radius_m == 0.2


def test_room_plan_extracts_current_failure_point_xy_hazard():
    zones = hazard_zones_from_artifact(
        {
            "failures": [
                {
                    "fatal": True,
                    "kind": "tension_limit",
                    "message": "tension exceeded 17.0 N during calibration",
                    "point_xy": [-0.23576692697794036, 0.7697733516138807],
                    "lines": [3],
                    "limit_n": 17.0,
                }
            ],
        },
        default_radius_m=0.2,
    )

    assert [zone.name for zone in zones] == ["hazard:tension_limit"]
    assert zones[0].center.x == pytest.approx(-0.23576692697794036)
    assert zones[0].center.y == pytest.approx(0.7697733516138807)
    assert zones[0].radius_m == 0.2


def test_room_plan_output_validates_as_calibration_safety():
    safety, _summary = build_room_plan(
        room_width_m=2.5,
        room_depth_m=2.0,
        grid_step_m=0.25,
        obstacle_margin_m=0.05,
        no_go_zones=[parse_circle_no_go("lamp,1.2,1.0,0.2")],
        line_endpoints=[parse_line_endpoint("anchor,0.0,0.0")],
    )

    errors, summary = validate_safety(safety)

    assert errors == []
    assert summary["probe_center"] == safety["safeProbeCenter"]
    assert summary["line_endpoint_count"] == 1
    assert summary["no_go_zone_count"] == 1


def test_documented_room_example_meets_usable_quality(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibration-room-plan",
            "--room-file",
            "docs/calibration_room.example.json",
            "--include-plan-summary",
            "--require-plan-quality",
            "usable",
            "--summary",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["roomPlan"]["planQuality"]["level"] in {"usable", "strong"}
    assert payload["roomPlan"]["candidateCounts"]["accepted"] > 0


def test_wrapped_room_plan_output_can_be_used_as_safety_payload():
    safety, room_plan = build_room_plan(
        room_width_m=2.0,
        room_depth_m=2.0,
        grid_step_m=0.25,
    )

    extracted = _extract_safety_payload(
        {
            "calibrationSafety": safety,
            "roomPlan": room_plan,
        }
    )
    errors, _summary = validate_safety(extracted)

    assert extracted == safety
    assert errors == []


def test_room_plan_svg_preview_contains_operational_layers():
    safety, room_plan = build_room_plan(
        room_width_m=2.0,
        room_depth_m=2.0,
        grid_step_m=0.25,
        no_go_zones=[parse_circle_no_go("lamp,1.0,1.0,0.2,0.05")],
        line_endpoints=[parse_line_endpoint("anchor,0.0,0.0")],
    )

    svg = build_room_plan_svg(safety, room_plan)

    assert "<svg" in svg
    assert "calibration zone" in svg
    assert "no-go lamp" in svg
    assert "cable sweep endpoint 1" in svg
    assert "selected probe center" in svg
    assert "adaptive probe diamond" in svg


def test_room_plan_recommended_actions_can_be_built_from_summary():
    actions = build_room_plan_recommended_actions(
        {
            "candidateCounts": {
                "searched": 100,
                "accepted": 2,
                "reasons": {
                    "inside_no_go": 10,
                    "cable_sweep_intersects_no_go": 20,
                    "probe_envelope_too_small": 30,
                },
            },
            "lineEndpointCount": 0,
            "obstacleCount": 0,
            "clearanceScore": 0.03,
            "selectedProbe": {
                "maxHalfWidthM": 0.05,
                "maxHalfHeightM": 0.02,
            },
        }
    )

    assert any("lineEndpoints" in action for action in actions)
    assert any("noGoZones" in action for action in actions)
    assert any("Review cable sweeps" in action for action in actions)
    assert any("Fewer than 5 percent" in action for action in actions)


def test_room_plan_quality_can_be_built_from_summary():
    quality = build_room_plan_quality(
        {
            "candidateCounts": {
                "searched": 100,
                "accepted": 2,
                "reasons": {"cable_sweep_intersects_no_go": 20},
            },
            "lineEndpointCount": 0,
            "obstacleCount": 0,
            "clearanceScore": 0.03,
            "selectedProbe": {
                "maxHalfWidthM": 0.05,
                "maxHalfHeightM": 0.02,
            },
        }
    )

    assert quality["level"] == "marginal"
    assert quality["acceptedCandidateRatio"] == 0.02
    assert "line endpoints missing" in quality["reasons"]


def test_room_plan_summary_text_surfaces_quality_and_recommendations():
    _safety, summary = build_room_plan(
        room_width_m=2.0,
        room_depth_m=2.0,
        grid_step_m=0.25,
    )

    text = format_room_plan_summary(summary)

    assert "selected safeProbeCenter=" in text
    assert "planQuality=marginal" in text
    assert "recommendedActions:" in text
    assert "--derive-line-endpoints-from-config" in text


def test_room_plan_cli_uses_hazards_from_artifact(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    output_path = tmp_path / "room_plan.json"
    artifact_path.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "kind": "calibration_hazard",
                        "hazard": {
                            "kind": "tension_limit",
                            "position": {"x": 0.6, "y": 0.8},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--room-width-m",
            "2.5",
            "--room-depth-m",
            "2.0",
            "--grid-step-m",
            "0.25",
            "--hazards-from-artifact",
            str(artifact_path),
            "--include-plan-summary",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["roomPlan"]["hazardAvoidanceCount"] == 1
    assert payload["calibrationSafety"]["noGoZones"][0]["name"] == "hazard:tension_limit"


def test_room_plan_selects_latest_artifacts_from_directory(tmp_path):
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(json.dumps({"observations": []}), encoding="utf-8")
    new_path.write_text(json.dumps({"observations": []}), encoding="utf-8")
    old_path.touch()
    new_path.touch()

    assert latest_artifact_paths_from_dir(tmp_path, limit=1) == [new_path]


def test_room_plan_cli_uses_hazards_from_latest_artifact_dir(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    old_path = artifact_dir / "old.json"
    new_path = artifact_dir / "new.json"
    output_path = tmp_path / "room_plan.json"
    old_path.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "kind": "calibration_hazard",
                        "hazard": {"kind": "old_tension", "position": {"x": 0.4, "y": 0.8}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    new_path.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "kind": "calibration_hazard",
                        "hazard": {"kind": "new_tension", "position": {"x": 0.7, "y": 0.8}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    old_path.touch()
    new_path.touch()

    exit_code = main(
        [
            "--room-width-m",
            "2.5",
            "--room-depth-m",
            "2.0",
            "--grid-step-m",
            "0.25",
            "--hazards-from-artifact-dir",
            str(artifact_dir),
            "--include-plan-summary",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["roomPlan"]["hazardAvoidanceCount"] == 1
    assert payload["roomPlan"]["hazardArtifactSources"] == [str(new_path)]
    assert payload["calibrationSafety"]["noGoZones"][0]["name"] == "hazard:new_tension"


def test_room_plan_cli_quality_gate_rejects_marginal_plan(tmp_path):
    output_path = tmp_path / "calibration_safety.json"
    exit_code = main(
        [
            "--room-width-m",
            "2.0",
            "--room-depth-m",
            "2.0",
            "--grid-step-m",
            "0.25",
            "--require-plan-quality",
            "usable",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 3
    assert not output_path.exists()


def test_room_plan_cli_writes_safety_json(tmp_path):
    output_path = tmp_path / "calibration_safety.json"
    exit_code = main(
        [
            "--room-width-m",
            "2.5",
            "--room-depth-m",
            "2.0",
            "--grid-step-m",
            "0.25",
            "--add-circle-no-go",
            "lamp,1.2,1.0,0.2",
            "--line-endpoint",
            "anchor,0.0,0.0",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "safeProbeCenter" in payload
    assert payload["noGoZones"][0]["name"] == "lamp"


def test_room_plan_cli_writes_svg_preview(tmp_path):
    output_path = tmp_path / "calibration_safety.json"
    svg_path = tmp_path / "room_plan.svg"
    exit_code = main(
        [
            "--room-width-m",
            "2.5",
            "--room-depth-m",
            "2.0",
            "--grid-step-m",
            "0.25",
            "--add-circle-no-go",
            "lamp,1.2,1.0,0.2",
            "--line-endpoint",
            "anchor,0.0,0.0",
            "--output",
            str(output_path),
            "--svg-output",
            str(svg_path),
        ]
    )

    assert exit_code == 0
    assert "<svg" in svg_path.read_text(encoding="utf-8")


def test_room_plan_cli_accepts_reusable_room_file(tmp_path):
    room_path = tmp_path / "room.json"
    output_path = tmp_path / "calibration_safety.json"
    room_path.write_text(
        json.dumps(
            {
                "room": {
                    "widthM": 2.5,
                    "depthM": 2.0,
                    "origin": {"x": 0.0, "y": 0.0},
                    "calibrationZoneInsetM": 0.15,
                },
                "mode": "constrained",
                "gridStepM": 0.25,
                "obstacleMarginM": 0.05,
                "lineEndpoints": [{"name": "anchor", "x": 0.0, "y": 0.0}],
                "recentHazards": [{"name": "catch", "x": 0.3, "y": 1.0, "radiusM": 0.12}],
                "noGoZones": [
                    {
                        "name": "lamp",
                        "type": "circle",
                        "center": [1.2, 1.0],
                        "radiusM": 0.2,
                        "marginM": 0.05,
                    },
                    {
                        "name": "table",
                        "type": "rect",
                        "min": {"x": 1.7, "y": 0.4},
                        "max": {"x": 2.2, "y": 1.2},
                    },
                    {
                        "name": "plant_cluster",
                        "type": "polygon",
                        "points": [
                            [0.4, 1.3],
                            [0.8, 1.25],
                            [0.9, 1.7],
                            [0.5, 1.8],
                        ],
                        "marginM": 0.05,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--room-file", str(room_path), "--output", str(output_path)])
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    errors, summary = validate_safety(payload)

    assert exit_code == 0
    assert payload["mode"] == "constrained"
    assert len(payload["lineEndpoints"]) == 1
    assert [zone["name"] for zone in payload["noGoZones"]] == [
        "lamp",
        "table",
        "plant_cluster",
        "hazard:catch",
    ]
    assert errors == []
    assert summary["no_go_zone_count"] == 4


def test_room_plan_cli_flags_override_room_file_scalars(tmp_path):
    room_path = tmp_path / "room.json"
    output_path = tmp_path / "room_plan.json"
    room_path.write_text(
        json.dumps(
            {
                "roomWidthM": 2.0,
                "roomDepthM": 2.0,
                "mode": "full",
                "gridStepM": 0.25,
                "obstacleMarginM": 0.2,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--room-file",
            str(room_path),
            "--mode",
            "manual_assisted",
            "--obstacle-margin-m",
            "0.05",
            "--include-plan-summary",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["calibrationSafety"]["mode"] == "manual_assisted"
    assert payload["calibrationSafety"]["obstacleMarginM"] == 0.05


def test_room_plan_cli_derives_line_endpoints_from_robot_config(tmp_path):
    room_path = tmp_path / "room.json"
    config_path = tmp_path / "robot.json"
    output_path = tmp_path / "calibration_safety.json"
    room_path.write_text(
        json.dumps(
            {
                "roomWidthM": 2.0,
                "roomDepthM": 2.0,
                "gridStepM": 0.25,
                "lineEndpoints": [[0.0, 0.0]],
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "anchors": [
                    {
                        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 2.0}},
                        "indirectLine": {"eyeletPos": {"x": 1.8, "y": 0.0, "z": 2.0}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--room-file",
            str(room_path),
            "--derive-line-endpoints-from-config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["lineEndpoints"] == [[0.0, 0.0], [1.8, 0.0]]


def test_room_plan_cli_can_overwrite_room_file_line_endpoints_with_derived_config(tmp_path):
    room_path = tmp_path / "room.json"
    config_path = tmp_path / "robot.json"
    output_path = tmp_path / "calibration_safety.json"
    room_path.write_text(
        json.dumps(
            {
                "roomWidthM": 2.0,
                "roomDepthM": 2.0,
                "gridStepM": 0.25,
                "lineEndpoints": [[0.0, 0.0]],
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "anchors": [
                    {
                        "pose": {"position": {"x": 1.8, "y": 0.0, "z": 2.0}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--room-file",
            str(room_path),
            "--derive-line-endpoints-from-config",
            str(config_path),
            "--overwrite-line-endpoints",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["lineEndpoints"] == [[1.8, 0.0]]


def test_room_plan_cli_rejects_empty_derived_line_endpoints_without_override(tmp_path):
    room_path = tmp_path / "room.json"
    config_path = tmp_path / "robot.json"
    output_path = tmp_path / "calibration_safety.json"
    room_path.write_text(
        json.dumps({"roomWidthM": 2.0, "roomDepthM": 2.0, "gridStepM": 0.25}),
        encoding="utf-8",
    )
    config_path.write_text(json.dumps({"anchors": [{"pose": {}}]}), encoding="utf-8")

    exit_code = main(
        [
            "--room-file",
            str(room_path),
            "--derive-line-endpoints-from-config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()


def test_room_plan_cli_can_disable_risky_room_file_booleans(tmp_path):
    room_path = tmp_path / "room.json"
    output_path = tmp_path / "calibration_safety.json"
    room_path.write_text(
        json.dumps(
            {
                "roomWidthM": 2.0,
                "roomDepthM": 2.0,
                "gridStepM": 0.25,
                "allowDegradedReference": True,
                "skipSafeMotionValidation": True,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--room-file",
            str(room_path),
            "--no-allow-degraded-reference",
            "--no-skip-safe-motion-validation",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["allowDegradedReference"] is False
    assert payload["skipSafeMotionValidation"] is False


def test_room_plan_cli_can_include_machine_summary(tmp_path):
    output_path = tmp_path / "room_plan.json"
    exit_code = main(
        [
            "--room-width-m",
            "2.0",
            "--room-depth-m",
            "2.0",
            "--grid-step-m",
            "0.25",
            "--include-plan-summary",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "calibrationSafety" in payload
    assert payload["roomPlan"]["candidateCounts"]["accepted"] > 0
