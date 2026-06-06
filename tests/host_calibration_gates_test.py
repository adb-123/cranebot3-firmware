import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np

from nf_robot.common.config_loader import create_default_config
from nf_robot.generated.nf import common
from nf_robot.host.arp_anchor_client import ArpeggioAnchorClient
from nf_robot.host.observer import AsyncObserver


class _Record:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)

    def getLast(self):
        return self.value


class _GantryBuffer:
    def __init__(self, rows):
        self.rows = np.array(rows, dtype=float)

    def deepCopy(self):
        return self.rows.copy()


class _DataStore:
    def __init__(self, line_records=None, gantry_rows=None):
        now = time.time()
        if line_records is None:
            line_records = [[now, 1.0, 0.0, 1.0] for _ in range(4)]
        if gantry_rows is None:
            gantry_rows = []
        self.anchor_line_record = [_Record(row) for row in line_records]
        self.gantry_pos = _GantryBuffer(gantry_rows)


class _KF:
    def __init__(self):
        self.reset_positions = []

    def reset_biases(self, position):
        self.reset_positions.append(np.array(position, dtype=float))


class _PositionEstimator:
    def __init__(self):
        self.anchor_points = np.array([
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
        ])
        self.visual_pos = np.array([0.5, 0.5, 1.0])
        self.gant_pos = np.array([0.5, 0.5, 1.0])
        self.kf = _KF()
        self.commanded_velocities = []

    def record_commanded_vel(self, velocity):
        self.commanded_velocities.append(np.array(velocity, dtype=float))

    def point_inside_work_area_2d(self, point):
        return True

    def point_inside_work_area(self, point):
        return True


class _AnchorClient:
    def __init__(self, anchor_num, commands):
        self.anchor_num = anchor_num
        self.commands = commands

    async def send_commands(self, command):
        self.commands.append((self.anchor_num, command))


class _ArpeggioAnchorClient(ArpeggioAnchorClient):
    def __init__(self, anchor_num, commands, line_action_states=None):
        self.anchor_num = anchor_num
        self.commands = commands
        self.line_action_states = list(line_action_states or [])

    async def send_commands(self, command):
        self.commands.append((self.anchor_num, command))


class _Artifact:
    def __init__(self):
        self.line_health_samples = []
        self.observations = []

    def record_line_health(self, **fields):
        self.line_health_samples.append(fields)

    def record_observation(self, **fields):
        self.observations.append(fields)


def _observer(line_records=None, gantry_rows=None):
    observer = AsyncObserver.__new__(AsyncObserver)
    observer.config = create_default_config()
    observer.config.anchor_type = common.AnchorType.PILOT
    observer.datastore = _DataStore(line_records=line_records, gantry_rows=gantry_rows)
    observer.pe = _PositionEstimator()
    observer.gripper_client = None
    observer.swing_cancellation_task = None
    observer.ui_messages = []
    observer.stop_calls = 0
    observer.sent_anchor_commands = []
    observer.anchors = {
        i: _AnchorClient(i, observer.sent_anchor_commands)
        for i in range(4)
    }
    observer.bot_clients = {}
    observer.input_velocities = {"default": np.zeros(3)}
    observer.active_set = {"default"}
    observer.send_ui = lambda **kwargs: observer.ui_messages.append(kwargs)
    observer.slow_stop_all_spools = lambda: setattr(observer, "stop_calls", observer.stop_calls + 1)
    return observer


class TestHostCalibrationGates(unittest.IsolatedAsyncioTestCase):
    def test_calibration_safety_report_separates_config_flag_from_effective_reset(self):
        observer = _observer()
        observer._diamond_center_xy = lambda: [0.0, 0.0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "robot.conf"
            config_path.write_text(
                json.dumps(
                    {
                        "calibrationSafety": {
                            "mode": "manual_assisted",
                            "allowDegradedReference": False,
                            "skipSafeMotionValidation": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
            observer.config_path = config_path

            report = observer._calibration_safety_report()

        self.assertFalse(report["allow_degraded_reference"])
        self.assertTrue(report["degraded_reference_reset_allowed"])

    def test_calibration_safety_report_includes_z_bounds(self):
        observer = _observer()
        observer._diamond_center_xy = lambda: [0.0, 0.0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "robot.conf"
            config_path.write_text(
                json.dumps(
                    {
                        "calibrationSafety": {
                            "mode": "manual_assisted",
                            "calibrationZMinM": 0.0,
                            "calibrationZMaxM": 1.524,
                        }
                    }
                ),
                encoding="utf-8",
            )
            observer.config_path = config_path

            report = observer._calibration_safety_report()

        self.assertEqual(report["calibration_z_bounds_m"], (0.0, 1.524))

    def test_calibration_safeguards_do_not_block_reachable_marker_targets(self):
        observer = _observer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "robot.conf"
            config_path.write_text(
                json.dumps(
                    {
                        "calibrationSafety": {
                            "mode": "manual_assisted",
                            "noGoZones": [
                                {
                                    "name": "hamper",
                                    "rect": [[0.0, 0.0], [1.0, 1.0]],
                                },
                                {
                                    "name": "controller",
                                    "center": [1.5, 1.5],
                                    "radiusM": 0.4,
                                },
                                {
                                    "name": "camera_clutter",
                                    "rect": [[2.0, 2.0], [3.0, 3.0]],
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            observer.config_path = config_path

            report = observer._calibration_safety_report()
            safe_marker_ok, safe_marker_reason = observer._calibration_probe_safe(
                np.array([0.5, 0.5]),
                "hamper_marker",
            )
            clutter_ok, clutter_reason = observer._calibration_probe_safe(
                np.array([2.5, 2.5]),
                "clutter",
            )

        self.assertIn("gamepad", report["reachable_marker_targets"])
        self.assertIn("hamper", report["reachable_marker_targets"])
        self.assertIn("trash", report["reachable_marker_targets"])
        self.assertEqual([zone["name"] for zone in report["no_go_zones"]], ["camera_clutter"])
        self.assertTrue(safe_marker_ok, safe_marker_reason)
        self.assertFalse(clutter_ok)
        self.assertIn("camera_clutter", clutter_reason)

    def test_spin_origin_alignment_scores_anchor_camera_bearings(self):
        observer = _observer()
        raw_obs = {
            "origin": [
                [(np.zeros(3), np.array([0.0, 0.0, 2.0]))],
                [(np.zeros(3), np.array([0.0, 0.0, 2.0]))],
            ],
            "gantry": [
                [(np.zeros(3), np.array([0.2, 0.0, 2.0]))],
                [(np.zeros(3), np.array([0.0, 0.1, 2.0]))],
            ],
        }

        snapshot = observer._spin_origin_anchor_alignment_snapshot(raw_obs)

        self.assertIsNotNone(snapshot)
        self.assertAlmostEqual(snapshot["score"], 0.075)
        self.assertEqual(snapshot["origin_counts"], [1, 1])
        self.assertEqual(snapshot["gantry_counts"], [1, 1])

    def test_partial_origin_card_hint_detects_clipped_gripper_card(self):
        observer = _observer()
        frame = np.full((384, 384, 3), 90, dtype=np.uint8)
        card = np.array([[0, 170], [230, 65], [360, 384], [0, 384]], dtype=np.int32)
        tag = np.array([[45, 245], [255, 150], [330, 384], [70, 384]], dtype=np.int32)
        cv2.fillPoly(frame, [card], (230, 230, 230))
        cv2.fillPoly(frame, [tag], (10, 10, 10))

        hint = observer._origin_card_partial_hint(frame)

        self.assertIsNotNone(hint)
        self.assertGreater(hint["area_fraction"], 0.03)
        self.assertTrue(hint["touches_frame_edge"]["bottom"])
        self.assertGreater(hint["center_px"][1], 250.0)

    async def test_spin_origin_staging_keeps_wall_camera_improving_probe(self):
        observer = _observer()
        observer.move_direction_speed = AsyncMock()
        observer._spin_staging_line_health_ok = MagicMock(return_value=True)
        observer._origin_card_partial_hint = MagicMock(return_value=None)
        observer._wait_for_gripper_marker = AsyncMock(side_effect=[
            None,
            None,
            {"n": "origin", "p": (np.zeros(3), np.zeros(3))},
        ])
        observer._spin_origin_anchor_alignment_snapshot = MagicMock(side_effect=[
            {
                "score": 1.0,
                "anchors": [],
                "origin_counts": [1, 1],
                "gantry_counts": [1, 1],
            },
            {
                "score": 0.8,
                "anchors": [],
                "origin_counts": [1, 1],
                "gantry_counts": [1, 1],
            },
        ])
        artifact = _Artifact()

        ok = await observer.stage_origin_for_spin_calibration(artifact)

        self.assertTrue(ok)
        observer.move_direction_speed.assert_awaited_once()
        args, kwargs = observer.move_direction_speed.await_args
        np.testing.assert_allclose(args[0], [0.0, -1.0, 0.0])
        self.assertEqual(kwargs["speed"], 0.025)
        self.assertTrue(any(
            obs.get("kind") == "spin_origin_staging_candidate"
            and obs.get("label") == "y_minus"
            and obs.get("accepted") is True
            and obs.get("origin_visible") is True
            for obs in artifact.observations
        ))

    async def test_spin_origin_staging_moves_up_when_origin_card_is_clipped(self):
        observer = _observer()
        observer.move_direction_speed = AsyncMock()
        observer._spin_staging_line_health_ok = MagicMock(return_value=True)
        observer._origin_card_partial_hint = MagicMock(return_value={
            "bbox_px": [40, 230, 260, 154],
            "center_px": [175.0, 322.0],
            "center_error_norm": [-0.09, 0.68],
            "area_fraction": 0.12,
            "touches_frame_edge": {
                "left": False,
                "top": False,
                "right": False,
                "bottom": True,
            },
        })
        observer._wait_for_gripper_marker = AsyncMock(side_effect=[
            None,
            None,
            {"n": "origin", "p": (np.zeros(3), np.zeros(3))},
        ])
        artifact = _Artifact()

        ok = await observer.stage_origin_for_spin_calibration(artifact)

        self.assertTrue(ok)
        observer.move_direction_speed.assert_awaited_once()
        args, kwargs = observer.move_direction_speed.await_args
        np.testing.assert_allclose(args[0], [0.0, 0.0, 1.0])
        self.assertEqual(kwargs["speed"], 0.025)
        self.assertTrue(any(
            obs.get("kind") == "spin_origin_partial_card_staging"
            and obs.get("origin_visible") is True
            for obs in artifact.observations
        ))

    def test_spin_staging_zone_tolerance_allows_small_origin_pose_drift(self):
        observer = _observer()
        safety_config = {
            "calibrationZone": [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            "calibrationZMinM": 0.0,
            "calibrationZMaxM": 2.0,
        }

        no_margin_ok, no_margin_reason = observer._calibration_point_safe(
            np.array([0.5, 1.03, 1.0]),
            "spin_origin",
            safety_config,
        )
        margin_ok, margin_reason = observer._calibration_point_safe(
            np.array([0.5, 1.03, 1.0]),
            "spin_origin",
            safety_config,
            zone_margin_m=0.05,
        )

        self.assertFalse(no_margin_ok)
        self.assertIn("outside configured calibration zone", no_margin_reason)
        self.assertTrue(margin_ok, margin_reason)

    async def test_wait_for_safe_calibration_start_rejects_z_outside_bounds(self):
        observer = _observer()
        observer.pe.gant_pos = np.array([0.5, 0.5, 1.6])
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "robot.conf"
            config_path.write_text(
                json.dumps(
                    {
                        "calibrationSafety": {
                            "mode": "constrained",
                            "calibrationZone": [
                                [0.0, 0.0],
                                [1.0, 0.0],
                                [1.0, 1.0],
                                [0.0, 1.0],
                            ],
                            "calibrationZMinM": 0.0,
                            "calibrationZMaxM": 1.524,
                        }
                    }
                ),
                encoding="utf-8",
            )
            observer.config_path = config_path

            ok = await observer.wait_for_safe_calibration_start_position()

        self.assertFalse(ok)

    async def test_return_to_calibration_start_envelope_moves_down_from_high_z(self):
        observer = _observer()
        observer.pe.gant_pos = np.array([0.5, 0.5, 1.6])
        observer.move_direction_speed = AsyncMock()
        artifact = _Artifact()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "robot.conf"
            config_path.write_text(
                json.dumps(
                    {
                        "calibrationSafety": {
                            "mode": "manual_assisted",
                            "calibrationZone": [
                                [0.0, 0.0],
                                [1.0, 0.0],
                                [1.0, 1.0],
                                [0.0, 1.0],
                            ],
                            "calibrationZMinM": 0.0,
                            "calibrationZMaxM": 1.524,
                        }
                    }
                ),
                encoding="utf-8",
            )
            observer.config_path = config_path

            async def _move_down(direction, **kwargs):
                np.testing.assert_allclose(direction, [0.0, 0.0, -1.0])
                observer.pe.gant_pos = np.array([0.5, 0.5, 1.494])

            observer.move_direction_speed.side_effect = _move_down

            ok = await observer.return_to_calibration_start_envelope(
                artifact,
                phase="test",
            )

        self.assertTrue(ok)
        observer.move_direction_speed.assert_awaited_once()
        self.assertEqual(observer.stop_calls, 1)
        self.assertTrue(any(
            obs.get("kind") == "calibration_start_envelope_return"
            and obs.get("skipped") is False
            for obs in artifact.observations
        ))
        self.assertTrue(any(
            obs.get("kind") == "calibration_start_envelope_return_result"
            and obs.get("ok") is True
            for obs in artifact.observations
        ))

    def test_calibration_keeps_passive_tension_safety_ceiling(self):
        observer = _observer()
        observer._calibration_active = True

        self.assertEqual(observer._passive_safety_tension_limit(), 17.0)

    def test_room_points_from_gantry_marker_observations_uses_gantry_tag_pose(self):
        observer = _observer()
        observer.config.anchor_type = common.AnchorType.ARPEGGIO

        points = observer._room_points_from_marker_observations(
            {
                "gantry": [
                    [(np.zeros(3), np.zeros(3))],
                    [],
                ]
            },
            [(np.zeros(3), np.zeros(3)), (np.zeros(3), np.zeros(3))],
            "gantry",
            (22.0, 22.0),
        )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].shape, (3,))
        self.assertTrue(np.all(np.isfinite(points[0])))

    async def test_wait_for_tension_rejects_opposing_speeds_that_cancel_in_sum(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.02, 1.0],
            [now, 1.0, -0.02, 1.0],
            [now, 1.0, 0.0, 1.0],
            [now, 1.0, 0.0, 1.0],
        ])

        ok = await observer.wait_for_tension(timeout_s=0.02, poll_interval_s=0.001)

        self.assertFalse(ok)
        self.assertEqual(observer.stop_calls, 1)

    async def test_wait_for_tension_accepts_taut_fresh_settled_lines(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 1.0],
            [now, 1.0, 0.0, 1.0],
            [now, 1.0, 0.0, 1.0],
            [now, 1.0, 0.0, 1.0],
        ])

        ok = await observer.wait_for_tension(timeout_s=0.02, poll_interval_s=0.001)

        self.assertTrue(ok)
        self.assertEqual(observer.stop_calls, 0)

    async def test_arpeggio_tension_lines_only_commands_lines_below_target(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 5.0],
            [now, 1.0, 0.0, 2.0],
            [now, 1.0, 0.0, 1.0],
            [now, 1.0, 0.0, 9.0],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        observer.sent_anchor_commands = []
        observer.anchors = {
            0: _ArpeggioAnchorClient(0, observer.sent_anchor_commands),
            1: _ArpeggioAnchorClient(1, observer.sent_anchor_commands),
        }

        await observer.tension_lines(target_tension_n=3.38)
        await asyncio.sleep(0)

        self.assertEqual(
            observer.sent_anchor_commands,
            [
                (0, {"tighten": {"spool": 1, "target_tension_n": 3.38}}),
                (1, {"tighten": {"spool": 0, "target_tension_n": 3.38}}),
            ],
        )

    async def test_calibration_tension_balance_move_commands_small_safe_move(self):
        observer = _observer()
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        observer.move_direction_speed = AsyncMock(return_value=np.zeros(3))

        moved = await observer._calibration_tension_balance_move(
            {
                "lines": [
                    {"line": 0, "tension_n": 0.0},
                    {"line": 1, "tension_n": 2.0},
                    {"line": 2, "tension_n": 0.5},
                    {"line": 3, "tension_n": 9.0},
                ],
                "high_tension_lines": [],
            },
            target_tension_n=3.38,
            calibration_artifact=None,
            phase="test",
            attempt=1,
        )

        self.assertTrue(moved)
        self.assertEqual(observer.move_direction_speed.await_count, 1)
        args, kwargs = observer.move_direction_speed.await_args
        direction = np.asarray(args[0], dtype=float)
        self.assertTrue(np.all(np.isfinite(direction)))
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=5)
        self.assertEqual(kwargs["key"], "default")
        self.assertEqual(kwargs["speed"], 0.025)
        self.assertEqual(observer.stop_calls, 1)

    def test_line_health_includes_arpeggio_line_action_states(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 5.0],
            [now, 1.0, 0.0, 5.0],
            [now, 1.0, 0.0, 0.5],
            [now, 1.0, 0.0, 5.0],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        observer.anchors = {
            0: _ArpeggioAnchorClient(0, [], [
                {"spool": 0, "action": "tighten", "status": "succeeded", "ts": now},
                {"spool": 1, "action": "tighten", "status": "succeeded", "ts": now},
            ]),
            1: _ArpeggioAnchorClient(1, [], [
                {"spool": 0, "action": "tighten", "status": "failed", "reason": "tension_timeout", "ts": now},
                {"spool": 1, "action": "tighten", "status": "idle", "ts": now},
            ]),
        }

        snapshot = observer._snapshot_line_health()

        self.assertEqual(snapshot["line_action_states"][2]["line"], 2)
        self.assertEqual(snapshot["lines"][2]["line_action_state"]["reason"], "tension_timeout")

    def test_line_tension_response_diagnostics_marks_reel_in_without_tension_gain_responsive(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 5.0],
            [now, 1.0, 0.0, 5.0],
            [now, 5.70, 0.0, 0.70],
            [now, 1.0, 0.0, 5.0],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        before = {
            "valid": True,
            "threshold_n": 1.38,
            "lines": [
                {"line": 0, "length_m": 1.0, "tension_n": 5.0},
                {"line": 1, "length_m": 1.0, "tension_n": 5.0},
                {"line": 2, "length_m": 6.05, "tension_n": 0.62},
                {"line": 3, "length_m": 1.0, "tension_n": 5.0},
            ],
        }
        artifact = _Artifact()

        snapshot = observer._record_calibration_line_health(
            artifact,
            requested_tension_lines=[2],
            response_baseline_snapshot=before,
            threshold_n=1.38,
        )

        self.assertEqual(snapshot["tension_response_fault_lines"], [])
        self.assertEqual(
            snapshot["tension_response_diagnostics"][0]["reason"],
            "reel_in_without_tension_gain",
        )
        self.assertTrue(snapshot["tension_response_diagnostics"][0]["responsive"])

    async def test_profile_wait_accepts_unequal_safe_tensions_and_low_responsive_line(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 3.7],
            [now, 1.0, 0.0, 9.0],
            [now, 1.0, 0.0, 0.7],
            [now, 1.0, 0.0, 4.1],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        profiles = [
            {"line": 0, "status": "healthy", "baseline_tension_n": 3.7},
            {"line": 1, "status": "high_friction_healthy", "baseline_tension_n": 9.0},
            {"line": 2, "status": "low_tension_but_responsive", "baseline_tension_n": 0.7},
            {"line": 3, "status": "healthy", "baseline_tension_n": 4.1},
        ]

        ok = await observer.wait_for_profile_tension(
            profiles,
            timeout_s=0.02,
            poll_interval_s=0.001,
        )

        self.assertTrue(ok)
        self.assertEqual(observer.stop_calls, 0)

    def test_profile_targets_only_repull_nonaccepted_or_below_window_lines(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 3.7],
            [now, 1.0, 0.0, 9.0],
            [now, 1.0, 0.0, 0.7],
            [now, 1.0, 0.0, 4.1],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        profiles = [
            {"line": 0, "status": "healthy", "baseline_tension_n": 3.7},
            {"line": 1, "status": "high_friction_healthy", "baseline_tension_n": 9.0},
            {"line": 2, "status": "low_tension_but_responsive", "baseline_tension_n": 0.7},
            {"line": 3, "status": "nonresponsive", "baseline_tension_n": 4.1},
        ]

        self.assertEqual(observer._lines_below_profile_targets(profiles), [3])

    async def test_diagnose_line_tension_profiles_accepts_low_responsive_probe(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 3.7],
            [now, 1.0, 0.0, 9.0],
            [now, 6.05, 0.0, 0.62],
            [now, 1.0, 0.0, 4.1],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        artifact = _Artifact()

        async def _fake_send_line_speed(line_no, speed, jog=False):
            self.assertEqual(line_no, 2)
            if speed < 0:
                observer.datastore.anchor_line_record[2].value = np.array(
                    [time.time(), 5.70, 0.0, 0.70],
                    dtype=float,
                )

        observer.send_line_speed = AsyncMock(side_effect=_fake_send_line_speed)
        with patch("nf_robot.host.observer.CAL_TENSION_PROFILE_PROBE_WAIT_S", 0.0), patch(
            "nf_robot.host.observer.CAL_TENSION_PROFILE_SETTLE_S",
            0.0,
        ):
            profiles = await observer.diagnose_line_tension_profiles(artifact)

        by_line = {profile["line"]: profile for profile in profiles}
        self.assertEqual(by_line[1]["status"], "high_friction_healthy")
        self.assertEqual(by_line[2]["status"], "low_tension_but_responsive")
        self.assertEqual(by_line[2]["reason"], "reel_in_without_tension_gain")
        self.assertEqual(observer.send_line_speed.await_count, 2)
        self.assertEqual(observer._line_profiles_blocking_reasons(profiles), [])

    async def test_diagnose_line_tension_profiles_waits_for_fresh_records(self):
        stale = time.time() - 10.0
        observer = _observer(line_records=[
            [stale, 1.0, 0.0, 3.7],
            [stale, 1.0, 0.0, 4.2],
            [stale, 1.0, 0.0, 2.1],
            [stale, 1.0, 0.0, 6.9],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO

        async def _freshen_line_records(*args, **kwargs):
            for record in observer.datastore.anchor_line_record:
                record.value[0] = time.time()
            return True

        observer.wait_for_fresh_line_records = AsyncMock(side_effect=_freshen_line_records)
        artifact = _Artifact()

        profiles = await observer.diagnose_line_tension_profiles(artifact)

        observer.wait_for_fresh_line_records.assert_awaited_once()
        self.assertEqual(
            [profile["status"] for profile in profiles],
            ["healthy", "healthy", "healthy", "high_friction_healthy"],
        )
        self.assertEqual(observer._line_profiles_blocking_reasons(profiles), [])

    async def test_profile_tension_recovery_uses_direct_speed_and_refreshes_profile(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 3.7],
            [now, 1.0, 0.0, 0.42],
            [now, 1.0, 0.0, 8.5],
            [now, 1.0, 0.0, 4.1],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        artifact = _Artifact()
        profiles = [
            {"line": 0, "status": "healthy", "baseline_tension_n": 3.7},
            {"line": 1, "status": "high_friction_healthy", "baseline_tension_n": 7.0},
            {"line": 2, "status": "high_friction_healthy", "baseline_tension_n": 8.5},
            {"line": 3, "status": "healthy", "baseline_tension_n": 4.1},
        ]

        async def _fake_send_line_speed(line_no, speed, jog=False):
            self.assertEqual(line_no, 1)
            if speed < 0:
                observer.datastore.anchor_line_record[1].value = np.array(
                    [time.time(), 0.76, 0.0, 0.74],
                    dtype=float,
                )

        observer.send_line_speed = AsyncMock(side_effect=_fake_send_line_speed)
        with patch("nf_robot.host.observer.CAL_TENSION_PROFILE_PROBE_WAIT_S", 0.0), patch(
            "nf_robot.host.observer.CAL_TENSION_PROFILE_SETTLE_S",
            0.0,
        ):
            updated = await observer.recover_profile_tension_lines(
                profiles,
                [1],
                calibration_artifact=artifact,
                phase="test",
                attempt=1,
                target_tension_n=1.38,
            )

        by_line = {profile["line"]: profile for profile in updated}
        self.assertEqual(by_line[1]["status"], "low_tension_but_responsive")
        self.assertTrue(by_line[1]["responsive"])
        self.assertAlmostEqual(by_line[1]["target_min_n"], 0.59)
        self.assertEqual(observer.send_line_speed.await_count, 2)
        self.assertEqual(observer.stop_calls, 1)
        self.assertTrue(any(
            sample.get("kind") == "line_tension_profile_recovery"
            for sample in artifact.line_health_samples
        ))

    async def test_profile_tension_recovery_skips_line_inside_accepted_window(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 3.7],
            [now, 1.0, 0.0, 0.62],
            [now, 1.0, 0.0, 8.5],
            [now, 1.0, 0.0, 4.1],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        artifact = _Artifact()
        profiles = [
            {"line": 0, "status": "healthy", "baseline_tension_n": 3.7},
            {"line": 1, "status": "low_tension_but_responsive", "baseline_tension_n": 0.7},
            {"line": 2, "status": "high_friction_healthy", "baseline_tension_n": 8.5},
            {"line": 3, "status": "healthy", "baseline_tension_n": 4.1},
        ]
        observer._run_bounded_line_tension_probe = AsyncMock()

        updated = await observer.recover_profile_tension_lines(
            profiles,
            [1],
            calibration_artifact=artifact,
            phase="test",
            attempt=1,
            target_tension_n=1.38,
        )

        by_line = {profile["line"]: profile for profile in updated}
        self.assertEqual(by_line[1]["status"], "low_tension_but_responsive")
        self.assertEqual(
            by_line[1]["reason"],
            "within_accepted_profile_window_before_recovery",
        )
        observer._run_bounded_line_tension_probe.assert_not_awaited()
        self.assertTrue(any(
            sample.get("kind") == "line_tension_profile_recovery_skipped"
            for sample in artifact.line_health_samples
        ))

    async def test_profile_tension_recovery_preserves_accepted_profile_after_small_probe_response(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 3.7],
            [now, 1.000, 0.0, 0.48],
            [now, 1.0, 0.0, 8.5],
            [now, 1.0, 0.0, 4.1],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        artifact = _Artifact()
        profiles = [
            {"line": 0, "status": "healthy", "baseline_tension_n": 3.7},
            {"line": 1, "status": "low_tension_but_responsive", "baseline_tension_n": 0.7},
            {"line": 2, "status": "high_friction_healthy", "baseline_tension_n": 8.5},
            {"line": 3, "status": "healthy", "baseline_tension_n": 4.1},
        ]

        async def _fake_probe(line_no, target_min):
            self.assertEqual(line_no, 1)
            self.assertAlmostEqual(target_min, 0.55)
            observer.datastore.anchor_line_record[1].value = np.array(
                [time.time(), 0.996, 0.0, 0.62],
                dtype=float,
            )
            return {
                "line": line_no,
                "target_min_n": target_min,
                "stopped_reason": "target_tension_reached",
            }

        observer._run_bounded_line_tension_probe = AsyncMock(side_effect=_fake_probe)
        with patch("nf_robot.host.observer.CAL_TENSION_PROFILE_SETTLE_S", 0.0):
            updated = await observer.recover_profile_tension_lines(
                profiles,
                [1],
                calibration_artifact=artifact,
                phase="test",
                attempt=1,
                target_tension_n=1.38,
            )

        by_line = {profile["line"]: profile for profile in updated}
        self.assertEqual(by_line[1]["status"], "low_tension_but_responsive")
        self.assertEqual(
            by_line[1]["reason"],
            "within_accepted_profile_window_after_recovery",
        )
        self.assertNotIn("responsive", by_line[1])
        observer._run_bounded_line_tension_probe.assert_awaited_once()
        after_sample = next(
            sample for sample in artifact.line_health_samples
            if sample.get("kind") == "after_line_tension_profile_recovery"
        )
        self.assertEqual(after_sample["tension_response_fault_lines"], [1])

    async def test_bounded_profile_probe_stops_when_target_reached(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 3.7],
            [now, 1.0, 0.0, 0.42],
            [now, 1.0, 0.0, 8.5],
            [now, 1.0, 0.0, 4.1],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO

        async def _fake_send_line_speed(line_no, speed, jog=False):
            self.assertEqual(line_no, 1)
            if speed < 0:
                observer.datastore.anchor_line_record[1].value = np.array(
                    [time.time(), 0.95, 0.0, 1.45],
                    dtype=float,
                )

        observer.send_line_speed = AsyncMock(side_effect=_fake_send_line_speed)

        result = await observer._run_bounded_line_tension_probe(
            1,
            1.38,
            duration_s=3.0,
            poll_interval_s=0.0,
        )

        self.assertEqual(result["stopped_reason"], "target_tension_reached")
        self.assertEqual(observer.send_line_speed.await_count, 2)
        self.assertEqual(observer.stop_calls, 1)

    async def test_bounded_profile_probe_stops_near_safe_ceiling(self):
        now = time.time()
        observer = _observer(line_records=[
            [now, 1.0, 0.0, 3.7],
            [now, 1.0, 0.0, 0.42],
            [now, 1.0, 0.0, 15.8],
            [now, 1.0, 0.0, 4.1],
        ])
        observer.config.anchor_type = common.AnchorType.ARPEGGIO

        async def _fake_send_line_speed(line_no, speed, jog=False):
            self.assertEqual(line_no, 1)
            if speed < 0:
                observer.datastore.anchor_line_record[2].value = np.array(
                    [time.time(), 1.0, 0.0, 16.1],
                    dtype=float,
                )

        observer.send_line_speed = AsyncMock(side_effect=_fake_send_line_speed)

        result = await observer._run_bounded_line_tension_probe(
            1,
            1.38,
            duration_s=3.0,
            poll_interval_s=0.0,
        )

        self.assertEqual(result["stopped_reason"], "near_safe_tension")
        self.assertEqual(result["max_tension_line"], 2)
        self.assertEqual(observer.send_line_speed.await_count, 2)
        self.assertEqual(observer.stop_calls, 1)

    async def test_calibration_tension_recovery_uses_configured_attempt_limit(self):
        observer = _observer()
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        observer.tension_lines = AsyncMock()
        observer.wait_for_profile_tension = AsyncMock(return_value=False)
        profiles = [
            {"line": 0, "status": "healthy", "baseline_tension_n": 3.0},
            {"line": 1, "status": "healthy", "baseline_tension_n": 3.0},
            {"line": 2, "status": "healthy", "baseline_tension_n": 3.0},
            {"line": 3, "status": "healthy", "baseline_tension_n": 3.0},
        ]
        observer.diagnose_line_tension_profiles = AsyncMock(return_value=profiles)
        observer.recover_profile_tension_lines = AsyncMock(return_value=profiles)
        observer._lines_below_profile_targets = MagicMock(return_value=[0, 1, 2, 3])
        observer._record_calibration_line_health = MagicMock(return_value={"high_tension_lines": []})
        observer._fail_on_calibration_hazard = lambda *args, **kwargs: False
        observer._line_tension_failure_message = lambda snapshot: "not settled"

        ok = await observer.tension_and_wait(calibration_artifact=object(), phase="test")

        self.assertFalse(ok)
        observer.tension_lines.assert_not_awaited()
        self.assertEqual(observer.recover_profile_tension_lines.await_count, 3)
        targets = [
            call.kwargs["target_tension_n"]
            for call in observer.recover_profile_tension_lines.await_args_list
        ]
        self.assertEqual(targets, [1.38, 2.38, 3.38])

    async def test_calibration_tension_uses_profile_recovery_instead_of_tighten(self):
        observer = _observer()
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        profiles = [
            {"line": 0, "status": "healthy", "baseline_tension_n": 3.0},
            {"line": 1, "status": "healthy", "baseline_tension_n": 3.0},
            {"line": 2, "status": "healthy", "baseline_tension_n": 3.0},
            {"line": 3, "status": "healthy", "baseline_tension_n": 3.0},
        ]
        observer.tension_lines = AsyncMock()
        observer.wait_for_profile_tension = AsyncMock(return_value=True)
        observer.diagnose_line_tension_profiles = AsyncMock(return_value=profiles)
        observer.recover_profile_tension_lines = AsyncMock(return_value=profiles)
        observer._lines_below_profile_targets = MagicMock(return_value=[1, 3])
        observer._record_calibration_line_health = MagicMock(return_value={"high_tension_lines": []})
        observer._fail_on_calibration_hazard = lambda *args, **kwargs: False

        ok = await observer.tension_and_wait(calibration_artifact=object(), phase="test")

        self.assertTrue(ok)
        observer.tension_lines.assert_not_awaited()
        observer.recover_profile_tension_lines.assert_awaited_once()
        self.assertEqual(
            observer.recover_profile_tension_lines.await_args.args[1],
            [1, 3],
        )

    async def test_calibration_tension_reuses_cached_line_profiles(self):
        observer = _observer()
        observer.config.anchor_type = common.AnchorType.ARPEGGIO
        profiles = [
            {"line": 0, "status": "healthy", "baseline_tension_n": 3.0},
            {"line": 1, "status": "healthy", "baseline_tension_n": 3.0},
            {"line": 2, "status": "low_tension_but_responsive", "baseline_tension_n": 0.7},
            {"line": 3, "status": "healthy", "baseline_tension_n": 3.0},
        ]
        observer._calibration_line_tension_profiles = profiles
        observer.diagnose_line_tension_profiles = AsyncMock(return_value=[])
        observer.tension_lines = AsyncMock()
        observer.recover_profile_tension_lines = AsyncMock(return_value=profiles)
        observer.wait_for_profile_tension = AsyncMock(return_value=True)
        observer._record_calibration_line_health = MagicMock(return_value={"high_tension_lines": []})
        observer._fail_on_calibration_hazard = lambda *args, **kwargs: False

        ok = await observer.tension_and_wait(calibration_artifact=object(), phase="test")

        self.assertTrue(ok)
        observer.diagnose_line_tension_profiles.assert_not_awaited()
        observer.tension_lines.assert_not_awaited()
        self.assertTrue(any(
            call.kwargs.get("kind") == "line_tension_profile_reused"
            for call in observer._record_calibration_line_health.call_args_list
        ))

    async def test_send_reference_lengths_rejects_stale_visual_data_before_sending(self):
        stale = time.time() - 10.0
        observer = _observer(gantry_rows=[
            [stale, 0, 0.5, 0.5, 1.0],
            [stale, 1, 0.5, 0.5, 1.0],
        ])

        ok = await observer.sendReferenceLengths(np.ones(4))

        self.assertFalse(ok)
        self.assertEqual(observer.sent_anchor_commands, [])
        self.assertEqual(observer.pe.kf.reset_positions, [])

    async def test_send_reference_lengths_accepts_fresh_visual_data_before_bias_reset(self):
        now = time.time()
        observer = _observer(gantry_rows=[
            [now, 0, 0.4, 0.5, 1.0],
            [now, 1, 0.6, 0.5, 1.0],
        ])

        ok = await observer.sendReferenceLengths(np.array([1.0, 1.1, 1.2, 1.3]))
        await asyncio.sleep(0)

        self.assertTrue(ok)
        self.assertEqual(len(observer.sent_anchor_commands), 4)
        np.testing.assert_allclose(observer.pe.kf.reset_positions[0], [0.5, 0.5, 1.0])

    async def test_half_calibration_aborts_before_reference_save_when_tension_fails(self):
        observer = _observer()
        observer.tension_and_wait = AsyncMock(return_value=False)
        observer.sendReferenceLengths = AsyncMock(return_value=True)
        observer.move_direction_speed = AsyncMock()

        ok = await observer.half_auto_calibration()

        self.assertFalse(ok)
        self.assertEqual(observer.sendReferenceLengths.await_count, 0)
        self.assertEqual(observer.move_direction_speed.await_count, 0)

    async def test_half_calibration_aborts_before_motion_when_reference_reset_fails(self):
        observer = _observer()
        observer.tension_and_wait = AsyncMock(return_value=True)
        observer.sendReferenceLengths = AsyncMock(return_value=False)
        observer.move_direction_speed = AsyncMock()

        ok = await observer.half_auto_calibration()

        self.assertFalse(ok)
        self.assertEqual(observer.sendReferenceLengths.await_count, 1)
        self.assertEqual(observer.move_direction_speed.await_count, 0)
        self.assertEqual(observer.stop_calls, 1)
