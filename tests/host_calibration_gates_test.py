import asyncio
import time
import unittest
from unittest.mock import AsyncMock

import numpy as np

from nf_robot.common.config_loader import create_default_config
from nf_robot.generated.nf import common
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
        self.kf = _KF()
        self.commanded_velocities = []

    def record_commanded_vel(self, velocity):
        self.commanded_velocities.append(np.array(velocity, dtype=float))


class _AnchorClient:
    def __init__(self, anchor_num, commands):
        self.anchor_num = anchor_num
        self.commands = commands

    async def send_commands(self, command):
        self.commands.append((self.anchor_num, command))


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
    observer.send_ui = lambda **kwargs: observer.ui_messages.append(kwargs)
    observer.slow_stop_all_spools = lambda: setattr(observer, "stop_calls", observer.stop_calls + 1)
    return observer


class TestHostCalibrationGates(unittest.IsolatedAsyncioTestCase):
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
