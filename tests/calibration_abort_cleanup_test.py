import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from nf_robot.common.config_loader import create_default_config
from nf_robot.generated.nf import common
from nf_robot.generated.nf import config as nf_config
from nf_robot.host.observer import AsyncObserver


class _FakeAnchor:
    def __init__(self, anchor_num):
        self.anchor_num = anchor_num
        self.commands = []
        self.save_raw = False
        self.calibrating_room_spin = False

    async def send_commands(self, command):
        self.commands.append(command)
        return True


class _LineRecord:
    def __init__(self, tensions, length=1.0, speed=0.0):
        self.tensions = list(tensions)
        self.length = length
        self.speed = speed

    def getLast(self):
        if len(self.tensions) > 1:
            tension = self.tensions.pop(0)
        else:
            tension = self.tensions[0]
        return [0.0, self.length, self.speed, tension]


def _cleanup_observer():
    observer = AsyncObserver.__new__(AsyncObserver)
    observer.config = create_default_config()
    observer.config.anchor_type = common.AnchorType.ARPEGGIO
    for anchor in observer.config.anchors[:2]:
        anchor.indirect_line = nf_config.IndirectLine(cam_tilt=0.0)
    observer.anchors = {0: _FakeAnchor(0), 1: _FakeAnchor(1)}
    observer.bot_clients = {}
    observer.gripper_client = SimpleNamespace(calibrating_room_spin=False)
    observer.motion_task = None
    observer.swing_cancellation_task = None
    observer.active_set = {'default'}
    observer.input_velocities = {'default': np.zeros(3)}
    observer.ui_messages = []
    observer.send_ui = lambda **kwargs: observer.ui_messages.append(kwargs)
    observer.slow_stop_calls = 0
    observer.slow_stop_all_spools = lambda: setattr(
        observer,
        'slow_stop_calls',
        observer.slow_stop_calls + 1,
    )
    return observer


def _anti_tangle_values(anchor):
    return [
        command['set_anti_tangle'][0]
        for command in anchor.commands
        if 'set_anti_tangle' in command
    ]


def test_collect_arp_eyelet_restores_anti_tangle_and_raw_capture_on_cancel():
    async def run():
        observer = _cleanup_observer()
        observer.datastore = SimpleNamespace(anchor_line_record=[
            _LineRecord([1.0]),
            _LineRecord([1.0]),
            _LineRecord([1.0]),
            _LineRecord([1.0]),
        ])
        observer.touch_floor = lambda: asyncio.sleep(0)

        async def cancel_on_line_speed(*_args, **_kwargs):
            raise asyncio.CancelledError

        observer.send_line_speed = cancel_on_line_speed

        with pytest.raises(asyncio.CancelledError):
            await observer.collect_arp_anchor_eyelet_experiment_data(anchor_poses=[None, None])

        for anchor in observer.anchors.values():
            assert _anti_tangle_values(anchor) == [False, True]
            assert anchor.save_raw is False
        assert observer.slow_stop_calls >= 1

    asyncio.run(run())


def test_collect_arp_eyelet_relaxes_each_direct_line_until_both_are_slack():
    async def run():
        observer = _cleanup_observer()
        observer.datastore = SimpleNamespace(anchor_line_record=[
            _LineRecord([0.05, 0.05]),
            _LineRecord([1.0]),
            _LineRecord([0.5, 0.05]),
            _LineRecord([1.0]),
        ])
        observer.touch_floor = lambda: asyncio.sleep(0)
        line_commands = []

        async def record_line_speed(line_no, speed, jog=False):
            line_commands.append((line_no, speed, jog))
            if jog and line_no == 0 and speed == 0.3:
                raise RuntimeError('stop after direct-line relaxation')

        observer.send_line_speed = record_line_speed

        with pytest.raises(RuntimeError, match='stop after direct-line relaxation'):
            await observer.collect_arp_anchor_eyelet_experiment_data(anchor_poses=[None, None])

        assert (0, 0, False) in line_commands
        assert (2, 0.1, False) in line_commands
        assert (2, 0, False) in line_commands
        for anchor in observer.anchors.values():
            assert _anti_tangle_values(anchor) == [False, True]
            assert anchor.save_raw is False

    asyncio.run(run())


def test_stop_all_clears_motion_keys_and_calibration_modes():
    async def run():
        observer = _cleanup_observer()
        observer.active_set = {'default', 'manual', 'swingc'}
        observer.input_velocities = {
            'default': np.array([1.0, 0.0, 0.0]),
            'manual': np.array([0.0, 1.0, 0.0]),
            'swingc': np.array([0.0, 0.0, 1.0]),
        }
        observer.gripper_client.calibrating_room_spin = True
        for anchor in observer.anchors.values():
            anchor.save_raw = True
            anchor.calibrating_room_spin = True

        async def motion():
            await asyncio.sleep(60)

        async def swing():
            await asyncio.sleep(60)

        observer.motion_task = asyncio.create_task(motion())
        observer.motion_task.set_name('calibration_motion')
        observer.swing_cancellation_task = asyncio.create_task(swing())

        await observer.stop_all()
        await asyncio.sleep(0)

        assert observer.motion_task is None
        assert observer.active_set == {'default'}
        assert set(observer.input_velocities) == {'default', 'manual', 'swingc'}
        for velocity in observer.input_velocities.values():
            np.testing.assert_allclose(velocity, np.zeros(3))
        assert observer.gripper_client.calibrating_room_spin is False
        assert observer.swing_cancellation_task.cancelled()
        for anchor in observer.anchors.values():
            assert anchor.save_raw is False
            assert anchor.calibrating_room_spin is False
            assert _anti_tangle_values(anchor) == [True]
        assert observer.slow_stop_calls == 1

    asyncio.run(run())


def test_calibrate_spin_restores_mode_when_detection_fails():
    class _FailedResult:
        def get(self, timeout=None):
            raise RuntimeError('detector failed')

    class _Pool:
        def apply_async(self, *_args, **_kwargs):
            return _FailedResult()

    async def run():
        observer = _cleanup_observer()
        observer.gripper_client = SimpleNamespace(
            last_frame_resized=np.zeros((8, 8, 3), dtype=np.uint8),
            calibrating_room_spin=False,
        )
        observer.pool = _Pool()
        observer.config_path = None

        with pytest.raises(RuntimeError, match='detector failed'):
            await observer.calibrate_spin(reset_wrist_first=False)

        assert observer.gripper_client.calibrating_room_spin is False

    asyncio.run(run())
