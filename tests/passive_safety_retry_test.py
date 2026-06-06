import asyncio
from types import SimpleNamespace

import numpy as np

from nf_robot.host import observer as observer_module
from nf_robot.host.observer import (
    AsyncObserver,
    PASSIVE_SAFE_RELEASE_MARGIN_N,
    PASSIVE_SAFE_RELEASE_RESET_MARGIN_N,
    PASSIVE_SAFE_TENSION_N,
)


def _observer():
    observer = AsyncObserver.__new__(AsyncObserver)
    observer.input_velocities = {'default': np.zeros(3)}
    observer.active_set = {'default'}
    observer.swing_cancellation_task = None
    observer.motion_task = None
    observer._passive_safety_tension_limit_extra_until = 0.0
    observer._passive_safety_release_armed = True
    observer._passive_safety_last_final_prompt_ts = 0.0
    observer.pe = SimpleNamespace(gant_pos=np.zeros(3))
    observer.run_command_loop = True
    observer.ui_messages = []
    observer.line_commands = []
    observer.stop_calls = 0
    observer.disable_calls = 0
    observer.enable_calls = 0
    observer.retry_calls = []
    observer.send_ui = lambda **kwargs: observer.ui_messages.append(kwargs)
    observer.slow_stop_all_spools = lambda: setattr(
        observer,
        'stop_calls',
        observer.stop_calls + 1,
    )

    async def disable_torque():
        observer.disable_calls += 1

    async def enable_torque():
        observer.enable_calls += 1

    async def retry_move(*args, **kwargs):
        observer.retry_calls.append((args, kwargs))
        return np.asarray(args[0], dtype=float)

    async def send_line_speed(line_no, speed, jog=False):
        observer.line_commands.append((line_no, speed, jog))

    observer._handle_disable_torque = disable_torque
    observer._handle_enable_torque = enable_torque
    observer.move_direction_speed = retry_move
    observer.send_line_speed = send_line_speed
    observer._line_records_for_tension = lambda max_age_s=0: None
    return observer


def _popup_texts(observer):
    return [
        message['pop_message'].message
        for message in observer.ui_messages
        if 'pop_message' in message
    ]


def test_passive_tension_limit_stops_without_retry(monkeypatch):
    async def run():
        observer = _observer()

        ok = await observer._handle_passive_tension_limit(
            np.array([17.2, 2.0, 2.0, 2.0]),
            PASSIVE_SAFE_TENSION_N,
        )

        assert ok is False
        assert observer.stop_calls == 1
        assert observer.disable_calls == 1
        assert observer.enable_calls == 1
        assert observer._passive_safety_tension_limit_extra_until == 0.0
        assert observer.retry_calls == []
        popup = _popup_texts(observer)[0]
        assert 'without retrying the same move' in popup
        assert 'temporarily raised the retry limit' not in popup

    async def fast_sleep(_delay):
        return None

    monkeypatch.setattr(observer_module.asyncio, 'sleep', fast_sleep)
    asyncio.run(run())


def test_passive_tension_limit_clears_stale_retry_window_without_retry(monkeypatch):
    async def run():
        observer = _observer()
        observer._passive_safety_tension_limit_extra_until = 9999999999.0

        ok = await observer._handle_passive_tension_limit(
            np.array([18.0, 2.0, 2.0, 2.0]),
            PASSIVE_SAFE_TENSION_N,
        )

        assert ok is False
        assert observer.stop_calls == 1
        assert observer.retry_calls == []
        assert observer._passive_safety_tension_limit_extra_until == 0.0
        popup = _popup_texts(observer)[0]
        assert 'without retrying the same move' in popup

    async def fast_sleep(_delay):
        return None

    monkeypatch.setattr(observer_module.asyncio, 'sleep', fast_sleep)
    asyncio.run(run())


def test_passive_safety_tension_limit_is_not_bumped():
    observer = _observer()
    observer._passive_safety_tension_limit_extra_until = 9999999999.0

    assert observer._passive_safety_tension_limit() == PASSIVE_SAFE_TENSION_N
    assert observer._passive_safety_release_threshold() == (
        PASSIVE_SAFE_TENSION_N - PASSIVE_SAFE_RELEASE_MARGIN_N
    )
    assert observer._passive_safety_release_reset_threshold() == (
        PASSIVE_SAFE_TENSION_N - PASSIVE_SAFE_RELEASE_RESET_MARGIN_N
    )


def test_passive_tension_limit_throttles_repeated_stop_popups(monkeypatch):
    async def run():
        observer = _observer()

        ok = await observer._handle_passive_tension_limit(
            np.array([17.2, 2.0, 2.0, 2.0]),
            PASSIVE_SAFE_TENSION_N,
        )

        assert ok is False
        assert observer.retry_calls == []
        popup = _popup_texts(observer)[0]
        assert 'Check the lines before moving again' in popup

    async def fast_sleep(_delay):
        return None

    monkeypatch.setattr(observer_module.asyncio, 'sleep', fast_sleep)
    asyncio.run(run())


def test_passive_near_limit_releases_high_line_and_biases_other_lines(monkeypatch):
    async def run():
        observer = _observer()
        monkeypatch.setattr(observer_module, 'PASSIVE_SAFE_RELEASE_LENGTH_M', 0.01)
        monkeypatch.setattr(observer_module, 'PASSIVE_SAFE_PICKUP_LENGTH_M', 0.002)
        monkeypatch.setattr(observer_module, 'PASSIVE_SAFE_RELEASE_MAX_SPEED_MPS', 0.01)
        monkeypatch.setattr(observer_module, 'PASSIVE_SAFE_RELEASE_RAMP_S', 0.2)
        monkeypatch.setattr(observer_module, 'PASSIVE_SAFE_RELEASE_POLL_S', 0.05)
        monkeypatch.setattr(observer_module, 'PASSIVE_SAFE_RELEASE_TIMEOUT_S', 2.0)

        async def fast_sleep(_delay):
            return None

        monkeypatch.setattr(observer_module.asyncio, 'sleep', fast_sleep)

        observer.input_velocities = {
            'default': np.array([0.1, 0.2, 0.3]),
            'swingc': np.array([0.2, 0.0, 0.0]),
        }
        observer.active_set = {'default', 'swingc'}

        ok = await observer._release_passive_safety_slack(
            np.array([PASSIVE_SAFE_TENSION_N - PASSIVE_SAFE_RELEASE_MARGIN_N, 2.0, 2.0, 2.0]),
            PASSIVE_SAFE_TENSION_N,
        )

        assert ok is False
        assert observer.stop_calls == 1
        assert observer.disable_calls == 0
        assert observer.enable_calls == 0
        assert observer.active_set == {'default'}
        assert all(np.allclose(value, np.zeros(3)) for value in observer.input_velocities.values())
        assert observer.line_commands[-4:] == [(0, 0, False), (1, 0, False), (2, 0, False), (3, 0, False)]
        assert all(jog is False for _line, _speed, jog in observer.line_commands)
        speed_batches = [
            observer.line_commands[i:i + 4]
            for i in range(0, len(observer.line_commands) - 4, 4)
        ]
        speeds = [batch[0][1] for batch in speed_batches]
        assert speeds[0] < speeds[1] < speeds[2]
        assert max(speeds) <= observer_module.PASSIVE_SAFE_RELEASE_MAX_SPEED_MPS
        assert speeds[-1] < max(speeds)
        for batch in speed_batches:
            assert [line for line, _speed, _jog in batch] == [0, 1, 2, 3]
            assert batch[0][1] > 0
            assert all(speed < 0 for _line, speed, _jog in batch[1:])
            pickup_scale = (
                observer_module.PASSIVE_SAFE_PICKUP_LENGTH_M
                / observer_module.PASSIVE_SAFE_RELEASE_LENGTH_M
            )
            assert all(
                np.isclose(abs(speed), batch[0][1] * pickup_scale)
                for _line, speed, _jog in batch[1:]
            )
        assert observer._passive_safety_release_armed is False
        popup = _popup_texts(observer)[0]
        assert 'softly releasing 6 inches on lines [0]' in popup
        assert 'lines [1, 2, 3] to pick up about 1 inch of slack' in popup

    asyncio.run(run())


def test_passive_safety_releases_once_until_tension_resets(monkeypatch):
    async def run():
        observer = _observer()
        observer._passive_safety_release_armed = False
        observer.pe.tension = np.array([
            (PASSIVE_SAFE_TENSION_N - PASSIVE_SAFE_RELEASE_MARGIN_N) * 10,
            0.0,
            0.0,
            0.0,
        ])

        async def stop_after_first_loop(_delay):
            observer.run_command_loop = False

        monkeypatch.setattr(observer_module.asyncio, 'sleep', stop_after_first_loop)
        await observer.passive_safety()

        assert observer.line_commands == []
        assert observer.stop_calls == 0
        assert observer._passive_safety_release_armed is False

        observer.run_command_loop = True
        observer.pe.tension = np.zeros(4)
        await observer.passive_safety()

        assert observer._passive_safety_release_armed is True

    asyncio.run(run())


def test_passive_safety_over_limit_hard_stop_does_not_release_slack(monkeypatch):
    async def run():
        observer = _observer()
        observer.pe.tension = np.array([(PASSIVE_SAFE_TENSION_N + 1.0) * 10, 0.0, 0.0, 0.0])

        async def fast_sleep(_delay):
            observer.run_command_loop = False

        monkeypatch.setattr(observer_module.asyncio, 'sleep', fast_sleep)
        await observer.passive_safety()

        assert observer.stop_calls == 1
        assert observer.disable_calls == 1
        assert observer.enable_calls == 1
        assert observer.line_commands == []
        popup = _popup_texts(observer)[0]
        assert 'Tension exceeded 17.0 N' in popup

    asyncio.run(run())
