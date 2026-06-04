import asyncio
import time
from types import SimpleNamespace

import numpy as np

from nf_robot.host import observer as observer_module
from nf_robot.host.observer import (
    AsyncObserver,
    PASSIVE_SAFE_TENSION_N,
    PASSIVE_SAFE_TENSION_RETRY_BUMP_N,
)


def _observer():
    observer = AsyncObserver.__new__(AsyncObserver)
    observer.input_velocities = {'default': np.zeros(3)}
    observer.active_set = {'default'}
    observer.swing_cancellation_task = None
    observer.last_retryable_move = None
    observer._passive_safety_tension_limit_extra_until = 0.0
    observer._passive_safety_retry_history = {}
    observer._passive_safety_last_final_prompt_ts = 0.0
    observer.pe = SimpleNamespace(gant_pos=np.zeros(3))
    observer.ui_messages = []
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

    observer._handle_disable_torque = disable_torque
    observer._handle_enable_torque = enable_torque
    observer.move_direction_speed = retry_move
    return observer


def _popup_texts(observer):
    return [
        message['pop_message'].message
        for message in observer.ui_messages
        if 'pop_message' in message
    ]


def test_record_retryable_move_only_keeps_default_nonzero_motion():
    observer = _observer()

    observer._record_retryable_move('swingc', np.array([0.2, 0.0, 0.0]))
    assert observer.last_retryable_move is None

    observer._record_retryable_move('default', np.zeros(3))
    assert observer.last_retryable_move is None

    observer._record_retryable_move('default', np.array([0.2, 0.0, -0.01]))

    assert observer.last_retryable_move['key'] == 'default'
    np.testing.assert_allclose(
        observer.last_retryable_move['velocity'],
        np.array([0.2, 0.0, -0.01]),
    )


def test_passive_tension_limit_backs_off_bumps_limit_and_retries_once(monkeypatch):
    async def run():
        observer = _observer()
        observer._record_retryable_move('default', np.array([0.1, 0.0, 0.0]))

        ok = await observer._handle_passive_tension_limit(
            np.array([17.2, 2.0, 2.0, 2.0]),
            PASSIVE_SAFE_TENSION_N,
        )

        assert ok is True
        assert observer.stop_calls == 1
        assert observer.disable_calls == 1
        assert observer.enable_calls == 1
        assert observer._passive_safety_tension_limit_extra_until > time.time()
        assert len(observer._passive_safety_retry_history) == 1
        assert len(observer.retry_calls) == 1
        args, kwargs = observer.retry_calls[0]
        np.testing.assert_allclose(args[0], np.array([0.1, 0.0, 0.0]))
        assert kwargs['key'] == 'default'
        assert kwargs['record_retry'] is False
        assert 'temporarily raised the retry limit to 17.5 N' in _popup_texts(observer)[0]

    async def fast_sleep(_delay):
        return None

    monkeypatch.setattr(observer_module.asyncio, 'sleep', fast_sleep)
    asyncio.run(run())


def test_second_tension_trip_during_retry_window_stops_without_repeat_retry(monkeypatch):
    async def run():
        observer = _observer()
        observer._record_retryable_move('default', np.array([0.1, 0.0, 0.0]))
        observer._passive_safety_tension_limit_extra_until = time.time() + 1.0

        ok = await observer._handle_passive_tension_limit(
            np.array([18.0, 2.0, 2.0, 2.0]),
            PASSIVE_SAFE_TENSION_N + PASSIVE_SAFE_TENSION_RETRY_BUMP_N,
        )

        assert ok is False
        assert observer.stop_calls == 1
        assert observer.retry_calls == []
        assert observer._passive_safety_tension_limit_extra_until == 0.0
        popup = _popup_texts(observer)[0]
        assert 'stayed above 17.5 N during the retry' in popup
        assert 'Check for a caught line' in popup

    async def fast_sleep(_delay):
        return None

    monkeypatch.setattr(observer_module.asyncio, 'sleep', fast_sleep)
    asyncio.run(run())


def test_retry_cooldown_prevents_repeating_same_last_move(monkeypatch):
    async def run():
        observer = _observer()
        observer._record_retryable_move('default', np.array([0.1, 0.0, 0.0]))
        observer._passive_safety_retry_history[
            observer.last_retryable_move['signature']
        ] = time.time()

        ok = await observer._handle_passive_tension_limit(
            np.array([17.2, 2.0, 2.0, 2.0]),
            PASSIVE_SAFE_TENSION_N,
        )

        assert ok is False
        assert observer.retry_calls == []
        popup = _popup_texts(observer)[0]
        assert 'Check the lines, then retry manually' in popup

    async def fast_sleep(_delay):
        return None

    monkeypatch.setattr(observer_module.asyncio, 'sleep', fast_sleep)
    asyncio.run(run())
