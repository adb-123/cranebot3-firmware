import inspect
import asyncio
import time

import numpy as np
import pytest

from nf_robot.common.config_loader import (
    DEFAULT_SWING_AUTO_MIN_ENERGY,
    DEFAULT_SWING_GAIN,
    DEFAULT_SWING_MAX_VELOCITY,
    DEFAULT_SWING_SIGN,
    DEFAULT_SWING_LATENCY,
    create_default_config,
    normalize_config_defaults,
)
from nf_robot.generated.nf import config as nf_config
from nf_robot.generated.nf import common
from nf_robot.generated.nf import control
from nf_robot.host.arp_gripper_client import ArpeggioGripperClient, OMEGA
from nf_robot.host import observer as observer_module
from nf_robot.host.observer import AsyncObserver


class _Series:
    def __init__(self, value):
        self.value = value

    def getLast(self):
        return self.value


class _DataStore:
    def __init__(self, wrist_degrees=0.0):
        self.winch_line_record = _Series([0.0, wrist_degrees, 0.0])


class _Observer:
    def __init__(self, config):
        self.config = config


def _client(frame_room_spin=np.pi / 2, wrist_degrees=0.0):
    config = create_default_config()
    config.gripper.frame_room_spin = frame_room_spin
    config.swing_gain = DEFAULT_SWING_GAIN
    config.swing_sign = DEFAULT_SWING_SIGN
    config.swing_max_velocity = DEFAULT_SWING_MAX_VELOCITY
    return ArpeggioGripperClient(
        "127.0.0.1",
        8765,
        _DataStore(wrist_degrees=wrist_degrees),
        _Observer(config),
        pool=None,
        stat=None,
        pe=None,
        local_telemetry=None,
    )


def _async_observer(config=None):
    observer = AsyncObserver.__new__(AsyncObserver)
    observer.config = config or create_default_config()
    observer.ui_messages = []
    observer.send_ui = lambda **kwargs: observer.ui_messages.append(kwargs)
    return observer


def test_swing_energy_uses_model_norm():
    client = _client()
    client.gripper_swing_model = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert client.swing_energy() == 0.5 * (1 + 4 + 9 + 16)


def test_compute_swing_correction_applies_gain_sign_and_identity_rotation():
    client = _client()
    client.swing_model_ts = 100.0
    client.gripper_swing_model = np.array([[0.0, 1.0], [0.0, 0.0]])

    vel = client.compute_swing_correction(
        100.0,
        gain=0.1,
        sign=-1.0,
        max_velocity=10.0,
        update_integrator=False,
    )

    np.testing.assert_allclose(vel, np.array([-OMEGA * 0.1, 0.0]), atol=1e-9)


def test_half_period_latency_flips_correction_phase():
    client = _client()
    client.swing_model_ts = 100.0
    client.gripper_swing_model = np.array([[0.0, 1.0], [0.0, 0.0]])

    now_vel = client.compute_swing_correction(
        100.0,
        gain=0.1,
        sign=-1.0,
        max_velocity=10.0,
        update_integrator=False,
    )
    half_period_vel = client.compute_swing_correction(
        100.0 + np.pi / OMEGA,
        gain=0.1,
        sign=-1.0,
        max_velocity=10.0,
        update_integrator=False,
    )

    np.testing.assert_allclose(half_period_vel, -now_vel, atol=1e-9)


def test_compute_swing_correction_clamps_velocity():
    client = _client()
    client.swing_model_ts = 100.0
    client.gripper_swing_model = np.array([[0.0, 100.0], [0.0, 0.0]])

    vel = client.compute_swing_correction(
        100.0,
        gain=1.0,
        sign=-1.0,
        max_velocity=0.05,
        update_integrator=False,
    )

    assert np.linalg.norm(vel) <= 0.0500001


def test_compute_swing_correction_preserves_explicit_zero_values():
    client = _client()
    client.swing_model_ts = 100.0
    client.gripper_swing_model = np.array([[0.0, 1.0], [0.0, 0.0]])

    zero_gain_vel = client.compute_swing_correction(
        100.0,
        gain=0.0,
        sign=-1.0,
        max_velocity=10.0,
        update_integrator=False,
    )
    zero_max_vel = client.compute_swing_correction(
        100.0,
        gain=0.1,
        sign=-1.0,
        max_velocity=0.0,
        update_integrator=False,
    )

    np.testing.assert_allclose(zero_gain_vel, np.zeros(2), atol=1e-9)
    np.testing.assert_allclose(zero_max_vel, np.zeros(2), atol=1e-9)


def test_swing_model_phase_uses_host_receive_time_not_pi_wall_time():
    client = _client()
    remote_pi_time = 12.0

    asyncio.run(client.handle_update_from_ws({
        "st": remote_pi_time,
        "sm": [[0.0, 1.0], [0.0, 0.0]],
    }))

    assert abs(client.swing_model_ts - time.time()) < 1.0
    assert client.swing_model_ts != remote_pi_time


def test_swing_model_remote_timestamp_without_model_does_not_refresh_phase_or_age():
    client = _client()
    client.swing_model_ts = 100.0
    client.swing_model_host_ts = 100.0

    asyncio.run(client.handle_update_from_ws({"st": 200.0}))

    assert client.swing_model_remote_ts == 200.0
    assert client.swing_model_ts == 100.0
    assert client.swing_model_host_ts == 100.0


def test_config_defaults_preserve_explicit_zero_latency():
    config = nf_config.StringmanPilotConfig()
    raw_config = {"swingLatency": 0.0}

    normalize_config_defaults(config, raw_config)

    assert config.swing_latency == 0.0
    assert config.swing_gain == DEFAULT_SWING_GAIN
    assert config.swing_sign == DEFAULT_SWING_SIGN
    assert config.swing_max_velocity == DEFAULT_SWING_MAX_VELOCITY


def test_config_defaults_raise_too_low_swing_calibration_energy_floor():
    config = nf_config.StringmanPilotConfig()
    raw_config = {"swingAutoMinEnergy": 0.0001}
    config.swing_auto_min_energy = 0.0001

    normalize_config_defaults(config, raw_config)

    assert config.swing_auto_min_energy == DEFAULT_SWING_AUTO_MIN_ENERGY


def test_config_defaults_clamp_unsafe_swing_runtime_values():
    config = nf_config.StringmanPilotConfig()
    raw_config = {
        "swingLatency": -0.2,
        "swingGain": 0.12,
        "swingMaxVelocity": 0.12,
    }
    config.swing_latency = -0.2
    config.swing_gain = 0.12
    config.swing_max_velocity = 0.12

    normalize_config_defaults(config, raw_config)

    assert config.swing_latency == DEFAULT_SWING_LATENCY
    assert config.swing_gain == DEFAULT_SWING_GAIN
    assert config.swing_max_velocity == DEFAULT_SWING_MAX_VELOCITY


def test_swing_calibration_abort_ignores_single_noise_sample_near_limit():
    observer = _async_observer()
    samples = [(float(i), 0.08) for i in range(11)]
    samples.append((11.0, 0.105))

    assert not observer._swing_should_abort_calibration(samples, abort_energy=0.1)


def test_swing_calibration_abort_requires_sustained_or_large_energy_increase():
    observer = _async_observer()

    assert observer._swing_should_abort_calibration([(0.0, 0.151)], abort_energy=0.1)
    assert observer._swing_should_abort_calibration(
        [(float(i), 0.102) for i in range(12)],
        abort_energy=0.1,
    )


def test_swing_calibration_trial_plan_uses_current_latency_and_low_gain():
    config = create_default_config()
    config.swing_latency = 0.61
    config.swing_gain = 0.12
    config.swing_sign = 1.0
    config.swing_max_velocity = 0.12

    plan = _async_observer(config)._swing_calibration_trial_plan()

    assert len(plan) == 18
    assert {trial['sign'] for trial in plan} == {1.0, -1.0}
    assert {trial['latency'] for trial in plan} == {0.51, 0.61, 0.71}
    assert {trial['gain'] for trial in plan} == {0.006, 0.012, 0.02}
    assert {trial['max_velocity'] for trial in plan} == {0.03}


def test_swing_runtime_validation_accepts_sustained_damping():
    observer = _async_observer()
    samples = [(float(i), 0.2) for i in range(30)]
    samples.extend((float(i), 0.18) for i in range(30, 180))

    validation = observer._swing_runtime_validation(samples)

    assert validation['enough_samples']
    assert not validation['amplified']
    assert validation['damped']


def test_swing_runtime_validation_rejects_live_amplification_case():
    observer = _async_observer()
    samples = [(float(i), 0.146572) for i in range(30)]
    samples.extend((float(i), 0.219584) for i in range(30, 180))

    validation = observer._swing_runtime_validation(samples)

    assert validation['enough_samples']
    assert validation['amplified']
    assert not validation['damped']
    assert validation['runtime_ratio'] > 1.2


def test_swing_runtime_validation_treats_small_noise_as_not_damped_not_amplified():
    observer = _async_observer()
    samples = [(float(i), 0.2) for i in range(30)]
    samples.extend((float(i), 0.202) for i in range(30, 180))

    validation = observer._swing_runtime_validation(samples)

    assert validation['enough_samples']
    assert not validation['amplified']
    assert not validation['damped']


def test_swing_runtime_validation_accepts_quiet_energy_floor():
    observer = _async_observer()
    samples = [(float(i), 0.0051) for i in range(30)]
    samples.extend((float(i), 0.0056) for i in range(30, 180))

    validation = observer._swing_runtime_validation(samples)

    assert validation['enough_samples']
    assert validation['quiet']
    assert not validation['amplified']
    assert validation['damped']


def test_swing_runtime_validation_accepts_low_floor_noise_from_live_log():
    observer = _async_observer()
    samples = [(float(i), 0.000725) for i in range(30)]
    samples.extend((float(i), 0.010726) for i in range(30, 180))

    validation = observer._swing_runtime_validation(samples)

    assert validation['quiet']
    assert not validation['amplified']
    assert validation['damped']


def test_non_swing_velocity_norm_ignores_swing_key_and_detects_other_motion():
    observer = _async_observer()
    observer.active_set = {'default', 'swingc', 'manual'}
    observer.input_velocities = {
        'default': np.zeros(3),
        'swingc': np.array([1.0, 0.0, 0.0]),
        'manual': np.array([0.02, 0.0, 0.0]),
    }

    assert observer._non_swing_velocity_norm() == pytest.approx(0.02)


def test_sample_swing_energy_rejects_stale_imu_model():
    observer = _async_observer()
    client = _client()
    client.connected = True
    client.gripper_swing_model = np.array([[1.0, 0.0], [0.0, 1.0]])
    client.swing_model_host_ts = time.time() - 3.0
    observer.gripper_client = client

    with pytest.raises(RuntimeError, match='Swing IMU model is stale'):
        asyncio.run(observer._sample_swing_energy(0.01))


def test_swing_calibration_trial_rejects_stale_imu_model():
    observer = _async_observer()
    client = _client()
    client.connected = True
    client.swing_model_host_ts = time.time() - 3.0
    observer.gripper_client = client
    observer.active_set = set()
    observer.input_velocities = {'default': np.zeros(3)}

    async def move_direction_speed(*args, **kwargs):
        return np.zeros(3)

    observer.move_direction_speed = move_direction_speed

    with pytest.raises(RuntimeError, match='Swing IMU model is stale'):
        asyncio.run(observer._run_swing_calibration_trial(
            latency=0.1,
            sign=1.0,
            gain=0.006,
            max_velocity=0.03,
            duration_s=0.01,
            abort_energy=0.1,
        ))


def test_runtime_swing_cancellation_disables_on_stale_imu_model():
    observer = _async_observer()
    observer.run_command_loop = True
    observer.active_set = set()
    observer.input_velocities = {'default': np.zeros(3)}
    observer.slow_stop_all_spools = lambda: None
    fake_client = type('FakeSwingClient', (), {})()
    fake_client.reset_swing_correction_integrator = lambda: None
    fake_client.swing_model_age = lambda: 3.0
    fake_client.compute_calls = 0

    def compute_swing_correction(*args, **kwargs):
        fake_client.compute_calls += 1
        return np.array([0.01, 0.0])

    fake_client.compute_swing_correction = compute_swing_correction
    observer.gripper_client = fake_client

    asyncio.run(observer.run_swing_cancellation())

    assert fake_client.compute_calls == 0
    assert any(
        msg.get('pop_message') is not None
        and 'IMU model stopped updating' in msg['pop_message'].message
        for msg in observer.ui_messages
    )


def test_swing_calibration_finish_emits_completion_or_failure_telemetry():
    observer = _async_observer()

    result = observer._finish_swing_calibration(
        False,
        'Failed: validation amplified swing',
        'Swing calibration rejected the best setting.',
    )

    assert result is False
    progress = observer.ui_messages[0]['operation_progress']
    popup = observer.ui_messages[1]['pop_message']
    assert progress.percent_complete == 100
    assert progress.current_action == 'Failed: validation amplified swing'
    assert popup.message == 'Swing calibration rejected the best setting.'


def test_handle_command_ignores_malformed_control_batch():
    observer = _async_observer()

    asyncio.run(observer.handle_command(b'\x0a\x01\x80'))

    popup = observer.ui_messages[0]['pop_message']
    assert popup.message == 'Ignored malformed control command.'


def test_full_auto_calibration_integrates_swing_calibration_directly():
    source = inspect.getsource(AsyncObserver.full_auto_calibration)

    assert "await self.calibrate_spin(reset_wrist_first=True)" in source
    assert "swing_calibration_ok = await self.auto_calibrate_swing_cancellation()" in source
    assert "invoke_motion_task(self.auto_calibrate_swing_cancellation())" not in source


def test_full_auto_calibration_invokes_swing_after_spin_for_arpeggio(monkeypatch):
    config = create_default_config()
    config.anchor_type = common.AnchorType.ARPEGGIO
    for anchor in config.anchors[:2]:
        anchor.indirect_line = nf_config.IndirectLine(cam_tilt=22)
    observer = _async_observer(config)
    observer.anchors = {}
    for anchor_num in range(observer_module.N_ANCHORS[config.anchor_type]):
        anchor = type('Anchor', (), {})()
        anchor.save_raw = False
        anchor.origin_poses = {'origin': list(range(observer_module.max_origin_detections))}
        observer.anchors[anchor_num] = anchor

    class _PoolResult:
        def get(self, timeout=None):
            return np.zeros((2, 6)), np.zeros((2, 3))

    class _Pool:
        def apply_async(self, *args, **kwargs):
            return _PoolResult()

    observer.pool = _Pool()
    observer.pe = type('PE', (), {})()
    observer.config_path = None
    observer.gripper_client = _client()
    observer.gripper_client.connected = True

    async def send_commands(_commands):
        return None

    observer.gripper_client.send_commands = send_commands
    events = []
    observer.snapshot_tag_observations = lambda: {}
    observer.save_poses_arp = lambda *_args, **_kwargs: events.append('save_poses')

    async def half_auto_calibration():
        events.append('half')
        return True

    async def collect_eyelets(_anchor_poses):
        events.append('diamond')
        return {}, np.zeros(4)

    async def calibrate_finger_servo():
        events.append('finger')

    async def seek_gantry_goal():
        events.append('seek')

    async def calibrate_spin(reset_wrist_first=True):
        events.append(('spin', reset_wrist_first))

    async def auto_calibrate_swing_cancellation():
        events.append('swingcal')
        return True

    async def fast_sleep(_duration):
        return None

    observer.half_auto_calibration = half_auto_calibration
    observer.collect_arp_anchor_eyelet_experiment_data = collect_eyelets
    observer.calibrate_finger_servo = calibrate_finger_servo
    observer.seek_gantry_goal = seek_gantry_goal
    observer.calibrate_spin = calibrate_spin
    observer.auto_calibrate_swing_cancellation = auto_calibrate_swing_cancellation
    monkeypatch.setattr(observer_module.asyncio, 'sleep', fast_sleep)

    asyncio.run(observer.full_auto_calibration())

    assert ('spin', True) in events
    assert 'swingcal' in events
    assert events.index(('spin', True)) < events.index('swingcal')
    final_progress = [
        msg['operation_progress']
        for msg in observer.ui_messages
        if msg.get('operation_progress') is not None
    ][-1]
    assert final_progress.name == 'Calibration'
    assert 'Calibration completed' in final_progress.current_action


def test_full_auto_calibration_reports_swing_failure_in_final_progress():
    source = inspect.getsource(AsyncObserver.full_auto_calibration)

    assert "swing_calibration_ok is False" in source
    assert "swing cancellation calibration failed" in source


def test_auto_calibrate_swing_automatically_excites_low_baseline():
    source = inspect.getsource(AsyncObserver._auto_calibrate_swing_cancellation)

    assert "await self._induce_swing_for_calibration()" in source
    assert "Swing is too small to calibrate. Gently induce swing" not in source


def test_full_auto_calibration_guards_optional_finger_task():
    source = inspect.getsource(AsyncObserver.full_auto_calibration)

    assert "if finger_task is not None:\n                await finger_task" in source


def test_auto_calibrate_swing_common_command_is_generated():
    assert control.Command.AUTO_CALIBRATE_SWING == control.Command(19)
    assert (
        control.Command.betterproto_value_to_renamed_proto_names()[19]
        == "COMMAND_AUTO_CALIBRATE_SWING"
    )
    assert (
        control.Command.betterproto_renamed_proto_names_to_value()[
            "COMMAND_AUTO_CALIBRATE_SWING"
        ]
        == 19
    )
