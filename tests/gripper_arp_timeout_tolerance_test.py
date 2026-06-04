import asyncio
import importlib
import sys
import types

import pytest


def _module(name, **attrs):
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


class _FakeComponentServer:
    def __init__(self):
        self.conf = {}
        self.run_server = True
        self.update = {}
        self.spooler = None
        self.reset_wifi_event = None


class _FakeRangefinder:
    def __init__(self, _i2c):
        self.model_info = (1, 2, 3)
        self.data_ready = False
        self.distance = None
        self.distance_mode = None

    def start_ranging(self):
        pass

    def clear_interrupt(self):
        pass


class _FakePressureSensor:
    def __init__(self, _ads, _pin):
        self.voltage = 3.3


class _FakeMotors:
    def __init__(self):
        self.feedback = {
            1: {"position": 0, "speed": 0, "load": 0, "voltage": 7.4, "temp": 30, "moving": 0},
            2: {"position": 2048, "speed": 0, "load": 0, "voltage": 7.4, "temp": 30, "moving": 0},
        }
        self.fail_next = {}
        self.feedback_calls = {}
        self.positions = []
        self.torque = []
        self.reset_midpoint_calls = []

    def configure_multiturn(self, _motor_id):
        pass

    def reset_encoder_to_midpoint(self, motor_id):
        self.reset_midpoint_calls.append(motor_id)

    def torque_enable(self, motor_id, enabled):
        self.torque.append((motor_id, enabled))

    def set_position(self, motor_id, position):
        self.positions.append((motor_id, position))

    def get_feedback(self, motor_id):
        self.feedback_calls[motor_id] = self.feedback_calls.get(motor_id, 0) + 1
        exc = self.fail_next.pop(motor_id, None)
        if exc is not None:
            raise exc
        return dict(self.feedback[motor_id])


class _PID:
    def __init__(self, kp, ki, kd, sample_rate):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = 0
        self._error_sum = 0
        self._last_value = 0
        self._sample_rate = sample_rate

    def calculate(self, value, value_change=None):
        error = self.setpoint - value
        self._error_sum += error * self._sample_rate
        rate_error = value_change
        if rate_error is None:
            rate_error = (value - self._last_value) / self._sample_rate
        self._last_value = value
        return (error * self.kp) + (self._error_sum * self.ki) - (rate_error * self.kd)


def _clamp(value, small, big):
    return max(min(value, big), small)


def _remap(value, ilow, ihigh, olow, ohigh):
    return (value - ilow) / (ihigh - ilow) * (ohigh - olow) + olow


@pytest.fixture()
def gripper_server_module(monkeypatch):
    motors = _FakeMotors()

    anchor_server = _module("nf_robot.robot.anchor_server", RobotComponentServer=_FakeComponentServer)
    simple_st3215 = _module("nf_robot.robot.simple_st3215", SimpleSTS3215=lambda: motors)
    util = _module("nf_robot.common.util", remap=_remap, clamp=_clamp, PID=_PID)

    monkeypatch.setitem(sys.modules, "nf_robot.robot.anchor_server", anchor_server)
    monkeypatch.setitem(sys.modules, "nf_robot.robot.simple_st3215", simple_st3215)
    monkeypatch.setitem(sys.modules, "nf_robot.common.util", util)
    monkeypatch.setitem(sys.modules, "getmac", _module("getmac", get_mac_address=lambda: "00:11:22:33:44:55"))
    monkeypatch.setitem(sys.modules, "board", _module("board", SCL=object(), SDA=object()))
    monkeypatch.setitem(sys.modules, "busio", _module("busio", I2C=lambda _scl, _sda: object()))
    monkeypatch.setitem(sys.modules, "adafruit_mpu6050", _module("adafruit_mpu6050", MPU6050=lambda _i2c: object()))
    monkeypatch.setitem(sys.modules, "adafruit_vl53l1x", _module("adafruit_vl53l1x", VL53L1X=_FakeRangefinder))
    monkeypatch.setitem(
        sys.modules,
        "adafruit_ads1x15",
        _module(
            "adafruit_ads1x15",
            ADS1015=lambda _i2c: object(),
            AnalogIn=_FakePressureSensor,
            ads1x15=types.SimpleNamespace(Pin=types.SimpleNamespace(A0=0)),
        ),
    )

    monkeypatch.delitem(sys.modules, "nf_robot.robot.gripper_arp_server", raising=False)
    module = importlib.import_module("nf_robot.robot.gripper_arp_server")
    return module, motors


@pytest.fixture()
def server(gripper_server_module, monkeypatch, tmp_path):
    module, motors = gripper_server_module
    monkeypatch.chdir(tmp_path)
    return module.GripperArpServer(), motors, module


def test_read_other_sensors_uses_stale_wrist_and_finger_state_on_wrist_timeout(server):
    gripper, motors, module = server
    gripper.saved_unrolled_wrist_angle = 270
    gripper.desired_wrist_angle = 270
    motors.feedback[module.WRIST]["position"] = int(270 / 360 * module.STEPS_PER_REV)
    gripper.last_finger_data = {"position": 500, "speed": 0, "load": 0, "voltage": 7.4, "temp": 30, "moving": 0}

    assert gripper.getWristAngle() == pytest.approx(270)

    motors.fail_next[module.WRIST] = TimeoutError("single dropped wrist feedback frame")
    gripper.readOtherSensors()

    sensors = gripper.update["grip_sensors"]
    assert sensors["wrist_a"] == pytest.approx(270)
    assert sensors["fing_a"] == pytest.approx(45)


async def _run_update_motors_timeout_check(gripper, motors, module):
    task = asyncio.create_task(gripper.updateMotors())
    try:
        await asyncio.sleep(0)

        assert motors.feedback_calls[module.FINGER] == 1
        assert not task.done()
        assert gripper.filtered_force == pytest.approx(0.0225)
    finally:
        gripper.run_server = False
        await asyncio.wait_for(task, timeout=1)


def test_update_motors_uses_stale_finger_force_on_single_finger_timeout(server):
    gripper, motors, module = server
    stale_finger_data = {
        "position": 250,
        "speed": 0,
        "load": 250,
        "voltage": 7.4,
        "temp": 30,
        "moving": 0,
    }
    gripper.last_finger_data = dict(stale_finger_data)
    motors.fail_next[module.FINGER] = TimeoutError("single dropped finger feedback frame")

    asyncio.run(_run_update_motors_timeout_check(gripper, motors, module))
    assert gripper.last_finger_data == stale_finger_data


async def _run_reset_wrist_timeout_check(gripper, motors, module):
    motors.fail_next[module.WRIST] = TimeoutError("single dropped wrist feedback frame after reset")
    await gripper.resetWrist()


def test_reset_wrist_clears_motor_pause_on_single_post_reset_timeout(server, monkeypatch):
    gripper, motors, module = server
    now = 0

    async def fast_sleep(delay):
        nonlocal now
        now += delay

    def fake_time():
        nonlocal now
        now += 1
        return now

    monkeypatch.setattr(module.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(module.time, "time", fake_time)
    monkeypatch.setattr(gripper, "getWristAngle", lambda: 538.0)

    asyncio.run(_run_reset_wrist_timeout_check(gripper, motors, module))

    assert motors.reset_midpoint_calls == [module.WRIST]
    assert not gripper.motor_loop_pause
    assert gripper.last_simple_wrist_angle == pytest.approx(180)
    assert gripper.unrolled_wrist_angle == pytest.approx(180)
    assert gripper.desired_wrist_angle == pytest.approx(540)
