import asyncio
import math

import pytest
import websockets

from nf_robot.generated.nf import common, control, telemetry
from nf_robot.ros2_bridge import (
    StringmanWebSocketClient,
    control_item_from_common_command,
    control_item_from_gripper_command,
    control_item_from_json,
    control_item_from_twist,
    item_payload_name,
    rodrigues_to_quaternion,
    sanitize_topic_token,
)


class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def test_common_command_names_accept_ros_style_aliases():
    item = control_item_from_common_command("stop_all")

    assert item.command.name == control.Command.STOP_ALL


def test_twist_maps_linear_velocity_and_wrist_rate():
    item = control_item_from_twist(_Vec(0.1, -0.2, 0.3), _Vec(z=math.pi), direction_is_in_gripper_frame=True)

    assert item.move.direction.x == 0.1
    assert item.move.direction.y == -0.2
    assert item.move.direction.z == 0.3
    assert item.move.wrist_speed == 180.0
    assert item.move.direction_is_in_gripper_frame is True


def test_gripper_command_maps_finger_wrist_and_winch():
    item = control_item_from_gripper_command([1.0, 2.0, 3.0])

    assert item.move.finger_speed == 1.0
    assert item.move.wrist_speed == 2.0
    assert item.move.winch == 3.0


def test_json_supports_every_major_control_family():
    cases = [
        ({"move": {"x": 1, "y": 2, "z": 3}}, "move"),
        ({"move_gripper_to": {"x": 1, "y": 2, "z": 3}}, "move_gripper_to"),
        ({"move_gripper_to": {"target_id": "target-a"}}, "move_gripper_to"),
        ({"gantry_goal_pos": {"x": 1, "y": 2, "z": 3}}, "gantry_goal_pos"),
        ({"jog_spool": {"is_gripper": False, "anchor_num": 1, "speed": 0.1}}, "jog_spool"),
        ({"episode_control": {"command": "eval_start", "prompt": "pick up cup"}}, "episode_control"),
        ({"scale_room": {"scale": 1.01, "tiltcams": 0.0}}, "scale_room"),
        ({"add_cam_target": {"anchor_num": 0, "img_norm_x": 0.5, "img_norm_y": 0.4}}, "add_cam_target"),
        ({"delete_target": "target-a"}, "delete_target"),
        ({"debug": "dump_state"}, "debug"),
        ({"set_swing_cancellation": True}, "set_swing_cancellation"),
        ({"single_component_action": {"is_gripper": True, "action": "identify"}}, "single_component_action"),
        ({"manage_lerobot_session": {"action": "start_record", "repo_id": "user/ds"}}, "manage_lerobot_session"),
    ]

    for payload, expected_field in cases:
        item = control_item_from_json(payload)
        assert getattr(item, expected_field) is not None


def test_json_control_round_trips_as_control_batch_bytes():
    item = control_item_from_json({"command": "park"})
    batch = control.ControlBatchUpdate(robot_id="ros2", updates=[item])

    parsed = control.ControlBatchUpdate().parse(bytes(batch))

    assert parsed.robot_id == "ros2"
    assert parsed.updates[0].command.name == control.Command.PARK


def test_telemetry_payload_detection_uses_oneof_presence_not_truthiness():
    item = telemetry.TelemetryItem(grip_sensors=telemetry.GripperSensors())

    assert item_payload_name(item) == "grip_sensors"


def test_rodrigues_to_quaternion_converts_z_rotation():
    qx, qy, qz, qw = rodrigues_to_quaternion(common.Vec3(z=math.pi / 2))

    assert qx == 0.0
    assert qy == 0.0
    assert qz == pytest.approx(math.sin(math.pi / 4))
    assert qw == pytest.approx(math.cos(math.pi / 4))


def test_sanitize_topic_token_keeps_ros_topic_names_valid():
    assert sanitize_topic_token("parking location") == "parking_location"
    assert sanitize_topic_token("123 target") == "item_123_target"


def test_websocket_client_receives_telemetry_and_sends_controls():
    asyncio.run(_exercise_websocket_client())


async def _exercise_websocket_client():
    received_batches = []
    received_controls = []
    batch_received = asyncio.Event()
    control_received = asyncio.Event()

    async def handler(websocket):
        batch = telemetry.TelemetryBatchUpdate(
            robot_id="fake",
            updates=[telemetry.TelemetryItem(pop_message=telemetry.Popup(message="hello"))],
        )
        await websocket.send(bytes(batch))
        message = await asyncio.wait_for(websocket.recv(), timeout=2)
        received_controls.append(control.ControlBatchUpdate().parse(message))
        control_received.set()

    server = await websockets.serve(handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]

        def on_batch(batch, _raw):
            received_batches.append(batch)
            batch_received.set()

        client = StringmanWebSocketClient(
            f"ws://127.0.0.1:{port}",
            robot_id="ros2-test",
            on_batch=on_batch,
            reconnect_delay=0.05,
        )
        task = asyncio.create_task(client.run_forever())
        await asyncio.wait_for(batch_received.wait(), timeout=2)

        client.queue_control_item(control_item_from_common_command("stop_all"))
        await asyncio.wait_for(control_received.wait(), timeout=2)

        client.stop()
        await asyncio.wait_for(task, timeout=2)
    finally:
        server.close()
        await server.wait_closed()

    assert received_batches[0].updates[0].pop_message.message == "hello"
    assert received_controls[0].robot_id == "ros2-test"
    assert received_controls[0].updates[0].command.name == control.Command.STOP_ALL
