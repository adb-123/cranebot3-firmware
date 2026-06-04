from __future__ import annotations

import asyncio
import json
import logging
import math
import queue
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from nf_robot.generated.nf import common, control, telemetry

logger = logging.getLogger(__name__)

TELEMETRY_PAYLOAD_FIELDS = (
    "pos_estimate",
    "pos_factors_debug",
    "gantry_sightings",
    "new_anchor_poses",
    "component_conn_status",
    "vid_stats",
    "named_position",
    "last_commanded_vel",
    "raw_commanded_vel",
    "pop_message",
    "grip_sensors",
    "grip_cam_preditions",
    "target_list",
    "video_ready",
    "uplink_status",
    "episode_control",
    "operation_progress",
    "last_commanded_grip",
    "swing_cancellation_state",
    "visibility_states",
)


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> common.Vec3:
    return common.Vec3(x=float(x), y=float(y), z=float(z))


def vec3_to_list(value: common.Vec3 | None) -> list[float]:
    if value is None:
        return [0.0, 0.0, 0.0]
    return [float(value.x), float(value.y), float(value.z)]


def item_payload_name(item: telemetry.TelemetryItem) -> str | None:
    for field in TELEMETRY_PAYLOAD_FIELDS:
        if getattr(item, field) is not None:
            return field
    return None


def message_to_json(message: Any) -> str:
    if hasattr(message, "to_dict"):
        return json.dumps(message.to_dict(), separators=(",", ":"), sort_keys=True)
    return json.dumps(message, separators=(",", ":"), sort_keys=True)


def sanitize_topic_token(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", value.strip()).strip("_").lower()
    safe = re.sub(r"_+", "_", safe)
    if not safe:
        return "unnamed"
    if not re.match(r"^[A-Za-z]", safe):
        return f"item_{safe}"
    return safe


def rodrigues_to_quaternion(rotation: common.Vec3 | None) -> tuple[float, float, float, float]:
    if rotation is None:
        return (0.0, 0.0, 0.0, 1.0)
    x, y, z = vec3_to_list(rotation)
    theta = math.sqrt(x * x + y * y + z * z)
    if theta < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    scale = math.sin(theta / 2.0) / theta
    return (x * scale, y * scale, z * scale, math.cos(theta / 2.0))


def _normalize_key(value: str) -> str:
    value = value.strip().upper()
    value = re.sub(r"^(COMMAND|COMPONENTACTION|LEROBOTSESSIONACTION|EPCOMMAND)_", "", value)
    value = re.sub(r"[^A-Z0-9]+", "_", value)
    return value.strip("_")


def _enum_from_name(enum_cls: type, value: Any, *, allow_zero: bool = False) -> Any:
    if isinstance(value, enum_cls):
        result = value
    else:
        key = _normalize_key(str(value))
        result = None
        for member in enum_cls:
            if _normalize_key(member.name) == key:
                result = member
                break
    if result is None:
        allowed = ", ".join(member.name.lower() for member in enum_cls if allow_zero or int(member) != 0)
        raise ValueError(f"Unknown {enum_cls.__name__} value {value!r}. Expected one of: {allowed}")
    if not allow_zero and int(result) == 0:
        raise ValueError(f"{enum_cls.__name__}.{result.name} is not a usable command")
    return result


def control_item_from_common_command(name: str) -> control.ControlItem:
    command_name = _enum_from_name(control.Command, name)
    return control.ControlItem(command=control.CommonCommand(name=command_name))


def control_item_from_twist(
    linear: Any,
    angular: Any | None = None,
    *,
    direction_is_in_gripper_frame: bool = False,
) -> control.ControlItem:
    wrist_speed = None
    if angular is not None:
        wrist_speed = math.degrees(float(getattr(angular, "z", 0.0)))
    return control.ControlItem(
        move=control.CombinedMove(
            direction=vec3(
                getattr(linear, "x", 0.0),
                getattr(linear, "y", 0.0),
                getattr(linear, "z", 0.0),
            ),
            wrist_speed=wrist_speed,
            direction_is_in_gripper_frame=direction_is_in_gripper_frame,
        )
    )


def control_item_from_gripper_command(values: Sequence[float]) -> control.ControlItem:
    if len(values) == 0:
        raise ValueError("gripper command requires at least finger_speed")
    move = control.CombinedMove(finger_speed=float(values[0]))
    if len(values) > 1:
        move.wrist_speed = float(values[1])
    if len(values) > 2:
        move.winch = float(values[2])
    return control.ControlItem(move=move)


def control_item_from_move_gripper_point(point: Any) -> control.ControlItem:
    return control.ControlItem(
        move_gripper_to=control.MoveGripperTo(
            pos=vec3(getattr(point, "x", 0.0), getattr(point, "y", 0.0), getattr(point, "z", 0.0))
        )
    )


def control_item_from_set_swing(enabled: bool) -> control.ControlItem:
    return control.ControlItem(
        set_swing_cancellation=control.SetSwingCancellation(enabled=bool(enabled), present=".")
    )


def control_item_from_debug(action: str) -> control.ControlItem:
    return control.ControlItem(debug=control.Debug(action=str(action)))


def _mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _coords(data: Mapping[str, Any], *, default_z: float = 0.0) -> common.Vec3:
    if "direction" in data and isinstance(data["direction"], Mapping):
        data = data["direction"]
    return vec3(data.get("x", 0.0), data.get("y", 0.0), data.get("z", default_z))


def _optional_float(data: Mapping[str, Any], key: str) -> float | None:
    if key not in data or data[key] is None:
        return None
    return float(data[key])


def control_item_from_json(payload: str | Mapping[str, Any]) -> control.ControlItem:
    data = json.loads(payload) if isinstance(payload, str) else payload
    data = _mapping(data, "control payload")

    if "command" in data:
        return control_item_from_common_command(str(data["command"]))

    if "move" in data:
        move_data = _mapping(data["move"], "move")
        return control.ControlItem(
            move=control.CombinedMove(
                direction=_coords(move_data) if ("direction" in move_data or {"x", "y", "z"} & set(move_data)) else None,
                speed=_optional_float(move_data, "speed"),
                winch=_optional_float(move_data, "winch"),
                wrist_speed=_optional_float(move_data, "wrist_speed"),
                finger_speed=_optional_float(move_data, "finger_speed"),
                direction_is_in_gripper_frame=move_data.get("direction_is_in_gripper_frame"),
            )
        )

    if "gripper_cmd" in data:
        value = data["gripper_cmd"]
        if isinstance(value, Mapping):
            return control.ControlItem(
                move=control.CombinedMove(
                    finger_speed=_optional_float(value, "finger_speed"),
                    wrist_speed=_optional_float(value, "wrist_speed"),
                    winch=_optional_float(value, "winch"),
                )
            )
        return control_item_from_gripper_command(value)

    if "move_gripper_to" in data:
        move_data = data["move_gripper_to"]
        if isinstance(move_data, str):
            return control.ControlItem(move_gripper_to=control.MoveGripperTo(target_id=move_data))
        move_data = _mapping(move_data, "move_gripper_to")
        if "target_id" in move_data:
            return control.ControlItem(move_gripper_to=control.MoveGripperTo(target_id=str(move_data["target_id"])))
        return control.ControlItem(move_gripper_to=control.MoveGripperTo(pos=_coords(move_data)))

    if "gantry_goal_pos" in data:
        return control.ControlItem(
            gantry_goal_pos=control.GantryGoalPos(pos=_coords(_mapping(data["gantry_goal_pos"], "gantry_goal_pos")))
        )

    if "jog_spool" in data:
        jog_data = _mapping(data["jog_spool"], "jog_spool")
        jog = control.JogSpool(
            is_gripper=bool(jog_data.get("is_gripper", False)),
            anchor_num=int(jog_data.get("anchor_num", 0)),
        )
        if "speed" in jog_data:
            jog.speed = float(jog_data["speed"])
        elif "offset" in jog_data:
            jog.offset = float(jog_data["offset"])
        else:
            raise ValueError("jog_spool requires speed or offset")
        return control.ControlItem(jog_spool=jog)

    if "episode_control" in data:
        ep_data = _mapping(data["episode_control"], "episode_control")
        ep = common.EpisodeControl()
        if "command" in ep_data:
            ep.command = _enum_from_name(common.EpCommand, ep_data["command"], allow_zero=True)
        if "prompt" in ep_data:
            ep.prompt = str(ep_data["prompt"])
        return control.ControlItem(episode_control=ep)

    if "scale_room" in data:
        scale_data = _mapping(data["scale_room"], "scale_room")
        return control.ControlItem(
            scale_room=control.ScaleRoom(
                scale=float(scale_data.get("scale", 1.0)),
                tiltcams=float(scale_data.get("tiltcams", 0.0)),
            )
        )

    if "add_cam_target" in data:
        target_data = _mapping(data["add_cam_target"], "add_cam_target")
        return control.ControlItem(
            add_cam_target=control.AddTargetFromAnchorCam(
                anchor_num=int(target_data["anchor_num"]),
                img_norm_x=float(target_data["img_norm_x"]),
                img_norm_y=float(target_data["img_norm_y"]),
                target_id=target_data.get("target_id"),
            )
        )

    if "delete_target" in data:
        target = data["delete_target"]
        if isinstance(target, Mapping):
            target = target["target_id"]
        return control.ControlItem(delete_target=control.DeleteTarget(target_id=str(target)))

    if "debug" in data:
        debug = data["debug"]
        if isinstance(debug, Mapping):
            debug = debug["action"]
        return control_item_from_debug(str(debug))

    if "set_swing_cancellation" in data:
        value = data["set_swing_cancellation"]
        enabled = value.get("enabled", False) if isinstance(value, Mapping) else value
        return control_item_from_set_swing(bool(enabled))

    if "single_component_action" in data or "component_action" in data:
        action_data = _mapping(
            data.get("single_component_action", data.get("component_action")),
            "single_component_action",
        )
        return control.ControlItem(
            single_component_action=control.SingleComponentAction(
                is_gripper=bool(action_data.get("is_gripper", False)),
                anchor_num=action_data.get("anchor_num"),
                action=_enum_from_name(control.ComponentAction, action_data["action"]),
                spool_num=action_data.get("spool_num"),
                cam_angle=action_data.get("cam_angle"),
            )
        )

    if "manage_lerobot_session" in data:
        session_data = _mapping(data["manage_lerobot_session"], "manage_lerobot_session")
        return control.ControlItem(
            manage_lerobot_session=control.ManageLerobotSession(
                action=_enum_from_name(control.LerobotSessionAction, session_data["action"]),
                repo_id=str(session_data["repo_id"]),
                suppress_upload=bool(session_data.get("suppress_upload", False)),
            )
        )

    raise ValueError(f"Unsupported control JSON keys: {', '.join(sorted(data.keys()))}")


class StringmanWebSocketClient:
    def __init__(
        self,
        uri: str,
        *,
        robot_id: str = "ros2",
        on_batch: Callable[[telemetry.TelemetryBatchUpdate, bytes], None],
        on_status: Callable[[str], None] | None = None,
        reconnect_delay: float = 2.0,
        open_timeout: float = 10.0,
    ) -> None:
        self.uri = uri
        self.robot_id = robot_id
        self.on_batch = on_batch
        self.on_status = on_status or (lambda status: None)
        self.reconnect_delay = reconnect_delay
        self.open_timeout = open_timeout
        self._outgoing: queue.Queue[control.ControlItem] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def queue_control_item(self, item: control.ControlItem) -> None:
        self._outgoing.put(item)

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=lambda: asyncio.run(self.run_forever()), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    async def run_forever(self) -> None:
        while not self._stop.is_set():
            send_task: asyncio.Task | None = None
            try:
                self.on_status(f"connecting {self.uri}")
                async with websockets.connect(self.uri, max_size=None, open_timeout=self.open_timeout) as websocket:
                    self.on_status(f"connected {self.uri}")
                    send_task = asyncio.create_task(self._send_loop(websocket))
                    async for message in websocket:
                        if isinstance(message, str):
                            message = message.encode()
                        batch = telemetry.TelemetryBatchUpdate().parse(message)
                        self.on_batch(batch, bytes(message))
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as exc:
                self.on_status(f"disconnected {exc.code} {exc.reason}".strip())
            except Exception as exc:
                logger.exception("Stringman websocket bridge error")
                self.on_status(f"error {type(exc).__name__}: {exc}")
            finally:
                if send_task is not None:
                    send_task.cancel()
                    await asyncio.gather(send_task, return_exceptions=True)
            if not self._stop.is_set():
                await asyncio.sleep(self.reconnect_delay)
        self.on_status("stopped")

    async def _send_loop(self, websocket: Any) -> None:
        while not self._stop.is_set():
            try:
                item = await asyncio.to_thread(self._outgoing.get, True, 0.2)
            except queue.Empty:
                continue
            batch = control.ControlBatchUpdate(robot_id=self.robot_id, updates=[item])
            await websocket.send(bytes(batch))
