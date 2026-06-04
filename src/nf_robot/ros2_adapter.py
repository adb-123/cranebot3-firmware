from __future__ import annotations

import json
import math
import queue
from typing import Any

from nf_robot.generated.nf import common, telemetry
from nf_robot.ros2_bridge import (
    StringmanWebSocketClient,
    control_item_from_common_command,
    control_item_from_debug,
    control_item_from_gripper_command,
    control_item_from_json,
    control_item_from_move_gripper_point,
    control_item_from_set_swing,
    control_item_from_twist,
    item_payload_name,
    message_to_json,
    rodrigues_to_quaternion,
    sanitize_topic_token,
    vec3_to_list,
)

try:
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped, Twist, TwistStamped
    from rclpy.node import Node
    from sensor_msgs.msg import Range
    from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32MultiArray, String, UInt8MultiArray
    from std_srvs.srv import SetBool, Trigger

    try:
        from tf2_ros import TransformBroadcaster
    except ImportError:
        TransformBroadcaster = None

    ROS2_IMPORT_ERROR = None
except ImportError as exc:
    rclpy = None
    Node = object
    DiagnosticArray = DiagnosticStatus = KeyValue = None
    PointStamped = PoseStamped = TransformStamped = Twist = TwistStamped = None
    Range = None
    Bool = Float32 = Float32MultiArray = Int32MultiArray = String = UInt8MultiArray = None
    SetBool = Trigger = None
    TransformBroadcaster = None
    ROS2_IMPORT_ERROR = exc


COMMON_COMMAND_SERVICE_NAMES = {
    "half_cal": "half_cal",
    "full_cal": "full_cal",
    "auto_calibrate_swing": "auto_calibrate_swing",
    "zero_winch": "zero_winch",
    "stop_all": "stop_all",
    "enable_lerobot": "enable_lerobot",
    "pick_and_drop": "pick_and_drop",
    "record_park": "record_park",
    "park": "park",
    "unpark": "unpark",
    "grasp": "grasp",
    "submit_targets_to_dataset": "submit_targets_to_dataset",
    "tighten_lines": "tighten_lines",
    "disable_torque": "disable_torque",
    "enable_torque": "enable_torque",
    "horizontal_check": "horizontal_check",
    "collect_gripper_images": "collect_gripper_images",
    "shutdown": "shutdown",
    "update_firmware": "update_firmware",
}


class StringmanRos2Adapter(Node):
    def __init__(self) -> None:
        if ROS2_IMPORT_ERROR is not None:
            raise RuntimeError(
                "ROS2 Python packages are not importable. Source a ROS2 environment "
                "before running stringman-ros2-adapter, for example: "
                "source /opt/ros/jazzy/setup.bash"
            ) from ROS2_IMPORT_ERROR

        super().__init__("stringman_ros2_adapter")
        self.declare_parameter("server_uri", "ws://127.0.0.1:4245")
        self.declare_parameter("robot_id", "ros2")
        self.declare_parameter("topic_prefix", "stringman")
        self.declare_parameter("world_frame_id", "stringman_world")
        self.declare_parameter("gantry_frame_id", "stringman_gantry")
        self.declare_parameter("gripper_frame_id", "stringman_gripper")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("reconnect_delay", 2.0)
        self.declare_parameter("command_velocity_is_gripper_frame", False)
        self.declare_parameter("telemetry_queue_size", 500)

        self.topic_prefix = self.get_parameter("topic_prefix").value.strip("/")
        self.world_frame_id = self.get_parameter("world_frame_id").value
        self.gantry_frame_id = self.get_parameter("gantry_frame_id").value
        self.gripper_frame_id = self.get_parameter("gripper_frame_id").value
        self.command_velocity_is_gripper_frame = bool(
            self.get_parameter("command_velocity_is_gripper_frame").value
        )
        telemetry_queue_size = int(self.get_parameter("telemetry_queue_size").value)
        self.telemetry_queue: queue.Queue[tuple[telemetry.TelemetryBatchUpdate, bytes]] = queue.Queue(
            maxsize=max(1, telemetry_queue_size)
        )
        self.status_queue: queue.Queue[str] = queue.Queue(maxsize=50)

        self.named_position_publishers: dict[str, Any] = {}
        self.anchor_pose_publishers: dict[int, Any] = {}
        self.eyelet_publishers: dict[int, Any] = {}

        self._build_publishers()
        self._build_subscribers()
        self._build_services()

        self.tf_broadcaster = None
        if bool(self.get_parameter("publish_tf").value) and TransformBroadcaster is not None:
            self.tf_broadcaster = TransformBroadcaster(self)

        self.client = StringmanWebSocketClient(
            self.get_parameter("server_uri").value,
            robot_id=self.get_parameter("robot_id").value,
            on_batch=self._enqueue_telemetry,
            on_status=self._enqueue_status,
            reconnect_delay=float(self.get_parameter("reconnect_delay").value),
        )
        self.create_timer(0.02, self._drain_telemetry)
        self.create_timer(0.2, self._drain_status)

    def start(self) -> None:
        self.client.start_background()

    def stop(self) -> None:
        self.client.stop()
        self.client.join(timeout=2.0)

    def _topic(self, suffix: str) -> str:
        suffix = suffix.strip("/")
        if not self.topic_prefix:
            return suffix
        return f"{self.topic_prefix}/{suffix}"

    def _build_publishers(self) -> None:
        qos = 10
        self.raw_json_pub = self.create_publisher(String, self._topic("telemetry/raw_json"), qos)
        self.raw_protobuf_pub = self.create_publisher(UInt8MultiArray, self._topic("telemetry/raw_protobuf"), qos)
        self.status_pub = self.create_publisher(String, self._topic("telemetry/status"), qos)
        self.payload_type_pub = self.create_publisher(String, self._topic("telemetry/payload_type"), qos)

        self.gantry_pose_pub = self.create_publisher(PointStamped, self._topic("pose/gantry"), qos)
        self.gantry_twist_pub = self.create_publisher(TwistStamped, self._topic("twist/gantry"), qos)
        self.gripper_pose_pub = self.create_publisher(PoseStamped, self._topic("pose/gripper"), qos)
        self.tension_pub = self.create_publisher(Float32MultiArray, self._topic("tension"), qos)
        self.slack_pub = self.create_publisher(Int32MultiArray, self._topic("slack"), qos)

        self.position_factors_pub = self.create_publisher(String, self._topic("position_factors"), qos)
        self.visual_pose_pub = self.create_publisher(PointStamped, self._topic("position_factors/visual"), qos)
        self.hanging_pose_pub = self.create_publisher(PointStamped, self._topic("position_factors/hanging"), qos)
        self.position_spin_pub = self.create_publisher(Float32, self._topic("position_factors/spin"), qos)
        self.gantry_sightings_pub = self.create_publisher(Float32MultiArray, self._topic("gantry_sightings"), qos)
        self.anchor_poses_pub = self.create_publisher(String, self._topic("anchors/poses"), qos)
        self.component_status_pub = self.create_publisher(DiagnosticArray, self._topic("components/status"), qos)
        self.component_status_json_pub = self.create_publisher(String, self._topic("components/status_json"), qos)
        self.video_stats_pub = self.create_publisher(Float32MultiArray, self._topic("video/stats"), qos)
        self.video_ready_pub = self.create_publisher(String, self._topic("video/ready"), qos)
        self.named_positions_pub = self.create_publisher(String, self._topic("named_positions"), qos)
        self.commanded_velocity_pub = self.create_publisher(TwistStamped, self._topic("commanded_velocity"), qos)
        self.raw_commanded_velocity_pub = self.create_publisher(
            TwistStamped, self._topic("raw_commanded_velocity"), qos
        )
        self.popup_pub = self.create_publisher(String, self._topic("popup"), qos)
        self.grip_sensors_pub = self.create_publisher(Float32MultiArray, self._topic("gripper/sensors"), qos)
        self.grip_range_pub = self.create_publisher(Range, self._topic("gripper/range"), qos)
        self.grip_angle_pub = self.create_publisher(Float32, self._topic("gripper/angle"), qos)
        self.grip_pressure_pub = self.create_publisher(Float32, self._topic("gripper/pressure"), qos)
        self.grip_wrist_pub = self.create_publisher(Float32, self._topic("gripper/wrist"), qos)
        self.grip_target_force_pub = self.create_publisher(Float32, self._topic("gripper/target_force"), qos)
        self.grip_predictions_pub = self.create_publisher(
            Float32MultiArray, self._topic("gripper/camera_predictions"), qos
        )
        self.targets_pub = self.create_publisher(String, self._topic("targets"), qos)
        self.uplink_pub = self.create_publisher(Bool, self._topic("uplink/online"), qos)
        self.episode_control_pub = self.create_publisher(String, self._topic("episode_control"), qos)
        self.operation_progress_pub = self.create_publisher(String, self._topic("operation_progress"), qos)
        self.commanded_grip_pub = self.create_publisher(Float32MultiArray, self._topic("gripper/last_commanded"), qos)
        self.swing_cancellation_pub = self.create_publisher(Bool, self._topic("swing_cancellation_state"), qos)
        self.visibility_states_pub = self.create_publisher(String, self._topic("visibility_states"), qos)

    def _build_subscribers(self) -> None:
        qos = 10
        self.create_subscription(Twist, self._topic("cmd_vel"), self._on_cmd_vel, qos)
        self.create_subscription(Float32MultiArray, self._topic("gripper/cmd"), self._on_gripper_cmd, qos)
        self.create_subscription(String, self._topic("command"), self._on_common_command, qos)
        self.create_subscription(String, self._topic("control/json"), self._on_control_json, qos)
        self.create_subscription(PointStamped, self._topic("move_gripper_to"), self._on_move_gripper_to, qos)
        self.create_subscription(String, self._topic("debug"), self._on_debug, qos)
        self.create_subscription(String, self._topic("episode_prompt"), self._on_episode_prompt, qos)

    def _build_services(self) -> None:
        for service_name, command_name in COMMON_COMMAND_SERVICE_NAMES.items():
            self.create_service(Trigger, self._topic(f"services/{service_name}"), self._trigger_common(command_name))
        self.create_service(SetBool, self._topic("services/swing_cancellation"), self._set_swing_service)

    def _enqueue_telemetry(self, batch: telemetry.TelemetryBatchUpdate, raw: bytes) -> None:
        try:
            self.telemetry_queue.put_nowait((batch, raw))
        except queue.Full:
            try:
                self.telemetry_queue.get_nowait()
            except queue.Empty:
                pass
            self.telemetry_queue.put_nowait((batch, raw))

    def _enqueue_status(self, status: str) -> None:
        try:
            self.status_queue.put_nowait(status)
        except queue.Full:
            pass

    def _drain_status(self) -> None:
        while True:
            try:
                status = self.status_queue.get_nowait()
            except queue.Empty:
                return
            self.status_pub.publish(String(data=status))
            self.get_logger().info(status)

    def _drain_telemetry(self) -> None:
        handled = 0
        while handled < 25:
            try:
                batch, raw = self.telemetry_queue.get_nowait()
            except queue.Empty:
                return
            self._publish_batch(batch, raw)
            handled += 1

    def _publish_batch(self, batch: telemetry.TelemetryBatchUpdate, raw: bytes) -> None:
        self.raw_json_pub.publish(String(data=message_to_json(batch)))
        raw_msg = UInt8MultiArray()
        raw_msg.data = list(raw)
        self.raw_protobuf_pub.publish(raw_msg)
        for item in batch.updates:
            payload = item_payload_name(item)
            if payload is None:
                continue
            self.payload_type_pub.publish(String(data=payload))
            self._publish_item(item, payload)

    def _publish_item(self, item: telemetry.TelemetryItem, payload: str) -> None:
        if payload == "pos_estimate":
            self._publish_pos_estimate(item.pos_estimate)
        elif payload == "pos_factors_debug":
            self._publish_pos_factors(item.pos_factors_debug)
        elif payload == "gantry_sightings":
            self._publish_gantry_sightings(item.gantry_sightings)
        elif payload == "new_anchor_poses":
            self._publish_anchor_poses(item.new_anchor_poses)
        elif payload == "component_conn_status":
            self._publish_component_status(item.component_conn_status)
        elif payload == "vid_stats":
            self.video_stats_pub.publish(self._float_array([
                item.vid_stats.detection_rate,
                item.vid_stats.video_latency,
                item.vid_stats.video_framerate,
            ]))
        elif payload == "named_position":
            self._publish_named_position(item.named_position)
        elif payload == "last_commanded_vel":
            self.commanded_velocity_pub.publish(self._twist_stamped(item.last_commanded_vel.velocity))
        elif payload == "raw_commanded_vel":
            self.raw_commanded_velocity_pub.publish(self._twist_stamped(item.raw_commanded_vel.velocity))
        elif payload == "pop_message":
            self.popup_pub.publish(String(data=item.pop_message.message))
        elif payload == "grip_sensors":
            self._publish_gripper_sensors(item.grip_sensors)
        elif payload == "grip_cam_preditions":
            self.grip_predictions_pub.publish(self._float_array([
                item.grip_cam_preditions.move_x,
                item.grip_cam_preditions.move_y,
                item.grip_cam_preditions.prob_target_in_view,
                item.grip_cam_preditions.prob_holding,
                item.grip_cam_preditions.grip_angle,
            ]))
        elif payload == "target_list":
            self.targets_pub.publish(String(data=message_to_json(item.target_list)))
        elif payload == "video_ready":
            self.video_ready_pub.publish(String(data=message_to_json(item.video_ready)))
        elif payload == "uplink_status":
            self.uplink_pub.publish(Bool(data=bool(item.uplink_status.online)))
        elif payload == "episode_control":
            self.episode_control_pub.publish(String(data=message_to_json(item.episode_control)))
        elif payload == "operation_progress":
            self.operation_progress_pub.publish(String(data=message_to_json(item.operation_progress)))
        elif payload == "last_commanded_grip":
            self.commanded_grip_pub.publish(self._float_array([
                item.last_commanded_grip.wrist_speed,
                item.last_commanded_grip.finger_speed,
            ]))
        elif payload == "swing_cancellation_state":
            self.swing_cancellation_pub.publish(Bool(data=bool(item.swing_cancellation_state.enabled)))
        elif payload == "visibility_states":
            self.visibility_states_pub.publish(String(data=message_to_json(item.visibility_states)))

    def _publish_pos_estimate(self, msg: telemetry.PositionEstimate) -> None:
        stamp = self._stamp_from_seconds(msg.data_ts)
        if msg.gantry_position is not None:
            point = self._point_stamped(msg.gantry_position, stamp=stamp)
            self.gantry_pose_pub.publish(point)
            self._broadcast_transform(self.gantry_frame_id, msg.gantry_position, None, stamp)
        if msg.gantry_velocity is not None:
            self.gantry_twist_pub.publish(self._twist_stamped(msg.gantry_velocity, stamp=stamp))
        if msg.gripper_pose is not None:
            pose = self._pose_stamped(msg.gripper_pose, stamp=stamp)
            self.gripper_pose_pub.publish(pose)
            self._broadcast_transform(
                self.gripper_frame_id,
                msg.gripper_pose.position,
                msg.gripper_pose.rotation,
                stamp,
            )
        if msg.tension:
            self.tension_pub.publish(self._float_array(msg.tension))
        if msg.slack:
            slack = Int32MultiArray()
            slack.data = [1 if value else 0 for value in msg.slack]
            self.slack_pub.publish(slack)

    def _publish_pos_factors(self, msg: telemetry.PositionFactors) -> None:
        self.position_factors_pub.publish(String(data=message_to_json(msg)))
        if msg.visual_pos is not None:
            self.visual_pose_pub.publish(self._point_stamped(msg.visual_pos))
        if msg.hanging_pos is not None:
            self.hanging_pose_pub.publish(self._point_stamped(msg.hanging_pos))
        self.position_spin_pub.publish(Float32(data=float(msg.spin)))

    def _publish_gantry_sightings(self, msg: telemetry.GantrySightings) -> None:
        values: list[float] = []
        for sighting in msg.sightings:
            values.extend(vec3_to_list(sighting))
        self.gantry_sightings_pub.publish(self._float_array(values))

    def _publish_anchor_poses(self, msg: telemetry.AnchorPoses) -> None:
        self.anchor_poses_pub.publish(String(data=message_to_json(msg)))
        for index, pose in enumerate(msg.poses):
            publisher = self.anchor_pose_publishers.get(index)
            if publisher is None:
                publisher = self.create_publisher(PoseStamped, self._topic(f"anchors/poses/anchor_{index}"), 10)
                self.anchor_pose_publishers[index] = publisher
            publisher.publish(self._pose_stamped(pose))
        for index, eyelet in enumerate(msg.eyelets):
            publisher = self.eyelet_publishers.get(index)
            if publisher is None:
                publisher = self.create_publisher(PointStamped, self._topic(f"anchors/eyelets/eyelet_{index}"), 10)
                self.eyelet_publishers[index] = publisher
            publisher.publish(self._point_stamped(eyelet))

    def _publish_component_status(self, msg: telemetry.ComponentConnStatus) -> None:
        self.component_status_json_pub.publish(String(data=message_to_json(msg)))
        status = DiagnosticStatus()
        status.name = "stringman/gripper" if msg.is_gripper else f"stringman/anchor_{msg.anchor_num}"
        status.hardware_id = "gripper" if msg.is_gripper else f"anchor_{msg.anchor_num}"
        status.level = self._diagnostic_level(msg.websocket_status)
        status.message = msg.error_message or msg.websocket_status.name.lower()
        status.values = [
            KeyValue(key="websocket_status", value=msg.websocket_status.name),
            KeyValue(key="video_status", value=msg.video_status.name),
            KeyValue(key="ip_address", value=msg.ip_address),
        ]
        if msg.gripper_model is not None:
            status.values.append(KeyValue(key="model", value=msg.gripper_model.name))
        if msg.temp is not None:
            status.values.append(KeyValue(key="temperature_c", value=str(float(msg.temp))))
        if msg.motor_enabled is not None:
            status.values.append(KeyValue(key="motor_enabled", value=msg.motor_enabled.name))
        diagnostic = DiagnosticArray()
        diagnostic.header.stamp = self.get_clock().now().to_msg()
        diagnostic.status = [status]
        self.component_status_pub.publish(diagnostic)

    def _publish_named_position(self, msg: telemetry.NamedObjectPosition) -> None:
        self.named_positions_pub.publish(String(data=message_to_json(msg)))
        if msg.position is None:
            return
        safe_name = sanitize_topic_token(msg.name)
        publisher = self.named_position_publishers.get(safe_name)
        if publisher is None:
            publisher = self.create_publisher(PointStamped, self._topic(f"named_positions/{safe_name}"), 10)
            self.named_position_publishers[safe_name] = publisher
        publisher.publish(self._point_stamped(msg.position))

    def _publish_gripper_sensors(self, msg: telemetry.GripperSensors) -> None:
        values = [msg.range, msg.angle, msg.pressure, msg.wrist]
        if msg.target_force is not None:
            values.append(msg.target_force)
        self.grip_sensors_pub.publish(self._float_array(values))
        range_msg = Range()
        range_msg.header.frame_id = self.gripper_frame_id
        range_msg.header.stamp = self.get_clock().now().to_msg()
        range_msg.radiation_type = Range.INFRARED
        range_msg.min_range = 0.0
        range_msg.max_range = 4.0
        range_msg.range = float(msg.range) if math.isfinite(float(msg.range)) else float("inf")
        self.grip_range_pub.publish(range_msg)
        self.grip_angle_pub.publish(Float32(data=float(msg.angle)))
        self.grip_pressure_pub.publish(Float32(data=float(msg.pressure)))
        self.grip_wrist_pub.publish(Float32(data=float(msg.wrist)))
        if msg.target_force is not None:
            self.grip_target_force_pub.publish(Float32(data=float(msg.target_force)))

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.client.queue_control_item(
            control_item_from_twist(
                msg.linear,
                msg.angular,
                direction_is_in_gripper_frame=self.command_velocity_is_gripper_frame,
            )
        )

    def _on_gripper_cmd(self, msg: Float32MultiArray) -> None:
        try:
            self.client.queue_control_item(control_item_from_gripper_command(msg.data))
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def _on_common_command(self, msg: String) -> None:
        try:
            self.client.queue_control_item(control_item_from_common_command(msg.data))
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def _on_control_json(self, msg: String) -> None:
        try:
            self.client.queue_control_item(control_item_from_json(msg.data))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            self.get_logger().warning(f"Invalid control JSON: {exc}")

    def _on_move_gripper_to(self, msg: PointStamped) -> None:
        self.client.queue_control_item(control_item_from_move_gripper_point(msg.point))

    def _on_debug(self, msg: String) -> None:
        self.client.queue_control_item(control_item_from_debug(msg.data))

    def _on_episode_prompt(self, msg: String) -> None:
        self.client.queue_control_item(control_item_from_json({"episode_control": {"prompt": msg.data}}))

    def _trigger_common(self, command_name: str) -> Any:
        def callback(_request: Any, response: Any) -> Any:
            self.client.queue_control_item(control_item_from_common_command(command_name))
            response.success = True
            response.message = f"queued {command_name}"
            return response

        return callback

    def _set_swing_service(self, request: Any, response: Any) -> Any:
        self.client.queue_control_item(control_item_from_set_swing(bool(request.data)))
        response.success = True
        response.message = f"queued swing_cancellation={bool(request.data)}"
        return response

    def _stamp_from_seconds(self, seconds: float | None) -> Any:
        if seconds is None or seconds <= 0:
            return self.get_clock().now().to_msg()
        stamp = self.get_clock().now().to_msg()
        stamp.sec = int(seconds)
        stamp.nanosec = int((seconds - int(seconds)) * 1_000_000_000)
        return stamp

    def _point_stamped(self, value: common.Vec3, *, stamp: Any | None = None) -> Any:
        msg = PointStamped()
        msg.header.frame_id = self.world_frame_id
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = vec3_to_list(value)
        return msg

    def _pose_stamped(self, value: common.Pose, *, stamp: Any | None = None) -> Any:
        msg = PoseStamped()
        msg.header.frame_id = self.world_frame_id
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        if value.position is not None:
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = vec3_to_list(value.position)
        qx, qy, qz, qw = rodrigues_to_quaternion(value.rotation)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def _twist_stamped(self, value: common.Vec3 | None, *, stamp: Any | None = None) -> Any:
        msg = TwistStamped()
        msg.header.frame_id = self.world_frame_id
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = vec3_to_list(value)
        return msg

    def _float_array(self, values: list[float] | tuple[float, ...]) -> Any:
        msg = Float32MultiArray()
        msg.data = [float(value) for value in values]
        return msg

    def _diagnostic_level(self, status: telemetry.ConnStatus) -> int:
        if status == telemetry.ConnStatus.CONNECTED:
            return DiagnosticStatus.OK
        if status == telemetry.ConnStatus.CONNECTING:
            return DiagnosticStatus.WARN
        return DiagnosticStatus.ERROR

    def _broadcast_transform(
        self,
        child_frame_id: str,
        position: common.Vec3 | None,
        rotation: common.Vec3 | None,
        stamp: Any,
    ) -> None:
        if self.tf_broadcaster is None or position is None:
            return
        msg = TransformStamped()
        msg.header.frame_id = self.world_frame_id
        msg.header.stamp = stamp
        msg.child_frame_id = child_frame_id
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = vec3_to_list(position)
        qx, qy, qz, qw = rodrigues_to_quaternion(rotation)
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(msg)


def main(args: list[str] | None = None) -> None:
    if ROS2_IMPORT_ERROR is not None:
        raise SystemExit(
            "ROS2 Python packages are not importable. Source your ROS2 environment first, "
            "for example: source /opt/ros/jazzy/setup.bash"
        )
    rclpy.init(args=args)
    node = StringmanRos2Adapter()
    node.start()
    try:
        rclpy.spin(node)
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
