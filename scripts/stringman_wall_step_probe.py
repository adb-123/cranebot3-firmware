#!/usr/bin/env python3
"""Run one bounded Stringman velocity pulse and measure wall-camera alignment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from std_srvs.srv import Trigger

from nf_robot.common.cv_common import locate_markers, locate_markers_gripper


WALL_CAMERAS = ("anchor_0", "anchor_1")
GRIPPER_CAMERA = "gripper"
ALL_CAMERAS = (*WALL_CAMERAS, GRIPPER_CAMERA)


def _now() -> float:
    return time.monotonic()


class StepProbe(Node):
    def __init__(self) -> None:
        super().__init__("stringman_wall_step_probe")
        self.tension: list[float] | None = None
        self.tension_at = 0.0
        self.slack: list[int] | None = None
        self.slack_at = 0.0
        self.pose: PointStamped | None = None
        self.pose_at = 0.0
        self.frames: dict[str, tuple[np.ndarray, float]] = {}
        self.camera_info: dict[str, CameraInfo] = {}

        self.publisher = self.create_publisher(Twist, "/stringman/cmd_vel", 10)
        self.stop_client = self.create_client(Trigger, "/stringman/services/stop_all")

        self.create_subscription(Float32MultiArray, "/stringman/tension", self._on_tension, 10)
        self.create_subscription(Int32MultiArray, "/stringman/slack", self._on_slack, 10)
        self.create_subscription(PointStamped, "/stringman/pose/gantry", self._on_pose, 10)
        for camera in ALL_CAMERAS:
            self.create_subscription(
                CameraInfo,
                f"/stringman/cameras/{camera}/camera_info",
                lambda msg, camera=camera: self.camera_info.__setitem__(camera, msg),
                10,
            )
            self.create_subscription(
                CompressedImage,
                f"/stringman/cameras/{camera}/image_raw/compressed",
                lambda msg, camera=camera: self._on_image(camera, msg),
                qos_profile_sensor_data,
            )

    def _on_tension(self, msg: Float32MultiArray) -> None:
        self.tension = [float(v) for v in msg.data]
        self.tension_at = _now()

    def _on_slack(self, msg: Int32MultiArray) -> None:
        self.slack = [int(v) for v in msg.data]
        self.slack_at = _now()

    def _on_pose(self, msg: PointStamped) -> None:
        self.pose = msg
        self.pose_at = _now()

    def _on_image(self, camera: str, msg: CompressedImage) -> None:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is not None:
            self.frames[camera] = (frame, _now())

    def wait_for_ready(self, timeout_s: float) -> bool:
        deadline = _now() + timeout_s
        while _now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (
                self.tension is not None
                and self.slack is not None
                and self.pose is not None
                and all(camera in self.frames for camera in ALL_CAMERAS)
                and all(camera in self.camera_info for camera in ALL_CAMERAS)
            ):
                return True
        return False

    def wait_for_fresh_images(self, since: float, timeout_s: float) -> bool:
        deadline = _now() + timeout_s
        while _now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if all(camera in self.frames and self.frames[camera][1] >= since for camera in ALL_CAMERAS):
                return True
        return False

    def publish_zero(self, count: int = 3) -> None:
        zero = Twist()
        for _ in range(count):
            self.publisher.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.03)

    def call_stop_all(self) -> bool:
        self.publish_zero()
        if not self.stop_client.wait_for_service(timeout_sec=2.0):
            return False
        future = self.stop_client.call_async(Trigger.Request())
        deadline = _now() + 5.0
        while _now() < deadline and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        return bool(future.done() and future.result() and future.result().success)

    def pulse(self, vx: float, vy: float, vz: float, duration_s: float, hz: float, max_tension: float) -> str:
        if duration_s <= 0.0:
            return "no_pulse"
        period = 1.0 / hz
        end_at = _now() + duration_s
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.linear.z = vz
        status = "completed"
        while _now() < end_at:
            self.publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=min(period, 0.05))
            if self.tension and max(self.tension) >= max_tension:
                status = f"stopped:tension_ge_{max_tension:g}"
                break
            remaining = end_at - _now()
            if remaining > 0.0:
                time.sleep(min(period, remaining))
        self.publish_zero(count=5)
        return status


def _camera_cal(info: CameraInfo) -> SimpleNamespace:
    return SimpleNamespace(
        intrinsic_matrix=list(info.k),
        distortion_coeff=list(info.d),
        resolution=SimpleNamespace(width=int(info.width), height=int(info.height)),
    )


def _mean_center(items: list[dict]) -> list[float] | None:
    if not items:
        return None
    arr = np.array([item["center"] for item in items], dtype=float)
    return [float(v) for v in arr.mean(axis=0)]


def _annotate(frame: np.ndarray, detections: list[dict], path: Path) -> dict:
    groups: dict[str, list[dict]] = {}
    for detection in detections:
        groups.setdefault(str(detection["n"]), []).append(detection)
        center = tuple(int(round(v)) for v in detection["center"])
        color = (0, 255, 0) if detection["n"] == "origin" else (0, 190, 255)
        cv2.circle(frame, center, 12, color, 3)
        cv2.putText(frame, str(detection["n"]), (center[0] + 14, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    origin = _mean_center(groups.get("origin", []))
    gantry = _mean_center(groups.get("gantry", []))
    error = None
    distance = None
    if origin and gantry:
        p0 = tuple(int(round(v)) for v in gantry)
        p1 = tuple(int(round(v)) for v in origin)
        cv2.arrowedLine(frame, p0, p1, (255, 0, 255), 3, tipLength=0.08)
        error = [origin[0] - gantry[0], origin[1] - gantry[1]]
        distance = math.hypot(error[0], error[1])
        cv2.putText(frame, f"err=({error[0]:.1f},{error[1]:.1f}) d={distance:.1f}px", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
    return {
        "markers": {name: len(items) for name, items in sorted(groups.items())},
        "origin_center_px": origin,
        "gantry_center_px": gantry,
        "error_px": error,
        "distance_px": distance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-tension", type=float, default=16.0)
    parser.add_argument("--out-root", default="/tmp/stringman_step_probes")
    args = parser.parse_args()

    rclpy.init()
    node = StepProbe()
    try:
        if not node.wait_for_ready(timeout_s=10.0):
            raise RuntimeError("timed out waiting for tension/slack/pose/camera topics")
        before_tension = list(node.tension or [])
        if before_tension and max(before_tension) >= args.max_tension:
            node.call_stop_all()
            raise RuntimeError(f"pre-pulse tension {max(before_tension):.2f} N exceeds {args.max_tension:.2f} N")

        pulse_status = node.pulse(args.x, args.y, args.z, args.duration, args.hz, args.max_tension)
        stop_ok = node.call_stop_all()
        stopped_at = _now()
        node.wait_for_fresh_images(since=stopped_at, timeout_s=5.0)
        deadline = _now() + 0.5
        while _now() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

        out_dir = Path(args.out_root) / args.label
        camera_results = {}
        for camera in WALL_CAMERAS:
            frame, received_at = node.frames[camera]
            raw_path = out_dir / f"{camera}.jpg"
            annotated_path = out_dir / f"{camera}_annotated.jpg"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(raw_path), frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = locate_markers(rgb, _camera_cal(node.camera_info[camera])) or []
            result = _annotate(frame.copy(), detections, annotated_path)
            result["raw_path"] = str(raw_path)
            result["annotated_path"] = str(annotated_path)
            result["received_age_s"] = max(0.0, _now() - received_at)
            camera_results[camera] = result

        gripper_frame, gripper_received_at = node.frames[GRIPPER_CAMERA]
        gripper_raw_path = out_dir / "gripper.jpg"
        gripper_annotated_path = out_dir / "gripper_annotated.jpg"
        cv2.imwrite(str(gripper_raw_path), gripper_frame)
        gripper_rgb = cv2.cvtColor(gripper_frame, cv2.COLOR_BGR2RGB)
        gripper_cal = _camera_cal(node.camera_info[GRIPPER_CAMERA])
        gripper_detections = locate_markers(gripper_rgb, gripper_cal) or []
        gripper_stabilized_detections = locate_markers_gripper(gripper_rgb, gripper_cal) or []
        gripper_result = _annotate(gripper_frame.copy(), gripper_detections, gripper_annotated_path)
        gripper_result["raw_path"] = str(gripper_raw_path)
        gripper_result["annotated_path"] = str(gripper_annotated_path)
        gripper_result["received_age_s"] = max(0.0, _now() - gripper_received_at)
        gripper_result["stabilized_markers"] = [
            {"name": str(item["n"]), "center_px": [float(v) for v in item["center"]]}
            for item in gripper_stabilized_detections
        ]

        pose = node.pose.point if node.pose else None
        payload = {
            "label": args.label,
            "command": {"x": args.x, "y": args.y, "z": args.z, "duration_s": args.duration, "hz": args.hz},
            "pulse_status": pulse_status,
            "stop_all_success": stop_ok,
            "tension_before": before_tension,
            "tension_after": list(node.tension or []),
            "slack_after": list(node.slack or []),
            "gantry_pose_after": None if pose is None else {"x": pose.x, "y": pose.y, "z": pose.z},
            "cameras": camera_results,
            "gripper_camera": gripper_result,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "measurement.json").open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        node.publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
