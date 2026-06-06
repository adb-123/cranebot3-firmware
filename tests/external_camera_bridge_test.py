import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from nf_robot.common.config_loader import create_default_config, load_config, save_config
from nf_robot.common.pose_functions import compose_poses, create_lookat_pose, invert_pose
from nf_robot.host import external_camera_bridge as bridge


def _camera_info(width=640, height=480):
    return {
        "width": width,
        "height": height,
        "k": [500.0, 0.0, width / 2.0, 0.0, 500.0, height / 2.0, 0.0, 0.0, 1.0],
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def _registry_payload():
    return {
        "enabled": True,
        "knownMarkers": {
            "origin": {
                "pose": {
                    "rotation": [0.0, 0.0, 0.0],
                    "position": [0.0, 0.0, 0.0],
                },
                "required": True,
            }
        },
        "fusion": {"mapSizePx": 320, "mapExtentM": 6.0},
        "cameras": [
            {
                "name": "cam_a",
                "enabled": True,
                "sourceType": "ros2_compressed_image",
                "rosDomainId": 11,
                "imageTopic": "/cameras/cam_a/image_raw/compressed",
                "cameraInfoTopic": "/cameras/cam_a/camera_info",
                "promoteAfterFrames": 2,
            }
        ],
    }


def test_external_room_camera_registry_is_config_driven():
    registry = bridge.ExternalRoomCameraRegistry.from_json(_registry_payload())

    assert registry.enabled
    assert registry.enabled_cameras()[0].name == "cam_a"
    assert registry.enabled_cameras()[0].ros_domain_id == 11
    assert registry.known_markers["origin"].required is True


def test_config_loader_preserves_external_room_camera_block(tmp_path):
    path = tmp_path / "robot.conf"
    config = create_default_config()
    save_config(config, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["externalRoomCameras"] = _registry_payload()
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    loaded = load_config(path)
    loaded.max_accel = 0.2
    save_config(loaded, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["externalRoomCameras"]["cameras"][0]["name"] == "cam_a"
    assert saved["externalRoomCameras"]["knownMarkers"]["origin"]["required"] is True


def test_registry_loads_known_marker_pose_files(tmp_path):
    marker_file = tmp_path / "markers.json"
    marker_file.write_text(
        json.dumps(
            {
                "markers": {
                    "cal_assist_3": {
                        "pose": {
                            "rotation": [0.0, 0.0, 0.0],
                            "position": [1.0, 2.0, 0.0],
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = {
        "externalRoomCameras": {
            **_registry_payload(),
            "knownMarkerPoseFiles": ["markers.json"],
        }
    }
    config_path = tmp_path / "robot.conf"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    registry = bridge.load_external_room_camera_registry(config_path)

    assert registry.known_markers["origin"].pose.position == [0.0, 0.0, 0.0]
    assert registry.known_markers["cal_assist_3"].pose.position == [1.0, 2.0, 0.0]


def test_pose_estimate_recovers_camera_pose_from_known_marker():
    marker_pose = bridge.CameraPose(rotation=[0.0, 0.0, 0.0], position=[0.0, 0.0, 0.0])
    known = {"origin": bridge.KnownMarker(name="origin", pose=marker_pose)}
    camera_pose = create_lookat_pose([0.5, -2.0, 2.2], [0.0, 0.0, 0.0])
    marker_pose_camera = compose_poses([invert_pose(camera_pose), marker_pose.as_tuple()])

    state = bridge.estimate_camera_pose_from_detections(
        [{"n": "origin", "p": marker_pose_camera, "center": (320.0, 240.0)}],
        known,
        min_known_markers=1,
    )

    assert state.state == "solving"
    assert state.known_marker_count == 1
    assert state.pose is not None
    assert state.pose.position == pytest.approx([0.5, -2.0, 2.2], abs=1e-6)


def test_camera_runtime_promotes_stable_rgb_self_calibration(monkeypatch):
    registry = bridge.ExternalRoomCameraRegistry.from_json(_registry_payload())
    runtime = bridge.ExternalCameraRuntime(registry.enabled_cameras()[0], registry)
    marker_pose = registry.known_markers["origin"].pose
    camera_pose = create_lookat_pose([0.5, -2.0, 2.2], [0.0, 0.0, 0.0])
    marker_pose_camera = compose_poses([invert_pose(camera_pose), marker_pose.as_tuple()])

    monkeypatch.setattr(
        bridge,
        "locate_markers",
        lambda _rgb, _cal: [{"n": "origin", "p": marker_pose_camera, "center": (320.0, 240.0)}],
    )

    runtime.set_camera_info(_camera_info())
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    runtime.handle_frame_bgr(frame)
    assert runtime.calibration.state == "solving"
    runtime.handle_frame_bgr(frame)

    assert runtime.calibration.state == "calibrated"
    assert runtime.calibration.pose.position == pytest.approx([0.5, -2.0, 2.2], abs=1e-6)
    assert runtime.overlay_jpeg is not None


def test_discovery_finds_rgb_camera_candidates():
    candidates = bridge.discover_ros_camera_candidates(
        [
            ("/cameras/mx_brio/camera_info", ["sensor_msgs/msg/CameraInfo"]),
            ("/cameras/mx_brio/image_raw/compressed", ["sensor_msgs/msg/CompressedImage"]),
            ("/camera/depth/camera_info", ["sensor_msgs/msg/CameraInfo"]),
        ]
    )

    mx = next(item for item in candidates if item["baseTopic"] == "/cameras/mx_brio")
    assert mx["imageTopic"] == "/cameras/mx_brio/image_raw/compressed"
    assert mx["sourceType"] == "ros2_compressed_image"


def test_fused_room_maps_emit_floor_gaussian_obstacle_and_disagreement_images():
    camera_cal = bridge.camera_calibration_from_camera_info(_camera_info())
    frame = np.full((480, 640, 3), 160, dtype=np.uint8)
    cv2.line(frame, (100, 100), (540, 360), (0, 0, 0), 8)
    camera_pose = create_lookat_pose([0.5, -2.0, 2.2], [0.0, 0.0, 0.0])
    client = SimpleNamespace(
        name="cam_a",
        last_frame_resized=frame,
        camera_pose=camera_pose,
        camera_cal=camera_cal,
    )

    fused = bridge.build_fused_room_maps([client], map_size_px=220, map_extent_m=6.0)

    assert fused.status == "ready"
    assert fused.summary["calibratedCameraCount"] == 1
    assert fused.floor_jpeg is not None
    assert fused.gaussian_jpeg is not None
    assert fused.obstacle_jpeg is not None
    assert fused.disagreement_jpeg is not None
