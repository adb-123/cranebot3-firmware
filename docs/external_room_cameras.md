# External Room Cameras

`externalRoomCameras` is a raw config block preserved by the protobuf-backed
Stringman config loader. It lets room-mapping cameras be added or removed with a
config edit instead of a code change.

## Config Shape

```json
{
  "externalRoomCameras": {
    "enabled": true,
    "knownMarkers": {
      "origin": {
        "pose": {
          "rotation": [0.0, 0.0, 0.0],
          "position": [0.0, 0.0, 0.0]
        },
        "required": true
      }
    },
    "knownMarkerPoseFiles": [
      "calibrations/bedroom/external_known_markers.json"
    ],
    "fusion": {
      "mapSizePx": 700,
      "mapExtentM": 12.0,
      "roomBounds": [-4.9, -4.9, 4.9, 5.4],
      "includeUnknown": true
    },
    "cameras": [
      {
        "name": "claw_mx_brio",
        "enabled": true,
        "sourceType": "ros2_compressed_image",
        "rosDomainId": 11,
        "rmw": "rmw_cyclonedds_cpp",
        "imageTopic": "/cameras/mx_brio/image_raw/compressed",
        "cameraInfoTopic": "/cameras/mx_brio/camera_info",
        "frameId": "mx_brio_cam_2444lvj1re28_optical_frame",
        "labels": ["external", "rgb", "overview", "mapping"]
      }
    ]
  }
}
```

Each RGB camera calibrates itself from visible configured markers. Cameras with
missing `camera_info`, stale frames, unstable pose estimates, or no known marker
visibility stay available for viewing but are excluded from fused maps.

## Running The Bridge

```bash
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
stringman-external-camera-bridge \
  --config bedroom.conf \
  --ros-domain-id 11 \
  --host 127.0.0.1 \
  --port 8091 \
  --artifact-dir logs/external_room_cameras/latest
```

Useful endpoints:

- `http://127.0.0.1:8091/healthz`
- `http://127.0.0.1:8091/cameras/claw_mx_brio/snapshot.jpg`
- `http://127.0.0.1:8091/cameras/claw_mx_brio/overlay.jpg`
- `http://127.0.0.1:8091/maps/latest.json`
- `http://127.0.0.1:8091/maps/floor.jpg`
- `http://127.0.0.1:8091/maps/gaussian.jpg`
- `http://127.0.0.1:8091/maps/obstacles.jpg`
- `http://127.0.0.1:8091/maps/disagreement.jpg`

## Adding Another Camera

Add another object under `externalRoomCameras.cameras` with its ROS topics and
domain. If a camera is on a different ROS domain, run a second bridge instance
with the same config and that domain id; the code does not need to change.

For stronger self-calibration, add more fixed marker poses under
`knownMarkers` or in files listed by `knownMarkerPoseFiles`. A camera can
calibrate from one visible known marker, but two or more fixed markers make pose
confidence and disagreement checks much stronger.

When the normal LAN web UI is started with
`--external-camera-bridge-uri http://127.0.0.1:8091`, these endpoints are also
proxied under `/external-cameras/...` on the Stringman web server.
