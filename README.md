# nf_robot

Control code for the Stringman household robotic crane from Neufangled Robotics

## [Build Guides and Documentation](https://neufangled.com/docs)

Purchase assembled robots or kits at [neufangled.com](https://neufangled.com)

## Installation of stringman controller (Users)

Linux (python 3.11 or later)

    sudo apt install python3-dev python3-virtualenv python3-pip ffmpeg
    python3 -m virtualenv venv
    source venv/bin/activate
    pip install "nf_robot[host]"

Start headless robot controller in LAN-only mode.
The particular robot details will be read from/saved to bedroom.conf

    stringman-headless --config=bedroom.conf

Serve the local browser UI on the LAN from the same machine:

    stringman-web-ui --host=0.0.0.0 --port=8080 --robot-uri=ws://127.0.0.1:4245

Open the printed `http://<lan-ip>:8080` URL from any browser on the same LAN.
The web UI serves the bundled `nf-viz` playroom, keeps the controller websocket
on localhost, and exposes a LAN-safe raw protobuf websocket proxy for the
browser. The simpler diagnostic page remains available at `/simple`.

The stringman motion controller (stringman-headless) is the program which communicates with the robot components over wifi and acts as the central brain of a single robot. It must be running on the same network as the powered on anchors and gripper in order for the robot to be active and controllable. The main entrypoint is observer.py

It listens on port 4245 for a connection from a UI or local AI policy. The UI can be opened at [neufangled.com/playroom](https://neufangled.com/playroom). Select LAN mode at first.

Refer to the [Usage Guide](https://neufangled.com/docs/usage_guide/) for more detailed instructions on setup and use.

### Arguments to stringman-headless

options:

  --config              A json file where the robot's ID and calibration data are stored. You may use one for a bedroom and one for a playroom for example, even if it is the same hardware being taken
                        down and put back up in another room 
  --telemetry_env {local,staging,production}
                        The cloud telemetry server to connect to (choices: local, staging, production) The default is None, which allows local connections on port 4245 only
                        When production is used if you have already bound the robot to an account at neufangled.com. This is completely optional.
  --no_ai               Disable the use of the target identificaiton model.
  --auto_start          Automatically unpark and start cleaning when all components connect
  --local_models        Use local models from models/ rather than downloading the production models from huggingface (applicable to the target identification model only)
  --arp_grasp           Use arp_execute_grasp (centering net) instead of act_execute_grasp (ACT policy) for the Arpeggio gripper
  --debug               Enable DEBUG level logging
  --observability-debug Record bounded telemetry payload summaries at DEBUG level in observability logs
  --metrics-host        Prometheus metrics bind host, default 0.0.0.0
  --metrics-port        Prometheus metrics port, default 9464
  --observability-log   JSON log path scraped by Promtail, default logs/nf_robot-observability.jsonl
  --no_observability    Disable local Prometheus metrics, OTel traces, and JSON observability logs

### Local observability

The repo includes a local Grafana, Loki, Tempo, Prometheus, OTel Collector, and
Promtail stack in [`observability/`](observability/README.md). It instruments
`stringman-headless` with Prometheus metrics, OTel traces, and JSON logs with
trace IDs.

### Minimum system specs

At least 8 cores and 8GB of ram.
In order to perform local inference, some kind of pytorch accelertion is necessary.
Mini PC's or laptops based on the Ryzen 7 7840HS are probably about the cheapest machines that can run stringman's motion controller since it has an NPU that can be used to accelerate pytorch. A mac mini is also a viable option.

Otherwise, any gaming PC is usually more than enough.

### Telemetry stream

stringman-headless listens on port 4245 locally for telemetry connections. This is a websocket sending and receiving protobufs defined in `src/nf_robot/protos`
Every message sent by stringman-headless is a serialized `TelemetryBatchUpdate` and every message received is expected to be a `ControlBatchUpdate`.

Within the telemetry stream, there are `VideoReady` messages containing URIs for connecting to the robot's video streams.

The UI at [neufangled.com/playroom](https://neufangled.com/playroom) sends controls and receives telemetry.
Any AI policy served by `src/nf_robot/ml/stringman_lerobot.py` also sends controls and receives telemetry.
Agents wishing to write code to interface with a stringman robot may also follow this pattern.

The expected inputs are basically marker box velocity and finger and wrist speeds. The gripper hangs 50 cm below the marker box.
Higher level control is achived by having a policy such as DIT or a VLA connected to the robot, and having another client sending `nf.common.EpisodeControl` commands with prompts.

See [Imitation Learning](https://neufangled.com/docs/imitation_learning/) for a more detailed guide.

### ROS2 adapter

`stringman-ros2-adapter` bridges the local `stringman-headless` websocket to a
ROS2 graph. Source ROS2 first, then run the adapter while `stringman-headless`
is listening on port 4245:

    source /opt/ros/jazzy/setup.bash
    stringman-headless --config=bedroom.conf
    stringman-ros2-adapter --ros-args -p server_uri:=ws://127.0.0.1:4245

The adapter publishes typed ROS2 topics under `stringman/` for the main sensor
and state surfaces, including:

    stringman/pose/gantry
    stringman/twist/gantry
    stringman/pose/gripper
    stringman/tension
    stringman/slack
    stringman/gripper/range
    stringman/gripper/sensors
    stringman/components/status
    stringman/video/ready
    stringman/targets
    stringman/named_positions
    stringman/telemetry/raw_json
    stringman/telemetry/raw_protobuf

It also publishes TF frames from `stringman_world` to `stringman_gantry` and
`stringman_gripper` when `tf2_ros` is available.

Control inputs are accepted through standard topics:

    stringman/cmd_vel                 # geometry_msgs/Twist, linear m/s, angular.z as wrist rad/s
    stringman/gripper/cmd             # Float32MultiArray: [finger_speed_deg_s, wrist_speed_deg_s, winch_m_s]
    stringman/command                 # std_msgs/String, e.g. "stop_all" or "park"
    stringman/move_gripper_to         # geometry_msgs/PointStamped, world-space meters
    stringman/debug                   # std_msgs/String debug action
    stringman/episode_prompt          # std_msgs/String LeRobot prompt
    stringman/control/json            # full protobuf control escape hatch

Common commands are also exposed as `std_srvs/Trigger` services under
`stringman/services/`, including `stop_all`, `park`, `unpark`, `grasp`,
`zero_winch`, `tighten_lines`, `enable_torque`, `disable_torque`,
`half_cal`, `full_cal`, `auto_calibrate_swing`, and `update_firmware`.
Swing cancellation is exposed as `std_srvs/SetBool` at
`stringman/services/swing_cancellation`.

Use `stringman/control/json` for controls that do not have a dedicated typed
topic. Examples:

    ros2 topic pub --once /stringman/control/json std_msgs/msg/String "{data: '{\"jog_spool\":{\"is_gripper\":false,\"anchor_num\":0,\"speed\":0.01}}'}"
    ros2 topic pub --once /stringman/control/json std_msgs/msg/String "{data: '{\"move_gripper_to\":{\"x\":0.2,\"y\":0.1,\"z\":0.4}}'}"
    ros2 topic pub --once /stringman/control/json std_msgs/msg/String "{data: '{\"single_component_action\":{\"is_gripper\":true,\"action\":\"identify\"}}'}"

### Calibration safety constraints

Full calibration can use optional `calibrationSafety` settings in the robot
config to adapt the Arpeggio probe size to the room, reject catch-risk no-go
zones, and run a post-solve safe-motion validation before reporting success.
See [`docs/calibration_safety.md`](docs/calibration_safety.md) and
[`docs/calibration_safety_implementation.md`](docs/calibration_safety_implementation.md).

Common setup commands:

    calibration-safety-apply bedroom.conf --safety docs/calibration_safety.example.json --summary
    calibration-safety-check bedroom.conf
    calibration-artifact-summary logs/calibration/<session>.json

## Cloud telemetry relay

When stringman-headless is in LAN mode (done by omitting the --telemetry_env argument) it only accepts local telemetry connections and only streams video locally.

If connected to a robot in lan mode from the UI at neufangled.com, you can click "Bind robot" in the run menu, log in with an identity profider, and that robot id (from the config.json file) will be marked as owned by you.
It is then possible to run with `--telemetry_env=production` and stringman will also send telemetry and video to neufangled.com so that you can view and control the robot remotely over the internet. This is accessed from the "My Robots" option when opening neufangled.com/playroom.

No video or telemetry is saved when you use the cloud relay. The only way video gets shared with us is if you record a public lerobot dataset and inform us of it.

## Installation of Robot Control Panel (developers)

    git clone https://github.com/nhnifong/cranebot3-firmware.git

    sudo apt install python3-dev python3-virtualenv python3-pip ffmpeg
    python -m virtualenv venv
    source venv/bin/activate
    pip install -e ".[host,dev,pi]"

### If you have an RTX 5090

    pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchcodec==0.6.0 --index-url https://download.pytorch.org/whl/cu129

### Run tests

    pytest tests

### Setting up a component

Robot components that boot from the [`stringman-zero2w.img`](https://storage.googleapis.com/stringman-models/stringman-zero2w.img) (1.6GB) image should begin looking for wifi share codes with their camera immediately. You can produce a code with [qifi.org](htts://qifi.org)

Once the pi sees the code it will connect to the network and remember those settings. It should then be discoverable by the control panel via multicast DNS (Bonjour)

## Starting from a base rpi image

Alternatively the software can be set up from a fresh raspberry pi lite 64 bit image.
After booting any raspberry pi from a fresh image, perform an update

    sudo apt update -y && sudo apt full-upgrade -y -o Dpkg::Options::="--force-confold" && sudo apt install -y git python3-dev python3-virtualenv rpicam-apps i2c-tools

Clone the [cranebot-firmware](https://github.com/nhnifong/cranebot3-firmware) repo

    git clone https://github.com/nhnifong/cranebot3-firmware.git && cd cranebot3-firmware

Set the component type by uncommenting the appropriate line in server.conf

    nano server.conf

Install stringman

    chmod +x install.sh
    sudo ./install.sh

### Additional settings for anchors

Setup for any raspberry pi that will be part of an anchor
Enable uart serial harware interface interactively.

    sudo raspi-config

In interface optoins, select serial port. disable the login shell, but enable hardware serial.

add the following lines lines to to `/boot/firmware/config.txt`  at the end this disables bluetooth, which would otherwise occupy the uart hardware.
Then reboot after this change

    enable_uart=1
    dtoverlay=disable-bt

### Additional settings for gripper

Setup for the raspberry pi in the gripper with the inventor hat mini
Enable i2c

    sudo raspi-config nonint do_i2c 0

Add this line to `/boot/firmware/config.txt` just under `dtparam=i2c_arm=on` and reboot

    dtparam=i2c_baudrate=400000

## Rebuilding the python module

within a venv install the build tools

    python3 -m pip install --upgrade build twine

Bump the version number in pyproject.toml
then at this repo's root, build the module. Artifacts will be in dist/

    python3 -m build

Upload the particular version you just built to PyPi

    python3 -m twine upload dist/nf_robot-3.4.4*

### QA scripts

Note that if you are proceeding to QA scripts right after doing the steps above you must reboot and then stop the service before running those scrips.

    sudo reboot now

log back in

    sudo systemctl stop cranebot.service

Run QA scripts for the specific component type

    /opt/robot/env/bin/qa-anchor anchor|power_anchor
    /opt/robot/env/bin/qa-gripper
    /opt/robot/env/bin/qa-gripper-arp

These scripts both check whether everything is connected as it should be and in the case of anchors, set whether it is a power anchor or not.

To update to the lastest nf_robot version in a component

    /opt/robot/env/bin/pip install --upgrade "nf_robot[pi]"

## Training models


## Windows

A self contained windows installer can be generated. The exact installation of stringman that ends up in the installer depends on what was in the virtualenv these commands are run from, so make a new one.

    python3 -m venv winvenv
    source winvenv/bin/activate
    pip install nf_robot[host]
    pip install pyinstaller
    pyinstaller --onefile --windowed --name "Stringman" win_main.py

## Support this project

[Donate on Ko-fi](https://ko-fi.com/neufangled)

## Dynamic-room calibration planning

For cluttered rooms, generate one conservative `calibrationSafety` block before running calibration:

```bash
python scripts/plan_calibration_room.py \
  --room-file docs/calibration_room.example.json \
  --derive-line-endpoints-from-config path/to/config.json \
  --overwrite-line-endpoints \
  --hazards-from-artifact-dir logs/calibration \
  --hazard-artifact-limit 1 \
  --require-plan-quality usable \
  --summary \
  --svg-output room_plan.svg \
  --output calibration_safety.generated.json
```

Apply the generated block to the robot config:

```bash
python scripts/apply_calibration_safety.py path/to/config.json \
  --safety calibration_safety.generated.json \
  --write
```

Restart the controller after updating config, then run calibration. If a run fails, inspect the newest `logs/calibration/*.json` with `scripts/summarize_calibration_artifact.py`, update the room file/no-go zones, regenerate `calibration_safety.generated.json`, and apply it again.

Use `--include-plan-summary` when you want a single JSON file containing both `calibrationSafety` and `roomPlan` evidence. Use `--svg-output` for an operator preview of the room, no-go zones, selected probe center, probe diamond, line endpoints, and cable sweeps.

External RGB room cameras can be added through the preserved
`externalRoomCameras` config block and served by
`stringman-external-camera-bridge`. See
[`docs/external_room_cameras.md`](docs/external_room_cameras.md) for the
multi-camera registry, self-calibration, fused map endpoints, and the MX Brio
ROS domain 11 setup.
