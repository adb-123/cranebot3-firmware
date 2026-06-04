import numpy as np
from pathlib import Path
import uuid
import json
from nf_robot.generated.nf import common, config as nf_config

DEFAULT_CONFIG_PATH = Path(__file__).parent / 'configuration.json'
DEFAULT_SWING_LATENCY = 0.18
DEFAULT_SWING_GAIN = 0.02
DEFAULT_SWING_SIGN = -1.0
DEFAULT_SWING_MAX_VELOCITY = 0.03
DEFAULT_SWING_AUTO_MIN_ENERGY = 1e-2
DEFAULT_SWING_AUTO_ABORT_RATIO = 1.2
MIN_SWING_GAIN = 0.0
MAX_SWING_GAIN = 0.02
MIN_SWING_MAX_VELOCITY = 0.0
MAX_SWING_MAX_VELOCITY = 0.03

# Anchors
# Defaults based on a square room setup, pointing towards center.
anchor_defs = [
    # (num, position_xyz, rotation_rvec_xyz)
    (0, (3.0, 3.0, 2.0),  (0.0, 0.0, -np.pi/4)),    # -45 deg
    (1, (3.0, -3.0, 2.0), (0.0, 0.0, -3*np.pi/4)),  # -135 deg
    (2, (-3.0, 3.0, 2.0), (0.0, 0.0, np.pi/4)),     # 45 deg
    (3, (-3.0, -3.0, 2.0),(0.0, 0.0, 3*np.pi/4)),   # 135 deg
]

def default_arp_anchors():
    anch_list = []
    for i in (0,1):
        anchor = nf_config.Anchor()
        anchor.num = i
        # leaving service_name None is a indicator that this anchor config is a placeholder
        # and no such service has been disovered yet and assigned this anchor number
        pos = anchor_defs[i*2][1]
        rot = anchor_defs[i*2][2]
        eye = anchor_defs[i*2+1][1]
        anchor.pose = common.Pose(
            rotation=common.Vec3(x=rot[0], y=rot[1], z=rot[2]),
            position=common.Vec3(x=pos[0], y=pos[1], z=pos[2]),
        )
        anchor.indirect_line = nf_config.IndirectLine(
            eyelet_pos=common.Vec3(x=eye[0], y=eye[1], z=eye[2]),
            cam_tilt=22,
        )
        anch_list.append(anchor)  
    return anch_list

def create_default_config() -> nf_config.StringmanPilotConfig:
    """
    Creates a protobuf configuration object populated with reasonable defaults.
    """
    config = nf_config.StringmanPilotConfig()
    # provision a random ID
    # once the robot tells the backend what this ID is, it has to stick to it, or the owner may see it disappear from their dashboard
    config.robot_id = str(uuid.uuid4())
    config.has_been_calibrated = False
    config.connect_cloud_telemetry = False

    for num, pos, rot in anchor_defs:
        anchor = nf_config.Anchor()
        anchor.num = num
        # leaving service_name None is a indicator that this anchor config is a placeholder
        # and no such service has been disovered yet and assigned this anchor number
        
        # Construct Pose using common.Vec3 for rvec (rotation) and tvec (translation)
        anchor.pose = common.Pose(
            rotation=common.Vec3(x=rot[0], y=rot[1], z=rot[2]),
            position=common.Vec3(x=pos[0], y=pos[1], z=pos[2]),
        )
        config.anchors.append(anchor)

    # Camera Calibration Standard
    config.camera_cal = nf_config.CameraCalibration()
    config.camera_cal.resolution = nf_config.Resolution(
        width=1920, 
        height=1080
    )

    # Default Intrinsic Matrix.
    # calcluated for the standard FOV Raspberry Pi Camera module 3
    # with autofocus set to fixed lens position 0.1
    # Derived by anchoring a known room height against solvePnP output
    # a chessboard calibration has been tried, but results were too far off center due to
    # the difficulty of positioning a large enough chessboard in the room.
    intrinsic_np = np.array([
        [1424.,    0., 960.],
        [   0., 1424., 540.],
        [   0.,    0.,   1.]
    ])
    config.camera_cal.intrinsic_matrix = intrinsic_np.flatten().tolist()

    # Default Distortion Coefficients
    distortion_np = np.array([ 0.0115842, 0.18723804, -0.00126164, 0.00058383, -0.38807272])
    config.camera_cal.distortion_coeff = distortion_np.flatten().tolist()

    # Camera Calibration Wide
    config.camera_cal_wide = nf_config.CameraCalibration()
    config.camera_cal_wide.resolution = nf_config.Resolution(width=1920, height=1080)
    intrinsic_np = np.array([
        [791.15,    0., 960.],
        [   0., 791.57, 540.],
        [   0.,    0.,   1.]
    ])
    config.camera_cal_wide.intrinsic_matrix = intrinsic_np.flatten().tolist()
    distortion_np = np.array([-0.06742619,  0.1546371, -0.00232347, 0.00080991, -0.13094542])
    config.camera_cal_wide.distortion_coeff = distortion_np.flatten().tolist()

    # Gripper
    config.gripper = nf_config.Gripper()
    config.gripper.frame_room_spin = (50.0 / 180.0) * np.pi
    
    # Preferred Cameras
    config.preferred_cameras = [0, 1]
    
    # Miscelleneous anchor vars
    config.max_accel = 0.3
    config.rec_mod = 1
    config.running_ws_delay = 0.03

    # Swing cancellation
    config.swing_latency = DEFAULT_SWING_LATENCY # seconds
    config.swing_gain = DEFAULT_SWING_GAIN
    config.swing_sign = DEFAULT_SWING_SIGN
    config.swing_max_velocity = DEFAULT_SWING_MAX_VELOCITY
    config.swing_auto_min_energy = DEFAULT_SWING_AUTO_MIN_ENERGY
    config.swing_auto_abort_ratio = DEFAULT_SWING_AUTO_ABORT_RATIO

    return config

def _raw_config_has(raw_config: dict | None, camel_name: str, snake_name: str) -> bool:
    if raw_config is None:
        return False
    return camel_name in raw_config or snake_name in raw_config

def normalize_config_defaults(config: nf_config.StringmanPilotConfig, raw_config: dict | None=None):
    """
    Fill defaults for fields introduced after older config files were written.

    Proto3 scalar fields deserialize to zero when absent, so use the raw JSON
    keys to avoid overwriting a deliberately saved zero latency.
    """
    if not _raw_config_has(raw_config, 'swingLatency', 'swing_latency'):
        config.swing_latency = DEFAULT_SWING_LATENCY
    if not np.isfinite(config.swing_latency) or config.swing_latency < 0.0 or config.swing_latency > 0.8:
        config.swing_latency = DEFAULT_SWING_LATENCY
    if not _raw_config_has(raw_config, 'swingGain', 'swing_gain'):
        config.swing_gain = DEFAULT_SWING_GAIN
    if not np.isfinite(config.swing_gain):
        config.swing_gain = DEFAULT_SWING_GAIN
    config.swing_gain = float(np.clip(abs(config.swing_gain), MIN_SWING_GAIN, MAX_SWING_GAIN))
    if not _raw_config_has(raw_config, 'swingSign', 'swing_sign'):
        config.swing_sign = DEFAULT_SWING_SIGN
    if config.swing_sign not in (-1.0, 1.0):
        config.swing_sign = DEFAULT_SWING_SIGN if config.swing_sign == 0 else float(np.sign(config.swing_sign))
    if not _raw_config_has(raw_config, 'swingMaxVelocity', 'swing_max_velocity'):
        config.swing_max_velocity = DEFAULT_SWING_MAX_VELOCITY
    if not np.isfinite(config.swing_max_velocity):
        config.swing_max_velocity = DEFAULT_SWING_MAX_VELOCITY
    config.swing_max_velocity = float(np.clip(abs(config.swing_max_velocity), MIN_SWING_MAX_VELOCITY, MAX_SWING_MAX_VELOCITY))
    if not _raw_config_has(raw_config, 'swingAutoMinEnergy', 'swing_auto_min_energy'):
        config.swing_auto_min_energy = DEFAULT_SWING_AUTO_MIN_ENERGY
    if config.swing_auto_min_energy < DEFAULT_SWING_AUTO_MIN_ENERGY:
        config.swing_auto_min_energy = DEFAULT_SWING_AUTO_MIN_ENERGY
    if not _raw_config_has(raw_config, 'swingAutoAbortRatio', 'swing_auto_abort_ratio'):
        config.swing_auto_abort_ratio = DEFAULT_SWING_AUTO_ABORT_RATIO
    if config.swing_auto_abort_ratio <= 1.0:
        config.swing_auto_abort_ratio = DEFAULT_SWING_AUTO_ABORT_RATIO

def save_config(config: nf_config.StringmanPilotConfig, path: Path=DEFAULT_CONFIG_PATH):
    """
    Writes the proto to a JSON file.
    """
    if path is None:
        return
    with open(path, 'w') as f:
        f.write(config.to_json(indent=2))

def load_config(path: Path=DEFAULT_CONFIG_PATH) -> nf_config.StringmanPilotConfig:
    """
    Loads the proto from a JSON file.
    """
    try:
        if path is None:
            raise FileNotFoundError # observer unit test path
        with open(path, 'r') as f:
            print(f'Loaded config from {path}')
            raw_text = f.read()
            raw_config = json.loads(raw_text)
            c = nf_config.StringmanPilotConfig().from_json(raw_text)
            if c.camera_cal is None or c.camera_cal_wide is None:
                default = create_default_config()
                if c.camera_cal is None:
                    c.camera_cal = default.camera_cal
                if c.camera_cal_wide is None:
                    c.camera_cal_wide = default.camera_cal_wide
            if c.park_data is None:
                c.park_data = nf_config.ParkData()

            # any existing config which had anchors must have had pilot anchors
            if c.anchor_type is None and len(c.anchors) > 0:
                c.anchor_type = common.AnchorType.PILOT

            # Set camera tilt on configs that existed before the field was added
            if c.anchor_type == common.AnchorType.ARPEGGIO:
                for anchor in c.anchors:
                    if anchor.indirect_line.cam_tilt is None:
                        anchor.indirect_line.cam_tilt = 26.0
            normalize_config_defaults(c, raw_config)

            return c

            
    except FileNotFoundError:
        print(f"No config found at {path}, creating default.")
        config = create_default_config()
        print(f"New robot id chosen {config.robot_id}.")
        save_config(config, path)
        return config

def config_has_any_address(config: nf_config.StringmanPilotConfig):
    """Return true if this config has the address of at least one component"""
    return any([c.address is not None for c in [config.gripper, *config.anchors]])

if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG_PATH)
    print(f"Loaded config for robot: {cfg.robot_id}")
    print(f"Gripper Spin: {cfg.gripper.frame_room_spin}")
