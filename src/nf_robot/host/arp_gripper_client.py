import asyncio
import logging
import numpy as np
from scipy.spatial.transform import Rotation
import json
import cv2
import time
import math

from nf_robot.host.anchor_client import ComponentClient
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, common
from nf_robot.common.cv_common import SF_TARGET_SHAPE, stabilize_frame_2
from nf_robot.common.config_loader import (
    DEFAULT_SWING_GAIN,
    DEFAULT_SWING_MAX_VELOCITY,
    DEFAULT_SWING_SIGN,
)

logger = logging.getLogger(__name__)

"""
"Arpeggio" is the codename of the 2nd revision of the Stringman gripper

It differs from the previous gripper in that it has a wrist instead of a winch.
Since it uses smart servos it can report the exact angle of either the fingers or wrist
It does not send 'line records' because there is no changing length of line, but wherever line
records were being used as a heartbeat signal, the grip sensors can be used instead.

It has a wide angle camera instead of standard, and the camera is pointed inward at a point 1m below the gripper

The gripper and gantry are now one model, with the gripper's origin being 57cm below the gantry's.
They are related by a chain of poses from the gantry tags, through the wrist rotation, 

"""

R_imu_to_cam = np.array([
    [1, 0,  0],
    [0,  -1, 0],
    [0,  0,  1]
])

# omega is the constant angular frequency of the pendulum. Effectuve length from pivot to center of gripper mass is 0.4526 meters
LENGTH = 0.4526
OMEGA = np.sqrt(9.81 / LENGTH)
SWING_CANCEL_GAIN = -DEFAULT_SWING_GAIN
CENTERING_GAIN = 0.4

def rotate_vector(vec, rad):
    """Rotates a 2D vector [x, y] by a given angle in radians."""
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    return np.array([
        vec[0] * cos_a - vec[1] * sin_a,
        vec[0] * sin_a + vec[1] * cos_a
    ])

class ArpeggioGripperClient(ComponentClient):
    def __init__(self, address, port, datastore, ob, pool, stat, pe, local_telemetry):
        super().__init__(address, port, datastore, ob, pool, stat, local_telemetry)
        self.conn_status = telemetry.ComponentConnStatus(
            is_gripper=True,
            websocket_status=telemetry.ConnStatus.NOT_DETECTED,
            video_status=telemetry.ConnStatus.NOT_DETECTED,
            gripper_model=telemetry.GripperModel.ARPEGGIO,
        )
        self.anchor_num = None
        self.pe = pe
        self.park_pose_relative_to_camera = None
        self.gripper_swing_model = np.zeros((2,2))
        self.swing_model_ts = time.time()
        self.swing_model_host_ts = 0.0
        self.swing_model_remote_ts = None
        self.finger_contact_calibration_complete = asyncio.Event()
        
        # State variables added to track and prevent platform drift
        self._swing_position_offset = np.zeros(2)
        self._last_future_time = 0

        # State for looking in direction of motion
        self.smoothed_error = 0.0
        self.ema_alpha = 0.3  # Smoothing factor (0 to 1)
        self.deadband = 0.02  # Radians (~1.1 degrees)
        self.p_gain = 2.0     # Proportional gain for speed calculation

    async def handle_update_from_ws(self, update):
        if 'st' in update:
            self.swing_model_remote_ts = float(update['st'])

        if 'sm' in update:
            self.gripper_swing_model = np.array(update['sm'])
            # Use host receipt time for phase projection. The Pi's st is useful
            # for diagnostics but not for closed-loop phase math because the two
            # machines' wall clocks can differ or step independently.
            swing_update_host_ts = time.time()
            self.swing_model_host_ts = swing_update_host_ts
            self.swing_model_ts = swing_update_host_ts
            
        if 'grip_sensors' in update:
            gs = update['grip_sensors']
            timestamp = gs['time']

            # rotation of gripper as quaternion. not present if IMU not installed.
            if 'quat' in gs:
                self.datastore.imu_quat.insert(np.concatenate([np.array([timestamp], dtype=float), gs['quat']]))

            distance_measurement = self.datastore.range_record.getLast()[1]
            if 'range' in gs:
                distance_measurement = float(gs['range'])
                self.datastore.range_record.insert([timestamp, distance_measurement])

            if 'raw_accel' in gs:
                logger.debug(f"raw_accel: {gs['raw_accel']}")

            if 'vel_from_imu' in gs:
                self.vel_from_imu = np.array(gs['vel_from_imu'])

            target_force = 0
            if 'dforce' in gs:
                target_force = float(gs['dforce'])

            # Note that finger angles are returned in the range of (-90, 90) even though these are not the actual angle
            # -90 is open
            finger_angle = float(gs['fing_a'])

            # finger pad pressure is indicated by this voltage with 3.3 being no pressure.
            # lower values indicate more pressure.
            voltage = float(gs['fing_v'])

            # wrist angle in degrees of rotation from the original zero point. can be more than one revolution.
            # the zero point is probably a safe bet for where the wire would be least twisted.
            # the angle at which it aligns with the gantry or the room must be determined in calibration
            wrist_angle = float(gs['wrist_a'])
            
            self.datastore.winch_line_record.insert([timestamp, wrist_angle, 0])
            self.datastore.finger.insert([timestamp, finger_angle, voltage])
            
            self.ob.send_ui(grip_sensors=telemetry.GripperSensors(
                range = distance_measurement,
                angle = finger_angle,
                pressure = voltage,
                wrist = wrist_angle,
                target_force = target_force,
            ))

        if 'finger_contact_calibration_complete' in update:
            self.finger_contact_calibration_complete.set()

    def reset_swing_correction_integrator(self):
        self._swing_position_offset = np.zeros(2)
        self._last_future_time = 0

    def _project_swing_model(self, timestamp=None):
        sm = self.gripper_swing_model
        st = self.swing_model_ts
        if sm is None or st is None:
            return None

        if timestamp is None:
            return np.asarray(sm, dtype=float)

        latency_comp = timestamp - st
        look_ahead_angle = OMEGA * latency_comp
        c_future, s_future = np.cos(look_ahead_angle), np.sin(look_ahead_angle)
        return np.asarray(sm, dtype=float) @ np.array([
            [c_future, -s_future],
            [s_future, c_future],
        ])

    def swing_energy(self, timestamp=None):
        """
        Return harmonic oscillator energy from the fitted gyro model.

        The model columns are angular velocity-like sine and cosine phase
        components. For a fixed-frequency oscillator, the squared norm of
        those columns is proportional to swing energy and is sufficient for
        ranking damping trials.
        """
        state = self._project_swing_model(timestamp)
        if state is None:
            return None
        return 0.5 * float(np.sum(state * state))

    def swing_model_age(self):
        if self.swing_model_host_ts <= 0:
            return float('inf')
        return time.time() - self.swing_model_host_ts

    def get_swing_to_room_angle(self):
        wrist = self.datastore.winch_line_record.getLast()[1]
        return wrist / 180 * np.pi + self.config.gripper.frame_room_spin - np.pi/2

    def compute_swing_correction(
        self,
        future_time,
        gain=None,
        sign=None,
        max_velocity=None,
        update_integrator=True,
    ):
        """Compute a corrective velocity to be applied at a future time in order to cancel the swing"""
        state = self._project_swing_model(future_time)
        if state is None:
            return None

        if gain is None:
            gain = getattr(self.config, 'swing_gain', DEFAULT_SWING_GAIN)
        if sign is None:
            sign = getattr(self.config, 'swing_sign', DEFAULT_SWING_SIGN)
        if max_velocity is None:
            max_velocity = getattr(self.config, 'swing_max_velocity', DEFAULT_SWING_MAX_VELOCITY)

        gain = abs(float(gain))
        sign = DEFAULT_SWING_SIGN if sign not in (-1.0, 1.0) else float(sign)
        max_velocity = max(0.0, float(max_velocity))

        # The angular acceleration (alpha) is the derivative of the velocity
        # component. Projecting first makes the configured latency/phase explicit.
        future_accel = OMEGA * state[:, 1]

        # A corrective velocity to the gantry inversely proportional to the angular velocity of the gripper cancels the swing
        raw_vel = future_accel * gain * sign

        # cancel accumulated drift introduced from swing cancellation
        # Calculate time elapsed since last call to update the integrator
        dt = 0.0
        if update_integrator:
            dt = future_time - self._last_future_time
            self._last_future_time = future_time

            # Ignore massive jumps in time if the control loop paused
            if dt > 0.5 or dt < 0:
                dt = 0.0

        # Apply a centering restorative velocity proportional to the accumulated position offset
        centering_vel = self._swing_position_offset * CENTERING_GAIN if update_integrator else np.zeros(2)
        vel = raw_vel - centering_vel

        vel_norm = float(np.linalg.norm(vel))
        if max_velocity == 0:
            vel = np.zeros(2)
        elif vel_norm > max_velocity:
            vel = vel * (max_velocity / vel_norm)

        # Track the accumulated position offset based on the clamped velocity
        # we are actually commanding.
        if update_integrator:
            self._swing_position_offset += vel * dt

        # rotate vector into room frame of reference
        return rotate_vector(vel, -self.get_swing_to_room_angle())

    def handle_detections(self, detections, timestamp):
        """
        handle a list of tag detections from the pool
        """
        self.stat.pending_frames_in_pool -= 1
        self.stat.detection_count += len(detections)
        # setting to none every frame so we know whether it's in frame by looking at this variable
        self.park_pose_relative_to_camera = None

        for detection in detections:
            if detection['n'] == 'park_target':
                # pose of parking target relative to gripper camera
                self.park_pose_relative_to_camera = detection['p']

    async def send_config(self):
        pass

    def get_gripper_rvec(self, timestamp=None):
        """
        Calculates the rotation of the gripper in its local frame of reference
        at a specific timestamp, not counting the wrist.
        """
        if timestamp is None:
            projected_state = self.gripper_swing_model
        else:
            # Calculate how much the phase has evolved between the model's last update and the requested timestamp.
            dt = timestamp - self.swing_model_ts
            angle = OMEGA * dt
            c, s = np.cos(angle), np.sin(angle)
            # Project the state matrix to the target timestamp using a rotation matrix.
            # This allows us to find the A*sin and A*cos components at that exact moment.
            projected_state = self.gripper_swing_model @ np.array([[c, -s], [s, c]])
        
        # In a harmonic oscillator, displacement is the integral of velocity.
        # For a model where Col 0 is Velocity (A*sin), the displacement is -A/omega * cos.
        # This corresponds to the negative of the phase tracker (Col 1) divided by omega.
        theta_x = projected_state[0, 1] / OMEGA
        theta_y = projected_state[1, 1] / OMEGA
        return np.array([theta_x, theta_y, 0])

    def get_spin(self, debug=False):
        # return the rotation of the gripper camera relative to the room in radians
        roomspin = self.datastore.winch_line_record.getLast()[1] / 180 * np.pi
        if not self.calibrating_room_spin and self.config.gripper.frame_room_spin is not None:
            # undo the rotation that the room would appear to have at the wrist's 540 position
            extra = self.config.gripper.frame_room_spin - np.pi
            if debug:
                print(f'gripper spin should be wrist {roomspin} plus extra spin from config {extra}')
            roomspin = roomspin + extra
        return roomspin

    def look_towards_vector(self, vec2):
        """
        Turn the head to face in the direction of the given XY vector in room space.
        vec2: A numpy array [x, y]
        """
        # Calculate target angle from vector
        target_angle_base = math.atan2(vec2[0], vec2[1]) # Result in (-pi, pi]

        # Spin ranges from 0 to 6*pi. Nose @ +Y is spin % 2pi == 0.
        current_spin = self.get_spin()
        
        # Determine the best target within the [0, 6*pi] range
        # There are 3 possible rotations that face the same direction:
        # base_angle (normalized to [0, 2pi]), base_angle + 2pi, and base_angle + 4pi.
        norm_target = target_angle_base % (2 * math.pi)
        candidates = [norm_target, norm_target + 2 * math.pi, norm_target + 4 * math.pi]
        
        # Determine proximity to bounds
        lower_bound = 0.5 * math.pi
        upper_bound = 5.5 * math.pi
        center_point = 3 * math.pi

        if current_spin < lower_bound:
            # Near lower limit: Force selection of a candidate that moves us toward center
            # Typically picking the candidate > current_spin
            target = min([c for c in candidates if c > current_spin] or [candidates[-1]])
        elif current_spin > upper_bound:
            # Near upper limit: Force selection of a candidate that moves us toward center
            target = max([c for c in candidates if c < current_spin] or [candidates[0]])
        else:
            # Normal operation: Pick the closest candidate
            target = min(candidates, key=lambda c: abs(c - current_spin))

        # Calculate raw error
        raw_error = target - current_spin

        # Apply Deadband
        if abs(raw_error) < self.deadband:
            raw_error = 0.0

        # Exponential Moving Average (EMA) Smoothing
        # smoothed = alpha * new + (1 - alpha) * old
        self.smoothed_error = (self.ema_alpha * raw_error) + (1.0 - self.ema_alpha) * self.smoothed_error

        # Convert Error to Speed (Degrees per Second)
        # Convert radians to degrees: radians * (180 / pi)
        # Apply a proportional gain
        wrist_speed_deg = self.smoothed_error * self.p_gain * (180.0 / math.pi)

        # 8. Clamp and Send
        wrist_speed = clamp(wrist_speed_deg, -120, 120)
        asyncio.create_task(self.send_commands({'set_wrist_speed': wrist_speed}))

    def process_frame(self, frame_to_encode):
        # an action space in which the gripper camera is not stabilized or rotated.
        # no matter what perspective the operator is driving with, the network is always seeing
        # control inputs relative to the gripper image. it will see a +Y direction when the motion is up in
        # the gripper image.
        input_shape = (frame_to_encode.shape[1], frame_to_encode.shape[0])
        if input_shape != SF_TARGET_SHAPE:
            temp_image = cv2.resize(frame_to_encode, SF_TARGET_SHAPE, interpolation=cv2.INTER_AREA)
        else:
            temp_image = frame_to_encode
        return temp_image
