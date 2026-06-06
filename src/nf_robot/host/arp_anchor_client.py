import asyncio
import math
import numpy as np
from collections import deque
import threading

from nf_robot.host.anchor_client import ComponentClient
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.generated.nf import telemetry, common
from nf_robot.common.cv_common import *
from nf_robot.common.pose_functions  import *
from nf_robot.common.util import *
from nf_robot.observability import OBS

# looking for cranebot-anchor-arpeggio-service

"""
"Arpeggio" is the codename of the 2nd revision of the Stringman

The new anchors differ drastically from the pilot version. Pairs of lines are combined into units,
So each server will report the length of two different lines. One line spans from the anchor to the marker box directly.
the other passes through a ceramic eyelet on an adjacent wall, referred to as the "indirect line"
What we pass as the reference_length to each spool controller determines the length we get back.
for the indirect line, we pass the length between the ceramic eyelet and the marker box.

Determining the real position of the ceramic eyelet is done during calibration.

These anchors also use a direct drive BLDC motor with built in FOC controller that continuously reports torque rather than
a binary tight/slack value.

"""

class ArpeggioAnchorClient(ComponentClient):
    def __init__(self, address, port, anchor_num, datastore, ob, pool, stat, telemetry_env):
        super().__init__(address, port, datastore, ob, pool, stat, telemetry_env)
        self.anchor_num = anchor_num
        self.conn_status = telemetry.ComponentConnStatus(
            is_gripper=False,
            anchor_num=self.anchor_num,
            websocket_status=telemetry.ConnStatus.NOT_DETECTED,
            video_status=telemetry.ConnStatus.NOT_DETECTED,
            gripper_model=telemetry.GripperModel.ARPEGGIO,
        )
        self.anchor_pose = np.zeros((2, 3))
        self.camera_pose = np.zeros((2, 3))
        self.eye_pos = np.zeros(3)
        self.raw_gant_poses = deque(maxlen=24)
        self.gantry_pos_sightings = deque(maxlen=100)
        self.gantry_pos_sightings_lock = threading.RLock()
        self.line_action_states = []
        self.last_line_action = None

        self.updatePoseAndEye(
            poseProtoToTuple(self.config.anchors[anchor_num].pose),
            tonp(self.config.anchors[anchor_num].indirect_line.eyelet_pos),
        )

    async def send_config(self):
        anchor_config_vars = {}
        # TODO
        if len(anchor_config_vars) > 0:
            await self.websocket.send(json.dumps({'set_config_vars': anchor_config_vars}))

    def updatePoseAndEye(self, pose=None, eye=None):
        if pose is not None:
            self.anchor_pose = pose
        if eye is not None:
            self.eye_pos = eye
        # 22 is the tilt of the camera in the model
        extratilt = 22 - self.config.anchors[self.anchor_num].indirect_line.cam_tilt
        self.camera_pose = np.array(compose_poses([
            self.anchor_pose,
            model_constants.arp_anchor_camera,
            (np.array([extratilt/180*np.pi, 0, 0], dtype=float), np.zeros(3, dtype=float)),
        ]))

    async def handle_update_from_ws(self, update):
        if 'spool0' in update:
            self.storeSpoolData(0, update['spool0'])
        if 'spool1' in update:
            self.storeSpoolData(1, update['spool1'])
        if 'line_action_states' in update:
            self.line_action_states = [
                dict(state)
                for state in update['line_action_states']
                if isinstance(state, dict)
            ]
        if 'line_action' in update and isinstance(update['line_action'], dict):
            self.last_line_action = dict(update['line_action'])

        if len(self.gantry_pos_sightings) > 0:
            with self.gantry_pos_sightings_lock:
                self.ob.send_ui(gantry_sightings=telemetry.GantrySightings(
                    sightings=[common.Vec3(*position) for position in self.gantry_pos_sightings]
                ))
                self.gantry_pos_sightings.clear()

    def storeSpoolData(self, spool_no, data):
        # data= [(time, line_length, line_speed, tension, optional_motor_metadata), ...]
        line_number = self.anchor_num * 2 + spool_no
        rows = []
        for sample in data or []:
            if not isinstance(sample, (list, tuple, np.ndarray)) or len(sample) < 4:
                continue
            try:
                row = [float(sample[0]), float(sample[1]), float(sample[2]), float(sample[3])]
            except (TypeError, ValueError):
                continue
            metadata = sample[4] if len(sample) > 4 and isinstance(sample[4], dict) else {}
            if not metadata:
                spool_rate = self._estimated_spool_unspool_rate(spool_no, row[1])
                metadata = {
                    "motor_controller_available": False,
                    "spool_unspool_rate_m_per_rev": spool_rate,
                }
            rows.append(row)
            OBS.record_line_motor_sample(
                line_number,
                timestamp_s=row[0],
                length_m=row[1],
                line_speed_m_s=row[2],
                tension_n=row[3],
                metadata=metadata,
            )
        if rows:
            self.datastore.anchor_line_record[line_number].insertList(np.array(rows, dtype=float))
            self.datastore.anchor_line_record_event.set()

    def _estimated_spool_unspool_rate(self, spool_no, length_m):
        full_length = model_constants.assumed_full_line_length
        empty_diameter = model_constants.damiao_empty_spool_diameter
        full_diameter = (
            model_constants.damiao_full_spool_diameter_power_line
            if spool_no == 0
            else model_constants.damiao_full_spool_diameter_fishing_line
        )
        try:
            spooled_length = np.clip(full_length - float(length_m), 0.0, full_length)
        except (TypeError, ValueError):
            spooled_length = 0.0
        effective_diameter_mm = empty_diameter + (full_diameter - empty_diameter) * (spooled_length / full_length)
        return math.pi * effective_diameter_mm * 0.001

    def handle_detections(self, detections, timestamp):
        """
        handle a list of apriltag detections from the pool
        """
        self.stat.pending_frames_in_pool -= 1
        self.stat.detection_count += len(detections)

        for detection in detections:
            name = detection['n']
            self.last_known_centers[name] = detection['center']

            if name in CAL_MARKERS:
                # save all the detections of the origin for later analysis
                self.origin_poses[detection['n']].append(detection['p'])
                # if detection['n'] == "origin":
                #     print(detection)

            if name == 'gantry':
                # rotate and translate to where that object's origin would be
                # given the position and rotation of the camera that made this observation (relative to the origin)
                # store the time and that position in the appropriate measurement array in observer.
                # you have the pose of gantry_front relative to a particular anchor camera
                # convert it to a pose relative to the origin
                pose = np.array(compose_poses([
                    self.camera_pose, # config dependent
                    detection['p'], # the pose obtained just now
                    gantry_april_inv, # constant
                ]))
                position = pose[1] # take only the position from the pose
                self.datastore.gantry_pos.insert(np.concatenate([[timestamp], [self.anchor_num], position]))
                # print(f'Inserted gantry pose ts={timestamp}, pose={pose}')
                self.datastore.gantry_pos_event.set()

                self.last_gantry_frame_coords = detection['p'][1] # second item in pose tuple is position
                with self.gantry_pos_sightings_lock:
                    self.gantry_pos_sightings.append(position)

                if self.save_raw:
                    self.raw_gant_poses.append(detection['p'])

            if name in OTHER_MARKERS:
                offset = model_constants.basket_offset_inv if name.endswith('back') else model_constants.basket_offset
                pose = np.array(compose_poses([
                    self.camera_pose, # config dependent
                    detection['p'], # the pose obtained just now
                    offset, # the named location is out in front of the tag.
                ]))
                position = pose.reshape(6)[3:]
                # save the position of this object for use in various planning tasks.
                self.ob.update_avg_named_pos(detection['n'], position)


    def process_frame(self, frame_to_encode):
        return frame_to_encode
