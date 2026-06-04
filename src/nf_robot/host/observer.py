from __future__ import annotations

import signal
import sys
import threading
import time
import socket
import asyncio
import argparse
import logging
import os
from zeroconf import IPVersion, ServiceStateChange, Zeroconf
from zeroconf.asyncio import (
    AsyncServiceBrowser,
    AsyncServiceInfo,
    AsyncZeroconf,
    AsyncZeroconfServiceTypes,
    InterfaceChoice,
)
from multiprocessing import Pool, Process
import numpy as np
import scipy.optimize as optimize
from scipy.spatial.transform import Rotation
from random import random
import traceback
import cv2
import pickle
from collections import deque, defaultdict
import uuid
import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from functools import partial
from pathlib import Path
import json
import re
import subprocess

logger = logging.getLogger(__name__)

DEFAULT_CV_THREADS = 1
DEFAULT_TORCH_THREADS = 4
DEFAULT_TORCH_INTEROP_THREADS = 1
MAX_SWING_MODEL_AGE_S = 2.0


def _env_int(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def configure_native_thread_pools(configure_torch=False):
    cv_threads = _env_int("NF_ROBOT_CV_THREADS", DEFAULT_CV_THREADS)
    try:
        cv2.setNumThreads(cv_threads)
    except Exception:
        logger.exception("Failed to configure OpenCV thread count")

    if configure_torch:
        try:
            import torch
            torch.set_num_threads(_env_int("NF_ROBOT_TORCH_THREADS", DEFAULT_TORCH_THREADS))
            torch.set_num_interop_threads(_env_int("NF_ROBOT_TORCH_INTEROP_THREADS", DEFAULT_TORCH_INTEROP_THREADS))
        except RuntimeError:
            # Torch only allows interop threads to be set before parallel work starts.
            logger.warning("Torch thread pools were already initialized; leaving existing settings")
        except ImportError:
            pass


def configure_worker_process():
    configure_native_thread_pools(configure_torch=False)


configure_native_thread_pools(configure_torch=False)

from nf_robot.common.pose_functions import compose_poses
from nf_robot.common.cv_common import *
from nf_robot.common.config_loader import *
import nf_robot.common.definitions as model_constants
from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, control, common
import nf_robot.generated.nf.config as nf_config
from nf_robot.host.data_store import DataStore
from nf_robot.host.stats import StatCounter
from nf_robot.host.target_queue import TargetQueue
from nf_robot.host.calibration import optimize_anchor_poses
from nf_robot.host.calibration_artifacts import CalibrationArtifactSession
from nf_robot.host.eyelet_calibration import optimize_arp_anchors, analyze_diamond_data, DIAMOND_SIZE
from nf_robot.host.anchor_client import RaspiAnchorClient, max_origin_detections
from nf_robot.host.gripper_client import RaspiGripperClient
from nf_robot.host.arp_gripper_client import ArpeggioGripperClient, rotate_vector
from nf_robot.host.arp_anchor_client import ArpeggioAnchorClient
from nf_robot.host.position_estimator import Positioner2
from nf_robot.observability import OBS, init_observability

# Define the service names for network discovery
anchor_service_name = 'cranebot-anchor-service'
anchor_power_service_name = 'cranebot-anchor-power-service'
gripper_service_name = 'cranebot-gripper-service'
arp_gripper_service_name = 'cranebot-gripper-arpeggio-service'
arp_anchor_service_name = 'cranebot-anchor-arpeggio-service'

N_ANCHORS = {
    common.AnchorType.PILOT: 4,
    common.AnchorType.ARPEGGIO: 2,
}
N_LINES = 4
TENSION_WAIT_TIMEOUT_S = 10.0
TENSION_POLL_INTERVAL_S = 0.1
TENSION_SPEED_NORM_THRESHOLD = 0.01  # m/s
TENSION_RECORD_MAX_AGE_S = 2.0
REFERENCE_VISUAL_MAX_AGE_S = 2.0
CAL_DIAMOND_MIN_HALF_HEIGHT = 0.02
CAL_DIAMOND_MIN_HALF_WIDTH = 0.05
PASSIVE_SAFE_TENSION_N = 17.0
PASSIVE_SAFE_TENSION_RETRY_BUMP_N = 0.5
PASSIVE_SAFE_RETRY_WINDOW_S = 3.0
PASSIVE_SAFE_RETRY_MOVE_MAX_AGE_S = 5.0
PASSIVE_SAFE_RETRY_COOLDOWN_S = 10.0
INFO_REQUEST_TIMEOUT_MS = 3000 # milliseconds
CONTROL_PLANE_PRODUCTION = "wss://neufangled.com"
CONTROL_PLANE_STAGING = "wss://nf-site-monolith-staging-690802609278.us-east1.run.app"
CONTROL_PLANE_LOCAL = "ws://localhost:8080"
UNPROCESSED_DIR = "square_centering_data_unlabeled"
USER_TARGETS_DIR = "user_targets_data"
METADATA_PATH = os.path.join(USER_TARGETS_DIR, "metadata.jsonl")

# threshold of non slack tension in newtons for arp anchors
TENSION_THRESH = 1.38

CRANEBOT_SERVICE_TYPES = [
    "_cranebot-gripper-arpeggio-service._tcp.local.",
    "_cranebot-gripper-service._tcp.local.",
    "_cranebot-anchor-power-service._tcp.local.",
    "_cranebot-anchor-service._tcp.local.",
]

# finger positions
OPEN = -30
CLOSED = 90

POLE = np.array([0,0,0.5334])
GRIPPER_HEIGHT_OVER_TARGET = np.array([0,0,0.3])

def capture_gripper_image(ndimage, gripper_occupied=False):
    """
    Saves an image to the unprocessed directory. 
    Encodes gripper state in filename: {uuid}_g{1|0}.jpg
    """
    if not os.path.exists(UNPROCESSED_DIR):
        os.makedirs(UNPROCESSED_DIR)
    
    h, w = ndimage.shape[:2]
        
    state_str = "g1" if gripper_occupied else "g0"
    file_id = str(uuid.uuid4())
    img_filename = f"{file_id}_{state_str}.jpg"
    img_full_path = os.path.join(UNPROCESSED_DIR, img_filename)
    
    # Save (ensure RGB/BGR consistency)
    cv2.imwrite(img_full_path, ndimage)
    logger.info(f"Captured: {img_filename} (Gripper: {gripper_occupied})")

class AsyncObserver:
    """
    Manager of multiple tasks running clients connected to each robot component
    The job of this class in a nutshell is to discover four anchors and a gripper on the network,
    connect to them, and forward data between them and the position estimator, shape tracker, and UI.

    It reads from the config file to find any components it already knows about.
    It starts zeroconf to discover any components it doesn't know about and add them to the config.
    it starts keep_robot_connected to continually reconnect to all known components.
    It starts position_estimator to continually run kalman filters on the observed variables.
    It starts run_perception to continually run inference on the camera feeds.
    It starts a websocket server to accept connections from local UIs 

    It starts a websocket server to accept connections from local UIs 
    It reads from the config file to find any components it already knows about.
    It starts zeroconf to discover any components it doesn't know about and add them to the config.
    As soon as a component in the config has a known address, it starts keep_robot_connected to continually reconnect to all known components.
    As soon as the first component websocket is connected, It starts position_estimator to continually run kalman filters on the observed variables.
    As soon as a feed from the first preferred camera is up, It starts run_perception to continually run inference on the camera feeds.

    Since this class serves as the coordination center of all the robot compnents, it also contains methods to perform
    various actions like calibration and the pick and place routine.
    """
    def __init__(
        self,
        terminate_with_ui,
        config_path,
        telemetry_env=None,
        run_ai=True,
        run_ortho=True,
        auto_start=False,
        local_models=False,
        port=4245,
        use_arp_grasp=False,
        debug=False,
        observability_debug=False,
        observability_metrics_host=None,
        observability_metrics_port=None,
        observability_log_path=None,
    ) -> None:
        self.port = port
        self.terminate_with_ui = terminate_with_ui
        self.position_update_task = None
        self.aiobrowser: AsyncServiceBrowser | None = None
        self.aiozc: AsyncZeroconf | None = None
        self.run_command_loop = True
        self.datastore = DataStore()
        self.pool = None
        # all clients by server name
        self.bot_clients = {}
        # all connected anchors keyed by anchor num
        self.anchors = {}
        # convenience reference to gripper client
        self.gripper_client = None
        # TODO allow a command line argument to override the config file path
        self.config_path = config_path
        self.config = load_config(config_path)
        self.debug = debug
        init_observability(
            service_name="stringman-headless",
            robot_id=getattr(self.config, "robot_id", None),
            metrics_host=observability_metrics_host,
            metrics_port=observability_metrics_port,
            log_path=observability_log_path,
            log_level="DEBUG" if debug or observability_debug else None,
        )
        OBS.set_expected_components(anchors=self.config.anchors, gripper=self.config.gripper is not None)
        self.started_at = time.time()
        self.telemetry_env = telemetry_env
        self.stat = StatCounter(self)
        self.enable_shape_tracking = False
        self.shape_tracker = None
        # Position Estimator. this used to be a seperate process so it's still somewhat independent.
        self.pe = Positioner2(self.datastore, self)
        self.locate_anchor_task = None
        # only one motion task can be active at a time
        self.motion_task = None
        # only used for integration test only to allow some code to run right after sending the gantry to a goal point
        self.test_gantry_goal_callback = None
        # event used to notify tasks that gripper is connected.
        self.gripper_client_connected = asyncio.Event()
        self.last_user_move_time = time.time()
        self.named_positions = {}
        self.target_model = None
        self.centering_model = None
        self.predicted_lateral_vector = None
        self.perception_task = None
        # targets
        self.target_queue = TargetQueue()
        self.last_snapshot_hash = None # to spare the UI from too many updates
        # websockets to locally connected UIs
        self.connected_local_clients = set()
        self.telemetry_buffer = deque(maxlen=100)
        self.telemetry_buffer_lock = threading.RLock()
        self.startup_complete = asyncio.Event()
        self.any_anchor_connected = asyncio.Event() # fires as soon as first anchor connects, starting pe
        self.cloud_telem_websocket = None
        self.gip_task = None
        self.cloud_telem = None
        self.passive_safety_task = None
        self.observability_task = None
        # last attempt to connect, keyed by service name
        self.connection_tasks: dict[str, asyncio.Task] = {}
        self.run_collect_images = False
        self.time_last_grip_sensors_retain_key = 0
        # dict of vectors representing last velocities commanded by different subsystems. all keys in active_set are summard
        self.input_velocities = {'default': np.zeros(3)}
        self.active_set = set(['default'])
        self.last_retryable_move = None
        self._passive_safety_tension_limit_extra_until = 0.0
        self._passive_safety_retry_history = {}
        self._passive_safety_last_final_prompt_ts = 0.0
        self.run_ai = run_ai
        self.run_ortho = run_ortho
        self.auto_start = auto_start
        self.use_arp_grasp = use_arp_grasp
        self.swing_cancellation_task = None
        self.local_models = local_models
        # ortho projection state - written by _ortho_worker thread, read by run_perception AI task
        self.ortho_event = threading.Event()
        self.last_ortho_bgr = None
        self.last_ortho_heatmap = None
        self.last_heatmaps_np = None
        # list of (NfVideoStreamer, feed_number) for ortho feeds, so send_setup_telemetry can replay them
        self.ortho_streamers: list = []
        self.lerobot_process_watcher = None
        self.last_ep_ctrl_status = common.LerobotStatus.NA
        self.lerobot_process_pid = None
        self.grip_angle = 0

    async def send_setup_telemetry(self):
        logger.debug('Sending setup telemetry')
        if self.config.anchor_type == common.AnchorType.ARPEGGIO:
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
                poses=[a.pose for a in self.config.anchors],
                eyelets=[a.indirect_line.eyelet_pos for a in self.config.anchors],
                tilt=[a.indirect_line.cam_tilt for a in self.config.anchors],
                swing_latency=self.config.swing_latency,
            ))
        else:
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
                poses=[a.pose for a in self.config.anchors]
            ))
        if self.config.park_data is not None:
            self.send_ui(named_position=telemetry.NamedObjectPosition(
                name = 'parking_location',
                position = self.config.park_data.pos
            ))
        for client in self.bot_clients.values():
            client.send_conn_status()
            if (client.local_video_uri is not None or client.remote_stream_path is not None) and client.anchor_num in [None, *self.config.preferred_cameras]:
                self.send_ui(video_ready=telemetry.VideoReady(
                    is_gripper=client.anchor_num is None,
                    anchor_num=client.anchor_num,
                    local_uri=client.local_video_uri,
                    feed_number=client.feed_number,
                    stream_path=client.remote_stream_path,
                ))
        for vs, feed_number in self.ortho_streamers:
            if vs._ready_sent:
                self.send_ui(video_ready=telemetry.VideoReady(
                    is_gripper=None,
                    anchor_num=None,
                    local_uri=vs.local_uri,
                    stream_path=vs.stream_path,
                    feed_number=feed_number,
                ))
        if self.lerobot_process_watcher is None or self.lerobot_process_watcher.done():
            self.last_ep_ctrl_status = common.LerobotStatus.NA
        self.send_ui(episode_control=common.EpisodeControl(status = self.last_ep_ctrl_status))
        r = await self.flush_tele_buffer()

    async def handle_local_client(self, websocket):
        # Called when Ursina connects to a websocket that is opened to accept control commands
        self.connected_local_clients.add(websocket)
        OBS.set_ui_clients(len(self.connected_local_clients))
        logger.info('Connection received from local UI process')

        with OBS.span("observer.local_ui.websocket", client_count=len(self.connected_local_clients)):
            # send anything that it would need up-front
            r = await self.send_setup_telemetry()
            try:
                async for message in websocket:
                    r = await self.handle_command(message) # Handle 'ControlBatchUpdate'
                    # warning, any uncaught exception here will kill this websocket connection
                    # but the observer would go on running, possibly in a bad state.
            except (ConnectionClosedError, ConnectionClosedOK) as e:
                pass
            # except Exception as e:
            #     print(e)
            #     traceback.print_exc()
            finally:
                self.connected_local_clients.remove(websocket)
                OBS.set_ui_clients(len(self.connected_local_clients))
                if len(self.connected_local_clients) == 0 and self.terminate_with_ui:
                    # The only local UI has disconnected and we were asked to shutdown when it disconnects
                    self.run_command_loop = False

    async def handle_command(self, message: bytes):
        """ Decodes a binary batch of commands """
        # betterproto .parse() returns a standard python dataclass
        started = time.time()
        try:
            batch = control.ControlBatchUpdate().parse(message)
        except Exception as exc:
            logger.warning(
                'Ignoring malformed control batch (%d bytes): %s',
                len(message),
                exc,
            )
            self.send_ui(pop_message=telemetry.Popup(
                message='Ignored malformed control command.'
            ))
            OBS.record_command("malformed_batch", time.time() - started)
            return
        with OBS.span("observer.handle_command_batch", bytes=len(message), updates=len(batch.updates)):
            for update in batch.updates:
                r = await self._dispatch_update(update)
        OBS.record_command("batch", time.time() - started)

    def _control_item_name(self, item: control.ControlItem) -> str:
        if item.command is not None:
            return f"command.{item.command.name}"
        for name in (
            "move",
            "gantry_goal_pos",
            "jog_spool",
            "episode_control",
            "scale_room",
            "add_cam_target",
            "delete_target",
            "debug",
            "set_swing_cancellation",
            "single_component_action",
            "manage_lerobot_session",
            "move_gripper_to",
        ):
            if getattr(item, name) is not None:
                return name
        return "unknown"

    async def _dispatch_update(self, item: control.ControlItem):
        # In betterproto2, 'oneof' fields appear as attributes. 
        # Only one will be non-None.
        # not that checking if the field is truthy is insufficient, as a default instance of the proto is false
        # and default instances can carry meaningful information such as zeroing out a value.
        command_name = self._control_item_name(item)
        started = time.time()
        try:
            with OBS.span("observer.dispatch_update", command=command_name):
                return await self._dispatch_update_inner(item)
        finally:
            OBS.record_command(command_name, time.time() - started)

    async def _dispatch_update_inner(self, item: control.ControlItem):
        
        # Standard Commands (Stop, Calibrate, Zero)
        if item.command is not None:
            r = await self._handle_common_command(item.command.name)

        # Movement Vector (Gamepad/AI Policy)
        elif item.move is not None:
            r = await self._handle_movement(item.move)

        # Setting gantry goal
        elif item.gantry_goal_pos is not None:
            r = await self._handle_gantry_goal_pos(tonp(item.gantry_goal_pos.pos))

        # Manual Spool Control
        elif item.jog_spool is not None:
            r = await self._handle_jog_spool(item.jog_spool)

        # Lerobot Episode Control (Start/Stop Recording)
        elif item.episode_control is not None:
            self._handle_add_episode_control_events(item.episode_control)

        elif item.scale_room is not None:
            self._handle_scale_room(item.scale_room)

        elif item.add_cam_target is not None:
            self._handle_add_cam_target(item.add_cam_target)

        elif item.delete_target is not None:
            self._handle_delete_target(item.delete_target)

        elif item.debug is not None:
            r = await self._handle_debug_command(item.debug)

        elif item.set_swing_cancellation is not None:
            r = await self._handle_set_swing_cancellation(item.set_swing_cancellation)

        elif item.single_component_action is not None:
            r = await self._handle_single_component_action(item.single_component_action)

        elif item.manage_lerobot_session is not None:
            self.lerobot_process_watcher = asyncio.create_task(self.lerobot_process(item.manage_lerobot_session))

        elif item.move_gripper_to is not None:
            r = await self._handle_move_gripper_to(item.move_gripper_to)

    async def _handle_move_gripper_to(self, item: control.MoveGripperTo):
        logging.debug(f'_handle_move_gripper_to {item}')
        goal_pos = None
        if item.target_id is not None:
            # derive target position from target
            target = self.target_queue.get_target_info(item.target_id)
            if target is not None:
                goal_pos = tonp(target.position) + GRIPPER_HEIGHT_OVER_TARGET + POLE
        elif item.pos is not None:
            goal_pos = tonp(item.pos) + POLE

        if goal_pos is None:
            return
        self.gantry_goal_pos = goal_pos
        r = await self.invoke_motion_task(self.seek_gantry_goal())

    async def _handle_single_component_action(self, item: control.SingleComponentAction):
        """Issue a special command to a single component"""
        client = None
        if item.is_gripper:
            client = self.gripper_client
        else:
            client = self.anchors.get(item.anchor_num, None)
        if client is not None:
            spool_actions = (
                control.ComponentAction.TIGHTEN,
                control.ComponentAction.RELAX,
                control.ComponentAction.STOW,
            )
            spool_num = None
            if (
                not item.is_gripper
                and self.config.anchor_type == common.AnchorType.ARPEGGIO
                and item.action in spool_actions
            ):
                if item.spool_num is None:
                    logger.warning(
                        "Ignoring ARP %s action for anchor %s without spool_num",
                        item.action,
                        item.anchor_num,
                    )
                    return
                spool_num = int(item.spool_num)

            if item.action == control.ComponentAction.REBOOT:
                r = await client.send_commands({'reboot': None})
            elif item.action == control.ComponentAction.IDENTIFY:
                r = await client.send_commands({'identify': None})
            elif item.action == control.ComponentAction.TIGHTEN:
                r = await client.send_commands({'tighten': spool_num})
            elif item.action == control.ComponentAction.RELAX:
                r = await client.send_commands({'relax': spool_num})
            elif item.action == control.ComponentAction.STOW:
                r = await client.send_commands({'stow': spool_num})
            elif item.action == control.ComponentAction.SET_CAM_ANGLE and self.config.anchor_type == common.AnchorType.ARPEGGIO:
                self.config.anchors[item.anchor_num].indirect_line.cam_tilt = item.cam_angle
                save_config(self.config, self.config_path)
                self.anchors[item.anchor_num].updatePoseAndEye()

    async def _handle_set_swing_cancellation(self, item: control.SetSwingCancellation):
        logger.info(f'Swing cancellation set {item.enabled}')
        if item.enabled:
            if not isinstance(self.gripper_client, ArpeggioGripperClient):
                self.send_ui(pop_message=telemetry.Popup(
                    message=f'Swing cancellation only supported on Arpeggio Gripper'
                ))
                return
            # Does it need to be enabled?
            if self.swing_cancellation_task is None or self.swing_cancellation_task.done():
                self.swing_cancellation_task = asyncio.create_task(self.run_swing_cancellation())
        else:
            # does it need to be disabled?
            if self.swing_cancellation_task is not None and not self.swing_cancellation_task.done():
                self.swing_cancellation_task.cancel()

    async def run_swing_cancellation(self):
        """ Task which adds swing cancellation inputs. """

        # TODO attempt to measure this. It is the round trip latency between IMU measurements on the grpper and when our inputs move the spools.
        # latency = 0.18 # works best for desktop machine?
        # latency = 0.61 # works best for laptop
        # when it seems wonky, sometimes it's because the gripper has a different timezone setting than the host!
        # come up with a way to sync them.
        try:
            self.gripper_client.reset_swing_correction_integrator()
            self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=True, present='.'))
            self.active_set.add('swingc')
            energy_window = deque(maxlen=120)
            initial_energy = None
            started = time.time()
            while self.run_command_loop:
                now = time.time()
                model_age = self.gripper_client.swing_model_age()
                if model_age > MAX_SWING_MODEL_AGE_S:
                    logger.warning('Swing cancellation disabled because IMU model is stale (%.2fs old)', model_age)
                    self.send_ui(pop_message=telemetry.Popup(
                        message='Swing cancellation disabled because the gripper IMU model stopped updating.'
                    ))
                    break
                gain = min(max(abs(self.config.swing_gain), 0.0), MAX_SWING_GAIN)
                max_velocity = min(max(abs(self.config.swing_max_velocity), 0.0), MAX_SWING_MAX_VELOCITY)
                vel2 = self.gripper_client.compute_swing_correction(
                    now + self.config.swing_latency,
                    gain=gain,
                    sign=self.config.swing_sign,
                    max_velocity=max_velocity,
                )
                if vel2 is not None:
                    await self.move_direction_speed(np.array([vel2[0], vel2[1], 0]), key='swingc', downward_bias=0)

                energy = self.gripper_client.swing_energy()
                if energy is not None and np.isfinite(energy):
                    energy_window.append(float(energy))
                    if initial_energy is None and len(energy_window) >= 30:
                        initial_energy = float(np.mean(energy_window))

                    energy_limit = self._swing_amplification_limit(initial_energy) if initial_energy is not None else None
                    if (
                        initial_energy is not None
                        and now - started > 4.0
                        and self._non_swing_velocity_norm() < 0.01
                        and np.mean(energy_window) > energy_limit
                    ):
                        logger.warning(
                            'Swing cancellation energy increased from %.6f to %.6f; disabling',
                            initial_energy,
                            float(np.mean(energy_window)),
                        )
                        self.send_ui(pop_message=telemetry.Popup(
                            message='Swing cancellation appears to amplify swing. Run swingcal before re-enabling.'
                        ))
                        break
                await asyncio.sleep(1/100)
        except asyncio.CancelledError:
            pass
        finally:
            self.input_velocities['swingc'] = np.zeros(3)
            self.active_set.discard('swingc')
            self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=False, present='.'))
            self.slow_stop_all_spools()

    def _non_swing_velocity_norm(self):
        total = np.zeros(3)
        for key in self.active_set:
            if key == 'swingc':
                continue
            total += self.input_velocities.get(key, np.zeros(3))
        return float(np.linalg.norm(total))

    def _swing_amplification_limit(self, initial_energy):
        floor = self.config.swing_auto_min_energy
        if initial_energy < floor:
            return floor * self.config.swing_auto_abort_ratio
        return max(initial_energy * self.config.swing_auto_abort_ratio, floor)

    def _swing_energy_summary(self, samples):
        if len(samples) == 0:
            return {
                'mean': float('inf'),
                'head_mean': float('inf'),
                'tail_mean': float('inf'),
                'ratio': float('inf'),
                'slope': float('inf'),
            }
        arr = np.array(samples, dtype=float)
        times = arr[:, 0] - arr[0, 0]
        energies = arr[:, 1]
        n = max(1, min(len(energies) // 4, 30))
        head_mean = float(np.mean(energies[:n]))
        tail_mean = float(np.mean(energies[-n:]))
        ratio = tail_mean / max(head_mean, 1e-12)
        slope = 0.0
        if len(samples) >= 3 and np.max(times) > 0:
            slope = float(np.polyfit(times, energies, 1)[0])
        return {
            'mean': float(np.mean(energies)),
            'head_mean': head_mean,
            'tail_mean': tail_mean,
            'ratio': ratio,
            'slope': slope,
        }

    def _swing_should_abort_calibration(self, samples, abort_energy):
        if len(samples) == 0 or not np.isfinite(abort_energy):
            return False
        energies = np.array([energy for _, energy in samples[-12:]], dtype=float)
        energies = energies[np.isfinite(energies)]
        if len(energies) == 0:
            return False
        if float(np.max(energies)) > abort_energy * 1.5:
            return True
        return len(energies) >= 12 and float(np.mean(energies)) > abort_energy

    def _swing_calibration_trial_plan(self):
        try:
            latency = float(getattr(self.config, 'swing_latency', DEFAULT_SWING_LATENCY))
        except (TypeError, ValueError):
            latency = DEFAULT_SWING_LATENCY
        if not np.isfinite(latency):
            latency = DEFAULT_SWING_LATENCY
        latency = min(max(latency, 0.0), 0.8)
        latency_candidates = []
        for offset in (-0.10, 0.0, 0.10):
            candidate_latency = round(min(max(latency + offset, 0.0), 0.8), 3)
            if candidate_latency not in latency_candidates:
                latency_candidates.append(candidate_latency)

        gain = getattr(self.config, 'swing_gain', DEFAULT_SWING_GAIN)
        gain = min(max(abs(gain or DEFAULT_SWING_GAIN), 0.006), 0.02)
        gain_candidates = [0.006, 0.012, 0.02]
        if gain not in gain_candidates:
            gain_candidates.append(gain)
        gain_candidates = sorted(set(gain_candidates))

        max_velocity = getattr(self.config, 'swing_max_velocity', DEFAULT_SWING_MAX_VELOCITY)
        max_velocity = min(max(max_velocity, 0.012), 0.03)

        configured_sign = getattr(self.config, 'swing_sign', DEFAULT_SWING_SIGN)
        configured_sign = -1.0 if configured_sign < 0 else 1.0
        signs = [configured_sign, -configured_sign]
        return [
            {
                'latency': candidate_latency,
                'sign': sign,
                'gain': candidate_gain,
                'max_velocity': max_velocity,
            }
            for candidate_latency in latency_candidates
            for sign in signs
            for candidate_gain in gain_candidates
        ]

    def _swing_runtime_validation(self, samples):
        summary = self._swing_energy_summary(samples)
        finite_energies = [
            float(energy)
            for _, energy in samples
            if energy is not None and np.isfinite(energy)
        ]
        if len(finite_energies) < 180:
            return {
                'summary': summary,
                'enough_samples': False,
                'initial_mean': float('inf'),
                'runtime_mean': float('inf'),
                'runtime_ratio': float('inf'),
                'amplified': True,
                'damped': False,
            }

        initial_mean = float(np.mean(finite_energies[:30]))
        runtime_mean = float(np.mean(finite_energies[-120:]))
        runtime_ratio = runtime_mean / max(initial_mean, 1e-12)
        energy_limit = self._swing_amplification_limit(initial_mean)
        score = max(summary['ratio'], runtime_ratio)
        quiet = (
            initial_mean < self.config.swing_auto_min_energy
            and runtime_mean <= energy_limit
            and summary['tail_mean'] <= energy_limit
        ) or (
            runtime_mean <= self.config.swing_auto_min_energy
            and summary['tail_mean'] <= self.config.swing_auto_min_energy
        )
        amplified = runtime_mean > energy_limit or (not quiet and score > 1.02)
        return {
            'summary': summary,
            'enough_samples': True,
            'initial_mean': initial_mean,
            'runtime_mean': runtime_mean,
            'runtime_ratio': runtime_ratio,
            'quiet': quiet,
            'amplified': amplified,
            'damped': quiet or (runtime_ratio <= 0.98 and summary['ratio'] <= 0.98),
        }

    def _finish_swing_calibration(self, succeeded, current_action, message=None):
        self.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=100,
            name='Swing calibration',
            current_action=current_action,
        ))
        if message:
            self.send_ui(pop_message=telemetry.Popup(message=message))
        return succeeded

    async def _sample_swing_energy(self, duration_s, interval_s=1/60):
        samples = []
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if not isinstance(self.gripper_client, ArpeggioGripperClient) or not self.gripper_client.connected:
                raise RuntimeError('Arpeggio gripper must be connected for swing calibration')
            model_age = self.gripper_client.swing_model_age()
            if model_age > MAX_SWING_MODEL_AGE_S:
                raise RuntimeError(f'Swing IMU model is stale ({model_age:.2f}s old)')
            energy = self.gripper_client.swing_energy()
            if energy is not None and np.isfinite(energy):
                samples.append((time.time(), float(energy)))
            await asyncio.sleep(interval_s)
        return samples

    async def _induce_swing_for_calibration(self):
        key = 'swingcal_excitation'
        speed = min(max(self.config.swing_max_velocity, 0.015), 0.03)
        self.active_set.add(key)
        try:
            for direction in (
                np.array([1.0, 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
            ):
                await self.move_direction_speed(direction, speed=speed, key=key, downward_bias=0)
                await asyncio.sleep(0.45)
            await self.move_direction_speed(np.zeros(3), key=key, downward_bias=0)
            await asyncio.sleep(1.0)
        finally:
            await self.move_direction_speed(np.zeros(3), key=key, downward_bias=0)
            self.active_set.discard(key)
            self.input_velocities.pop(key, None)
            self.slow_stop_all_spools()

    async def _run_swing_calibration_trial(
            self,
            latency,
            sign,
            gain,
            max_velocity,
            duration_s,
            abort_energy,
            update_integrator=True,
    ):
        key = 'swingcal'
        samples = []
        aborted = False
        self.gripper_client.reset_swing_correction_integrator()
        self.active_set.add(key)
        try:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                if not isinstance(self.gripper_client, ArpeggioGripperClient) or not self.gripper_client.connected:
                    raise RuntimeError('Arpeggio gripper disconnected during swing calibration')
                model_age = self.gripper_client.swing_model_age()
                if model_age > MAX_SWING_MODEL_AGE_S:
                    raise RuntimeError(f'Swing IMU model is stale ({model_age:.2f}s old)')
                now = time.time()
                vel2 = self.gripper_client.compute_swing_correction(
                    now + latency,
                    gain=gain,
                    sign=sign,
                    max_velocity=max_velocity,
                    update_integrator=update_integrator,
                )
                if vel2 is not None:
                    await self.move_direction_speed(np.array([vel2[0], vel2[1], 0]), key=key, downward_bias=0)
                energy = self.gripper_client.swing_energy()
                if energy is not None and np.isfinite(energy):
                    energy = float(energy)
                    samples.append((now, energy))
                    if self._swing_should_abort_calibration(samples, abort_energy):
                        aborted = True
                        logger.warning(
                            'Swing calibration abort latency=%.3f sign=%s gain=%.3f energy=%.6f limit=%.6f',
                            latency,
                            sign,
                            gain,
                            float(np.mean([e for _, e in samples[-12:]])),
                            abort_energy,
                        )
                        break
                await asyncio.sleep(1/60)
        finally:
            await self.move_direction_speed(np.zeros(3), key=key, downward_bias=0)
            self.active_set.discard(key)
            self.input_velocities.pop(key, None)
        return samples, aborted

    async def auto_calibrate_swing_cancellation(self):
        try:
            return await self._auto_calibrate_swing_cancellation()
        except asyncio.CancelledError:
            raise
        except RuntimeError as exc:
            logger.warning('Swing calibration failed: %s', exc)
            return self._finish_swing_calibration(
                False,
                f'Failed: {exc}',
                str(exc),
            )

    async def _auto_calibrate_swing_cancellation(self):
        """
        Learn swing cancellation phase/sign/gain by running bounded low-speed trials.
        This is a motion task and should only run while the gantry is otherwise idle.
        """
        if not isinstance(self.gripper_client, ArpeggioGripperClient) or not self.gripper_client.connected:
            return self._finish_swing_calibration(
                False,
                'Failed: Arpeggio gripper is not connected',
                'Swing auto calibration requires a connected Arpeggio gripper.',
            )

        was_running = self.swing_cancellation_task is not None and not self.swing_cancellation_task.done()
        if was_running:
            self.swing_cancellation_task.cancel()
            try:
                await self.swing_cancellation_task
            except asyncio.CancelledError:
                pass

        await self.move_direction_speed(np.zeros(3), key='default', downward_bias=0)
        self.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=0,
            name='Swing calibration',
            current_action='Measuring baseline swing',
        ))

        baseline_samples = await self._sample_swing_energy(3.0)
        baseline = self._swing_energy_summary(baseline_samples)
        logger.info('Swing calibration baseline: %s', baseline)
        if baseline['mean'] < self.config.swing_auto_min_energy:
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=5,
                name='Swing calibration',
                current_action='Inducing measurable swing',
            ))
            await self._induce_swing_for_calibration()
            baseline_samples = await self._sample_swing_energy(3.0)
            baseline = self._swing_energy_summary(baseline_samples)
            logger.info('Swing calibration baseline after excitation: %s', baseline)
            if baseline['mean'] < self.config.swing_auto_min_energy:
                return self._finish_swing_calibration(
                    False,
                    'Failed: swing remained too small to calibrate',
                    'Swing is too small to calibrate after automatic excitation.',
                )

        abort_energy = max(
            baseline['tail_mean'] * self.config.swing_auto_abort_ratio,
            self.config.swing_auto_min_energy * self.config.swing_auto_abort_ratio,
        )
        baseline_tail = max(baseline['tail_mean'], self.config.swing_auto_min_energy)

        candidates = []
        trial_plan = self._swing_calibration_trial_plan()
        for i, trial in enumerate(trial_plan):
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=10 + 65 * (i + 1) / len(trial_plan),
                name='Swing calibration',
                current_action=(
                    f'Testing latency {trial["latency"]:.2f}s '
                    f'sign {int(trial["sign"])} gain {trial["gain"]:.3f}'
                ),
            ))
            samples, aborted = await self._run_swing_calibration_trial(
                latency=trial['latency'],
                sign=trial['sign'],
                gain=trial['gain'],
                max_velocity=trial['max_velocity'],
                duration_s=2.0,
                abort_energy=abort_energy,
            )
            summary = self._swing_energy_summary(samples)
            candidate = {
                **trial,
                'aborted': aborted,
                'summary': summary,
                'score': max(summary['ratio'], summary['tail_mean'] / baseline_tail),
            }
            logger.info('Swing calibration candidate: %s', candidate)
            if not aborted:
                candidates.append(candidate)
            await asyncio.sleep(0.25)

        if len(candidates) == 0:
            return self._finish_swing_calibration(
                False,
                'Failed: every candidate amplified swing',
                'Swing calibration aborted every candidate. Check spin calibration and reduce swing.',
            )

        best = min(candidates, key=lambda c: (c['score'], c['summary']['tail_mean']))

        improvement = baseline['ratio'] - best['summary']['ratio']
        if best['summary']['ratio'] >= 0.98 and improvement < 0.03:
            logger.warning('Swing calibration found no useful improvement. baseline=%s best=%s', baseline, best)
            return self._finish_swing_calibration(
                False,
                'Failed: no candidate damped swing',
                'Swing calibration did not find a damping setting better than baseline.',
            )

        await self.move_direction_speed(np.zeros(3), key='default', downward_bias=0)
        await asyncio.sleep(0.5)
        self.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=90,
            name='Swing calibration',
            current_action='Validating best setting against runtime guard',
        ))
        validation_abort_energy = max(
            baseline_tail * self.config.swing_auto_abort_ratio,
            self.config.swing_auto_min_energy * self.config.swing_auto_abort_ratio,
        )
        validation_samples, validation_aborted = await self._run_swing_calibration_trial(
            latency=best['latency'],
            sign=best['sign'],
            gain=best['gain'],
            max_velocity=best['max_velocity'],
            duration_s=8.0,
            abort_energy=validation_abort_energy,
        )
        validation = self._swing_runtime_validation(validation_samples)
        if validation_aborted or validation['amplified']:
            logger.warning(
                'Swing calibration validation failed: baseline=%s validation=%s best=%s aborted=%s',
                baseline,
                validation,
                best,
                validation_aborted,
            )
            return self._finish_swing_calibration(
                False,
                'Failed: validation amplified swing',
                'Swing calibration rejected the best setting because sustained validation still amplified swing.',
            )
        if not validation['damped']:
            logger.warning(
                'Swing calibration validation found no sustained damping: baseline=%s validation=%s best=%s',
                baseline,
                validation,
                best,
            )
            return self._finish_swing_calibration(
                False,
                'Failed: validation did not damp swing',
                'Swing calibration found a stable setting, but it did not reduce swing enough to save.',
            )

        self.config.swing_latency = best['latency']
        self.config.swing_sign = best['sign']
        self.config.swing_gain = best['gain']
        self.config.swing_max_velocity = best['max_velocity']
        save_config(self.config, self.config_path)
        await self.send_setup_telemetry()
        logger.info('Swing calibration saved: baseline=%s validation=%s best=%s', baseline, validation, best)
        return self._finish_swing_calibration(
            True,
            'Completed: validated and saved',
            (
                f'Swing calibration saved latency={best["latency"]:.2f}s '
                f'sign={int(best["sign"])} gain={best["gain"]:.3f}'
            ),
        )

    async def _handle_debug_command(self, item: control.Debug):
        logger.debug(f'Debug action "{item.action}"')
        if item.action == "spincal":
            r = await self.calibrate_spin()
        if item.action == 'fingercal':
            asyncio.create_task(self.calibrate_finger_servo())
        if item.action == 'eyelets':
            r = await self.invoke_motion_task(self.collect_arp_anchor_eyelet_experiment_data())
        if item.action == 'stow':
            r = await self.stow_lines()
        if item.action.startswith('swinglatency '):
            parts = item.action.split(' ')
            self.config.swing_latency = min(max(float(parts[1]), 0.0), 0.8)
            save_config(self.config, self.config_path)
        if item.action.startswith('swinggain '):
            parts = item.action.split(' ')
            self.config.swing_gain = min(max(abs(float(parts[1])), 0.0), MAX_SWING_GAIN)
            save_config(self.config, self.config_path)
        if item.action.startswith('swingsign '):
            parts = item.action.split(' ')
            sign = float(parts[1])
            self.config.swing_sign = -1.0 if sign < 0 else 1.0
            save_config(self.config, self.config_path)
        if item.action.startswith('swingmaxvel '):
            parts = item.action.split(' ')
            self.config.swing_max_velocity = min(max(abs(float(parts[1])), 0.0), MAX_SWING_MAX_VELOCITY)
            save_config(self.config, self.config_path)
        if item.action == 'swingcal':
            r = await self.invoke_motion_task(self.auto_calibrate_swing_cancellation())
        if item.action == 'reset_wrist':
             await asyncio.create_task(self.gripper_client.send_commands({'reset_wrist': None}))
        if item.action == 'spind':
            print(self.gripper_client.get_spin(True))
        if item.action == 'chaset':
            # keep the gripper 10cm over the "trash" tag
            r = await self.invoke_motion_task(self.chase_tag('trash'))
        if item.action == 'ferry':
            r = await self.invoke_motion_task(self.ferry('hamper', 'trash'))
        if item.action == 'sync_timezone':
            self.sync_timezone_to_bots()

    def sync_timezone_to_bots(self):
        tz = subprocess.check_output(['timedatectl', 'show', '--property=Timezone', '--value']).decode().strip()
        for client in self.bot_clients.values():
            asyncio.create_task(client.send_commands({'set_timezone': tz}))

    async def chase_tag(self, name):
        """Keep the gripper at the named location"""
        try:
            chase_task = None
            while self.run_command_loop:
                await asyncio.sleep(0.1)
                if not name in self.named_positions:
                    continue
                goal = self.named_positions[name] + POLE
                self.gantry_goal_pos = goal
                if chase_task is None or chase_task.done():
                    chase_task = asyncio.create_task(self.seek_gantry_goal())
        except asyncio.CancelledError:
            if chase_task is not None:
                chase_task.cancel()
            raise

    async def ferry(self, source, dest):
        """Carry objectes between one named tag and another.
        Moves to source, attempt auto grasp, move to test, drop, repeat"""
        try:
            while self.run_command_loop:
                await asyncio.sleep(0.1)

                # wait for source position to be seen
                while not source in self.named_positions:
                    await asyncio.sleep(0.5)
                # go to position
                goal = self.named_positions[source] + POLE + GRIPPER_HEIGHT_OVER_TARGET
                self.gantry_goal_pos = goal
                await self.seek_gantry_goal()

                # auto grasp
                # await self.gripper_client.send_commands({'set_finger_angle': 30})
                # await asyncio.sleep(1)
                await self.execute_grasp()

                # wait for destination position to be seen
                while not dest in self.named_positions:
                    await asyncio.sleep(0.5)
                # go to position
                goal = self.named_positions[dest] + POLE + GRIPPER_HEIGHT_OVER_TARGET
                self.gantry_goal_pos = goal
                await self.seek_gantry_goal()

                # drop
                await self.gripper_client.send_commands({'set_finger_angle': -30})
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            raise

    async def lerobot_process(self, item: control.ManageLerobotSession):
        if self.lerobot_process_pid is not None:
            logger.warning(f"Cannot start lerobot session, one is already active.")
            return

        repo_id = item.repo_id
        action = item.action
        # Sanitize and validate repo_id to prevent code injection.
        # Enforces the Hugging Face Hub format: 'namespace/dataset_name'
        if not re.match(r"^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+$", str(repo_id)):
            logger.warning(f"Invalid repo_id format '{repo_id}'. Expected 'namespace/dataset_name'. Aborting.")
            return

        # Run the python function as a command-line script to hook into its stdout and stderr streams asynchronously and use the same virtualenv
        if action == control.LerobotSessionAction.START_RECORD:
            func_name = 'record_until_disconnected'
        elif action == control.LerobotSessionAction.START_EVAL:
            func_name = 'eval_until_disconnected'

        up = ''
        if item.suppress_upload:
            up = ' upload=False'

        # A lerobot session running on the local machine must connect to the telemetry socket of the robot.
        # When telemetry_env is not None, there are two options. connect to the remote stream - this introduces needless latency and requires a token
        # Or spin up the local telemetry socket and the MJepeg streamers while the lerobot process is active.
        tele_addr = 'ws://localhost:4245'

        command = [
            sys.executable,
            '-u', '-c',
            f"from nf_robot.ml.stringman_lerobot import {func_name}; "
            f"{func_name}('{tele_addr}', '{repo_id}', '{self.config.robot_id}'{up})"
        ]

        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        logger.info(f"Lerobot process started with PID: {process.pid}")
        self.lerobot_process_pid = process.pid

        async def log_stream(stream, stream_name):
            while True:
                line = await stream.readline()
                if not line:
                    break
                sline = line.decode('utf-8').rstrip()
                if not sline.startswith('[swscaler'):
                    logger.info(f"[{stream_name}] {sline}")

        # Create concurrent background tasks to monitor stdout and stderr
        stdout_task = asyncio.create_task(log_stream(process.stdout, "LEROBOT STDOUT"))
        stderr_task = asyncio.create_task(log_stream(process.stderr, "LEROBOT STDERR"))

        try:
            return_code = await process.wait()
            logger.info(f"Lerobot process exited with code: {return_code}")
            
        except asyncio.CancelledError:
            logger.info("Cancellation requested. Terminating Lerobot process...")
            try:
                process.terminate()
            except ProcessLookupError:
                pass # Process already died
            await process.wait()
            logger.info("Lerobot process terminated.")
            
        finally:
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            self.lerobot_process_pid = None

    async def calibrate_finger_servo(self):
        self.gripper_client.finger_contact_calibration_complete.clear()
        await asyncio.create_task(self.gripper_client.send_commands({'measure_finger_contact': None}))
        await asyncio.wait_for(self.gripper_client.finger_contact_calibration_complete.wait(), 20)

    def _handle_delete_target(self, item: control.DeleteTarget):
        if item.target_id is not None:
            self.target_queue.remove_target(item.target_id);

    def _handle_add_cam_target(self, item: control.AddTargetFromAnchorCam):
        # Add the target
        targets2d = [[item.img_norm_x, item.img_norm_y]]
        if item.anchor_num not in self.anchors:
            return
        floor_points = project_pixels_to_floor(targets2d, self.anchors[item.anchor_num].camera_pose, self.config.camera_cal)
        logger.info(f'Adding target at floor point ({floor_points}) from image point ({targets2d[0]}) in anchor cam {item.anchor_num}')
        if (len(floor_points) == 1):
            if item.target_id is not None:
                self.target_queue.set_target_position(item.target_id, floor_points[0])
            else:   
                new_id = self.target_queue.add_user_target(floor_points[0], dropoff='hamper')
        self.send_tq_to_ui()

    def submitTargets(self):
        """snapshot any active cameras at 1920x1080 and save images in the raw dir"""
        images = []
        for anchor in self.anchors.values():
            if anchor.frame is not None:
                images.append(anchor.frame.copy())

        def save_data(images):
            directory_path = Path("target_heatmap_data_unlabeled")
            directory_path.mkdir(exist_ok=True, parents=True)
            
            for img in images:
                img_filename = f"{str(uuid.uuid4())}.jpg"
                # write the image
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_full_path = directory_path / img_filename
                cv2.imwrite(str(img_full_path), rgb_image)

        threading.Thread(target=save_data, args=(images,)).start()

    def _handle_scale_room(self, item: control.ScaleRoom):
        # not implemented for arpeggio anchor
        if item.scale:
            # move positions of anchors towards or away from origin
            logger.info(f'Scaling by {item.scale}')
            anchor_poses = [(client.anchor_pose[0], client.anchor_pose[1]*item.scale) for client in self.anchors.values()]

            # update everything
            for client in self.anchors.values():
                self.config.anchors[client.anchor_num].pose = poseTupleToProto(anchor_poses[client.anchor_num])
                client.updatePose(anchor_poses[client.anchor_num])
            save_config(self.config, self.config_path)
            # inform UI
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(poses=[
                poseTupleToProto(p)
                for p in anchor_poses
            ]))
            # inform position estimator
            anchor_points = np.array([compose_poses([pose, model_constants.anchor_grommet])[1] for pose in anchor_poses])
            self.pe.set_anchor_points(anchor_points)

        if item.tiltcams:
            logger.info(f'Tilting cams inward by {item.tiltcams} deg')
            for client in self.anchors.values():
                client.extratilt += item.tiltcams
                client.updatePose(client.anchor_pose)


    async def _handle_common_command(self, cmd: control.Command):
        # betterproto Enums are IntEnums, comparable directly
        match cmd:
            case control.Command.STOP_ALL:
                r = await self.stop_all()
            case control.Command.TIGHTEN_LINES:
                r = await self.tension_lines()
            case control.Command.ZERO_WINCH:
                asyncio.create_task(self._handle_zero_winch_line())
            case control.Command.HALF_CAL:
                r = await self.invoke_motion_task(self.half_auto_calibration())
            case control.Command.FULL_CAL:
                r = await self.invoke_motion_task(self.full_auto_calibration())
            case control.Command.AUTO_CALIBRATE_SWING:
                r = await self.invoke_motion_task(self.auto_calibrate_swing_cancellation())
            case control.Command.PICK_AND_DROP:
                r = await self.invoke_motion_task(self.pick_and_place_loop())
            case control.Command.HORIZONTAL_CHECK:
                r = await self.invoke_motion_task(self.horizontal_line_task())
            case control.Command.COLLECT_GRIPPER_IMAGES:
                self._handle_collect_images()
            case control.Command.SHUTDOWN:
                self.run_command_loop = False
            case control.Command.RECORD_PARK:
                r = await self.record_park()
            case control.Command.PARK:
                r = await self.invoke_motion_task(self.park())
            case control.Command.UNPARK:
                r = await self.invoke_motion_task(self.unpark())
            case control.Command.GRASP:
                r = await self.invoke_motion_task(self.execute_grasp())
            case control.Command.SUBMIT_TARGETS_TO_DATASET:
                self.submitTargets()
            case control.Command.UPDATE_FIRMWARE:
                r = await self._handle_update_firmware()
            case control.Command.DISABLE_TORQUE:
                await self._handle_disable_torque()
            case control.Command.ENABLE_TORQUE:
                await self._handle_enable_torque()

    async def _handle_update_firmware(self):
        r = await self.stop_all()
        async def update_bar_task():
            for i in range(100):
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=float(i),
                    name="Update Component Firmware",
                    current_action="updating...",
                ))
                if not self.run_command_loop:
                    break
                await asyncio.sleep(0.5)
        bar = asyncio.create_task(update_bar_task())
        tasks = []
        keys = []
        for name, client in self.bot_clients.items():
            tasks.append(client.firmware_update())
            keys.append(name)
        results = await asyncio.gather(*tasks)
        bar.cancel()
        lines = []
        for i, r in enumerate(results):
            a = "Not supported"
            if r == True:
                a = "Success"
            elif r == False:
                a = "Failed"
            lines.append(f"({self.bot_clients[keys[i]].address}) {a}")
        table = '\n'.join(lines)
        if any(x is False for x in results):
            message = f"Failed on one or more components \n\n{table}"
        elif all(results):
            message = "Completed successfully"
        else:
            message = f"Successful on some components, others require manual updating \n\n{table}"
        self.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=float(100),
            name="Update Component Firmware",
            current_action=message,
        ))

    async def _handle_disable_torque(self):
        if self.config.anchor_type != common.AnchorType.ARPEGGIO:
            return
        for client in self.anchors.values():
            asyncio.create_task(client.send_commands({'disable_torque': None}))

    async def _handle_enable_torque(self):
        if self.config.anchor_type != common.AnchorType.ARPEGGIO:
            return
        for client in self.anchors.values():
            asyncio.create_task(client.send_commands({'enable_torque': None}))

    async def _handle_jog_spool(self, jog: control.JogSpool):
        """Handles manually jogging a spool motor."""
        # identify the client we need to send the command to
        client = None
        if jog.is_gripper:
            if jog.speed is not None:
                asyncio.create_task(self.gripper_client.send_commands({'aim_speed': jog.speed}))
            elif jog.offset is not None:
                asyncio.create_task(self.gripper_client.send_commands({'jog': jog.offset}))
        else:
            if jog.speed is not None:
                await self.send_line_speed(jog.anchor_num, jog.speed)
            elif jog.offset is not None:
                await self.send_line_speed(jog.anchor_num, jog.offset, jog=True)

    async def _handle_gantry_goal_pos(self, goal_pos: np.ndarray):
        """Handles moving the gantry to a specific goal position."""
        self.gantry_goal_pos = goal_pos
        await self.invoke_motion_task(self.seek_gantry_goal())

    async def _handle_slow_stop_one(self, stop_data: dict):
        """Handles stopping a single spool motor."""
        if stop_data.get('id') == 'gripper' and self.gripper_client:
            asyncio.create_task(self.gripper_client.slow_stop_spool())
        else:
            for client in self.anchors.values():
                if client.anchor_num == stop_data.get('id'):
                    asyncio.create_task(client.slow_stop_spool())

    async def _handle_zero_winch_line(self):
        if self.gripper_client is not None and isinstance(self.gripper_client, RaspiGripperClient):
            await self.gripper_client.zero_winch()

    async def _handle_movement(self, move: control.CombinedMove):
        winch = None
        wrist = None
        if self.gripper_client is not None:
            # if we have to clip these values to legal limits, save what they were clipped to
            if move.finger_speed is not None or move.wrist_speed is not None:
                winch, finger, wrist = await self.send_gripper_move(move.winch, move.finger_speed, move.wrist_speed)
            else:
                # this type of message may be sent from older UIs. probably safe to removed by end of Feb.
                winch, finger, wrist = await self.send_gripper_move_legacy(move.winch, move.finger, move.wrist)

        direction = np.zeros(3)
        if move.direction:
            direction = tonp(move.direction)

            if self.gripper_client is not None and isinstance(self.gripper_client, ArpeggioGripperClient):
                if move.direction_is_in_gripper_frame:
                    if move.speed is not None:
                        velocity = direction * move.speed # make sure the network receives information on speed as well
                    else:
                        velocity = direction
                    self.send_ui(raw_commanded_vel=telemetry.CommandedVelocity(velocity=fromnp(velocity)))
                    # rotate later component of direction into room frame
                    direction[:2] = rotate_vector(direction[:2], -self.gripper_client.get_spin())
                else:
                    # direction is already in room frame, and we can use it, but we still want to send the lerobot record script a direction in gripper frame
                    gf_direction = direction.copy()
                    gf_direction[:2] = rotate_vector(gf_direction[:2], self.gripper_client.get_spin())
                    if move.speed is not None:
                        velocity = gf_direction * move.speed # make sure the network receives information on speed as well
                    else:
                        velocity = gf_direction
                    self.send_ui(raw_commanded_vel=telemetry.CommandedVelocity(velocity=fromnp(velocity)))

        commanded_vel = await self.move_direction_speed(direction, move.speed)

        self.last_user_move_time = time.time()

    def _passive_safety_tension_limit(self, now=None):
        now = time.time() if now is None else now
        extra = PASSIVE_SAFE_TENSION_RETRY_BUMP_N if now < getattr(self, '_passive_safety_tension_limit_extra_until', 0.0) else 0.0
        return PASSIVE_SAFE_TENSION_N + extra

    def _retryable_move_signature(self, key, velocity):
        velocity = np.asarray(velocity, dtype=float)
        return (key, tuple(np.round(velocity, 3)))

    def _record_retryable_move(self, key, velocity):
        if key != 'default':
            return
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            return
        if float(np.linalg.norm(velocity)) < 0.005:
            return
        signature = self._retryable_move_signature(key, velocity)
        self.last_retryable_move = {
            'key': key,
            'velocity': velocity.copy(),
            'signature': signature,
            'ts': time.time(),
        }

    def _get_passive_safety_retry_move(self, now=None):
        now = time.time() if now is None else now
        move = getattr(self, 'last_retryable_move', None)
        if not move:
            return None
        if now - move.get('ts', 0.0) > PASSIVE_SAFE_RETRY_MOVE_MAX_AGE_S:
            return None
        history = getattr(self, '_passive_safety_retry_history', {})
        last_retry = history.get(move['signature'], 0.0)
        if now - last_retry < PASSIVE_SAFE_RETRY_COOLDOWN_S:
            return None
        return move

    async def _stop_for_passive_tension_limit(self):
        if self.swing_cancellation_task is not None and not self.swing_cancellation_task.done():
            self.swing_cancellation_task.cancel()
        for key in list(self.active_set):
            self.input_velocities[key] = np.zeros(3)
        self.active_set = set(['default'])
        self.slow_stop_all_spools()
        await self._handle_disable_torque()
        await asyncio.sleep(1)
        await self._handle_enable_torque()
        await asyncio.sleep(1)

    def _send_passive_safety_popup(self, message, now=None, throttle_s=0.0):
        now = time.time() if now is None else now
        if throttle_s > 0 and now - getattr(self, '_passive_safety_last_final_prompt_ts', 0.0) < throttle_s:
            return
        if throttle_s > 0:
            self._passive_safety_last_final_prompt_ts = now
        self.send_ui(pop_message=telemetry.Popup(message=message))

    async def _handle_passive_tension_limit(self, ema, limit):
        now = time.time()
        high_lines = [int(i) for i in np.where(np.asarray(ema) > limit)[0]]
        retry_window_active = now < getattr(self, '_passive_safety_tension_limit_extra_until', 0.0)
        retry_move = None if retry_window_active else self._get_passive_safety_retry_move(now)

        await self._stop_for_passive_tension_limit()

        if retry_move is None:
            self._passive_safety_tension_limit_extra_until = 0.0
            if retry_window_active:
                message = (
                    f'Tension stayed above {limit:.1f} N during the retry on lines {high_lines}. '
                    'I stopped and left the robot idle. Check for a caught line before moving again.'
                )
            else:
                message = (
                    f'Tension exceeded {limit:.1f} N on lines {high_lines}. '
                    'I stopped and left the robot idle. Check the lines, then retry manually.'
                )
            logger.warning(message)
            self._send_passive_safety_popup(message, now=now, throttle_s=5.0)
            return False

        retry_limit = PASSIVE_SAFE_TENSION_N + PASSIVE_SAFE_TENSION_RETRY_BUMP_N
        self._passive_safety_retry_history[retry_move['signature']] = now
        self._passive_safety_tension_limit_extra_until = now + PASSIVE_SAFE_RETRY_WINDOW_S
        message = (
            f'Tension exceeded {limit:.1f} N on lines {high_lines}. '
            f'I backed off, temporarily raised the retry limit to {retry_limit:.1f} N, '
            'and I am trying the last move once more.'
        )
        logger.warning(message)
        self._send_passive_safety_popup(message)
        await self.move_direction_speed(
            retry_move['velocity'],
            speed=None,
            starting_pos=self.pe.gant_pos,
            downward_bias=0,
            key=retry_move.get('key', 'default'),
            record_retry=False,
        )
        return True

    async def passive_safety(self):
        """If any line becomes too tight, switch all motors to damped movement for one second."""
        ema = np.zeros(4)
        while self.run_command_loop and self.pe.tension is not None:
            ema = ema * 0.9 + self.pe.tension * 0.1
            max_safe_tension = self._passive_safety_tension_limit()
            OBS.record_tension(ema, max_safe_tension)
            if np.any(ema > max_safe_tension):
                OBS.record_safety_stop()
                logger.warning('Tension limit reached! backing off.')
                await self._handle_passive_tension_limit(ema, max_safe_tension)
            await asyncio.sleep(0.2)

    async def update_observability_runtime(self):
        while self.run_command_loop:
            OBS.set_uptime(self.started_at)
            OBS.set_ui_clients(len(self.connected_local_clients))
            with self.telemetry_buffer_lock:
                telemetry_buffer_size = len(self.telemetry_buffer)
                OBS.set_telemetry_buffer(telemetry_buffer_size)
            connected_components = {}
            for client in self.bot_clients.values():
                kind = "gripper" if getattr(client, "anchor_num", None) is None else "anchor"
                component = "gripper" if kind == "gripper" else f"anchor_{client.anchor_num}"
                status = getattr(getattr(client, "conn_status", None), "websocket_status", None)
                connected_components[(component, kind)] = status == telemetry.ConnStatus.CONNECTED
            target_count = len(getattr(self.target_queue.get_queue_snapshot(), "targets", []) or [])
            OBS.record_runtime_state(
                connected_components=connected_components,
                gripper_present=self.gripper_client is not None,
                anchor_count=len(self.anchors),
                active_velocity_keys=self.active_set,
                telemetry_buffer_size=telemetry_buffer_size,
                target_count=target_count,
            )
            await asyncio.sleep(1.0)

    def update_avg_named_pos(self, key: str, position: np.ndarry):
        """Update the running average of the named position"""
        if key not in self.named_positions:
            self.named_positions[key] = position
        # exponential moving average
        self.named_positions[key] = self.named_positions[key] * 0.75 + position * 0.25
        pos = self.named_positions[key]
        self.send_ui(named_position=telemetry.NamedObjectPosition(
            position=fromnp(pos),
            name=key,
        ))

    async def invoke_motion_task(self, coro):
        """
        Cancel whatever else is happening and start a new long running motion task
        Any task that can be called this way is known in this file as a "motion task"
        The defining feature of a motion task is that it could send a second motion command to any client after any amount of sleeping
        every motion task must have the follwing structure

        try:
            # do something
        except asyncio.CancelledError:
            raise
        finally:
            # perform any clean up work

        Do not call invoke_motion_task from within a motion task or it will cancel itself.
        It is ok to call a motion task from within another, just don't start it with invoke_motion_task
        Do not call stop_all from within a motion task. use slow_stop_all_spools instead

        """
        if self.motion_task is not None and not self.motion_task.done():
            logger.info(f"Cancelling previous motion task: {self.motion_task.get_name()}")
            self.motion_task.cancel()
            try:
                # Wait briefly for the old task's cleanup to complete.
                result = await self.motion_task
            except asyncio.CancelledError:
                pass # Expected behavior

        self.motion_task = asyncio.create_task(coro)
        self.motion_task.set_name(coro.__name__)

    async def tension_lines(self):
        """Request all anchors to reel in all lines until tight.
        This is a fire and forget function"""
        for client in self.anchors.values():
            if isinstance(client, RaspiAnchorClient):
                asyncio.create_task(client.send_commands({'tighten': None}))
            elif isinstance(client, ArpeggioAnchorClient):
                asyncio.create_task(client.send_commands({'tighten': 0}))
                asyncio.create_task(client.send_commands({'tighten': 1}))
        # This function does not  wait for confirmation from every anchor, as it would just hold up the processing of the ob_q
        # this is similar to sending a manual move command. it can be overridden by any subsequent command.
        # thus, it should be done while paused.

    async def stow_lines(self):
        """Request all anchors to reel in all lines until tight and then disable motors"""
        for client in self.anchors.values():
            if isinstance(client, RaspiAnchorClient):
                asyncio.create_task(client.send_commands({'stow': None}))
            elif isinstance(client, ArpeggioAnchorClient):
                asyncio.create_task(client.send_commands({'stow': 0}))
                asyncio.create_task(client.send_commands({'stow': 1}))

    def _line_records_for_tension(self, max_age_s=TENSION_RECORD_MAX_AGE_S):
        try:
            records = np.array([alr.getLast() for alr in self.datastore.anchor_line_record], dtype=float)
        except Exception:
            logger.exception('Failed to read anchor line records while waiting for tension')
            return None

        if records.ndim != 2 or records.shape[0] != N_LINES or records.shape[1] < 4:
            logger.warning(f'Invalid line record shape while waiting for tension: {records.shape}')
            return None

        now = time.time()
        timestamps = records[:, 0]
        values = records[:, 1:4]
        if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(values)):
            logger.warning(f'Invalid line records while waiting for tension: {records}')
            return None
        if np.any(timestamps <= 0) or np.any(now - timestamps > max_age_s):
            logger.warning(f'Stale line records while waiting for tension: ages={now - timestamps}')
            return None
        return records

    async def wait_for_tension(
        self,
        timeout_s=TENSION_WAIT_TIMEOUT_S,
        poll_interval_s=TENSION_POLL_INTERVAL_S,
    ):
        """Return True only after all lines are taut and settled; stop spools on failure."""
        threshold = 0.5
        if self.config.anchor_type == common.AnchorType.ARPEGGIO:
            threshold = TENSION_THRESH

        last_tension = np.full(N_LINES, np.nan)
        last_speed_norm = np.nan
        timeout = time.time() + timeout_s
        while time.time() < timeout:
            await asyncio.sleep(poll_interval_s)
            records = self._line_records_for_tension()
            if records is None:
                continue

            speeds = records[:, 2]
            tension = records[:, 3]
            last_tension = tension
            last_speed_norm = float(np.linalg.norm(speeds))
            if np.all(tension > threshold) and last_speed_norm < TENSION_SPEED_NORM_THRESHOLD:
                logger.debug(f'tension on lines = {tension}, speed_norm={last_speed_norm}')
                return True

        logger.warning(
            f'Timed out waiting for line tension. tension={last_tension}, '
            f'speed_norm={last_speed_norm}, threshold={threshold}'
        )
        self.slow_stop_all_spools()
        return False

    async def tension_and_wait(self):
        """Send tightening command and wait until lines appear tight. This is not a motion task"""
        logger.info('Tightening all lines')
        await self.tension_lines()
        ok = await self.wait_for_tension()
        if not ok:
            logger.warning('Line tension failed; calibration/motion caller must abort before saving references')
        return ok

    def _fresh_gantry_reference_position(
        self,
        max_age_s=REFERENCE_VISUAL_MAX_AGE_S,
        min_unique_anchors=None,
    ):
        try:
            data = np.array(self.datastore.gantry_pos.deepCopy(), dtype=float)
        except Exception:
            logger.exception('Failed to read gantry visual data before reference length reset')
            return None

        if data.ndim != 2 or data.shape[1] < 5:
            logger.warning(f'Invalid gantry visual data shape before reference length reset: {data.shape}')
            return None

        expected_anchors = len(getattr(self, 'anchors', {}))
        if min_unique_anchors is None:
            min_unique_anchors = min(2, max(1, expected_anchors))

        now = time.time()
        timestamps = data[:, 0]
        positions = data[:, 2:5]
        valid = (
            (timestamps > 0)
            & np.all(np.isfinite(positions), axis=1)
            & np.isfinite(data[:, 1])
        )
        fresh = data[valid & ((now - timestamps) <= max_age_s)]
        if len(fresh) == 0:
            logger.warning('No fresh finite gantry visual observations available for reference length reset')
            return None

        unique_anchors = np.unique(fresh[:, 1].astype(int))
        if len(unique_anchors) < min_unique_anchors:
            logger.warning(
                f'Not enough fresh gantry visual anchors for reference length reset: '
                f'{len(unique_anchors)} < {min_unique_anchors}'
            )
            return None

        position = np.mean(fresh[:, 2:5], axis=0)
        if not np.all(np.isfinite(position)):
            logger.warning(f'Invalid gantry visual mean before reference length reset: {position}')
            return None
        return position

    async def sendReferenceLengths(self, lengths):
        lengths = np.asarray(lengths, dtype=float)
        if lengths.ndim != 1 or lengths.shape[0] != N_LINES:
            logger.warning(f'Cannot send {lengths.shape} ref lengths to anchors')
            return False
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            logger.warning(f'Cannot send invalid reference lengths to anchors: {lengths}')
            return False

        position = self._fresh_gantry_reference_position()
        if position is None:
            logger.warning('Cannot send reference lengths without fresh visual gantry data')
            return False

        if self.config.anchor_type == common.AnchorType.PILOT:
            # any anchor that receives this and is slack would ignore it
            # If only some anchors are connected, this would still send reference lengths to those
            for client in self.anchors.values():
                asyncio.create_task(client.send_commands({'reference_length': lengths[client.anchor_num]}))
        elif self.config.anchor_type == common.AnchorType.ARPEGGIO:
            for client in self.anchors.values():
                # which two lines is this anchor responsible for?
                asyncio.create_task(client.send_commands({
                    'two_reference_lengths': (lengths[client.anchor_num*2], lengths[client.anchor_num*2+1])
                }))

        # use swing to estimate winch line length in pilot gripper
        if self.gripper_client is not None and isinstance(self.gripper_client, RaspiGripperClient):
            winch_length = self.pe.get_pendulum_length()
            if winch_length is not None:
                asyncio.create_task(self.gripper_client.send_commands({'reference_length': winch_length}))

        # reset biases on kalman filter
        logger.debug(f'Resetting filter biases with assumed position of {position}')
        self.pe.kf.reset_biases(position)
        return True

    async def stop_all(self):
        # If lerobot scripts are connected this must also stop them
        self.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.ABANDON))

        if self.swing_cancellation_task is not None and not self.swing_cancellation_task.done():
            self.swing_cancellation_task.cancel()

        # Cancel any active motion task
        if self.motion_task is not None:
            # Store the handle and clear the class attribute immediately.
            # This prevents race conditions if another command comes in.
            task_to_stop = self.motion_task
            self.motion_task = None

            # Only cancel the task if it's actually still running.
            if not task_to_stop.done():
                logger.info(f"Cancelling motion task: {task_to_stop.get_name()}")
                task_to_stop.cancel()

            # Now, await the task's completion.
            try:
                # Awaiting a task will re-raise any exception it had,
                # or raise CancelledError if we just cancelled it.
                await task_to_stop
            except asyncio.CancelledError:
                # This is the expected, non-error outcome of a clean cancellation.
                logger.info(f"Task '{task_to_stop.get_name()}' was successfully stopped.")
            except Exception as e:
                # If any other exception occurred, print it now.
                logger.error(f"An unhandled exception occurred in motion task '{task_to_stop.get_name()}':\n{e}")
                traceback.print_exc()

        self._clear_motion_inputs()
        self.slow_stop_all_spools()
        await self._restore_calibration_cleanup()

    def _clear_motion_inputs(self):
        if not hasattr(self, 'input_velocities') or self.input_velocities is None:
            self.input_velocities = {}
        for key in list(self.input_velocities):
            self.input_velocities[key] = np.zeros(3)
        self.input_velocities['default'] = np.zeros(3)
        self.active_set = set(['default'])

    async def _set_arp_direct_line_anti_tangle(self, enabled):
        if getattr(self, 'config', None) is None or self.config.anchor_type != common.AnchorType.ARPEGGIO:
            return
        for anchor_num in (0, 1):
            client = self.anchors.get(anchor_num, None)
            if client is None:
                continue
            try:
                await client.send_commands({'set_anti_tangle': (enabled, 0)})
            except Exception:
                logger.exception(
                    'Failed to set anti-tangle enabled=%s on anchor %s during cleanup',
                    enabled,
                    anchor_num,
                )

    async def _restore_calibration_cleanup(self):
        gripper_client = getattr(self, 'gripper_client', None)
        if gripper_client is not None and hasattr(gripper_client, 'calibrating_room_spin'):
            gripper_client.calibrating_room_spin = False
        for client in getattr(self, 'anchors', {}).values():
            if hasattr(client, 'save_raw'):
                client.save_raw = False
            if hasattr(client, 'calibrating_room_spin'):
                client.calibrating_room_spin = False
        await self._set_arp_direct_line_anti_tangle(True)

    def slow_stop_all_spools(self):
        for name, client in self.bot_clients.items():
            # Slow stop all spools. gripper too
            asyncio.create_task(client.slow_stop_spool())
        self.pe.record_commanded_vel(np.zeros(3))

    def snapshot_tag_observations(self):
        """Recent origin detections and cal_assist marker detections

        returns a dict of raw observations of various markers
        the shape of a pose is (2,3) with rotation coming first
        the first dimension is anchor number, the next is observation
        # for the arp anchor, the shape would be (2,12,2,3)

        'marker_name': array(n_anchors, n_observations, 2, 3)
        """
        markers = ['origin', 'cal_assist_1', 'cal_assist_2', 'cal_assist_3', 'gantry']
        raw_obs = defaultdict(lambda: [[]]*N_ANCHORS[self.config.anchor_type])
        for client in self.anchors.values():
            # copy each list of detections, but leave them in the camera's reference frame.
            for marker in markers:
                if marker == 'gantry':
                    raw_obs[marker][client.anchor_num] = list(client.raw_gant_poses)
                else:
                    raw_obs[marker][client.anchor_num] = list(client.origin_poses[marker])
                # print(f'anchor {client.anchor_num} has {len(raw_obs[marker][client.anchor_num])} observations of {marker}')
        return dict(raw_obs)

    def save_poses_arp(self, anchor_poses, eyelet_positions):
        # Use the optimization output to update anchor poses and spool params
        for anum, client in self.anchors.items():
            self.config.anchors[anum].pose = poseTupleToProto(anchor_poses[anum])
            self.config.anchors[anum].indirect_line.eyelet_pos = fromnp(eyelet_positions[anum])
            client.updatePoseAndEye(anchor_poses[anum], eyelet_positions[anum])
        save_config(self.config, self.config_path)
        # inform UI
        self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
            poses=[poseTupleToProto(p) for p in anchor_poses],
            eyelets=[fromnp(e) for e in eyelet_positions]
        ))
        # inform position estimator
        anchor_points = np.array([
            compose_poses([anchor_poses[0], model_constants.arp_anchor_right_eyelet])[1],
            eyelet_positions[0],
            compose_poses([anchor_poses[1], model_constants.arp_anchor_right_eyelet])[1],
            eyelet_positions[1],
        ])
        self.pe.set_anchor_points(anchor_points)

    def _new_calibration_artifact(self, calibration_name):
        anchor_type = getattr(getattr(self, 'config', None), 'anchor_type', None)
        metadata = {
            'calibration_name': calibration_name,
            'anchor_type': getattr(anchor_type, 'name', str(anchor_type)),
            'config_path': getattr(self, 'config_path', None),
        }
        return CalibrationArtifactSession(metadata=metadata)

    def _write_calibration_artifact(self, artifact, status=None, message=None):
        if artifact is None:
            return None
        try:
            return artifact.write(status=status, message=message)
        except Exception:
            logger.exception('Failed to write calibration artifact')
            return None

    def _origin_detection_counts(self):
        return {
            int(getattr(client, 'anchor_num', anchor_num)): len(client.origin_poses['origin'])
            for anchor_num, client in self.anchors.items()
        }

    def _origin_visible_anchor_nums(self, counts):
        return sorted([anum for anum, count in counts.items() if count > 0])

    def _diamond_center_xy(self):
        pe = getattr(self, 'pe', None)
        for attr in ('visual_pos', 'gant_pos', 'hang_pos'):
            point = getattr(pe, attr, None)
            if point is None:
                continue
            point = np.asarray(point, dtype=float)
            if point.shape[0] >= 2 and np.all(np.isfinite(point[:2])):
                return point[:2]

        work_area = getattr(pe, 'work_area', None)
        if work_area is not None:
            try:
                area = np.asarray(work_area, dtype=float)
                if area.ndim == 3 and area.shape[1] == 1:
                    area = area[:, 0, :]
                if area.ndim == 2 and area.shape[1] >= 2 and len(area) > 0:
                    area_xy = area[:, :2]
                    if np.all(np.isfinite(area_xy)):
                        return np.mean(area_xy, axis=0)
            except (TypeError, ValueError):
                pass
        return np.zeros(2, dtype=float)

    def _diamond_probe_points(self, center_xy, half_h, half_w):
        center_xy = np.asarray(center_xy, dtype=float)
        return [
            center_xy + np.array([0.0, -half_h]),
            center_xy + np.array([half_w, 0.0]),
            center_xy + np.array([0.0, half_h]),
            center_xy + np.array([-half_w, 0.0]),
        ]

    def _adaptive_diamond_size(self, default_size=DIAMOND_SIZE):
        """Shrink the Arpeggio diamond to fit the configured work area, if any."""
        half_h, half_w = [float(x) for x in default_size]
        pe = getattr(self, 'pe', None)
        work_area = getattr(pe, 'work_area', None)
        if pe is None or work_area is None:
            return half_h, half_w

        try:
            if np.asarray(work_area).size == 0:
                return half_h, half_w
        except (TypeError, ValueError):
            raise RuntimeError(f'Invalid work area for diamond calibration: {work_area}')

        center_xy = self._diamond_center_xy()
        for _ in range(16):
            if half_h < CAL_DIAMOND_MIN_HALF_HEIGHT or half_w < CAL_DIAMOND_MIN_HALF_WIDTH:
                break
            probe_points = self._diamond_probe_points(center_xy, half_h, half_w)
            if all(pe.point_inside_work_area_2d(point) for point in probe_points):
                return half_h, half_w
            half_h *= 0.75
            half_w *= 0.75

        raise RuntimeError(
            'No safe Arpeggio eyelet calibration diamond fits the configured work area '
            f'around center {center_xy}'
        )

    def _require_gantry_observations(self, label, min_anchor_count=1):
        gantry_obs = self.snapshot_tag_observations().get('gantry', [])
        counts = [len(obs) for obs in gantry_obs]
        anchors_with_obs = sum(count > 0 for count in counts)
        if anchors_with_obs < min_anchor_count:
            raise RuntimeError(
                f'No usable gantry observations for diamond {label}; counts={counts}'
            )
        return gantry_obs

    async def _wait_for_diamond_lines_to_stop(self, deadband=0.05, timeout=30):
        await asyncio.sleep(2)
        deadline = asyncio.get_event_loop().time() + timeout
        speed1 = np.nan
        speed3 = np.nan
        while asyncio.get_event_loop().time() < deadline:
            speed1 = abs(self.datastore.anchor_line_record[1].getLast()[2])
            speed3 = abs(self.datastore.anchor_line_record[3].getLast()[2])
            if speed1 < deadband and speed3 < deadband:
                await asyncio.sleep(2)
                return True
            await asyncio.sleep(1/30)

        await self.send_line_speed(1, 0)
        await self.send_line_speed(3, 0)
        raise RuntimeError(
            f'Diamond lines did not settle before timeout: '
            f'line1_speed={speed1:.4f}m/s line3_speed={speed3:.4f}m/s'
        )

    async def touch_floor(self):
        await self.gripper_client.send_commands({'set_finger_angle': -30})
        laser_range = self.datastore.range_record.getLast()[1]
        logger.info(f'Touch the floor. current range: {laser_range}')
        try:
            await self.move_direction_speed(np.array([0, 0, -0.1]))
            timeout = time.time()+20
            while laser_range > 0.12 and time.time() < timeout:
                await asyncio.sleep(0.1)
                laser_range = self.datastore.range_record.getLast()[1]
                logger.debug(f'Laser range: {laser_range}')
        finally:
            self.slow_stop_all_spools()


    async def collect_arp_anchor_eyelet_experiment_data(self, anchor_poses):
        """  
        Perform experiments in which only the eyelet lines are tight and a diamond pattern is observed
        """
        tilts = (self.config.anchors[0].indirect_line.cam_tilt, self.config.anchors[1].indirect_line.cam_tilt)

        try:
            for a in self.anchors.values():
                a.save_raw = True
            
            # move to the center of the room.

            # touch the floor using the rangefinder
            await self.touch_floor()

            self.slow_stop_all_spools()

            logger.info('Relax the direct lines, tighten the indirect line')

            def get_direct_tensions():
                t0 = self.datastore.anchor_line_record[0].getLast()[3]
                t2 = self.datastore.anchor_line_record[2].getLast()[3]
                return t0,t2

            # relax direct lines
            await self._set_arp_direct_line_anti_tangle(False)
            t0,t2 = get_direct_tensions()
            direct_relax_deadline = time.time() + 10
            while t0 > 0.1 or t2 > 0.1:
                if time.time() > direct_relax_deadline:
                    raise RuntimeError(f'Direct lines did not relax before timeout: line0={t0:.3f}N line2={t2:.3f}N')
                await self.send_line_speed(0,  0.1 if t0 > 0.1 else 0)
                await self.send_line_speed(2,  0.1 if t2 > 0.1 else 0)
                await asyncio.sleep(0.1)
                t0,t2 = get_direct_tensions()
                print((t0,t2))
            await self.send_line_speed(0, 0)
            await self.send_line_speed(2, 0)
            # another 30 cm
            await self.send_line_speed(0,  0.3, jog=True)
            await self.send_line_speed(2,  0.3, jog=True)

            # tighten indirect lines
            await self.send_line_speed(1, -0.02, jog=True)
            await self.send_line_speed(3, -0.02, jog=True)

            await asyncio.sleep(1)
            self.slow_stop_all_spools()

            half_h, half_w = self._adaptive_diamond_size()
            logger.info(
                f'Using Arpeggio calibration diamond half-height={half_h:.3f}m '
                f'half-width={half_w:.3f}m'
            )

            results = {}
            line_deltas = {}


            def get_eyelet_lengths():
                l1 = self.datastore.anchor_line_record[1].getLast()[1]
                l3 = self.datastore.anchor_line_record[3].getLast()[1]
                return l1, l3

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=20.0,
                name="Calibration",
                current_action="Observe diamond bottom",
            ))
            logger.info('This position is the bottom of the diamond. Observe gantry for 2 seconds')
            await asyncio.sleep(5)
            results['bottom'] = self._require_gantry_observations('bottom')

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=25.0,
                name="Calibration",
                current_action="Observe diamond right",
            ))
            # RIGHT:
            logger.info('Move to RIGHT')
            l1_before, l3_before = get_eyelet_lengths()
            await self.send_line_speed(1, -half_w-half_h, jog=True)
            await self.send_line_speed(3, half_w-half_h, jog=True)
            await self.send_line_speed(0,  0.3, jog=True)
            await self.send_line_speed(2,  0.3, jog=True)
            await self._wait_for_diamond_lines_to_stop()
            await self.send_line_speed(1, 0)
            await self.send_line_speed(3, 0)
            l1_after, l3_after = get_eyelet_lengths()
            line_deltas['bot_to_rig'] = (l1_after - l1_before, l3_after - l3_before)
            logger.info(f'bot_to_rig actual deltas: line1={line_deltas["bot_to_rig"][0]:.4f}, line3={line_deltas["bot_to_rig"][1]:.4f}')
            await asyncio.sleep(5)
            results['right'] = self._require_gantry_observations('right') # it is to the right from the perspective of camera 0

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=30.0,
                name="Calibration",
                current_action="Observe diamond top",
            ))
            # TOP:
            logger.info('Move to TOP')
            l1_before, l3_before = get_eyelet_lengths()
            await self.send_line_speed(1, half_w-half_h, jog=True)
            await self.send_line_speed(3, -half_w-half_h, jog=True)
            await self._wait_for_diamond_lines_to_stop()
            await self.send_line_speed(1, 0)
            await self.send_line_speed(3, 0)
            l1_after, l3_after = get_eyelet_lengths()
            line_deltas['rig_to_top'] = (l1_after - l1_before, l3_after - l3_before)
            logger.info(f'rig_to_top actual deltas: line1={line_deltas["rig_to_top"][0]:.4f}, line3={line_deltas["rig_to_top"][1]:.4f}')
            await asyncio.sleep(5)
            results['top'] = self._require_gantry_observations('top')

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=35.0,
                name="Calibration",
                current_action="Observe diamond left",
            ))
            # LEFT:
            logger.info('Move to LEFT')
            l1_before, l3_before = get_eyelet_lengths()
            await self.send_line_speed(1, half_w+half_h, jog=True)
            await self.send_line_speed(3, -half_w+half_h, jog=True)
            await self.send_line_speed(0,  0.1, jog=True)
            await self.send_line_speed(2,  0.1, jog=True)
            await self._wait_for_diamond_lines_to_stop()
            await self.send_line_speed(1, 0)
            await self.send_line_speed(3, 0)
            l1_after, l3_after = get_eyelet_lengths()
            line_deltas['top_to_lef'] = (l1_after - l1_before, l3_after - l3_before)
            logger.info(f'top_to_lef actual deltas: line1={line_deltas["top_to_lef"][0]:.4f}, line3={line_deltas["top_to_lef"][1]:.4f}')
            await asyncio.sleep(5)
            results['left'] = self._require_gantry_observations('left')

            # set back anti tangle to normal function 
            await self._set_arp_direct_line_anti_tangle(True)

            logger.info('Return result')
            for a in self.anchors.values():
                a.save_raw = False

            analyze_diamond_data(results, anchor_poses, tilts)

            return results, line_deltas

        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()
            await self._restore_calibration_cleanup()
    
    async def half_auto_calibration(self):
        """
        Set line lengths from observation
        tighten, wait for obs, estimate line lengths, move up slightly, estimate line lengths, move down slightly
        This is a motion task
        """
        NUM_SAMPLE_POINTS = 3
        OPTIMIZER_TIMEOUT_S = 60  # seconds
        
        try:
            if len(self.anchors) < N_ANCHORS[self.config.anchor_type]:
                logger.warning('Cannot run half calibration until all anchors are connected')
                return

            need_sc_restart = False
            if self.swing_cancellation_task is not None and not self.swing_cancellation_task.done():
                self.swing_cancellation_task.cancel()
                need_sc_restart = True

            for direction in [[0,0,1], [0,0,-1]]:
                if not await self.tension_and_wait():
                    self.send_ui(operation_progress=telemetry.OperationProgress(
                        percent_complete=100.0,
                        name="Calibration",
                        current_action="Calibration failed: line tension did not settle",
                    ))
                    return False
                # wait for some new obs
                await asyncio.sleep(0.5)
                lengths = np.linalg.norm(self.pe.anchor_points - self.pe.visual_pos, axis=1)
                if not await self.sendReferenceLengths(lengths):
                    self.slow_stop_all_spools()
                    self.send_ui(operation_progress=telemetry.OperationProgress(
                        percent_complete=100.0,
                        name="Calibration",
                        current_action="Calibration failed: reference length data was invalid",
                    ))
                    return False
                await asyncio.sleep(0.25)
                # move in direction for short time
                await self.move_direction_speed(direction, 0.05, downward_bias=0)
                await asyncio.sleep(0.25)
                self.slow_stop_all_spools()

            if need_sc_restart:
                self.swing_cancellation_task = asyncio.create_task(self.run_swing_cancellation())
            return True

        except asyncio.CancelledError:
            raise

    async def full_auto_calibration(self):
        """Automatically determine anchor poses and zero angles
        This is a motion task"""
        calibration_artifact = self._new_calibration_artifact('full_auto_calibration')
        calibration_artifact.set_phase('start')
        self.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=0.0,
            name="Calibration",
            current_action="Observing markers",
        ))
        finger_task = None
        swing_calibration_ok = None
        DETECTION_WAIT_S = 1.0 # seconds
        try:
            if len(self.anchors) < N_ANCHORS[self.config.anchor_type]:
                calibration_artifact.fail(
                    'not all anchors connected',
                    connected_anchors=len(self.anchors),
                    expected_anchors=N_ANCHORS[self.config.anchor_type],
                )
                self._write_calibration_artifact(calibration_artifact)
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=100.0,
                    name="Calibration",
                    current_action='Cannot run full calibration until all anchors are connected',
                ))
                return False
            elif len(self.anchors) > N_ANCHORS[self.config.anchor_type]:
                logger.warning(f'Too many anchors found for type {self.config.anchor_type} \n{self.anchors}')
            # collect observations of origin card aruco marker to get initial guess of anchor poses.
            #   origin pose detections are actually always stored by all connected clients,
            #   it is only necessary to ensure enough have been collected from each client and average them.
            for a in self.anchors.values():
                a.save_raw = True
            origin_counts = {}
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=2.0,
                name="Calibration",
                current_action="Observing markers",
            ))
            calibration_artifact.set_phase('origin_marker_capture')
            detecting_start = time.time()
            while (
                len(origin_counts) == 0
                or len(origin_counts) < N_ANCHORS[self.config.anchor_type]
                or min(origin_counts.values()) < max_origin_detections
            ):
                logger.debug(f'Waiting for enough origin card detections from every anchor camera {origin_counts}')
                self.send_ui(visibility_states=telemetry.VisibilityStates(anchors_seeing_origin_card=list(
                    self._origin_visible_anchor_nums(origin_counts)
                )))

                await asyncio.sleep(DETECTION_WAIT_S)
                origin_counts = self._origin_detection_counts()
            logger.info(f'Collected enough observations {origin_counts}')
            calibration_artifact.record_observation(
                kind='origin_visibility',
                counts=origin_counts,
                elapsed_s=time.time() - detecting_start,
            )
            self.send_ui(visibility_states=telemetry.VisibilityStates(anchors_seeing_origin_card=list(
                self._origin_visible_anchor_nums(origin_counts)
            )))

            raw_obs = self.snapshot_tag_observations()
            calibration_artifact.record_observation(
                kind='raw_marker_snapshot',
                marker_counts={
                    marker: [len(anchor_obs) for anchor_obs in sightings]
                    for marker, sightings in raw_obs.items()
                },
            )

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=12.0,
                name="Calibration",
                current_action="Determining anchor positions",
            ))

            if self.config.anchor_type == common.AnchorType.ARPEGGIO:
                calibration_artifact.set_phase('arpeggio_anchor_solve')
                tilts = (self.config.anchors[0].indirect_line.cam_tilt, self.config.anchors[1].indirect_line.cam_tilt)
                # determine position of two anchors visually and guess at external eyelets.
                async_result = self.pool.apply_async(optimize_arp_anchors, (raw_obs, None, None, None, None, tilts))
                anchor_poses, eyelet_positions = async_result.get(timeout=30)
                calibration_artifact.record_optimizer_report(
                    name='arpeggio_anchor_initial',
                    success=anchor_poses is not None and eyelet_positions is not None,
                )
                logger.info(f'Obtained result from optimize_arp_anchors anchor_poses=\n{anchor_poses}\neyelet_positions=\n{eyelet_positions}')

                self.save_poses_arp(anchor_poses, eyelet_positions)
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=15.0,
                    name="Calibration",
                    current_action="Collecting proprioceptive data",
                ))
                if await self.half_auto_calibration() is False:
                    calibration_artifact.fail('half calibration failed before eyelet solve')
                    self._write_calibration_artifact(calibration_artifact)
                    return False

                # measure finger contact and reset wrist while doing the diamond pattern to save time.
                async def wait_then_finger():
                    await asyncio.sleep(10)
                    await self.calibrate_finger_servo()
                    # if you want to re-enable this to make calibration faster, prevent the seek goal function from turning the wrist
                    # await self.gripper_client.send_commands({'reset_wrist': None})
                finger_task = asyncio.create_task(wait_then_finger())

                # collect length_change_data data to estimate eyelets better
                calibration_artifact.set_phase('arpeggio_eyelet_probe')
                diamond_data, line_deltas = await self.collect_arp_anchor_eyelet_experiment_data(anchor_poses)
                calibration_artifact.record_observation(
                    kind='arpeggio_diamond',
                    line_deltas=line_deltas,
                    gantry_counts={
                        key: [len(anchor_obs) for anchor_obs in value]
                        for key, value in diamond_data.items()
                    },
                )
                # stop saving raw poses
                for a in self.anchors.values():
                    a.save_raw = False
                # debug: save args for experimentation
                args = (raw_obs, diamond_data, None, None, line_deltas, tilts)
                # with open('arp_opt_data.pkl', 'wb') as f:
                #     pickle.dump(args, f)
                # optimize again with length_change_data
                calibration_artifact.set_phase('arpeggio_eyelet_solve')
                async_result = self.pool.apply_async(optimize_arp_anchors, args)
                anchor_poses, eyelet_positions = async_result.get(timeout=30)
                calibration_artifact.record_optimizer_report(
                    name='arpeggio_eyelet',
                    success=anchor_poses is not None and eyelet_positions is not None,
                )
                logger.info(f'Obtained result from optimize_arp_anchors anchor_poses=\n{anchor_poses}\neyelet_positions=\n{eyelet_positions}')

                self.save_poses_arp(anchor_poses, eyelet_positions)

            else:
                calibration_artifact.set_phase('pilot_anchor_solve')
                for a in self.anchors.values():
                    a.save_raw = False

                # run optimization in pool
                async_result = self.pool.apply_async(optimize_anchor_poses, (raw_obs,))
                anchor_poses = async_result.get(timeout=30)
                calibration_artifact.record_optimizer_report(
                    name='pilot_anchor',
                    success=anchor_poses is not None,
                )
                logger.info(f'Obtained result from find_cal_params anchor_poses=\n{anchor_poses}')
                anchor_poses = np.array(anchor_poses)

                # Use the optimization output to update anchor poses and spool params
                for client in self.anchors.values():
                    self.config.anchors[client.anchor_num].pose = poseTupleToProto(anchor_poses[client.anchor_num])
                    client.updatePose(anchor_poses[client.anchor_num])
                save_config(self.config, self.config_path)
                # inform UI
                self.send_ui(new_anchor_poses=telemetry.AnchorPoses(poses=[
                    poseTupleToProto(p)
                    for p in anchor_poses
                ]))
                # inform position estimator
                anchor_points = np.array([compose_poses([pose, model_constants.anchor_grommet])[1] for pose in anchor_poses])
                self.pe.set_anchor_points(anchor_points)


            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=40.0,
                name="Calibration",
                current_action="Tensioning lines and Locating Gripper",
            ))
            if await self.half_auto_calibration() is False:
                calibration_artifact.fail('half calibration failed after anchor solve')
                self._write_calibration_artifact(calibration_artifact)
                return False

            # open grip enough that we can see an unobstructed view from the palm camera
            if finger_task is not None:
                await finger_task
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': -30}))

            # move over the origin card
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=60.0,
                name="Calibration",
                current_action="Moving gripper to origin",
            ))
            self.gantry_goal_pos = np.array([0,0,1.2])
            await self.seek_gantry_goal()

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=90.0,
                name="Calibration",
                current_action="Measuring spin",
            ))
            # there should be some swing when we get there. 
            if await self.half_auto_calibration() is False:
                calibration_artifact.fail('half calibration failed before spin calibration')
                self._write_calibration_artifact(calibration_artifact)
                return False

            # roomspin
            calibration_artifact.set_phase('spin_calibration')
            await self.calibrate_spin(reset_wrist_first=True) # already did that during diamond to save time

            if isinstance(self.gripper_client, ArpeggioGripperClient) and self.gripper_client.connected:
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=95.0,
                    name="Calibration",
                    current_action="Calibrating swing cancellation",
                ))
                calibration_artifact.set_phase('swing_cancellation_calibration')
                swing_calibration_ok = await self.auto_calibrate_swing_cancellation()
                calibration_artifact.record_optimizer_report(
                    name='swing_cancellation',
                    success=swing_calibration_ok is not False,
                )

            # TODO "Calibration complete. Would you like stringman to pick up the cards and put them in the trash? yes/no"
            completion_action = "Calibration completed. Sanity check anchor positions before moving. Cards can be removed from the floor. Parking location must be re-recorded."
            if swing_calibration_ok is False:
                completion_action = (
                    "Calibration completed, but swing cancellation calibration failed. "
                    "Leave swing cancellation disabled and rerun swingcal after checking logs."
                )
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=100.0,
                name="Calibration",
                current_action=completion_action,
            ))
            self._write_calibration_artifact(
                calibration_artifact,
                status='completed',
                message=completion_action,
            )
            return True

        except asyncio.CancelledError:
            calibration_artifact.fail('cancelled by user')
            self._write_calibration_artifact(calibration_artifact)
            if finger_task is not None:
                finger_task.cancel()
                await finger_task
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=100.0,
                name="Calibration",
                current_action="Cancelled by user",
            ))
            raise
        except Exception as e:
            calibration_artifact.fail('calibration failed', exception=repr(e))
            self._write_calibration_artifact(calibration_artifact)
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=100.0,
                name="Calibration",
                current_action='Calibration failed, see motion controller console',
            ))
            raise

    async def calibrate_spin(self, reset_wrist_first=True):
        """Calibration of the relationship between the wrist and the room frame of reference.
        Must be done over the origin card.
        """
        if self.gripper_client.last_frame_resized is None:
            logger.warning('Cannot calibrate the relationship between gripper zero angle and camera if gripper camera is offline!')
            return None

        # record the z rotation of the gantry card from the perspective of the gripper camera's stabilized frame
        # when the stabilization is done without any existing z rotation term
        self.gripper_client.calibrating_room_spin = True
        try:
            if isinstance(self.gripper_client, ArpeggioGripperClient):
                # measurement must be taken at the wrist's zero point
                center_angle = 540
                if reset_wrist_first:
                    asyncio.create_task(self.gripper_client.send_commands({'reset_wrist': None}))
                    await asyncio.sleep(10)
                # wait till within 1 degree of target
                actual_wrist = 100
                end_time = time.time() + 2
                logger.info(f'Moved wrist to {center_angle}, waiting to reach position')
                while abs(actual_wrist - center_angle) > 2.0 and time.time() < end_time:
                    await asyncio.sleep(0.2)
                    actual_wrist = self.datastore.winch_line_record.getLast()[1]
                logger.info(f'Actual wrist position = {actual_wrist}')

            # detect origin card
            try:
                await asyncio.sleep(0.1)
                origin_card_pose = [None]
                def special_handle_det(timestamp, detections):
                    for d in detections:
                        if d['n'] == 'origin':
                            # a pose of the origin card in the frame of reference of the stabilized gripper cam.
                            origin_card_pose[0] = d['p']
                end_time = time.time() + 10
                logger.info('Collecting observations of origin card from gripper cam')
                while origin_card_pose[0] is None and time.time() < end_time:
                    async_result = self.pool.apply_async(
                        locate_markers_gripper,
                        (self.gripper_client.last_frame_resized, self.config.camera_cal_wide),
                        callback=partial(special_handle_det, time.time()))
                    detections = async_result.get(timeout=5)
            except Exception as e:
                logger.exception(e)
                raise
            if origin_card_pose[0] is None:
                raise RuntimeError("Gripper camera was unable to make any observations of the origin card.")

            euler_rot = Rotation.from_rotvec(origin_card_pose[0][0]).as_euler('zyx')
            logger.info(f'Euler rotation of origin card relative to stabilized gripper camera {euler_rot}')
            roomspin = euler_rot[0]
            self.config.gripper.frame_room_spin = roomspin
            self.config.has_been_calibrated = True
            save_config(self.config, self.config_path)
        finally:
            self.gripper_client.calibrating_room_spin = False

    async def horizontal_line_task(self):
        """
        Attempt to move the gantry in a perfectly horizontal line. How hard could this be?
        This is a motion task
        """
        await self.tension_and_wait()
        await asyncio.sleep(1)
        range_at_start = self.datastore.range_record.getLast()[1]
        result = await self.move_direction_speed([1,0,0], 0.2, downward_bias=0)
        await asyncio.sleep(4)
        self.slow_stop_all_spools()
        await asyncio.sleep(1)
        range_at_end = self.datastore.range_record.getLast()[1]
        logger.info(f'During attempted horizontal move, height rose by {range_at_end - range_at_start} meters')

    async def record_park(self):
        """Record that the current location is reseted in the parking saddle and save in the config"""
        # confirm we can actually see the parking target in the grip camera
        if self.gripper_client.park_pose_relative_to_camera is not None:
            self.config.park_data.pos = fromnp(self.pe.gant_pos)

            # save marker pose in rested position
            self.config.park_data.marker_resting = poseTupleToProto(self.gripper_client.park_pose_relative_to_camera)

            # move up 10cm
            await self.move_direction_speed(np.array([0, 0, 0.1]))
            await asyncio.sleep(1.0)
            self.slow_stop_all_spools()
            await asyncio.sleep(1.0)

            # save marker pose while 10cm over target
            self.config.park_data.marker_over = poseTupleToProto(self.gripper_client.park_pose_relative_to_camera)

            # move down 10cm
            await self.move_direction_speed(np.array([0, 0, -0.1]))
            await asyncio.sleep(1.0)
            self.slow_stop_all_spools()
            await asyncio.sleep(1.0)

            save_config(self.config, self.config_path)
            self.send_ui(named_position=telemetry.NamedObjectPosition(
                name = 'parking_location',
                position = self.config.park_data.pos
            ))
            self.send_ui(pop_message=telemetry.Popup(
                message=f'Saved parking location as {self.config.park_data.pos}'
            ))
        else:
            self.send_ui(pop_message=telemetry.Popup(
                message=f'Cannot save location here. The parking marker is not in view of the gripper camera.'
            ))


    async def park(self):
        """ Park on the parking hook for safe power down. """
        FINGER_ANGLE_FOR_CLEAR_VIEW = -30
        STAGING_HOR_OFFSET_M = 0.2
        STAGING_VER_OFFSET_M = 0.0
        LOOK_FOR_MARKER_INITIAL_S = 2.0
        HOMING_TIME_S = 16.0
        MARKER_DIST_CLOSE_ENOUGH = 0.16
        HOMING_SPEED_MPS = 0.02
        HOMING_LOOP_DELAY = 0.1

        if isinstance(self.gripper_client, RaspiGripperClient):
            logger.warning("Self park unsupported in pilot gripper")
            return

        try:
            # TODO check if holding something, if so warn user and do not proceed.

            # perform half cal.

            # open gripper
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': FINGER_ANGLE_FOR_CLEAR_VIEW}))

            # move to position above and in front of saddle,
            parkpos = tonp(self.config.park_data.pos)
            away = get_inward_wall_normal(parkpos, self.pe.anchor_points) * STAGING_HOR_OFFSET_M
            self.gantry_goal_pos = parkpos + np.array([away[0], away[1], STAGING_VER_OFFSET_M])
            await self.seek_gantry_goal()

            # TODO rotate to face wall because camera is under nose and it lets us see a little further.

            # use observed position of park marker to adjust slowly towards
            # the park-over position
            park_over_pose = poseProtoToTuple(self.config.park_data.marker_over)
            over = park_over_pose[1]


            pos = None
            timeout = time.time()+LOOK_FOR_MARKER_INITIAL_S
            while time.time() < timeout:
                try:
                    pos = self.gripper_client.park_pose_relative_to_camera[1]
                    direction = pos - over
                    # since the gripper's camera is stabilized and rotated into the room frame of reference
                    # a vector pointing from the desired position of the marker to the current position in image space
                    # is the same direction we'd need to move the gantry in the room.
                    break
                except TypeError:
                    continue
            if pos is None:
                logger.warning("Can't see parking tag right now")
                return

            timeout = time.time()+HOMING_TIME_S
            while np.linalg.norm(direction) > MARKER_DIST_CLOSE_ENOUGH  and time.time() < timeout:
                move = np.array([direction[1], direction[0], 0])
                await self.move_direction_speed(move, HOMING_SPEED_MPS)
                logger.debug(f'Distance {np.linalg.norm(direction)} and moving {move}')
                await asyncio.sleep(HOMING_LOOP_DELAY)
                try:
                    pos = self.gripper_client.park_pose_relative_to_camera[1]
                    direction = pos - over
                except TypeError:
                    pass
                
            self.slow_stop_all_spools()

            # move down 20cm
            # TODO or until any two lines become slack
            # or until laser range reaches same distance recorded during set park
            await self.move_direction_speed(np.array([0, 0, -0.1]))
            await asyncio.sleep(2.0)
            self.slow_stop_all_spools()

            # for looks, as well as to let me know it finished.
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': 10}))

        except asyncio.CancelledError:
            logger.info('Park cancelled')
            raise
        finally:
            self.slow_stop_all_spools()
            await self.clear_gantry_goal()


    async def unpark(self):
        """ Unpark from the saddle and move clear of it. """
        try:
            # assume gantry position based on parking location since we probably can't see it
            parkpos = tonp(self.config.park_data.pos)
            self.pe.kf.reset_biases(parkpos)
            # move up 10cm
            await self.move_direction_speed(np.array([0, 0, 0.1]))
            await asyncio.sleep(1.0)
            # move directly away from the wall.
            away = get_inward_wall_normal(parkpos, self.pe.anchor_points)
            await self.move_direction_speed(np.array([away[0], away[1], 0]), 0.15)
            await asyncio.sleep(2.0)
            # move towards center of room.
            self.gantry_goal_pos = np.array([0,0,1])
            task = asyncio.create_task(self.seek_gantry_goal())
            # but don't go all the way, just stop after a bit
            await asyncio.sleep(5.0)
            await self.clear_gantry_goal()
            await self.half_auto_calibration()
        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()
            await self.clear_gantry_goal()

    def on_service_state_change(self, 
        zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        if 'cranebot' in name:
            if state_change is ServiceStateChange.Added:
                asyncio.create_task(self.add_service(zeroconf, service_type, name))
            if state_change is ServiceStateChange.Updated:
                asyncio.create_task(self.update_service(zeroconf, service_type, name))
            if state_change is ServiceStateChange.Removed:
                asyncio.create_task(self.remove_service(service_type, name))
            elif state_change is ServiceStateChange.Updated:
                pass

    async def add_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """Records the information about a discovered service in the config"""
        info = AsyncServiceInfo(service_type, name)
        await info.async_request(zc, INFO_REQUEST_TIMEOUT_MS)
        if not info or info.server is None or info.server == '':
            return None;
        namesplit = name.split('.')
        kind = namesplit[1]
        key  = ".".join(namesplit[:3])

        address = socket.inet_ntoa(info.addresses[0])
        logger.debug(f'Service discovered: {namesplit}')

        is_power_anchor = kind == anchor_power_service_name
        is_standard_anchor = kind == anchor_service_name
        is_standard_gripper = kind == gripper_service_name
        is_arp_gripper = kind == arp_gripper_service_name
        is_arp_anchor = kind == arp_anchor_service_name

        # -- BEFORE --
        # the number of anchors is decided ahead of time (in main.py)
        # but they are assigned numbers as we find them on the network
        # and the chosen numbers are persisted in configuration.json

        # -- AFTER --
        # the number of lines is always four.
        # the number of anchors may be four pilot anchors controlling one line each,
        # or two arpeggio anchors controlling two lines each.
        # they cannot be mixed. As soon as one type is discovered, this config will be locked to that type.
        # when the anchor type is arpeggio, anchor_num is 0 or 1.
        # refrerences to anchor num that referred to a service, a camera or its pose can still reference anchor num.
        # references to anchor num that were referring to grommet positions or line lengths and speeds,
        # must now refer line numbers 0-3. sending a command to jog a spool or set a line speed must be abstracted through
        # a class that will send the message to the connected server that manages that line.

        if is_power_anchor or is_standard_anchor or is_arp_anchor:
            found_type = common.AnchorType.ARPEGGIO if is_arp_anchor else common.AnchorType.PILOT
            
            if self.config.anchor_type == common.AnchorType.UNSPECIFIED:
                # the first discovered anchor locks the config to an anchor type
                self.config.anchor_type = found_type
                if is_arp_anchor:
                    # replace the four default pilot anchors in the config with two default arp anchors having unset addresses and service names
                    self.config.anchors = default_arp_anchors() # imported from config_loader

            elif self.config.anchor_type != found_type:
                logger.warning(f'Ignored {found_type} anchor at {address} because config is locked to {self.config.anchor_type}')
                return

            # create a map from service name to anchor num
            anchor_num_map = {a.service_name: a.num for a in self.config.anchors if a.service_name is not None}
            if key in anchor_num_map:
                anchor_num = anchor_num_map[key]
            else:
                anchor_num = len(anchor_num_map)
                if anchor_num >= N_ANCHORS[self.config.anchor_type]:
                    # Discovering more that four anchors could be a sign that another robot in the same network is turned on.
                    # We need a way to know that, but for now, you'll have to make sure only one is one at a time while discovering.
                    # After discovery, it should be ok to have more than one on at a time.
                    logger.warning(f"Discovered another {found_type} server on the network, but we already know of {N_ANCHORS[self.config.anchor_type]} {key} {address}")
                    return None
            if self.config.anchors[anchor_num].address != address or self.config.anchors[anchor_num].port != info.port:
                self.config.anchors[anchor_num].num = anchor_num
                self.config.anchors[anchor_num].service_name = key
                self.config.anchors[anchor_num].address = address
                self.config.anchors[anchor_num].port = info.port
                save_config(self.config, self.config_path)

        elif is_standard_gripper or is_arp_gripper:
            # a gripper has been discovered, assume it is ours only if we have never seen one before
            if self.config.gripper.service_name is None or self.config.gripper.service_name == "":
                self.config.gripper.service_name = key
                self.config.gripper.address = address
                self.config.gripper.port = info.port
                save_config(self.config, self.config_path)
                logger.info(f'Discovered gripper at "{address}" and adopted it as the gripper for this robot')
            elif address != self.config.gripper.address:
                logger.info(f'Discovered gripper at "{address}" and ignored it because ours is at {self.config.gripper.address}')

    async def update_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        # when zerconf has detected a change in address or port
        pass

    async def remove_service(self, service_type: str, name: str) -> None:
        """
        Finds if we have a client connected to this service. if so, ends the task if it is running, and deletes the client
        """
        namesplit = name.split('.')
        kind = namesplit[1]
        key  = ".".join(namesplit[:3])

        # only in this dict if we are connected to it.
        if key in self.bot_clients:
            # await self._handle_set_swing_cancellation(item=control.SetSwingCancellation(enabled=False, present='.'))
            client = self.bot_clients[key]
            await client.shutdown()
            if kind == anchor_service_name or kind == anchor_power_service_name or kind == arp_anchor_service_name:
                del self.anchors[client.anchor_num]
            elif kind == gripper_service_name or kind == arp_gripper_service_name:
                self.gripper_client = None
            del self.bot_clients[key]

    async def startup_action(self, event):
        """A sequence of actions to run when all components are discovered."""
        # wait for event
        await event.wait()

        # unpark if we were parked.
        r = await self.unpark()
        # start pick_and_place_loop
        r = await self.pick_and_place_loop()
        # pick and place finishes if no targets appear during a timeout
        # park robot
        r = await self.park()
        # disconnect all components and set flag that they should not reconnect unless control input is received.

    async def keep_robot_connected(self):
        """
        Keep a connection open to every robot component known in the config
        components are keyed by their service name which is the first three components of info.name, eg
        123.cranebot-anchor-service.2ccf67bc3fc4
        """
        # If config is empty (first time startup) sleep until zeroconf discovers robot components
        while not config_has_any_address(self.config) and self.run_command_loop:
            await asyncio.sleep(0.5)

        ready = asyncio.Event()
        if self.auto_start:
            s_task = asyncio.create_task(self.startup_action(ready))

        while self.run_command_loop:
            # is everything up the way we want it to be?
            if len([b for b in self.bot_clients.values() if b.connected])==5:
                ready.set()
                await asyncio.sleep(0.5)
                continue # All websocket connections are up.

            # make sure we have either a live connection to, or an ongoing attempt to connect to every component we know about.
            for cpt in [self.config.gripper, *self.config.anchors]:
                # assume only the common attributes between those two types
                key = cpt.service_name
                if key is None or cpt.address is None or cpt.port is None:
                    continue

                if key not in self.connection_tasks:
                    # Start a connection to this component. connect_component will also remove it when it completes regardless of success or failure.
                    self.connection_tasks[key] = asyncio.create_task(self.connect_component(key))

            await asyncio.sleep(0.5)

        if self.auto_start:
            s_task.cancel()
            r = await s_task

        for task in self.connection_tasks.values():
            task.cancel()
        result = await asyncio.gather(*self.connection_tasks.values())

    async def connect_component(self, service_name):
        """Connect to the component with the given name using the address stored in the config."""
        client = None
        try:
            name_component = service_name.split('.')[1]
        except IndexError:
            logger.warning(f'Invalid service name "{service_name}"')
            return

        is_power_anchor = name_component == anchor_power_service_name
        is_standard_anchor = name_component == anchor_service_name
        is_standard_gripper = name_component == gripper_service_name
        is_arp_gripper = name_component == arp_gripper_service_name
        is_arp_anchor = name_component == arp_anchor_service_name

        if is_standard_gripper:
            client = RaspiGripperClient(self.config.gripper.address, self.config.gripper.port, self.datastore, self, self.pool, self.stat, self.pe, self.telemetry_env)
            self.gripper_client_connected.clear()
            client.connection_established_event = self.gripper_client_connected
            self.gripper_client = client
            self.pe.set_gripper_type('pilot')
        if is_arp_gripper:
            client = ArpeggioGripperClient(self.config.gripper.address, self.config.gripper.port, self.datastore, self, self.pool, self.stat, self.pe, self.telemetry_env)
            self.gripper_client_connected.clear()
            client.connection_established_event = self.gripper_client_connected
            self.gripper_client = client
            self.pe.set_gripper_type('arp')
        elif is_power_anchor or is_standard_anchor:
            for a in self.config.anchors:
                if a.service_name != service_name:
                    continue
                client = RaspiAnchorClient(a.address, a.port, a.num, self.datastore, self, self.pool, self.stat, self.telemetry_env)
                client.connection_established_event = self.any_anchor_connected
                self.anchors[a.num] = client
        elif is_arp_anchor:
            for a in self.config.anchors:
                if a.service_name != service_name:
                    continue
                client = ArpeggioAnchorClient(a.address, a.port, a.num, self.datastore, self, self.pool, self.stat, self.telemetry_env)
                client.connection_established_event = self.any_anchor_connected
                self.anchors[a.num] = client
        else:
            logger.warning(f"Don't know how to connect to {name_component}")

        if client:
            self.bot_clients[service_name] = client
            # this function runs as long as the client is connected and returns true if the client was forced to disconnect abnormally
            abnormal_close = await client.startup()
            # remove client
            r = await self.remove_service(None, service_name)
            if abnormal_close:
                self.send_ui(pop_message=telemetry.Popup(
                    message=f'lost connection to {service_name}'
                ))
                await self.stop_all()
            # delete this task from the dict as it ends, so keep_robot_connected will try agian. 
            del self.connection_tasks[service_name]

    async def connect_cloud_telemetry(self):
        ws_protocol_and_host = CONTROL_PLANE_LOCAL
        if self.telemetry_env == 'staging':
            ws_protocol_and_host = CONTROL_PLANE_STAGING
        if self.telemetry_env == 'production':
            ws_protocol_and_host = CONTROL_PLANE_PRODUCTION

        while self.run_command_loop:
            try:
                use_id = self.config.robot_id
                ws_path = f"{ws_protocol_and_host}/telemetry/{use_id}"
                async with websockets.connect(ws_path, max_size=None, open_timeout=10) as websocket:
                    self.cloud_telem_websocket = websocket
                    logger.info(f'Connected to control plane {ws_path}')
                    # send anything that it would need up-front
                    await self.send_setup_telemetry()
                    try:
                        async for message in websocket:
                            r = await self.handle_command(message)
                            if not self.run_command_loop:
                                r = await websocket.close()
                    except ConnectionClosedOK as e:
                        logger.info(f'ConnectionClosedOK from {ws_path}')
                    except ConnectionClosedError as e:
                        logger.error(e)
                    finally:
                        logger.info(f'Disconnected from control plane {ws_path}')
                        self.cloud_telem_websocket = None
            except (asyncio.exceptions.CancelledError, websockets.exceptions.ConnectionClosedOK):
                pass # normal close
            except ConnectionRefusedError:
                logger.warning(f'Connection to control plane refused')
            except websockets.exceptions.InvalidMessage:
                logger.warning('Connection to control plane ended due to invalid message')
            await asyncio.sleep(2)

    def send_ui(self, **kwargs):
        """
        Ensure that the given telemetry item is sent to every connected UI
        keyword args are passed directly to telemetry item, so you can construct one like this

        self.send_ui(pop_message=telemetry.Popup('hello'))
        """
        if len(kwargs.keys()) != 1:
            raise ValueError
        key, msg = list(kwargs.items())[0]

        # mark certain messages with a retain key. the server will resend them to new UIs
        item = telemetry.TelemetryItem(**kwargs)
        if key == 'new_anchor_poses':
            item.retain_key = 'new_anchor_poses'
        if key == 'component_conn_status':
            if msg.is_gripper:
                item.retain_key = f'component_conn_status_g'
            else:
                item.retain_key = f'component_conn_status_{msg.anchor_num}'
        if key == 'video_ready':
            item.retain_key = f'video_ready_{msg.feed_number}'
        if key == 'episode_control' and item.episode_control.status is not None:
            self.last_ep_ctrl_status = item.episode_control.status
            item.retain_key = f'lerobot_status'

        # Add item to batch
        with self.telemetry_buffer_lock:
            self.telemetry_buffer.append(item)
            OBS.set_telemetry_buffer(len(self.telemetry_buffer))
        OBS.record_telemetry_item(key)
        OBS.record_telemetry_payload(key, msg)

    async def flush_tele_buffer(self):
        """
        Flush the teloperation buffer. sending all data to all UI clients.
        Normally called within position estimator's 60hz loop
        """
        started = time.time()
        with self.telemetry_buffer_lock:
            batch = telemetry.TelemetryBatchUpdate(
                robot_id=self.config.robot_id,
                updates=list(self.telemetry_buffer)
            )
            self.telemetry_buffer.clear()
            OBS.set_telemetry_buffer(0)
        to_send = bytes(batch)
        # copy list to prevent RuntimeError: Set changed size during iteration
        connected_clients = self.connected_local_clients.copy()
        if self.cloud_telem_websocket:
            connected_clients.add(self.cloud_telem_websocket) # will only be connected when self.telemetry_env is not None
        recipients = len(connected_clients)
        with OBS.span("observer.flush_telemetry", bytes=len(to_send), recipients=recipients):
            for ui_websocket in connected_clients:
                try:
                    r = await ui_websocket.send(to_send)
                except (ConnectionClosedOK, ConnectionClosedError) as e:
                    pass # stale connection
        OBS.record_telemetry_flush(bytes_sent=len(to_send), recipients=recipients, duration=time.time() - started)

    async def start_pe_when_ready(self):
        await self.any_anchor_connected.wait()
        r = await self.pe.main()

    async def main(self) -> None:
        self.startup_complete.clear()

        from nf_robot.host.loop_monitor import LoopMonitor
        monitor = LoopMonitor(interval=0.5, threshold=0.2)
        monitor.start()

        self.passive_safety_task = asyncio.create_task(self.passive_safety())
        self.observability_task = asyncio.create_task(self.update_observability_runtime())

        if self.telemetry_env is not None:
            self.cloud_telem = asyncio.create_task(self.connect_cloud_telemetry())

        # statistic counter - measures things like average camera frame latency
        asyncio.create_task(self.stat.stat_main())

        # A task that continuously estimates the position of the gantry
        # remains asleep until at least one anchor connects.
        self.pe_task = asyncio.create_task(self.start_pe_when_ready())

        # main process must own pool, and there's only one. multiple subprocesses may submit work.
        with Pool(processes=3, initializer=configure_worker_process) as pool:
            self.pool = pool

            # zeroconf only discovers services and keeps their addresses and ports up to date in the config.
            # start a task to connect and reconnect to all known robot components.
            self.keeper = asyncio.create_task(self.keep_robot_connected())

            # the only reason it might not be none is if a unit test set before calling main.
            if self.aiozc is None:
                self.aiozc = AsyncZeroconf(ip_version=IPVersion.V4Only, interfaces=InterfaceChoice.All)

            try:
                services = list(
                    await AsyncZeroconfServiceTypes.async_find(aiozc=self.aiozc, ip_version=IPVersion.V4Only)
                )
                self.aiobrowser = AsyncServiceBrowser(
                    self.aiozc.zeroconf, services, handlers=[self.on_service_state_change]
                )
            except asyncio.exceptions.CancelledError:
                await self.aiozc.async_close()
                return

            # perception model
            if self.run_ai or self.run_ortho:
                # task remains in a lightweight sleep until frames arrive.
                self.perception_task = asyncio.create_task(self.run_perception())

            # start a websocket server to accept incoming connections from either a local UI or local Lerobot session
            async with websockets.serve(self.handle_local_client, "127.0.0.1", self.port):
                # await something that will end when the program closes to keep serving and
                # keep zeroconf alive and discovering services.
                try:
                    self.startup_complete.set()

                    if self.telemetry_env == None:
                        message = f'Listening on localhost:{self.port} To control visit https://neufangled.com/playroom?robotid=lan on this machine'
                    elif self.telemetry_env == 'local':
                        message = f'To control visit http://localhost:5173/playroom?robotid={self.config.robot_id}'
                    elif self.telemetry_env == 'production':
                        message = f'To control visit https://neufangled.com/playroom?robotid={self.config.robot_id}'
                    elif self.telemetry_env == 'staging':
                        message = f'To control visit https://nf-site-monolith-staging-690802609278.us-east1.run.app/playroom?robotid={self.config.robot_id}'
                    else:
                        print(f'invalid telemetry_env {self.telemetry_env}')

                    bar = '=' * (len(message) + 12)
                    print(bar)
                    print(f'===== {message} =====')
                    print(bar)

                    result = await self.keeper
                except asyncio.exceptions.CancelledError:
                    pass

            await self.async_close()

    async def async_close(self) -> None:
        print('Stringman Controller Shutdown')
        result = await self.stop_all()
        self.run_command_loop = False
        self.stat.run = False
        self.pe.run = False
        self.pe_task.cancel()
        tasks = [self.pe_task, self.keeper]
        if self.cloud_telem:
            self.cloud_telem.cancel()
            tasks.append(self.cloud_telem)
        if self.aiobrowser is not None:
            tasks.append(self.aiobrowser.async_cancel())
        if self.aiozc is not None:
            tasks.append(self.aiozc.async_close())
        if self.locate_anchor_task is not None:
            tasks.append(self.locate_anchor_task)
        if self.gip_task is not None:
            tasks.append(self.gip_task)
        if self.swing_cancellation_task is not None:
            self.swing_cancellation_task.cancel()
            tasks.append(self.swing_cancellation_task)
        if self.lerobot_process_watcher is not None:
            self.lerobot_process_watcher.cancel()
            tasks.append(self.lerobot_process_watcher)
        if self.perception_task is not None:
            self.perception_task.cancel()
            tasks.append(self.perception_task)
        if self.passive_safety_task is not None:
            self.passive_safety_task.cancel()
            tasks.append(self.passive_safety_task)
        if self.observability_task is not None:
            self.observability_task.cancel()
            tasks.append(self.observability_task)

        tasks.extend([client.shutdown() for client in self.bot_clients.values()])
        try:
            result = await asyncio.gather(*tasks)
        except asyncio.exceptions.CancelledError:
            pass

    async def add_simulated_data_point2point(self):
        """Simulate the gantry moving from random point to random point.
        The only purpose of this simulation at the moment is to test the position estimator and it's feedback
        """
        LOWER_Z_BOUND = 1.0 # meters
        UPPER_Z_OFFSET = 0.3 # meters
        MAX_SPEED_MPS = 0.25 # m/s
        GOAL_PROXIMITY_THRESHOLD = 0.03 # meters
        SOFT_SPEED_FACTOR = 0.25
        RANDOM_EVENT_CHANCE = 0.5
        CAM_BIAS_STD_DEV = 0.2 # meters
        OBSERVATION_NOISE_STD_DEV = 0.01 # meters
        WINCH_LINE_LENGTH = 1.0 # meters
        RANGEFINDER_OFFSET = 1.0 # meters
        LOOP_SLEEP_S = 0.05 # seconds
        
        # each camera produces measurements with a position bias that can be around 20x larger than the position noise from a given camera.
        cam_bias = np.random.normal(0, CAM_BIAS_STD_DEV, (4, 3))

        pending_obs = deque()

        lower = np.min(self.pe.anchor_points, axis=0)
        upper = np.max(self.pe.anchor_points, axis=0)
        lower[2] = LOWER_Z_BOUND
        upper[2] = upper[2] - UPPER_Z_OFFSET
        # starting position
        gantry_real_pos = np.random.uniform(lower, upper)
        # initial goal
        travel_goal = np.random.uniform(lower, upper)
        t = time.time()
        while self.run_command_loop:
            try:
                now = time.time()
                elapsed_time = now - t
                t = now
                # move the gantry towards the goal
                to_goal_vec = travel_goal - gantry_real_pos
                dist_to_goal = np.linalg.norm(to_goal_vec)
                if dist_to_goal < GOAL_PROXIMITY_THRESHOLD:
                    # choose new goal
                    travel_goal = np.random.uniform(lower, upper)
                else:
                    soft_speed = dist_to_goal * SOFT_SPEED_FACTOR
                    # normalize
                    to_goal_vec = to_goal_vec / dist_to_goal
                    velocity = to_goal_vec * min(soft_speed, MAX_SPEED_MPS)
                    gantry_real_pos = gantry_real_pos + velocity * elapsed_time
                if random() > RANDOM_EVENT_CHANCE:
                    anchor_num = np.random.randint(4) # which camera it was observed from.
                    observed_position = gantry_real_pos + cam_bias[anchor_num] + np.random.normal(0, OBSERVATION_NOISE_STD_DEV, (3,))
                    dp = np.concatenate([[t], [anchor_num], observed_position])
                    # simulate delayed data
                    pending_obs.appendleft(dp)
                    if len(pending_obs) > 10:
                        dp = pending_obs.pop()
                        self.datastore.gantry_pos.insert(dp)
                        self.datastore.gantry_pos_event.set()
                        self.send_ui(gantry_sightings=telemetry.GantrySightings(sightings=[fromnp(dp[2:])]))
                
                # winch line always 1 meter
                self.datastore.winch_line_record.insert(np.array([t, WINCH_LINE_LENGTH, 0.0]))
                
                # range always perfect
                self.datastore.range_record.insert(np.array([t, gantry_real_pos[2]-RANGEFINDER_OFFSET]))

                # anchor lines always perfectly agree with gripper position
                for i, simanc in enumerate(self.pe.anchor_points):
                    dist = np.linalg.norm(simanc - gantry_real_pos)
                    last = self.datastore.anchor_line_record[i].getLast()
                    timesince = t-last[0]
                    travel = dist-last[1]
                    speed = travel/timesince # referring to the specific speed of this line, not the gantry
                    self.datastore.anchor_line_record[i].insert(np.array([t, dist, speed, 1.0]))
                    self.datastore.anchor_line_record_event.set()
                tt = self.datastore.anchor_line_record[0].getLast()[0]
                await asyncio.sleep(LOOP_SLEEP_S)
            except asyncio.exceptions.CancelledError:
                break

    async def send_gripper_move(self, line_speed, finger_speed, wrist_speed):
        """Command the gripper's motors in one update.
        finger speed is in degrees per second (but it's the fake degrees of the finger which range from -90 (open) to 90 (closed))
        positive values close the fingers.
        wrist speed is in real degrees per second."""
        update = {}

        if isinstance(self.gripper_client, ArpeggioGripperClient):

            # arpeggio gripper. Update finger and wrist speed
            cg = telemetry.CommandedGrip()
            if finger_speed is not None:
                finger_speed = clamp(finger_speed, -90, 90)
                update['set_finger_speed'] = finger_speed
                cg.finger_speed = finger_speed
            if wrist_speed is not None:
                wrist_speed = clamp(wrist_speed, -120, 120)
                update['set_wrist_speed'] = wrist_speed
                cg.wrist_speed = wrist_speed
            self.send_ui(last_commanded_grip=cg)
            r = await self.flush_tele_buffer()

        elif isinstance(self.gripper_client, RaspiGripperClient):

            # pilot gripper, update winch speed and finger angle
            if line_speed is not None:
                update['aim_speed'] = line_speed # winch
            if finger_speed is not None and abs(finger_speed) > 1.0:
                finger_speed = clamp(finger_speed, -90, 90)
                await self.gripper_client.set_finger_speed(finger_speed)

        if update:
            asyncio.create_task(self.gripper_client.send_commands(update))
        return line_speed, finger_speed, wrist_speed

    async def send_gripper_move_legacy(self, line_speed, finger_angle, wrist_angle):
        """Command the gripper's motors in one update."""
        update = {}
        if line_speed is not None:
            update['aim_speed'] = line_speed
        if finger_angle is not None:
            update['set_finger_angle'] = clamp(finger_angle, -90, 90)
        if wrist_angle is not None:
            clamped = clamp(wrist_angle, 0, 1080)
            update['set_wrist_angle'] = clamped
        if update and self.gripper_client is not None:
            asyncio.create_task(self.gripper_client.send_commands(update))
        return line_speed, finger_angle, wrist_angle

    async def clear_gantry_goal(self):
        self.gantry_goal_pos = None
        self.send_ui(named_position=telemetry.NamedObjectPosition(name='gantry_goal_marker')) # not setting position causes it to be hidden

    async def seek_gantry_goal(self):
        """
        Move towards a goal position, using the constantly updating gantry position provided by the position estimator
        This is a motion task
        """
        GOAL_PROXIMITY_M = 0.07 
        MAX_SPEED = 0.24 # GANTRY_SPEED_MPS
        ACCEL = 0.15     # m/s^2
        LOOP_SLEEP_S = 0.1

        # Calculate the distance needed to stop from MAX_SPEED: d = v^2 / (2a)
        braking_distance = (MAX_SPEED**2) / (2 * ACCEL)
        start_pos = self.pe.gant_pos
        current_speed = 0.0
        
        try:
            self.send_ui(named_position=telemetry.NamedObjectPosition(position=fromnp(self.gantry_goal_pos), name='gantry_goal_marker'))
            dist_to_goal = 10
            while self.gantry_goal_pos is not None:
                vector = self.gantry_goal_pos - self.pe.gant_pos
                dist_to_goal = np.linalg.norm(vector)
                dist_from_start = np.linalg.norm(self.pe.gant_pos - start_pos)

                if dist_to_goal < GOAL_PROXIMITY_M:
                    break

                # Calculate target speed based on distance from start (ramp up) 
                # and distance to goal (ramp down)
                # v = sqrt(2 * a * d)
                speed_ramp_up = np.sqrt(2 * ACCEL * max(dist_from_start, 0.01))
                speed_ramp_down = np.sqrt(2 * ACCEL * dist_to_goal)
                
                # Target speed is the lowest of the ramps or the max allowable speed
                target_speed = min(speed_ramp_up, speed_ramp_down, MAX_SPEED)
                
                # Smoothly interpolate current_speed toward target_speed to prevent 
                # instantaneous velocity jumps between loop iterations
                step = ACCEL * LOOP_SLEEP_S
                if current_speed < target_speed:
                    current_speed = min(current_speed + step, target_speed)
                else:
                    current_speed = max(current_speed - step, target_speed)

                self.gripper_client.look_towards_vector(vector[:2])

                # Normalize vector and command movement
                await self.move_direction_speed(vector / dist_to_goal, current_speed, self.pe.gant_pos)
                await asyncio.sleep(LOOP_SLEEP_S)

            logger.info(f'Goal reached {tuple(self.gantry_goal_pos)}')
        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()
            await self.clear_gantry_goal()

    async def send_line_speed(self, line_no, speed, jog=False):
        # send the line speed to the client that controls that line
        # when jog==True, speed is interpreted as a length in meters by which to lengthen the line
        command = 'jog' if jog else 'aim_speed'
        if self.config.anchor_type == common.AnchorType.PILOT:
            if line_no in self.anchors:
                asyncio.create_task(self.anchors[line_no].send_commands({command: speed}))
        elif self.config.anchor_type == common.AnchorType.ARPEGGIO:
            if line_no//2 in self.anchors:
                spool_no = line_no%2
                # we consider the lower line number to be the direct line
                asyncio.create_task(self.anchors[line_no//2].send_commands({command: (speed, spool_no)}))

    async def move_direction_speed(
        self,
        uvec,
        speed=None,
        starting_pos=None,
        downward_bias=-0.04,
        key='default',
        record_retry=True,
    ):
        """Move in the direction of the given unit vector at the given speed.
        Any move must be based on some assumed starting position. if none is provided,
        we will use the last one sent from position_estimator

        Due to inaccuaracy in the positions of the anchors and lengths of the lines,
        the speeds we command from the spools will not be perfect.
        On average, half will be too high, and half will be too low.
        Because there are four lines and the gantry only hangs stably from three,
        the actual point where the gantry ends up hanging after any move will always be higher than intended
        So a small downward bias is introduced into the requested direction to account for this.
        The size of the bias should theoretically be a function of the the magnitude of position and line errors,
        but we don't have that info. alternatively we could calibrate the bias to make horizontal movements level
        according to the laser rangefinder.

        if speed is None, uvec is assumed to be velocity and used directly with no bias

        If key is supplied, the resulting vector overwrites the last one with the same key
        Whenever one of the keys from the set that is being combined changes, all keys in the active set are summed and sent to the anchors.
        """
        KINEMATICS_STEP_SCALE = 10.0 # Determines the size of the virtual step to calculate line speed derivatives

        if starting_pos is None:
            starting_pos = self.pe.gant_pos

        # when speed is not provided, use uvec as a velocity vector in m/s (mode used with lerobot)
        if speed is None:
            speed = np.linalg.norm(uvec)

        # when a very small speed is provided, clamp it to zero.
        if speed < 0.005:
            speed = 0

        if speed == 0:
            velocity = np.zeros(3)
        else:
            # normalize, apply downward bias and renormalize
            uvec  = uvec / (np.linalg.norm(uvec) + 1e-5)
            uvec = uvec + np.array([0,0,downward_bias])
            uvec  = uvec / (np.linalg.norm(uvec) + 1e-5)
            velocity = uvec * speed

        # this commanded velocity overwrites the last velocity with the same key and all velocities are summed
        # currently this is only used to combine swing cancellation with user inputs.
        self.input_velocities[key] = velocity
        total_velocity = np.sum([self.input_velocities.get(k, 0) for k in self.active_set], axis=0)
        
        # Determine the total requested speed before limits
        speed = np.linalg.norm(total_velocity)

        # enforce a model dependent speed limit
        speed_limit = 0.5
        if self.config.anchor_type == common.AnchorType.PILOT:

            # On pilot stringman, also enforce a height dependent speed limit on the total combined velocity.
            # the reason being that as gantry height approaches anchor height, the line tension increases exponentially,
            # and a slower speed is need to maintain enough torque from the stepper motors.
            # The speed limit is proportional to how far the gantry hangs below a level 10cm below the average anchor.
            # This makes the behavior consistent across installations of different heights.
            hang_distance = np.mean(self.pe.anchor_points[:, 2]) - starting_pos[2]
            speed_limit = clamp(0.28 * (hang_distance - 0.1), 0.01, 0.25)
            # If the combined total speed exceeds the limit, scale the vector down
        elif self.config.anchor_type == common.AnchorType.ARPEGGIO:
            speed_limit = 1.0

        if speed > speed_limit:
            total_velocity = total_velocity * (speed_limit / speed)
            speed = speed_limit

        # line lengths at starting pos
        lengths_a = np.linalg.norm(starting_pos - self.pe.anchor_points, axis=1)
        # line lengths at new pos
        new_pos = starting_pos + (total_velocity / KINEMATICS_STEP_SCALE)
        
        # zero the speed if this would move the gantry out of the work area
        if not self.pe.point_inside_work_area(new_pos):
            speed = 0
            total_velocity = np.zeros(3)

        if record_retry:
            self._record_retryable_move(key, total_velocity)
            
        lengths_b = np.linalg.norm(new_pos - self.pe.anchor_points, axis=1)
        deltas = lengths_b - lengths_a
        line_speeds = deltas * KINEMATICS_STEP_SCALE

        # send move on each line
        for i, line_speed in enumerate(line_speeds):
            await self.send_line_speed(i, line_speed)
            
        self.pe.record_commanded_vel(total_velocity)
        return total_velocity

    def get_last_frame(self, camera_key):
        """gets the last frame of video from the given camera if possible
        camera_key should be one of 'g' 0, 1, 2, 3
        """
        image = None
        if camera_key == 'g':
            if self.gripper_client is not None:
                image = self.gripper_client.lerobot_jpeg_bytes
        else:
            image = self.anchors[int(camera_key)].lerobot_jpeg_bytes
        if image is not None:
            return image
        return bytes()

    def _handle_add_episode_control_events(self, data: nf.common.EpisodeControl):
        # forward episode control events back to all telemetry listeners
        self.send_ui(episode_control=data)
        asyncio.create_task(self.flush_tele_buffer())
        # TODO if the EpisodeControl message has a command, and we are running a session as a subprocess, forward it directly to that subprocess.
        # the subprocess may also send us EpisodeControl messages containing a status. forward these as telemetry.

    def send_tq_to_ui(self):
        snapshot = self.target_queue.get_queue_snapshot()
        # Create a deterministic hash
        current_hash = hash(bytes(snapshot))
        if current_hash != self.last_snapshot_hash:
            self.send_ui(target_list=snapshot)
            self.last_snapshot_hash = current_hash

    def _ortho_worker(self, ortho_floor_vs, heatmap_floor_vs):
        """
        Sync thread driven by self.ortho_event, which anchor frame_resizer_loops set on every
        new processed frame.  Projects all anchor views onto the floor and stores the result so
        the AI task can read it without re-running the projection.
        """
        from nf_robot.host.floor_view import generate_orthographic_floor_maps
        EXTENT = 5.0
        while self.run_command_loop:
            if not self.ortho_event.wait(timeout=1.0):
                continue
            self.ortho_event.clear()
            try:
                valid_clients = [
                    c for c in list(self.anchors.values())
                    if c.last_frame_resized is not None and c.anchor_num in self.config.preferred_cameras
                ]
                if not valid_clients:
                    continue

                heatmaps = self.last_heatmaps_np
                if heatmaps is None or len(heatmaps) != len(valid_clients):
                    heatmaps = np.zeros(
                        (len(valid_clients),) + valid_clients[0].last_frame_resized.shape[:2],
                        dtype=np.float32,
                    )

                ortho_heatmap, ortho_bgr = generate_orthographic_floor_maps(
                    valid_clients, heatmaps, self.config.camera_cal,
                    map_size_px=1000, map_extent_meters=EXTENT,
                )
                self.last_ortho_bgr = ortho_bgr
                self.last_ortho_heatmap = ortho_heatmap

                if ortho_floor_vs is not None:
                    ortho_floor_vs.send_frame(cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2RGB))
                if heatmap_floor_vs is not None:
                    heatmap_floor_vs.send_frame(
                        cv2.applyColorMap((ortho_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    )
            except Exception:
                logger.exception('_ortho_worker iteration failed')

    async def run_perception(self):
        """
        Orthographic floor projection and target heatmap inference.
        run_ortho and run_ai are independent: either or both may be active.
        """
        TARGETING_MODEL_REPOID = "naavox/targeting"
        CENTERING_MODEL_REPOID = "naavox/centering"
        LOOP_DELAY = 0.1
        FIND_TARGETS_EVERY = 5
        EXTENT = 5.0

        # wait until at least one preferred camera is producing frames
        logging.info('waiting for camera frames')
        while True:
            await asyncio.sleep(1)
            have_frames = (
                (self.gripper_client is not None and self.gripper_client.last_frame_resized is not None)
                or any(
                    anum in self.config.preferred_cameras and c.last_frame_resized is not None
                    for anum, c in self.anchors.items()
                )
            )
            if have_frames:
                break

        if self.run_ai:
            import torch
            configure_native_thread_pools(configure_torch=True)
            from huggingface_hub import hf_hub_download
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            from nf_robot.ml.target_heatmap import TargetHeatmapNet, extract_targets_from_heatmap, HM_IMAGE_RES
            if self.use_arp_grasp:
                from nf_robot.ml.centering import CenteringNet

            def load_models_sync():
                if self.local_models:
                    target_path = "models/target_heatmap.pth"
                else:
                    target_path = hf_hub_download(repo_id=TARGETING_MODEL_REPOID, filename="target_heatmap.pth")
                logger.info(f"Loading model from {target_path}...")
                t_model = TargetHeatmapNet().to(DEVICE)
                t_model.load_state_dict(torch.load(target_path, map_location=DEVICE))
                t_model.eval()

                if self.use_arp_grasp:
                    if self.local_models:
                        center_path = "models/square_centering.pth"
                    else:
                        center_path = hf_hub_download(repo_id=CENTERING_MODEL_REPOID, filename="square_centering.pth")
                    logger.info(f"Loading model from {center_path}...")
                    c_model = CenteringNet().to(DEVICE)
                    c_model.load_state_dict(torch.load(center_path, map_location=DEVICE))
                    c_model.eval()
                    return t_model, c_model
                else:
                    return t_model, None

            self.target_model, self.centering_model = await asyncio.to_thread(load_models_sync)

        ortho_floor_vs = None
        heatmap_floor_vs = None
        if self.run_ortho:
            from nf_robot.host.video_streamer import NfVideoStreamer

            def _make_on_ready(feed_number):
                def on_ready(local_uri, stream_path):
                    t = telemetry.VideoReady(
                        is_gripper=None,
                        anchor_num=None,
                        local_uri=local_uri,
                        stream_path=stream_path,
                        feed_number=feed_number,
                    )
                    logger.debug(f'sending {t}')
                    self.send_ui(video_ready=t)
                return on_ready

            ortho_floor_vs = NfVideoStreamer(
                width=1000, height=1000, fps=10,
                mjpeg_port=8747,
                stream_path=f'stringman/{self.config.robot_id}/3',
                telemetry_env=self.telemetry_env,
                on_ready=_make_on_ready(3),
            )
            ortho_floor_vs.start()
            heatmap_floor_vs = NfVideoStreamer(
                width=1000, height=1000, fps=10,
                mjpeg_port=8748,
                stream_path=f'stringman/{self.config.robot_id}/4',
                telemetry_env=self.telemetry_env,
                on_ready=_make_on_ready(4),
            )
            heatmap_floor_vs.start()
            self.ortho_streamers = [(ortho_floor_vs, 3), (heatmap_floor_vs, 4)]

        ortho_thread = threading.Thread(
            target=self._ortho_worker,
            args=(ortho_floor_vs, heatmap_floor_vs),
            daemon=True,
        )
        ortho_thread.start()

        counter = 0
        while self.run_command_loop:
            await asyncio.sleep(LOOP_DELAY)
            if not self.run_ai:
                continue
            counter += 1
            if counter < FIND_TARGETS_EVERY:
                continue
            counter = 0

            valid_anchor_clients = [
                c for c in self.anchors.values()
                if c.last_frame_resized is not None and c.anchor_num in self.config.preferred_cameras
            ]
            if not valid_anchor_clients:
                continue

            img_tensors = [
                torch.from_numpy(cv2.resize(c.last_frame_resized, HM_IMAGE_RES, interpolation=cv2.INTER_AREA))
                     .permute(2, 0, 1).float() / 255.0
                for c in valid_anchor_clients
            ]
            batch = torch.stack(img_tensors).to(DEVICE)

            def infer_sync():
                with torch.no_grad():
                    return self.target_model(batch).squeeze(1).cpu().numpy()

            heatmaps_np = await asyncio.to_thread(infer_sync)
            self.last_heatmaps_np = heatmaps_np

            ortho_heatmap = self.last_ortho_heatmap
            if ortho_heatmap is None:
                continue

            results = extract_targets_from_heatmap(ortho_heatmap)
            if len(results) > 0:
                targets2d = (results[:, :2] + np.array([-0.5, -0.5])) * EXTENT
                floor_targets = [
                    {'position': np.array([p[0], p[1], 0]), 'dropoff': 'hamper'}
                    for p in targets2d
                    if self.pe.point_inside_work_area_2d(p)
                ]
            else:
                floor_targets = []
            self.target_queue.add_ai_targets(floor_targets)
            self.send_tq_to_ui()

        if self.run_ortho:
            ortho_floor_vs.stop()
            heatmap_floor_vs.stop()

    async def pick_and_place_loop(self):
        """
        Long running motion task that repeatedly identifies targets picks them up and drops them over the hamper
        """
        GANTRY_HEIGHT_OVER_TARGET = 0.9
        GANTRY_HEIGHT_OVER_DROPOFF = 0.9
        RELAXED_OPEN = 0 # enough to drop something
        DELAY_AFTER_DROP = 0.6 # long enough that the payload is not visible anymore in the hand
        LOOP_DELAY = 0.5
        END_LOOP_TIMEOUT = 10

        drop_point = np.zeros(3)
        target_seen_t = time.time()
        try:
            gtask = None
            while self.run_command_loop:

                # hover over the hamper
                # await asyncio.sleep(1)
                # if 'hamper' in self.named_positions:
                #     self.gantry_goal_pos = self.named_positions['hamper'] + np.array([0,0,GANTRY_HEIGHT_OVER_DROPOFF])
                #     await self.seek_gantry_goal()
                # continue

                next_target = self.target_queue.get_best_target()
                if next_target is None:
                    if gtask is not None:
                        gtask.cancel()
                    self.gantry_goal_pos = None
                    if time.time() > target_seen_t + END_LOOP_TIMEOUT:
                        logger.info('Looks clean enough to me!')
                        return
                    await asyncio.sleep(LOOP_DELAY)
                    continue
                target_seen_t = time.time()

                self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SELECTED)
                self.send_tq_to_ui()

                # pick Z position for gantry
                # if we are too close to the drop point right now, the z position has to be our current z so we don't get hung up on the basket by going down too soon.
                # otherwise use the normal value
                if np.linalg.norm(self.pe.gant_pos - (drop_point + np.array([0,0,GANTRY_HEIGHT_OVER_DROPOFF]))) < 0.5:
                    z_pos = self.pe.gant_pos[2]
                else:
                    z_pos = GANTRY_HEIGHT_OVER_TARGET
                goal_pos = next_target.position + np.array([0, 0, z_pos])
                self.gantry_goal_pos = goal_pos

                # gantry is now heading for a position over next_target
                # wait only one second for it to arrive.
                if gtask is None or gtask.done():
                    gtask = asyncio.create_task(self.seek_gantry_goal())
                done, pending = await asyncio.wait([gtask], timeout=1)
                
                if gtask in pending:
                    # if doesn't arrive in one second, run target selection again since a better one might have appeared or the user might have put one in their queue
                    self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SEEN)
                    continue

                if self.gripper_client is None:
                    logger.warning('Pick and place aborted because we lost the gripper connection')
                    break

                # when we reach this point we arrived over the item. commit to it unless it proves impossible to pick up.
                logger.info('Attempt grasp')
                start = time.time()
                success = await self.execute_grasp()
                logger.info(f'Grasp succeeded={success} took {time.time() - start:.2f}s')
                if not success:
                    # just pick another target, but consider downranking this object or something.
                    self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SEEN)
                    self.send_tq_to_ui()
                    await asyncio.sleep(LOOP_DELAY)
                    continue
                else:
                    self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.PICKED_UP)
                    self.send_tq_to_ui()
                    logger.info('Object picked up')

                # tension now just in case.
                # await self.tension_and_wait()

                # If user specified drop point...
                if not isinstance(next_target.dropoff, str):
                    drop_point = next_target.dropoff
                # otherwise go to the named drop point
                if next_target.dropoff in self.named_positions:
                    drop_point = self.named_positions[next_target.dropoff]
                else:
                    # otherwise use the origin as a drop point :/
                    # TODO this is not ideal, as we will continue to pick things up from this spot most likely now that we are close to it.
                    # either need to drop it somewhere we know we won't ever see it again, or have a sign for this drop point so we don't touch things inside it.
                    logger.warning("No drop point specified, using (0,0,0) as a drop point")
                    drop_point = np.zeros(3)

                # fly to to drop point
                logger.info(f'Flying to drop point {drop_point}')
                self.gantry_goal_pos = drop_point + np.array([0,0,GANTRY_HEIGHT_OVER_DROPOFF])
                await self.seek_gantry_goal()
                # open gripper
                asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': RELAXED_OPEN}))
                # don't immediately select a new target, because there's a chance it'll be the sock you're holding.
                # TODO train network on more data containing examples of this, so it knows that only socks on the floor count.
                await asyncio.sleep(DELAY_AFTER_DROP)
                self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.DROPPED)
                self.send_tq_to_ui()
                # keep score


        except asyncio.CancelledError:
            raise
        finally:
            if gtask is not None:
                logger.info('Pick and place cancelled')
                gtask.cancel()
            self.slow_stop_all_spools()
            await self.clear_gantry_goal()

    async def execute_grasp(self):
        """Try to grasp whatever is directly below the gripper"""
        if isinstance(self.gripper_client, ArpeggioGripperClient):
            if self.use_arp_grasp:
                return await self.arp_execute_grasp()
            return await self.act_execute_grasp()
        else:
            return await self.pilot_execute_grasp()

    async def pilot_execute_grasp(self):
        FINGER_LENGTH = 0.1 # length between rangefinder and floor when fingers touch in meters
        HALF_VIRTUAL_FOV = model_constants.rpi_cam_3_fov * SF_SCALE_FACTOR / 2 * (np.pi/180)
        DOWNWARD_SPEED = -0.06
        VISUAL_CONF_THRESHOLD = 0.1 # level below which we give up on the target
        COMMIT_HEIGHT = 0.3 # height below which giving up due to visual disconfidence is not allowed.
        LAT_TRAVEL_FRACTION = 0.75 # try to finish lateral travel by this fraction of the time spent travelling downwards
        LAT_SPEED_ADJUSTMENT = 5.00 # final adjustment to lateral speed
        LOOP_DELAY = 0.1
        PRESSURE_SENSE_WAIT = 2.0

        smooth_grip_angle = self.grip_angle

        try:
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))
            attempts = 3
            while not self.pe.holding and attempts > 0 and self.run_command_loop:
                attempts -= 1
                asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))

                # move laterally until target is centered
                # at the same time, move downward until tip is detected.

                nothing_seen_countdown = 15
                self.pe.tip_over.clear()
                while (self.predicted_lateral_vector is not None and not self.pe.tip_over.is_set()):
                    distance_to_floor = self.datastore.range_record.getLast()[1]
                    if distance_to_floor < FINGER_LENGTH:
                        logger.debug(f'Stop going down, distance to floor is {distance_to_floor}')
                        break

                    if self.gripper_sees_target < VISUAL_CONF_THRESHOLD and distance_to_floor > COMMIT_HEIGHT:
                        nothing_seen_countdown -= 1
                        if nothing_seen_countdown == 0:
                            logger.debug('Nothing seen during centering loop')
                            break
                    else:
                        nothing_seen_countdown = 15

                    # calculate eta to the floor using laser range, we want to finish lateral travel at 0.75 of that eta
                    lat_travel_seconds = (distance_to_floor-FINGER_LENGTH)/(-DOWNWARD_SPEED)*LAT_TRAVEL_FRACTION
                    lateral_vector = np.zeros(3)
                    if lat_travel_seconds > 0:
                        # determine which direction we'd have to move laterally to center the object
                        # you get a normalized u,v coordinate in the [-1,1] range
                        # for now assume that the up direction in the gripper image is -Y in world space 
                        # stabilize_frame produced this direction and I think it depends on the compass.
                        # the direction in world space depends on how the user placed the origin card on the ground
                        # we need to capture a number during calibration to relate these two.
                        # +1 is the edge of the image. how far laterally that would be depends on how far from the ground the gripper is.
                        pred_vector = self.predicted_lateral_vector
                        pred_vector[1] *= -1
                        # lateral distance to object
                        lateral_vector = np.sin(pred_vector * HALF_VIRTUAL_FOV) * distance_to_floor
                        # lateral distance in meters
                        lateral_distance = np.linalg.norm(lateral_vector)
                        # speed to travel that lateral distance in lat_travel_seconds
                        lateral_speed = lateral_distance / lat_travel_seconds * LAT_SPEED_ADJUSTMENT
                    else:
                        # once we get too close, go straight down, stop relying on the camera
                        lateral_speed = 0
                    lateral_vector *= lateral_speed

                    logger.debug(f'Moving {[lateral_vector[0],lateral_vector[1],DOWNWARD_SPEED]}')
                    await self.move_direction_speed([lateral_vector[0],lateral_vector[1],DOWNWARD_SPEED])

                    try:
                        # the normal sleep on this loop would be LOOP_DELAY s, but if tip is detected
                        # we want to stop immediately.
                        await asyncio.wait_for(self.pe.tip_over.wait(), LOOP_DELAY)
                        logger.debug('Detected tip over, must be floor')
                        break
                    except TimeoutError:
                        pass

                self.slow_stop_all_spools()
                self.pe.tip_over.clear()

                if nothing_seen_countdown == 0:
                    logger.debug('Nothing seen')
                    continue # find new target?

                logger.info('Close gripper')
                await self.gripper_client.send_commands({'set_finger_angle': CLOSED})
                logger.debug(f'Wait up to {PRESSURE_SENSE_WAIT} seconds for pad to sense object.')
                try:
                    await asyncio.wait_for(self.pe.finger_pressure_rising.wait(), PRESSURE_SENSE_WAIT)
                    self.pe.finger_pressure_rising.clear()
                except TimeoutError:
                    pressure = self.datastore.finger.getLast()[2]
                    logger.debug(f'Did not detect a successful hold. pressure=({pressure}) open and go back up high enough to get a view of the object')
                    # move up slowly at first, till fingers just touch ground and we are veritical. this keeps unwanted swinging to a minimum
                    await self.move_direction_speed([0,0,0.06])
                    await asyncio.sleep(1.0)
                    # now move up a little faster in a slightly random direction
                    direction = np.concatenate([np.random.uniform(-0.025, 0.025, (2)), [0.12]])
                    await self.move_direction_speed(direction)
                    asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))
                    await asyncio.sleep(2.0)
                    self.slow_stop_all_spools()
                    continue
                logger.info('Successful grasp')
                return True
            logger.info(f'Gave up on grasp after {attempts} attempts. self.pe.holding={self.pe.holding}')
            return False

        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()

    async def arp_execute_grasp(self):
        """Try to grasp whatever is directly below the gripper"""
        FINGER_LENGTH = 0.1 # length between rangefinder and floor when fingers touch in meters
        FLOOR_GRIPPER_HEIGHT = 0.11 # distance above floor (gripper origin) when grasp should be started
        RANGE_ITEM = 0.04 # range to item below which grip should be started
        HALF_VIRTUAL_FOV = model_constants.rpi_cam_3_wide_fov * SF_SCALE_FACTOR / 2 * (np.pi/180)
        DOWNWARD_SPEED = -0.07
        VISUAL_CONF_THRESHOLD = 0.1 # level below which we give up on the target
        COMMIT_HEIGHT = 0.3 # height below which giving up due to visual disconfidence is not allowed.
        LAT_TRAVEL_FRACTION = 0.75 # try to finish lateral travel by this fraction of the time spent travelling downwards
        LAT_SPEED_ADJUSTMENT = 5.00 # final adjustment to lateral speed. so huge because network outputs small values (why?)
        LOOP_DELAY = 0.1
        PRESSURE_SENSE_WAIT = 10.0
        NUM_ATTEMPTS = 3
        CLOSING_FINGER_SPEED = 30
        WRIST_SMOOTH_FACTOR = 0.9

        smooth_grip_angle = self.grip_angle

        try:
            attempts = NUM_ATTEMPTS
            while not self.pe.holding and attempts > 0 and self.run_command_loop:
                attempts -= 1
                logger.debug(f'Open fingers to {OPEN} to clear camera')
                asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))

                # move laterally until target is centered
                # at the same time, move downward until tip is detected.

                nothing_seen_countdown = 15
                approach_timeout = time.time()+10
                self.pe.tip_over.clear()
                while (self.predicted_lateral_vector is not None and not self.pe.tip_over.is_set() and time.time() < approach_timeout):
                    range_to_target = self.datastore.range_record.getLast()[1]
                    # compare this rangefinder distance to the distance estimated from other methods
                    gripper_height = self.pe.grip_pose[1][2]

                    # for bulky objects, we want to close range_to_target to about zero to get the fingers all the way around
                    # for small objects, we don't want to, we can't get that low, the fingers would touch the floor and the object
                    # would still be a few cm away from the rangefinder. 

                    logger.debug(f'range_to_target {range_to_target} gripper_height = {gripper_height}')
                    if range_to_target < RANGE_ITEM or gripper_height < FLOOR_GRIPPER_HEIGHT:
                        logger.debug(f'Reached target at height {gripper_height} and range {range_to_target}')
                        break

                    if self.gripper_sees_target < VISUAL_CONF_THRESHOLD and range_to_target > COMMIT_HEIGHT:
                        nothing_seen_countdown -= 1
                        if nothing_seen_countdown == 0:
                            logger.debug('Nothing seen during centering loop')
                            break
                    else:
                        nothing_seen_countdown = 15

                    # calculate eta to the floor using laser range, we want to finish lateral travel at 0.75 of that eta
                    lat_travel_seconds = (range_to_target-FINGER_LENGTH)/(-DOWNWARD_SPEED)*LAT_TRAVEL_FRACTION
                    lateral_vector = np.zeros(2)
                    if lat_travel_seconds > 0:
                        # determine which direction we'd have to move laterally to center the object
                        # you get a normalized u,v coordinate in the [-1,1] range
                        # for now assume that the up direction in the gripper image is -Y in world space 
                        # stabilize_frame produced this direction and I think it depends on the compass.
                        # the direction in world space depends on how the user placed the origin card on the ground
                        # we need to capture a number during calibration to relate these two.
                        # +1 is the edge of the image. how far laterally that would be depends on how far from the ground the gripper is.
                        pred_vector = self.predicted_lateral_vector
                        pred_vector[1] *= -1
                        # lateral distance to object
                        lateral_vector = np.sin(pred_vector * HALF_VIRTUAL_FOV) * range_to_target
                        # lateral distance in meters
                        lateral_distance = np.linalg.norm(lateral_vector)
                        # speed to travel that lateral distance in lat_travel_seconds
                        lateral_speed = lateral_distance / lat_travel_seconds * LAT_SPEED_ADJUSTMENT
                    else:
                        # once we get too close, go straight down, stop relying on the camera
                        lateral_speed = 0
                    lateral_vector *= lateral_speed

                    # rotate later component of direction from gripper frame into room frame
                    lateral_vector = rotate_vector(lateral_vector, -self.gripper_client.get_spin())

                    await self.move_direction_speed([lateral_vector[0],lateral_vector[1],DOWNWARD_SPEED])

                    # move wrist to predicted grip angle with smoothing
                    smooth_grip_angle = smooth_grip_angle*WRIST_SMOOTH_FACTOR + self.grip_angle*(1-WRIST_SMOOTH_FACTOR)
                    await self.gripper_client.send_commands({'set_wrist_angle': smooth_grip_angle/np.pi*180})

                    try:
                        # the normal sleep on this loop would be LOOP_DELAY s, but if tip is detected
                        # we want to stop immediately.
                        await asyncio.wait_for(self.pe.tip_over.wait(), LOOP_DELAY)
                        logger.debug('Detected tip over, must be floor')
                        break
                    except TimeoutError:
                        pass

                self.slow_stop_all_spools()
                self.pe.tip_over.clear()

                if nothing_seen_countdown == 0:
                    logger.debug('Nothing seen')
                    continue # find new target?

                logger.info('Close gripper')
                end_time = time.time() + PRESSURE_SENSE_WAIT
                self.pe.finger_pressure_rising.clear()

                await self.gripper_client.send_commands({'set_finger_speed': CLOSING_FINGER_SPEED})
                # finger speed commands take effect for 200ms only. they must be sent repeatedly.
                t, angle, pressure = self.datastore.finger.getLast()
                while time.time() < end_time and not self.pe.finger_pressure_rising.is_set() and angle < CLOSED:
                    await asyncio.sleep(0.03)
                    await self.gripper_client.send_commands({'set_finger_speed': CLOSING_FINGER_SPEED})
                    t, angle, pressure = self.datastore.finger.getLast()
                logger.debug(f'End grip finger_pressure_rising={self.pe.finger_pressure_rising.is_set()} angle={self.datastore.finger.getLast()[1]}')
                await self.gripper_client.send_commands({'set_finger_speed': 0})

                if not self.pe.finger_pressure_rising.is_set():
                    pressure = self.datastore.finger.getLast()[2]
                    logger.debug(f'Did not detect a successful hold, pressure=({pressure}) open and go back up high enough to get a view of the object')
                    # move up slowly at first, till fingers just touch ground and we are veritical. this keeps unwanted swinging to a minimum
                    await self.move_direction_speed([0,0,0.06])
                    await asyncio.sleep(1.0)
                    asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))
                    # now move up a little faster in a slightly random direction
                    direction = np.concatenate([np.random.uniform(-0.025, 0.025, (2)), [0.12]])
                    await self.move_direction_speed(direction)
                    await asyncio.sleep(2.0)
                    self.slow_stop_all_spools()
                    continue

                self.pe.finger_pressure_rising.clear()
                logger.info('Successful grasp')
                # slowly at first
                await self.move_direction_speed(np.array([0,0,0.05]))
                await asyncio.sleep(1.0)
                # and then all at once
                await self.move_direction_speed(np.array([0,0,0.15]))
                await asyncio.sleep(2.0)
                logger.info('Stop moving')
                self.slow_stop_all_spools()
                return True
            logger.info(f'Gave up on grasp after {NUM_ATTEMPTS-attempts} attempts. self.pe.holding={self.pe.holding}')
            return False

        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()

    async def act_execute_grasp(self):
        """
        Execute a grasp on an arp gripper using a lerobot ACT policy.
        End the episode either when a timeout is reached, when motion ceases for some time, or when a grasp condition is reached.
        A grasp condition is a certain amount of force being exerted by the fingers while being at a certain altitude off the floor.
        
        A seperate process must be connected to the telemetry stream to manage the act policy at this time. It can be started with

        python -m nf_robot.ml.stringman_lerobot eval   --robot_id=lan   --server_address=ws://localhost:4245   --policy_id=outputs/train/grasp_remote_act_eggs_2/checkpoints/last/pretrained_model/   --dataset_id=naavox/grasping_dataset_eggs_fix
        """
        self.pe.finger_pressure_rising.clear()
        try:
            self.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.EVAL_START))
            timeout = time.time() + 60
            lifted = False
            applying_force = False
            while not (lifted and applying_force) and time.time() < timeout:
                await asyncio.sleep(0.2)
                applying_force = self.pe.finger_pressure_rising.is_set()
                gripper_height = self.pe.grip_pose[1][2]
                lifted = gripper_height > 0.2
            logger.info(f'Ended grasp lifted={lifted} applying_force={applying_force} time_rem={timeout - time.time():.1f}s')
            # return value indicates whether grasp was successful
            return lifted and applying_force
        except asyncio.CancelledError:
            raise
        finally:
            self.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.EVAL_STOP))
            self.slow_stop_all_spools()

    def _handle_collect_images(self):
        if self.run_collect_images:
            self.run_collect_images = False # ends the task
        else:
            self.run_collect_images = True
            self.gip_task = asyncio.create_task(self.collect_images())

    async def collect_images(self):
        """Collects data for the centering network"""
        while self.run_command_loop and self.run_collect_images:
            if self.gripper_client.last_frame_resized is not None:
                logger.debug(f'Gripper frame shape: {self.gripper_client.last_frame_resized.shape}')
                rgb_image = cv2.cvtColor(self.gripper_client.last_frame_resized, cv2.COLOR_BGR2RGB)
                capture_gripper_image(rgb_image, gripper_occupied=self.pe.holding)
            else:
                logger.debug('No resized frame available from gripper')
            await asyncio.sleep(3)

def main():
    """
    Run stringman in a headless manner

    note that connecting to a local telemetry enviroment is distinct from lan mode
    To run in LAN mode, do not pass --telemetry_env
    observer.py will listen on port 4245
    
    Whenever --telemetry_env is set, observer.py is connecting to some telemetry server
    even if it is the full stack running on the local machine
    """
    parser = argparse.ArgumentParser(description="Stringman motion controller")
    parser.add_argument("--config", type=str, default='configuration.json')
    parser.add_argument(
            '--telemetry_env',
            type=str,
            choices=['local', 'staging', 'production'],
            default=None,
            help="The cloud telemetry server to connect to (choices: local, staging, production) Used in development only. The default is None, which allows local connections on port 4245 only"
        )
    parser.add_argument("--no_ai", action="store_true", help="Disable target finding and centering model evaluation")
    parser.add_argument("--no_ortho", action="store_true", help="Disable orthographic floor projection and its video streams")
    parser.add_argument("--auto_start", action="store_true", help="Automatically unpark and start cleaning when all components connect")
    parser.add_argument("--local_models", action="store_true", help="Use local models from models/ rather than downloading the production models from huggingface")
    parser.add_argument("--arp_grasp", action="store_true", help="Use arp_execute_grasp (centering net) instead of act_execute_grasp (ACT policy) for the Arpeggio gripper")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")
    parser.add_argument("--observability-debug", action="store_true", help="Record bounded telemetry payload summaries at DEBUG level in observability logs")
    parser.add_argument("--no_observability", action="store_true", help="Disable local Prometheus metrics, OTel traces, and JSON observability logs")
    parser.add_argument("--metrics-host", default=os.environ.get("NF_PROMETHEUS_HOST", "0.0.0.0"), help="Prometheus metrics bind host")
    parser.add_argument("--metrics-port", type=int, default=int(os.environ.get("NF_PROMETHEUS_PORT", "9464")), help="Prometheus metrics port")
    parser.add_argument("--observability-log", default=os.environ.get("NF_OBSERVABILITY_LOG", "logs/nf_robot-observability.jsonl"), help="JSON log path scraped by Promtail")
    args = parser.parse_args()

    if args.no_observability:
        os.environ["NF_OBSERVABILITY_ENABLED"] = "0"
    if args.observability_debug:
        os.environ["NF_OBSERVABILITY_DEBUG"] = "1"

    if args.debug:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')
        logging.getLogger('nf_robot').setLevel(logging.DEBUG)

    async def run_async():
        runner = AsyncObserver(
            False,
            args.config,
            telemetry_env=args.telemetry_env,
            run_ai=(not args.no_ai),
            run_ortho=(not args.no_ortho),
            auto_start=args.auto_start,
            local_models=args.local_models,
            use_arp_grasp=args.arp_grasp,
            debug=args.debug,
            observability_debug=args.observability_debug,
            observability_metrics_host=args.metrics_host,
            observability_metrics_port=args.metrics_port,
            observability_log_path=args.observability_log,
        )

        # Idempotent stop trigger
        def stop():
            runner.run_command_loop = False
            time.sleep(0.5)
            if runner.cloud_telem_websocket is not None:
                runner.cloud_telem_websocket.transport.abort()

        # On Unix, register signal handler.
        # On Windows, catch keyboard interrupt
        if sys.platform != "win32":
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, stop)
        
        try:
            r = await runner.main()
        except KeyboardInterrupt:
            stop()

    asyncio.run(run_async())

if __name__ == "__main__":
    main()
