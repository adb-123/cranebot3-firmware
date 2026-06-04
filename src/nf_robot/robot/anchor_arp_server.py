import asyncio
from getmac import get_mac_address
import json
import threading
import time
import logging
import argparse

from damiao_motor import DaMiaoController

import nf_robot.common.definitions as model_constants
from nf_robot.robot.anchor_server import RobotComponentServer
from nf_robot.robot.spool_dm import DamiaoSpoolController

""" Server for Arpeggio Anchor

A double anchor containing two damiao hub motors and a custom hat that provides a CAN bus interface.

"""

default_anchor_conf = {
    # speed to reel in when the 'tighten' command is received. Meters of line per second
    'TIGHTENING_SPEED': -0.12,
    'TIGHTEN_DESIRED_TENSION_N': 1.38,
    'TIGHTEN_MAX_RETRIES': 5,
    'TIGHTEN_ATTEMPT_TIMEOUT_S': 8.0,
    'TIGHTEN_MONITOR_DURATION_S': 3.0,
    'STOW_TIMEOUT_S': 8.0,
    'LINE_ACTION_CHECK_INTERVAL_S': 0.05,
    'LINE_ACTION_STALE_TIMEOUT_S': 1.0,
    'LINE_ACTION_MIN_LENGTH_DELTA_M': 0.002,
    'LINE_ACTION_MIN_TENSION_DELTA_N': 0.05,
    'RELAX_SPEED': 0.04,
    'RELAX_DURATION_S': 1.0,
    'RELAX_DISTANCE_M': 0.05,
}


class AnchorArpServer(RobotComponentServer):
    def __init__(self, power):
        super().__init__()
        self.conf.update(default_anchor_conf)

        self.has_power_line = power

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = 'cranebot-anchor-arpeggio-service.' + unique

        # https://jia-xie.github.io/python-damiao-driver/dev/package-usage/python-api/
        self.controller = DaMiaoController(channel="can0", bustype="socketcan")
        # h6220 is probaly the closest to DM-H6215 but they all seem the same to me.
        self.motor1 = self.controller.add_motor(motor_id=0x02, feedback_id=0x02, motor_type="G6215") # high motor
        self.motor2 = self.controller.add_motor(motor_id=0x01, feedback_id=0x01, motor_type="G6215") # lower motor
        self.motors = [self.motor1, self.motor2]

        # consider the direct line (high) spool 0 and the indirect line (low) spool 1

        # the power line, if present is always on the high spool
        fulld = model_constants.damiao_full_spool_diameter_power_line if self.has_power_line else model_constants.damiao_full_spool_diameter_fishing_line
        spooler1 = DamiaoSpoolController(
            self.motor1,
            empty_diameter=model_constants.damiao_empty_spool_diameter,
            full_diameter=fulld,
            full_length=model_constants.assumed_full_line_length,
            config=self.conf, direction=-1)

        # Create a spool controller for each spool
        spooler2 = DamiaoSpoolController(
            self.motor2,
            empty_diameter=model_constants.damiao_empty_spool_diameter,
            full_diameter=model_constants.damiao_full_spool_diameter_fishing_line,
            full_length=model_constants.assumed_full_line_length,
            config=self.conf, direction=1)

        # parent class would use this to send line updates. setting it to None supresses that. we send our own.
        self.spooler = None
        self.spools = [spooler1, spooler2]
        self.line_action_states = [
            {'spool': 0, 'action': 'idle', 'status': 'idle'},
            {'spool': 1, 'action': 'idle', 'status': 'idle'},
        ]

    async def processOtherUpdates(self, updates, tg):
        if 'tighten' in updates:
            spool_no = updates['tighten']
            tg.create_task(self.tighten(spool_no))
        if 'stow' in updates:
            spool_no = updates['stow']
            tg.create_task(self.stow(spool_no))
        if 'relax' in updates:
            spool_no = updates['relax']
            tg.create_task(self.relax(spool_no))
        if 'identify' in updates:
            self.identify()
        if 'two_reference_lengths' in updates:
            ref0, ref1 = updates['two_reference_lengths']
            self.spools[0].setReferenceLength(float(ref0))
            self.spools[1].setReferenceLength(float(ref1))
        if 'aim_speed' in updates:
            if updates['aim_speed'] == 0:
                self.spools[0].setAimSpeed(0)
                self.spools[1].setAimSpeed(0)
            else:
                try:
                    speed, spool_no = updates['aim_speed']
                    speed = float(speed)
                    parsed_spool_no = self._parse_spool_no(spool_no, 'aim_speed')
                    if parsed_spool_no is not None:
                        self.spools[parsed_spool_no].setAimSpeed(speed)
                except (TypeError, ValueError):
                    logging.warning(f'invalid aim_speed command. expected (speed, spool_no). got {updates["aim_speed"]}')
        if 'jog' in updates:
            try:
                delta, spool_no = updates['jog']
                parsed_spool_no = self._parse_spool_no(spool_no, 'jog')
                if parsed_spool_no is not None:
                    self.spools[parsed_spool_no].jog(float(delta))
            except (TypeError, ValueError):
                logging.warning(f'invalid jog command: {updates["jog"]}')
        if 'disable_torque' in updates:
            for spool in self.spools:
                spool.pauseTrackingLoop(disable_torque=True)
                self.update['torque'] = False
        if 'enable_torque' in updates:
            for spool in self.spools:
                spool.resumeTrackingLoop()
                self.update['torque'] = True
        if 'set_anti_tangle' in updates:
            try:
                val, spool_no = updates['set_anti_tangle']
                parsed_spool_no = self._parse_spool_no(spool_no, 'set_anti_tangle')
                if parsed_spool_no is not None:
                    self.spools[parsed_spool_no].setAntiTangle(bool(val))
            except (TypeError, ValueError):
                logging.warning(f'invalid set_anti_tangle command: {updates["set_anti_tangle"]}')

    def readOtherSensors(self):
        """ Sends updates about both spools with the form
        {
            'spool0' : [
                (time, line_length, line_speed, torque),
                ...
            ],
            'spool1': [...]
        }
        """
        for i, spool in enumerate(self.spools):
            meas = spool.popMeasurements()
            if len(meas) > 0:
                meas = meas[:50]
            self.update[f'spool{i}'] = meas
        self.update['line_action_states'] = self.line_action_states

    def startOtherTasks(self):
        return list([
            asyncio.create_task(asyncio.to_thread(spool.trackingLoop))
            for spool in self.spools
        ])

    async def tighten(self, spool_no):
        """
        Pulls in the line until tight. If the line slips within 3 seconds,
        it reduces the speed by 30% and retries, up to 5 times.
        """
        spool_no = self._parse_spool_no(spool_no, 'tighten')
        if spool_no is None:
            return False

        max_retries = int(self.conf['TIGHTEN_MAX_RETRIES'])
        monitoring_duration_s = float(self.conf['TIGHTEN_MONITOR_DURATION_S'])
        
        current_speed = self.conf['TIGHTENING_SPEED']

        try:
            for attempt in range(1, max_retries + 1):
                self._publish_line_action(
                    'tighten', spool_no, 'running',
                    phase='pulling',
                    attempt=attempt,
                    speed=current_speed)
                reached_tension, reason = await self._wait_for_tension(
                    spool_no,
                    'tighten',
                    current_speed,
                    float(self.conf['TIGHTEN_ATTEMPT_TIMEOUT_S']))
                self._stop_spool(spool_no)

                if not reached_tension:
                    self._publish_line_action(
                        'tighten', spool_no, 'failed',
                        reason=reason,
                        attempt=attempt)
                    return False

                self._publish_line_action(
                    'tighten', spool_no, 'running',
                    phase='monitoring',
                    attempt=attempt)
                if await self._monitor_tension_held(spool_no, monitoring_duration_s):
                    self._publish_line_action(
                        'tighten', spool_no, 'succeeded',
                        reason='tension_held',
                        attempt=attempt)
                    return True

                self._publish_line_action(
                    'tighten', spool_no, 'retrying',
                    reason='line_slipped',
                    attempt=attempt)
                current_speed *= 0.7

            self._publish_line_action(
                'tighten', spool_no, 'failed',
                reason='max_retries_exhausted',
                attempts=max_retries)
            return False
        except asyncio.CancelledError:
            self._publish_line_action('tighten', spool_no, 'cancelled')
            raise
        finally:
            self._stop_spool(spool_no)

    async def stow(self, spool_no):
        """ Pulls the line till tight, then disables the motor for storage """
        spool_no = self._parse_spool_no(spool_no, 'stow')
        if spool_no is None:
            return False

        try:
            self._publish_line_action(
                'stow', spool_no, 'running',
                phase='pulling',
                speed=self.conf['TIGHTENING_SPEED'])
            reached_tension, reason = await self._wait_for_tension(
                spool_no,
                'stow',
                self.conf['TIGHTENING_SPEED'],
                float(self.conf['STOW_TIMEOUT_S']))
            self._stop_spool(spool_no)
            if not reached_tension:
                self._publish_line_action(
                    'stow', spool_no, 'failed',
                    reason=reason)
                return False

            self.spools[spool_no].pauseTrackingLoop()
            self.motors[spool_no].disable()
            self._publish_line_action(
                'stow', spool_no, 'succeeded',
                reason='tension_reached_motor_disabled')
            return True
        except asyncio.CancelledError:
            self._publish_line_action('stow', spool_no, 'cancelled')
            raise
        finally:
            self._stop_spool(spool_no)

    async def relax(self, spool_no):
        """Lets out a bounded amount of line at low speed."""
        spool_no = self._parse_spool_no(spool_no, 'relax')
        if spool_no is None:
            return False

        speed = abs(float(self.conf['RELAX_SPEED']))
        duration_s = max(0.0, float(self.conf['RELAX_DURATION_S']))
        distance_m = max(0.0, float(self.conf['RELAX_DISTANCE_M']))
        if speed == 0 or (duration_s == 0 and distance_m == 0):
            self._publish_line_action(
                'relax', spool_no, 'failed',
                reason='invalid_relax_limits',
                speed=speed,
                duration_s=duration_s,
                distance_m=distance_m)
            return False

        spool = self.spools[spool_no]
        start_length = self._line_float(spool, 'last_length')
        deadline = time.monotonic() + duration_s if duration_s > 0 else None
        last_change_at = time.monotonic()
        last_length = start_length
        last_tension = self._line_float(spool, 'last_tension')
        check_interval_s = float(self.conf['LINE_ACTION_CHECK_INTERVAL_S'])
        stale_timeout_s = float(self.conf['LINE_ACTION_STALE_TIMEOUT_S'])

        try:
            self._publish_line_action(
                'relax', spool_no, 'running',
                phase='unwinding',
                speed=speed,
                duration_s=duration_s,
                distance_m=distance_m)
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    self._publish_line_action(
                        'relax', spool_no, 'succeeded',
                        reason='duration_elapsed')
                    return True

                current_length = self._line_float(spool, 'last_length')
                if (
                    start_length is not None
                    and current_length is not None
                    and distance_m > 0
                    and current_length - start_length >= distance_m
                ):
                    self._publish_line_action(
                        'relax', spool_no, 'succeeded',
                        reason='distance_reached',
                        length_delta=current_length - start_length)
                    return True

                spool.setAimSpeed(speed)
                await asyncio.sleep(check_interval_s)

                current_tension = self._line_float(spool, 'last_tension')
                if self._line_state_changed(last_length, current_length, last_tension, current_tension):
                    last_change_at = time.monotonic()
                    last_length = current_length
                    last_tension = current_tension
                elif self._line_state_available(last_length, last_tension) and time.monotonic() - last_change_at >= stale_timeout_s:
                    self._publish_line_action(
                        'relax', spool_no, 'failed',
                        reason='line_state_stale')
                    return False
        except asyncio.CancelledError:
            self._publish_line_action('relax', spool_no, 'cancelled')
            raise
        finally:
            self._stop_spool(spool_no)

    def _parse_spool_no(self, spool_no, action):
        try:
            if isinstance(spool_no, bool):
                raise ValueError
            parsed = int(spool_no)
            if isinstance(spool_no, float) and not spool_no.is_integer():
                raise ValueError
            if isinstance(spool_no, str) and str(parsed) != spool_no.strip():
                raise ValueError
        except (TypeError, ValueError):
            self._publish_line_action(action, spool_no, 'failed', reason='invalid_spool')
            return None
        if parsed not in (0, 1):
            self._publish_line_action(action, parsed, 'failed', reason='invalid_spool')
            return None
        return parsed

    def _publish_line_action(self, action, spool_no, status, reason=None, **fields):
        state = {
            'spool': spool_no,
            'action': action,
            'status': status,
            'ts': time.time(),
        }
        if reason is not None:
            state['reason'] = reason
        state.update(fields)
        if isinstance(spool_no, int) and 0 <= spool_no < len(self.line_action_states):
            self.line_action_states[spool_no] = state
        self.update['line_action'] = state

        message = f'ARP line action {action} spool={spool_no} status={status}'
        if reason is not None:
            message += f' reason={reason}'
        if status == 'failed':
            logging.error(message)
        elif status == 'cancelled':
            logging.warning(message)
        else:
            logging.info(message)

    def _stop_spool(self, spool_no):
        try:
            self.spools[spool_no].setAimSpeed(0)
        except (IndexError, TypeError, AttributeError) as exc:
            logging.warning(f'failed to stop ARP spool {spool_no}: {exc}')

    def _line_float(self, spool, attr):
        try:
            return float(getattr(spool, attr))
        except (AttributeError, TypeError, ValueError):
            return None

    def _line_state_available(self, length, tension):
        return length is not None or tension is not None

    def _line_state_changed(self, previous_length, current_length, previous_tension, current_tension):
        if previous_length is not None and current_length is not None:
            if abs(current_length - previous_length) >= float(self.conf['LINE_ACTION_MIN_LENGTH_DELTA_M']):
                return True
        if previous_tension is not None and current_tension is not None:
            if abs(current_tension - previous_tension) >= float(self.conf['LINE_ACTION_MIN_TENSION_DELTA_N']):
                return True
        return False

    def _line_is_slack(self, spool_no):
        tension = self._line_float(self.spools[spool_no], 'last_tension')
        if tension is None:
            return True
        return tension < float(self.conf['TIGHTEN_DESIRED_TENSION_N'])

    async def _wait_for_tension(self, spool_no, action, speed, timeout_s):
        spool = self.spools[spool_no]
        deadline = time.monotonic() + timeout_s
        last_change_at = time.monotonic()
        last_length = self._line_float(spool, 'last_length')
        last_tension = self._line_float(spool, 'last_tension')
        check_interval_s = float(self.conf['LINE_ACTION_CHECK_INTERVAL_S'])
        stale_timeout_s = float(self.conf['LINE_ACTION_STALE_TIMEOUT_S'])

        while self._line_is_slack(spool_no):
            if time.monotonic() >= deadline:
                return False, 'tension_timeout'

            spool.setAimSpeed(speed)
            await asyncio.sleep(check_interval_s)

            current_length = self._line_float(spool, 'last_length')
            current_tension = self._line_float(spool, 'last_tension')
            if self._line_state_changed(last_length, current_length, last_tension, current_tension):
                last_change_at = time.monotonic()
                last_length = current_length
                last_tension = current_tension
            elif self._line_state_available(last_length, last_tension) and time.monotonic() - last_change_at >= stale_timeout_s:
                return False, 'line_state_stale'

        return True, None

    async def _monitor_tension_held(self, spool_no, duration_s):
        end_time = time.monotonic() + duration_s
        check_interval_s = float(self.conf['LINE_ACTION_CHECK_INTERVAL_S'])
        while time.monotonic() < end_time:
            if self._line_is_slack(spool_no):
                return False
            await asyncio.sleep(check_interval_s)
        return True

    def identify(self, spool_no=1):
        """ make a noise """
        self.spools[spool_no].pauseTrackingLoop()
        m = self.motors[spool_no]

        m.send_cmd_vel(target_velocity=0.0)
        for i in range(20):
            time.sleep(0.005)
            m.send_cmd_vel(target_velocity=0.2 * (i%2-0.5))
        m.send_cmd_vel(target_velocity=0.0)
        
        self.spools[spool_no].resumeTrackingLoop()

    async def process_imu(self, ws):
        """Runs when a new client connects.
        TODO don't just piggyback off this, organize it"""
        for m in self.motors:
            m.enable()
        for s in self.spools:
            s.resumeTrackingLoop()

    def shutdown(self):
        """must be a synchronous call. triggered by signal handler"""
        super().shutdown()
        for spool in self.spools:
            spool.fastStop()
        time.sleep(0.1)
        self.controller.shutdown()
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--power", action="store_true",
                        help="Configures this anchor as the one which has the power line")
    args = parser.parse_args()

    ras = AnchorArpServer(args.power)
    asyncio.run(ras.main())
