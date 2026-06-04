# Autocalibrate Fix TODO

Status: Watch pass completed; no fixes executed.

Last updated: 2026-06-03 15:24:45 PDT

Scope:
- Observe host `stringman-headless` logs, gripper `/opt/robot/cranebot.log`, controller telemetry, and local MJPEG output.
- Record only evidence-backed fixes.
- Do not execute fixes, edit code, restart services, or interrupt the running app.

## Evidence-Backed Fixes

### 1. Make `stringman-headless` logs durable and tailable

Symptom:
- During this live autocalibration watch, the running host app could not be tailed from a normal shell after the original exec session handle was unavailable.

Log evidence timestamp/message:
- 2026-06-03 15:18 PDT: `stringman-headless` was still running and listening on `127.0.0.1:4245` plus MJPEG ports `4246`, `4247`, `4248`, `8747`, and `8748`.
- 2026-06-03 15:18 PDT: `/home/sarah/cranebot3-firmware/logs/stringman-headless.log size=0 mtime=2026-06-03 14:18:35.283530435 -0700`.
- 2026-06-03 15:12 PDT: running app stdout/stderr were attached to `/dev/pts/5`; the prior exec session IDs were unavailable from this turn.
- Code evidence: [src/nf_robot/host/observer.py](/home/sarah/cranebot3-firmware/src/nf_robot/host/observer.py:3025) only configures logging under `--debug`, and that setup does not add a file handler.

Likely root cause:
- The host controller depends on the launching terminal/PTY for useful runtime logs. If that handle is lost, there is no durable app log even though the process is still running.

Proposed fix:
- Add durable host logging for `stringman-headless`, preferably an explicit `--log-file` option defaulting to `logs/stringman-headless.log`.
- Configure logging at startup for normal runs, not only `--debug`, with a file handler and a console handler.
- Include timestamps in host log lines and preserve DEBUG behavior by raising the `nf_robot` logger level when `--debug` is set.

Files likely involved:
- [src/nf_robot/host/observer.py](/home/sarah/cranebot3-firmware/src/nf_robot/host/observer.py:2997)
- [pyproject.toml](/home/sarah/cranebot3-firmware/pyproject.toml:64) only if the console script or defaults need packaging changes.

Verification command/procedure:
- Start `stringman-headless --config=bedroom.conf --no_ai --no_ortho`.
- Confirm `logs/stringman-headless.log` becomes nonzero and receives startup/component/video/calibration messages.
- Run autocalibrate and tail `logs/stringman-headless.log` while also checking `ws://127.0.0.1:4245` and `http://localhost:4246/stream.mjpeg`.

### 2. Make component self-update restart deterministic

Symptom:
- During the live session, the gripper Pi reported that self-update completed and it was restarting, but the service did not actually restart.

Log evidence timestamp/message:
- 2026-06-03 23:21:51 Pi log / 2026-06-03 15:21:51 PDT: `Performing Update`.
- 2026-06-03 23:22:03 Pi log / 2026-06-03 15:22:03 PDT: pip notices were logged as `ERROR`, including `[notice] A new release of pip is available: 26.0.1 -> 26.1.2`.
- 2026-06-03 23:22:05 Pi log / 2026-06-03 15:22:05 PDT: `Self update complete. Restarting.`
- 2026-06-03 15:23 PDT read-only Pi state: `ExecMainPID=706`, `NRestarts=0`, `ActiveState=active`, `SubState=running`; process start time remained `Wed Jun 3 23:07:06 2026`.
- Code evidence: [src/nf_robot/robot/anchor_server.py](/home/sarah/cranebot3-firmware/src/nf_robot/robot/anchor_server.py:204) sends the update result, sleeps `0.2`, then calls `self.shutdown()` and `quit()` when the pip return code is zero.

Likely root cause:
- `quit()` inside the async websocket/update task is not a reliable service restart mechanism. The code logs "Restarting" before proving that the process exited, and systemd never observed a restart in this run.
- The update subprocess stderr is logged wholesale as `ERROR`, so pip notices look like operational failures even when the install may have succeeded.

Proposed fix:
- Replace the task-local `quit()` path with a deterministic shutdown path that terminates the component process after the update result is sent, so systemd can restart it.
- Consider a shutdown event consumed by the main server loop, or a deliberate process exit after cleanup. Verify the systemd unit has the intended restart policy.
- Classify update subprocess stderr more carefully: log pip notices as info/warning unless the install return code is nonzero, and log the final return code explicitly.

Files likely involved:
- [src/nf_robot/robot/anchor_server.py](/home/sarah/cranebot3-firmware/src/nf_robot/robot/anchor_server.py:204)
- [src/nf_robot/host/anchor_client.py](/home/sarah/cranebot3-firmware/src/nf_robot/host/anchor_client.py:341)
- Pi `cranebot.service` unit, for restart policy verification only.

Verification command/procedure:
- In a controlled window, trigger firmware update once.
- Confirm Pi log records update result, process exit, new startup lines, and `rpicam-vid appears to be ready`.
- Confirm `systemctl show cranebot.service -p ExecMainPID -p NRestarts -p ActiveState -p SubState` shows a changed `ExecMainPID` and incremented restart count.
- Confirm host telemetry briefly marks the component disconnected/reconnecting, then returns gripper websocket/video to `CONNECTED`; confirm `http://localhost:4246/stream.mjpeg` decodes after reconnect.

## Live Watch Notes

- 2026-06-03 15:10:34 PDT: TODO created before the autocalibration attempt. Watching for failures, warnings, disconnects, timeouts, camera/video restarts, calibration exceptions, event-loop stalls, and MJPEG decode failures.
- 2026-06-03 15:11:32 PDT / 2026-06-03 23:11:32 Pi log: Gripper autocalibration stage reached `Calibrating finger servo...`; subsequent Pi log lines reported `reset midpoint position is now 2051`, `Motor encoder position at finger touch = 2042`, and `Motor encoder position at finger touch = 240`.
- 2026-06-03 15:12 PDT: Host telemetry reported gripper `192.168.5.39` with websocket `CONNECTED`, video `CONNECTED`, gripper stream `http://localhost:4246/stream.mjpeg`, and feed `0`.
- 2026-06-03 15:12 PDT: MJPEG probe decoded a fresh gripper frame from `http://localhost:4246/stream.mjpeg` at `384x384`.
- 2026-06-03 15:12 PDT: Prior host exec session handles were unavailable in this turn. The running host process stdout/stderr are attached to `/dev/pts/5`; direct host console log capture is not available without interfering, so host-side monitoring is using websocket telemetry and stream probes.
- 2026-06-03 15:13:18-15:18:22 PDT: Five-minute combined watcher completed. Gripper MJPEG decoded successfully on every probe; frame sizes varied from 19,180 to 48,746 bytes at `384x384`.
- 2026-06-03 15:13:19 PDT: Telemetry reported gripper `ws=CONNECTED`, `video=CONNECTED`, `ip=192.168.5.39`, and gripper `video_ready` local URI `http://localhost:4246/stream.mjpeg`.
- 2026-06-03 15:18 PDT: Telemetry remained live over a 12-second snapshot: `638` `pos_estimate`, `638` `pos_factors_debug`, `218` `grip_sensors`, `234` `gantry_sightings`, and `24` `vid_stats` updates. Last gripper status remained websocket `CONNECTED`, video `CONNECTED`.
- 2026-06-03 23:12:35 Pi log / 2026-06-03 15:12:35 PDT: Last new gripper calibration line observed was `Resest wrist. should be 540. (539.033203125)`. No Pi-side disconnect, timeout, camera restart, or calibration exception was observed during the watch window.
- 2026-06-03 15:19:59-15:22:34 PDT: Second watch pass saw telemetry heartbeats with gripper websocket/video still `CONNECTED` and repeated successful `384x384` MJPEG decodes.
- 2026-06-03 23:21:51-23:22:05 Pi log / 2026-06-03 15:21:51-15:22:05 PDT: Gripper Pi ran `Performing Update`, logged pip notices on stderr, then logged `Self update complete. Restarting.`
- 2026-06-03 15:23 PDT: Post-update snapshot still showed gripper telemetry/video `CONNECTED`; three additional MJPEG probes decoded `384x384` frames.
- 2026-06-03 15:23 PDT: Pi service state showed `ExecMainPID=706`, `NRestarts=0`, and process start time `Wed Jun 3 23:07:06 2026`, so the "Restarting" log did not correspond to an actual service restart.

## 2026-06-03 20:46-21:12 PDT Live RCA Notes

Scope:
- Read-only monitor while host/Pi gripper fixes were being patched and restarted by others.
- Host app session log: `logs/stringman-headless.log`.
- Pi gripper evidence: `journalctl -u cranebot.service` plus `/opt/robot/cranebot.log`.

Timeline:
- 2026-06-04 04:28:12 BST: Pi `cranebot.service` stopped and started; systemd also killed old `rpicam-vid`.
- 2026-06-04 04:39:03 BST: Pi log recorded `Killing rpicam-vid subprocess the task is being cancelled`, `Client disconnected`, and handler completion during a host app restart.
- 2026-06-04 04:39:28-04:39:31 BST: Pi log recorded new websocket connection, measurement streaming, `Restarting rpi-cam_vid`, and `rpicam-vid appears to be ready`.
- 2026-06-03 20:46:43 PDT: Host PID `3941631` owned exactly one socket to `192.168.5.39:8765` and one socket to `192.168.5.39:8888`; Pi process snapshot matched one `8765` and one `8888` connection.
- 2026-06-03 20:46-20:52 PDT: No duplicate `.39:8888` sockets observed in periodic `ss` snapshots; host PID remained `3941631`.
- 2026-06-03 20:46 PDT: Host log contained `EVENT LOOP BLOCKED`, repeated `Tension limit reached! backing off.`, `ERROR:websockets.server:connection handler failed`, and `Swing cancellation energy increased from 0.146572 to 0.219584; disabling`.
- 2026-06-03 20:48-20:49 PDT: New host log lines showed another `Tension limit reached! backing off.` plus swing cancellation energy increases `0.264260 to 0.518685` and `0.538903 to 0.780212`, each disabling cancellation.
- 2026-06-03 20:53 PDT / 2026-06-04 04:53 BST: During a patch/restart window, host log recorded `Connection to 192.168.5.39 closed. no close frame received or sent`, then `logs/stringman-headless.log` was truncated and reloaded config. Snapshot at 20:53:50 PDT saw no host `stringman-headl` process and zero `.39:8765/.39:8888` sockets.
- 2026-06-04 04:53:42 BST: Pi `cranebot.service` stopped and started; systemd killed `IPAProxyRPi`.
- 2026-06-04 04:53:48-04:54:09 BST: Pi log recorded server startup, websocket reconnect, `Restarting rpi-cam_vid`, and `rpicam-vid appears to be ready`.
- 2026-06-03 20:54-20:55 PDT: New host PID `3984341` re-established exactly one `.39:8765` socket and one `.39:8888` socket. No duplicate video socket observed after the reconnect. A new host `EVENT LOOP BLOCKED` warning appeared shortly after restart.
- 2026-06-03 20:58 PDT / 2026-06-04 04:58 BST: Another patch/restart cycle truncated the host log and reloaded config. Pi log recorded `Killing rpicam-vid subprocess the task is being cancelled` and `Client disconnected` at 04:58:31 BST, then websocket reconnect at 04:58:39 BST and `rpicam-vid appears to be ready` at 04:58:42 BST.
- 2026-06-03 20:58:50 PDT: New host PID `3995326` owned exactly one `.39:8765` socket and one `.39:8888` socket. No duplicate video socket observed after the second reconnect.
- 2026-06-03 21:02 PDT: Host log recorded multiple `Swing calibration abort` candidates with energy slightly above `limit=0.012000`, followed by `Swing calibration validation failed`. The selected best candidate was `aborted=False`, but validation reported `amplified=True` and `damped=False`.
- 2026-06-03 21:05 PDT / 2026-06-04 05:05 BST: Another patch/restart cycle truncated the host log and reloaded config. Pi log recorded `Killing rpicam-vid subprocess the task is being cancelled` and `Client disconnected` at 05:05:27 BST, then websocket reconnect at 05:05:35 BST and `rpicam-vid appears to be ready` at 05:05:38 BST.
- 2026-06-03 21:05:51 PDT: New host PID `4015305` owned exactly one `.39:8765` socket and one `.39:8888` socket. No duplicate video socket observed after the third reconnect.
- 2026-06-03 21:06 PDT: Host log recorded several `Swing calibration abort` candidates followed by `Swing calibration failed: Swing IMU model is stale (0.76s old)`.
- 2026-06-03 21:07 PDT / 2026-06-04 05:07 BST: Pi log recorded `Killing rpicam-vid subprocess the task is being cancelled`, `Client disconnected`, websocket reconnect, `Restarting rpi-cam_vid`, and `rpicam-vid appears to be ready`.
- 2026-06-03 21:07:51 PDT: New host PID `4047329` owned exactly one `.39:8765` socket and one `.39:8888` socket. No duplicate video socket observed after the fourth reconnect.
- 2026-06-03 21:08 PDT: Host log recorded another `EVENT LOOP BLOCKED` line and a swing calibration abort cluster at latency 0.080/0.180 with energies above `limit=0.016446`. Subsequent socket snapshots through 21:11:52 PDT kept the same host PID `4047329`, so this was not confirmed as another reconnect.

Suggested fixes to investigate:
- Keep the video socket lifecycle patch: it appears to prevent duplicate `.39:8888` sockets across the observed reconnect window.
- Add timestamps to host log output; the current host session log still emits untimestamped warning/error lines, which makes RCA correlation depend on external tail timestamps.
- Treat swing cancellation as still failing after the reconnect. Investigate why calibration attempts continue after `Tension limit reached` and why the cancellation candidate is accepted far enough to increase energy before being disabled.
- Tighten final swing calibration acceptance: a candidate that is locally best but later validates as `amplified=True` / `damped=False` should produce a clear failed-calibration state and avoid leaving operators with a false-positive "best" candidate.
- Investigate swing IMU freshness during calibration. The 21:06 failure shows calibration can get through candidate evaluation and then fail because the model is already stale by 0.76s.
- Capture websocket handler exception details in the host log, not only `ERROR:websockets.server:connection handler failed`, so the next restart/disconnect can be tied to a specific close/error path.
- Treat restart recovery as working only if both sockets return to exactly one each and the Pi log reaches `rpicam-vid appears to be ready`; the 20:53/04:53 restart satisfied that condition after a short disconnect window.
