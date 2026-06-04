# DGX Spark robotics model deep dive

Date: 2026-06-03
Repo: `/home/sarah/cranebot3-firmware`
Remote: `https://github.com/adb-123/cranebot3-firmware`

## Executive recommendation

Use the host observer websocket as the only model-control boundary. Do not let
models talk to Pi component JSON sockets, spool jog APIs, firmware update,
debug commands, torque toggles, or calibration commands.

The practical model path is:

1. Fix CUDA PyTorch on DGX Spark or use an NVIDIA ARM64 CUDA container. The
   current repo venv imports Torch but it is CPU-only.
2. Validate the in-repo perception models first:
   `TargetHeatmapNet` for floor target discovery and `CenteringNet` for gripper
   centering/grasp geometry.
3. Start policy learning with LeRobot ACT because this repo already maps its
   5-D action vector into `CombinedMove`.
4. Compare Diffusion Policy, VQ-BeT, and Multi-task DiT after ACT has a working
   dataset baseline.
5. Try SmolVLA, pi0, pi0-FAST, pi0.5, X-VLA, OpenVLA, and GR00T only after the
   LeRobot dataset features, camera keys, action scaling, and watchdog are
   proven. These are adaptation projects, not drop-in zero-shot controllers for
   this cable-driven robot.
6. Use Qwen-VL/Qwen3-Coder-class models for scene understanding, tool calling,
   and high-level planning only. They should output plans or target queue
   updates, not raw velocities.

## Team split

Robotics controls SME:

- Owns the observer websocket contract, action limits, workspace limits,
  stop-on-disconnect behavior, and hardware rollout procedure.
- Relevant files:
  - `src/nf_robot/host/observer.py`
  - `src/nf_robot/protos/control.proto`
  - `src/nf_robot/protos/telemetry.proto`
  - `src/nf_robot/host/position_estimator.py`

ML policy SME:

- Owns LeRobot datasets, ACT/Diffusion/VQ-BeT/DiT/VLA experiments, feature
  ordering, policy output validation, and evaluation scripts.
- Relevant files:
  - `src/nf_robot/ml/stringman_lerobot.py`
  - `src/nf_robot/ml/useful_commands.md`
  - `experiments/fix_dataset.py`
  - `experiments/lerobot_train_modal.py`

Perception SME:

- Owns anchor heatmaps, gripper centering, open-vocabulary detection,
  segmentation, depth, VLM scene summaries, and dataset labeling.
- Relevant files:
  - `src/nf_robot/ml/target_heatmap.py`
  - `src/nf_robot/ml/centering.py`
  - `src/nf_robot/host/video_streamer.py`
  - `docs/video_data_flow.md`

DGX runtime SME:

- Owns ARM64/CUDA/PyTorch compatibility, CUDA container selection, benchmark
  scripts, model cache layout, video encode acceleration, and resource limits.

QA and safety SME:

- Owns offline tests, simulator bringup, no-motion policy dry runs, telemetry
  acceptance criteria, stale-frame detection, and hardware go/no-go gates.

## Repo control architecture

The host controller is `AsyncObserver` in `src/nf_robot/host/observer.py`. It
discovers components, connects to anchors/gripper, starts `Positioner2`, starts
perception, and listens on a local websocket, normally `127.0.0.1:4245`.

External model/UI control uses binary protobuf:

- Inbound: `ControlBatchUpdate` containing `ControlItem`.
- Safest model command: `CombinedMove`.
- Outbound: `TelemetryBatchUpdate` containing `TelemetryItem`.

The direct learned-policy adapter already exists:

- `StringmanLeRobot.get_observation()` returns:
  - scalar state: velocity, pose, gripper sensors, target bearings/distances,
    tensions, gantry/visual/hang positions;
  - images: `gripper_camera` and `overhead_camera`, both `384x384x3` in the
    LeRobot adapter.
- `StringmanLeRobot.action_features` is exactly:
  - `vel_x`
  - `vel_y`
  - `vel_z`
  - `wrist_speed`
  - `finger_speed`
- `StringmanLeRobot.send_action()` sends a `CombinedMove` with
  `direction_is_in_gripper_frame=True`.

The physical component layer is separate JSON over LAN websockets. A model
should not use it directly.

## Current DGX Spark runtime facts

Observed locally:

- OS/arch: Linux aarch64.
- CPU: 20 ARM cores.
- GPU: NVIDIA GB10, driver `580.159.03`, CUDA runtime shown by `nvidia-smi` as
  `13.0`.
- `nvcc`: CUDA compilation tools `13.0`, V13.0.88.
- Repo venv Python: 3.12.3.
- Repo venv packages:
  - `nf_robot 3.18.1`
  - `lerobot 0.5.1`
  - `torch 2.10.0`
  - `torchvision 0.25.0`
  - `opencv-contrib-python-headless 4.13.0.92`
  - `av 15.1.0`
- Important blocker: `torch.cuda.is_available()` is `False` and
  `torch.version.cuda` is `None` in the repo venv. The PyPI/uv install selected
  a CPU-only Torch wheel on ARM64.

CPU-only smoke timings in the current venv:

- `TargetHeatmapNet`: about `0.40s/frame` for one `960x544` frame.
- `CenteringNet`: about `0.07s/frame` for one `384x384` frame.
- Peak RSS during smoke: about `1.4GB`.

This is enough for one-camera offline experimentation. It is not enough to
declare DGX GPU inference working.

## Model shortlist

### Already aligned with this repo

| Model | Type | Local fit | First experiment | Notes |
|---|---|---|---|---|
| `TargetHeatmapNet` / `naavox/targeting` | Anchor-camera floor target heatmap | Best first perception model | Load `models/target_heatmap.pth`, run image/stream eval, compare against `target_heatmap_data` | Input is `960x544`; observer projects heatmaps to floor targets and updates `TargetQueue`. |
| `CenteringNet` / `naavox/centering` | Gripper centering and grasp geometry | Best first gripper model | Load `models/square_centering.pth`, run eval on `square_centering_data`, verify vector/probability/angle ranges | Input is `384x384`; used by `arp_execute_grasp()` when `--arp_grasp` is enabled. |
| LeRobot ACT / `naavox/grasp_remote_act` | Imitation policy | Best first control policy | Record small dataset, train/eval ACT, verify 5-D action ordering | LeRobot docs call ACT the first recommended starting point because it is lightweight and data efficient. |
| LeRobot Diffusion Policy / `naavox/grasp_remote_diffusion_policy` | Imitation policy | Second control baseline | Train on same dataset as ACT; compare smoothness, latency, success | Likely heavier than ACT; still in installed LeRobot policies. |
| LeRobot VQ-BeT | Imitation policy | Candidate after ACT/diffusion | Use same 5-D action vector and compare discrete/action-token behavior | Installed in current LeRobot package; not a first hardware candidate. |
| LeRobot Multi-task DiT / `naavox/multitask-dit-*` | Flow/DiT policy | Good for richer tasks after data cleanup | Reproduce maintainer command in `useful_commands.md` with local CUDA | More knobs and likely more compute; use after ACT baseline. |

### Foundation policies worth adapting

| Model | Type | Local fit | First experiment | Notes |
|---|---|---|---|---|
| `lerobot/smolvla_base` | 0.5B VLA | Strong candidate after ACT | Fine-tune on Stringman LeRobot dataset; test inference offline | Inputs are multi-view images, state, optional language; outputs continuous actions. |
| `lerobot/pi0_base` | VLA flow policy | Research candidate | Try train-expert-only / frozen vision path on Stringman dataset | LeRobot describes pi0 as visual/language general robot control with continuous actions. |
| `lerobot/pi0_fast_base` plus `physical-intelligence/fast` | Autoregressive VLA with FAST action tokenizer | Good action-tokenization experiment | Train or reuse FAST tokenizer on 5-D action chunks, then fine-tune pi0-FAST | FAST maps action chunks to discrete tokens; local cache already contains `physical-intelligence/fast` metadata. |
| `lerobot/pi05_base` | Upgraded pi0 family | Later-stage VLA | Use only after pi0/pi0-FAST pipeline works | Repo notes already include a `pi05` command with expert-only training. |
| `lerobot/xvla-base` | Soft-prompted VLA | Useful if embodiment mismatch is the core problem | Learn Stringman soft prompts/adapters after dataset audit | X-VLA is designed around cross-embodiment heterogeneity; still not a drop-in controller. |
| `openvla/openvla-7b` | Open VLA | Useful research baseline | Fine-tune with LoRA/OFT-style adaptation; map 7-DoF deltas to Stringman 5-D action carefully | OpenVLA outputs normalized 7-DoF end-effector deltas, so an adapter is required. |
| `nvidia/GR00T-N1.5-3B` | NVIDIA VLA / humanoid-oriented foundation policy | DGX-aligned but setup-heavy | Use NVIDIA/LeRobot GR00T container path, then post-train on Stringman data | Requires CUDA and Flash Attention in LeRobot docs; license is more restrictive than Apache models. |

### Perception and planning helpers

| Model | Type | Local fit | First experiment | Notes |
|---|---|---|---|---|
| YOLO11n / YOLO11s / YOLO11-seg | Real-time detection/segmentation | Good for custom object detector baseline | Fine-tune on target/household-item frames, export ONNX/TensorRT later | Use as a complement to `TargetHeatmapNet`, not a replacement at first. |
| SAM 2.1 tiny/small/base | Promptable segmentation and video object masks | Good for dataset labeling and tracking | Use point/box prompts from heatmap peaks or VLM boxes to generate masks | Meta's repo supports image/video prediction and SAM 2.1 checkpoints. |
| Depth Anything V2 Small/Base | Monocular relative depth | Useful diagnostic/planning feature | Run on anchor/gripper frames and compare depth ordering with laser/floor geometry | Do not treat monocular relative depth as a metric safety sensor. |
| Qwen2.5-VL-7B-Instruct or Qwen3-VL-8B-Instruct | Local VLM scene parser | Good for semantic target descriptions and UI/planner summaries | Ask for JSON boxes/points on snapshots, then validate against heatmap/target queue | Qwen2.5-VL explicitly supports boxes/points and structured output. |
| Local Qwen3-Coder 30B A3B GGUF / Qwen3-Coder Next GGUF | Tool-calling planner/code assistant | Good for high-level planning and operator assistant | Run behind an OpenAI-compatible local server and restrict tools to read-only status plus target-queue proposals | Already present under `/home/sarah/models`; do not let it emit raw motor commands. |

## Recommended experiment phases

### Phase 0: Fix and prove DGX CUDA runtime

Create a separate GPU experiment env or container. Do not mutate the working
repo `.venv` until the CUDA path is known-good.

Acceptance:

```bash
nvidia-smi
nvcc --version
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Pass condition: `torch.cuda.is_available()` is `True` on the GB10.

### Phase 1: Offline repo checks

Use ROS plugin isolation on this host:

```bash
cd /home/sarah/cranebot3-firmware
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -rs
.venv/bin/python -m py_compile src/nf_robot/host/observer.py
```

Known result from this run: `74 passed, 7 skipped`.

### Phase 2: Perception-only baselines

Target heatmap:

```bash
cd /home/sarah/cranebot3-firmware
.venv/bin/python -m nf_robot.ml.target_heatmap eval \
  --model_path models/target_heatmap.pth
```

Centering:

```bash
.venv/bin/python -m nf_robot.ml.centering eval \
  --model_path models/square_centering.pth
```

If the weights are not local, let observer download from:

- `naavox/targeting` -> `target_heatmap.pth`
- `naavox/centering` -> `square_centering.pth`

For robot bringup without model load:

```bash
stringman-headless --config=<config> --no_ai --no_ortho --debug
```

For perception-only validation:

```bash
stringman-headless --config=<config> --no_ai --debug
```

Only enable `--local_models` after the local `models/*.pth` files exist.

### Phase 3: Dataset audit and small ACT baseline

Audit before training:

- dataset FPS: reconcile `FPS = 30` in `stringman_lerobot.py` with repair code
  that assumes `60.0` FPS in `experiments/fix_dataset.py`;
- camera keys and shapes;
- action ordering exactly `[vel_x, vel_y, vel_z, wrist_speed, finger_speed]`;
- action scale and clipping;
- dataset stats used by LeRobot preprocess/postprocess.

Train baseline:

```bash
lerobot-train \
  --dataset.repo_id=<user/stringman_dataset> \
  --policy.type=act \
  --output_dir=outputs/train/act_stringman_0 \
  --job_name=act_stringman_0 \
  --policy.device=cuda \
  --wandb.enable=false \
  --steps=100000 \
  --batch_size=8
```

Offline eval should run through `StringmanLeRobot` only after action validation
is added.

### Phase 4: Compare policy families

Run each policy on the same train/eval split:

1. ACT: first baseline.
2. Diffusion Policy: compare action smoothness and latency.
3. VQ-BeT: compare discrete/action-token behavior.
4. Multi-task DiT: compare larger policy behavior on richer tasks.
5. SmolVLA: first language-conditioned model worth trying.
6. pi0/pi0-FAST/pi0.5/X-VLA/GR00T/OpenVLA: later adaptation work.

### Phase 5: Simulator/no-motion integration

The safest intended path is simulator plus observer plus LeRobot eval. Current
simulator smoke failed in the existing `.venv` because `getmac` is missing.
Use `uv pip install` for missing simulator dependencies; the venv currently has
no `pip` module.

Example target flow:

```bash
cd /home/sarah/cranebot3-firmware
uv pip install --python .venv/bin/python getmac
PYTHONPATH=src .venv/bin/python experiments/robot_simulator.py
PYTHONPATH=src .venv/bin/python -m nf_robot.host.observer \
  --config /tmp/stringman-sim.json --no_ai --no_ortho --debug
PYTHONPATH=src .venv/bin/python -m nf_robot.ml.stringman_lerobot eval \
  --robot_id=lan \
  --server_address=ws://localhost:4245 \
  --policy_id=<policy>
```

### Phase 6: Guarded live run

Before real motion:

- hardware type confirmed: Pilot vs Arpeggio;
- model process has an action gate;
- model process sends zero action on exception, stale telemetry, stale image,
  high tension, low range, NaN/Inf, out-of-bounds plan, or operator stop;
- observer has an effective workspace polygon;
- command rate and stop latency are measured;
- max velocity is lower than repo defaults for initial trials;
- all logs capture observation age, action vector, clamped action vector,
  tension, gripper range, finger pressure, and policy latency.

## Required fixes before hardware policy eval

1. Add client-side action validation in `StringmanLeRobot.send_action()` or a
   wrapper:
   - reject NaN/Inf;
   - clamp velocity and wrist/finger speeds;
   - validate action vector length and feature names;
   - send zero action on exception/disconnect.
2. Add a local control heartbeat/watchdog. A stale `CombinedMove` should not
   leave line speeds active if a model process dies.
3. Implement or externally enforce real workspace limits. The current
   `Positioner2.point_inside_work_area*()` guards are effectively stubs.
4. Fix `AsyncObserver.lerobot_process()`:
   - support local checkpoint paths safely, not only `namespace/name` repo IDs;
   - fix malformed `suppress_upload` interpolation.
5. Fix `TargetHeatmapNet` training loss mismatch. The model forward applies
   sigmoid, but training uses `BCEWithLogitsLoss`; use raw logits or `BCELoss`.
6. Update docs/test command for this host:
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -rs`.
7. Keep `JogSpool`, `Debug`, firmware update, calibration, torque toggles, and
   component actions out of any planner tool surface.

## Source links

- Repo README and local code: `/home/sarah/cranebot3-firmware`
- LeRobot docs: https://huggingface.co/docs/lerobot/en/index
- LeRobot ACT: https://huggingface.co/docs/lerobot/en/act
- LeRobot SmolVLA model card: https://huggingface.co/lerobot/smolvla_base
- LeRobot pi0: https://huggingface.co/docs/lerobot/en/pi0
- LeRobot pi0-FAST: https://huggingface.co/docs/lerobot/en/pi0fast
- Physical Intelligence FAST tokenizer: https://huggingface.co/physical-intelligence/fast
- Physical Intelligence OpenPI: https://github.com/Physical-Intelligence/openpi
- LeRobot GR00T N1.5: https://huggingface.co/docs/lerobot/groot
- NVIDIA GR00T N1.5 research: https://research.nvidia.com/labs/gear/gr00t-n1_5/
- OpenVLA model card: https://huggingface.co/openvla/openvla-7b
- OpenVLA repo: https://github.com/openvla/openvla
- Qwen2.5-VL model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Qwen3-VL model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen3-Coder model card: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- SAM 2 repo: https://github.com/facebookresearch/sam2
- Depth Anything V2 repo: https://github.com/DepthAnything/Depth-Anything-V2
- YOLO11 docs: https://docs.ultralytics.com/models/yolo11/
