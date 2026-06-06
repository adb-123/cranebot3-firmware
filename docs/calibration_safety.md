# Calibration safety configuration

`stringman-headless` accepts optional room constraint settings from the robot config JSON under `calibrationSafety`. These settings let an operator or UI describe the safe calibration zone and catch-risk objects before full calibration runs.

Machine-readable schema: [`calibration_safety.schema.json`](calibration_safety.schema.json).

Artifact summary schema: [`calibration_artifact_summary.schema.json`](calibration_artifact_summary.schema.json).

Copyable example: [`calibration_safety.example.json`](calibration_safety.example.json).

Implementation checklist: [`calibration_safety_implementation.md`](calibration_safety_implementation.md).

Validate a robot config locally:

```bash
python scripts/check_calibration_safety.py bedroom.conf
python scripts/check_calibration_safety.py bedroom.conf --json
calibration-safety-check bedroom.conf
calibration-safety-check bedroom.conf --json
python -m nf_robot.host.calibration_safety_cli bedroom.conf
```

The validator checks shape/type errors and also catches `safeProbeCenter` values
that are outside the configured `calibrationZone` or inside a no-go object,
including its configured margin. If `lineEndpoints` are provided, it also
checks cable sweeps from those endpoints to `safeProbeCenter`. It also rejects
probe-size bounds where minimum half-width/height exceed maximum half-width/height.

Create or merge a safety block into a robot config:

```bash
python scripts/apply_calibration_safety.py bedroom.conf > bedroom.with-safety.conf
python scripts/apply_calibration_safety.py bedroom.conf --summary
python scripts/apply_calibration_safety.py bedroom.conf --safety docs/calibration_safety.example.json > bedroom.with-safety.conf
python scripts/apply_calibration_safety.py bedroom.conf --derive-line-endpoints --summary
python scripts/apply_calibration_safety.py bedroom.conf --max-probe-half-width-m 0.35 --max-probe-half-height-m 0.04 --summary
python scripts/apply_calibration_safety.py bedroom.conf --add-circle-no-go lamp,0.75,-0.35,0.22,0.12 --summary
python scripts/apply_calibration_safety.py bedroom.conf --add-rect-no-go table_edge,0.25,0.55,1.15,0.75,0.08 --summary
python scripts/apply_calibration_safety.py bedroom.conf --safety docs/calibration_safety.example.json --write
calibration-safety-apply bedroom.conf --safety docs/calibration_safety.example.json --write
python -m nf_robot.host.calibration_safety_apply_cli bedroom.conf --mode manual_assisted --write
```

`calibration-safety-apply` is dry-run by default. With `--write`, it writes the
config in place and creates a timestamped `<config>.<timestamp>.bak` unless
`--no-backup` is supplied.
Use `--summary` for a concise dry-run report instead of printing the full merged
config. The `--json` write summary includes the exact backup path.
Merge mode is non-destructive for existing no-go zones; use `--replace` only
when intentionally replacing the full safety block. Empty lists in merge input
do not erase existing non-empty lists, and non-empty lists append to existing
lists with duplicate entries removed; use `--replace` for full list replacement.
Use `--derive-line-endpoints` to populate pre-run `lineEndpoints` from existing
anchor and Arpeggio eyelet positions in the config when possible. If derivation
finds no endpoints, the command fails unless
`--allow-empty-derived-line-endpoints` is supplied. Derived endpoints are
validated immediately, so a derived cable sweep through a no-go zone fails
before any write occurs.
Use numeric overrides such as `--max-probe-half-width-m`,
`--max-probe-half-height-m`, `--validation-distance-m`,
`--validation-speed-mps`, `--manual-assist-timeout-s`,
`--obstacle-margin-m`, and `--hazard-avoid-radius-m` to tune a cluttered room
without hand-editing JSON. Use `--allow-degraded-reference` /
`--no-allow-degraded-reference` and `--skip-safe-motion-validation` /
`--no-skip-safe-motion-validation` to set fallback policy explicitly.
Use `--add-circle-no-go NAME,X,Y,RADIUS[,MARGIN]` and
`--add-rect-no-go NAME,X1,Y1,X2,Y2[,MARGIN]` to append no-go objects without
hand-editing JSON. Radius and margin fields must be non-negative and are
validated before merge/write. Names must be non-empty so artifacts and
recommended actions can identify the object.

Summarize a calibration artifact after a run:

```bash
python scripts/summarize_calibration_artifact.py logs/calibration/<session>.json
python scripts/summarize_calibration_artifact.py logs/calibration/<session>.json --json
calibration-artifact-summary logs/calibration/<session>.json
calibration-artifact-summary logs/calibration/<session>.json --json
python -m nf_robot.host.calibration_artifact_cli logs/calibration/<session>.json
```

The `--json` output from these summary commands follows
[`calibration_artifact_summary.schema.json`](calibration_artifact_summary.schema.json)
and includes `recommended_actions` derived from artifact evidence, including
manual-assisted repositioning guidance when safe-zone, no-go, or cable-sweep
constraints block calibration, plus line-delta, visual-coverage, and
near-minimum-probe guidance when the health gate reports those issues.
Reference-length reset records include gantry visual freshness by anchor so
operators can identify stale or missing anchor camera views, including on
successful degraded constrained/manual-assisted resets.
Artifact summaries recommend a full-mode rerun after any degraded one-anchor
reference reset once both anchor cameras can see the gantry marker.

The controller uses this data to:

- require the current calibration start position to be inside the safe envelope
- shrink the Arpeggio calibration diamond to the current room and safe zone
- reject probe points outside the solved workspace or configured calibration zone
- reject probe points inside no-go zones
- reject cable sweeps that cross no-go zones
- avoid recent tension hazard locations during the same calibration attempt
- run a separate tiny safe-motion validation before reporting calibration success

Example:

```json
{
  "calibrationSafety": {
    "mode": "full",
    "safeProbeCenter": [0.0, 0.0],
    "calibrationZone": [
      [-1.2, -1.0],
      [1.2, -1.0],
      [1.2, 1.0],
      [-1.2, 1.0]
    ],
    "noGoZones": [
      {
        "name": "lamp",
        "center": [0.75, -0.35],
        "radiusM": 0.22,
        "marginM": 0.12
      },
      {
        "name": "chair",
        "polygon": [
          [-0.8, 0.4],
          [-0.2, 0.4],
          [-0.2, 0.9],
          [-0.8, 0.9]
        ],
        "marginM": 0.1
      }
    ],
    "maxProbeHalfWidthM": 0.65,
    "maxProbeHalfHeightM": 0.08,
    "minProbeHalfWidthM": 0.08,
    "minProbeHalfHeightM": 0.02,
    "obstacleMarginM": 0.08,
    "hazardAvoidRadiusM": 0.35,
    "validationDistanceM": 0.04,
    "validationSpeedMps": 0.03,
    "validationSettleS": 0.35,
    "manualAssistTimeoutS": 20,
    "allowDegradedReference": false,
    "skipSafeMotionValidation": false
  }
}
```

Fields:

- `mode`: `full`, `constrained`, or `manual_assisted`. `full` requires strict two-anchor fresh visual reference reset. `constrained` allows one-anchor degraded reference reset and scores the run lower. `manual_assisted` first asks the operator to move the gantry/target into a visible safe zone, then can fall back to the degraded constrained path.
- `safeProbeCenter`: optional `[x, y]` center for calibration probes. If absent, the controller uses the current visual, gantry, or hang position.
- `lineEndpoints`: optional list of `[x, y]` line endpoint positions for pre-run validation. When present, `calibration-safety-check` verifies cable sweeps from each endpoint to `safeProbeCenter` do not cross margin-expanded no-go zones. Runtime calibration uses the controller's active solved line endpoints instead.
- `calibrationZone`: optional polygon limiting where calibration probe points may land.
- `noGoZones`: optional catch-risk objects. Each zone can be a `polygon`, `rect` or `rectangle`, or `center` plus `radiusM`.
- `marginM`: optional per-zone expansion margin. This should include object thickness, cable clearance, and pose uncertainty.
- `obstacleMarginM`: default margin for zones that do not set `marginM`.
- `maxProbeHalfWidthM` and `maxProbeHalfHeightM`: cap the adaptive Arpeggio diamond size.
- `minProbeHalfWidthM` and `minProbeHalfHeightM`: lower bound before calibration fails with “no safe diamond fits”.
- `hazardAvoidRadiusM`: radius around a tension hazard that later probes avoid during the same calibration attempt.
- `validationDistanceM`: tiny validation move distance after geometry and reference solve.
- `validationSpeedMps`: tiny validation move speed.
- `validationSettleS`: wait time after each validation probe before sampling line health.
- `manualAssistTimeoutS`: extra wait time in `manual_assisted` mode while the operator moves the gantry/target into a visible safe zone.
- `allowDegradedReference`: allows one-anchor visual reference reset only when two-anchor fresh visual reset times out. Keep this `false` unless the operator accepts lower confidence and relies on the health gate.
- `skipSafeMotionValidation`: bypasses the post-solve safe-motion validation. This is for bench/debug use, not normal operation. Do not enable it for cluttered rooms.

Behavior:

- The controller does not automatically move to a different probe center. If the current start is unsafe, use `manual_assisted` mode and move the gantry/target into the configured safe zone before calibration proceeds.
- If no safe diamond fits the configured constraints, full calibration fails before the probe path starts.
- If the current gantry position is outside the safe envelope, calibration fails before reference reset, floor-touch, and line setup. In `manual_assisted` mode, the controller prompts the operator and waits before failing.
- If tension exceeds the passive safety limit during calibration, the controller stops and records a hazard instead of retrying the same move.
- If passive safety records a hazard during floor touch, tensioning, or diamond probe movement, the calibration phase aborts before the next motion step.
- During safe-motion validation, high tension on a probe or return is fatal. Stale or invalid non-tension samples are recorded as recoverable hazards and the controller tries another safe direction before failing the validation phase. Healthy probes are followed by a short return move and a post-return line-health sample.
- If the final health gate sees failures, weak visual coverage, missing/invalid/tiny measured line deltas, near-minimum probe coverage, high tension, stale line health, recorded hazards, degraded full-mode references, or failed optimizers, calibration does not report success.
- New calibration geometry is provisional during the run. The config file is written only after the final health gate passes; failed or cancelled runs restore the previous active pose state.
- The controller preserves raw `calibrationSafety`, `calibration_safety`, `roomSafety`, and `room_safety` blocks when writing the protobuf-backed robot config.
- Calibration artifacts under `logs/calibration/` include a top-level `summary`, fatal/recoverable hazard counts, the normalized safety constraints used for the run, the adaptive diamond size search, selected diamond plan, commanded diamond transitions, measured line deltas, rejection reasons, line-health samples, rejected validation probes, and health-gate report.
- Generic calibration failures are summarized from artifact evidence before being sent to the UI, so operators see actionable causes such as high-tension lines, stale line health, rejected validation moves, or no safe adaptive diamond.

## Dynamic-room planning workflow

Use `calibration-room-plan` when the room size or catch-risk objects change between calibration runs. The planner searches the room grid, rejects probe centers inside no-go zones, rejects cable sweeps that cross expanded catch-risk geometry, fits a conservative probe diamond, and emits a `calibrationSafety` block for the controller.

Example:

```bash
python scripts/plan_calibration_room.py \
  --room-width-m 3.6 \
  --room-depth-m 2.4 \
  --mode manual_assisted \
  --line-endpoint front_left,0.0,0.0 \
  --line-endpoint front_right,3.6,0.0 \
  --add-circle-no-go tripod,1.4,1.0,0.25,0.10 \
  --add-rect-no-go table,2.0,0.4,2.8,1.3,0.10 \
  --obstacle-margin-m 0.12 \
  --output calibration_safety.generated.json
```

The default output is a bare `calibrationSafety` block. Add `--include-plan-summary` when you want machine-readable evidence showing the selected center, probe size, accepted candidate count, and rejection counts for no-go, cable-sweep, and probe-envelope failures.

The plan summary includes `recommendedActions`. These are pre-run operator suggestions generated from rejection counts, line endpoint coverage, object coverage, clearance score, and selected probe size. Treat them as setup guidance before writing the config or starting calibration.

The plan summary also includes `planQuality.level`: `marginal`, `usable`, or `strong`. Use `--require-plan-quality usable` or `--require-plan-quality strong` to fail before writing output when the planned room setup is too weak for the intended calibration run. Add `--summary` to print the selected center, quality level, candidate counts, probe size, clearance score, and recommended actions to stderr while preserving machine-readable JSON output.

To apply the generated block:

```bash
python scripts/apply_calibration_safety.py path/to/config.json \
  --safety calibration_safety.generated.json \
  --write
```

If a second artifact is needed for audit evidence, rerun with `--include-plan-summary --output room_plan.json`. `apply_calibration_safety.py --safety room_plan.json` accepts that wrapped planner output directly and extracts the `calibrationSafety` object. Keep `manual_assisted` as the default for rooms with moving people, furniture, camera tripods, monitors, plants, or other cable catch risks.

Planner output with `--include-plan-summary` is documented by `docs/calibration_room_plan.schema.json`.

### Reusable room files

For rooms with many catch-risk objects, keep the room description in JSON and pass it to the planner:

```bash
python scripts/plan_calibration_room.py \
  --room-file docs/calibration_room.example.json \
  --include-plan-summary \
  --svg-output room_plan.svg \
  --output room_plan.json
```

The room file supports nested `room.widthM`, `room.depthM`, `room.origin`, `lineEndpoints`, and `noGoZones`. No-go zones can be circles, rectangles, or polygons. Use polygons for irregular footprints such as angled chair legs, plant clusters, tripod spreads, or partial furniture outlines.

If the robot config already contains anchor or indirect-eyelet positions, derive cable sweep endpoints directly from that config:

```bash
python scripts/plan_calibration_room.py \
  --room-file docs/calibration_room.example.json \
  --derive-line-endpoints-from-config path/to/config.json \
  --overwrite-line-endpoints \
  --output calibration_safety.generated.json
```

Derived endpoints are merged with endpoints from the room file and `--line-endpoint`, then de-duplicated by coordinate. Add `--overwrite-line-endpoints` when the robot config should replace stale endpoints in the room file. By default, endpoint derivation fails if no anchors or eyelets are found; use `--allow-empty-derived-line-endpoints` only for planning environments where cable-sweep checks are intentionally unavailable.

To avoid the area around a prior catch or high-tension event during the next plan, add `recentHazards` to the room file, pass `--add-hazard NAME,X,Y[,RADIUS[,MARGIN]]`, pass `--hazards-from-artifact logs/calibration/<artifact>.json`, or point at the artifact directory with `--hazards-from-artifact-dir logs/calibration --hazard-artifact-limit 1`. Hazards are emitted as temporary circular no-go zones named `hazard:<name>` and counted in `roomPlan.hazardAvoidanceCount`. When artifact files are used, their paths are reported in `roomPlan.hazardArtifactSources`.

Polygon command-line example:

```bash
python scripts/plan_calibration_room.py \
  --room-width-m 3.6 \
  --room-depth-m 2.4 \
  --add-polygon-no-go plant_cluster,0.45,1.45,0.85,1.35,0.95,1.75,0.55,1.85,0.08 \
  --output calibration_safety.generated.json
```

Command-line scalar flags override the room file. Use `--no-allow-degraded-reference` and `--no-skip-safe-motion-validation` to force those risky booleans off when reusing an older room file.

Schema: `docs/calibration_room_input.schema.json`.

Use `--svg-output room_plan.svg` to produce an operator preview showing the room boundary, calibration zone, no-go objects, selected probe center, adaptive probe diamond, line endpoints, and cable sweeps.

## Failed-run recovery sequence

For a failed calibration in a cluttered or changing room, keep the recovery path explicit:

```bash
python scripts/summarize_calibration_artifact.py logs/calibration/failed_artifact.json

python scripts/plan_calibration_room.py \
  --room-file docs/calibration_room.example.json \
  --derive-line-endpoints-from-config path/to/config.json \
  --overwrite-line-endpoints \
  --hazards-from-artifact logs/calibration/failed_artifact.json \
  --require-plan-quality usable \
  --summary \
  --svg-output room_plan.svg \
  --output calibration_safety.generated.json

python scripts/apply_calibration_safety.py path/to/config.json \
  --safety calibration_safety.generated.json \
  --write
```

Read the generated SVG before applying the config. The apply step only writes the config; restart the controller before the next calibration run uses the new safety block.
