# Calibration hardening implementation checklist

This file maps the calibration hardening work to concrete repo artifacts.

## Implemented controller behavior

- Adaptive probe planner:
  `stringman-headless` computes an adaptive Arpeggio diamond size, shrinks it against room/workspace constraints, and records every candidate in the calibration artifact.
- Measured line deltas:
  Arpeggio eyelet solve continues to use measured `line_deltas`, and the final health gate now scores missing, invalid, non-finite, or suspiciously tiny deltas.
- Catch-risk map:
  Optional `calibrationSafety` no-go zones are checked against probe endpoints and cable sweeps from line endpoints to candidate probe points.
- No retry of unsafe calibration motion:
  Passive-safety tension hazards during calibration stop the attempt and record a fatal hazard instead of retrying the same move.
- Fresh visual reacquire:
  Reference-length reset waits for fresh gantry visuals before failing, with explicit degraded fallback only in constrained/manual-assisted modes.
- Fresh visual diagnostics:
  Reference-length reset records, including degraded successes and failures, include fresh/stale gantry visual observations by anchor camera and post-run summaries recommend which views to fix.
- Degraded modes:
  `full`, `constrained`, and `manual_assisted` modes are implemented and reflected in artifacts and health scoring.
- Health gate:
  Completion scoring includes failures, optimizer status, visual coverage, probe coverage, line deltas, line health, high tension, hazards, and degraded references.
- Artifact line health:
  Calibration artifacts record line health before/after tension, reference reset, diamond observations, validation probes, and validation returns.
- Geometry vs validation separation:
  Calibration geometry is provisional until the final health gate passes; config is persisted only after successful safe-motion validation and health scoring.
- Validation cleanup:
  Fatal safe-motion validation failures clear the default motion input and stop spools before returning failure.
- Operator constraints:
  `calibrationSafety` JSON, JSON Schema, example config, apply/merge CLI, validator CLI, and artifact-summary CLI provide the repo-side contract for UI/setup tools.
- Post-run recommendations:
  The artifact summary CLI emits `recommended_actions` from high tension, stale line health, rejected diamonds, validation failures, failed optimizers, and hazards.
- Reversible setup workflow:
  The apply/merge CLI is dry-run by default, supports concise `--summary` output, and writes a timestamped backup path when `--write` is used.
- Non-destructive safety merge:
  Default apply/merge preserves existing no-go zones and non-empty list fields, appending non-empty list patches with duplicate entries removed; `--replace` is required to intentionally replace the full safety block or clear lists.
- Setup-time probe tuning:
  The apply/merge CLI exposes numeric overrides for adaptive probe size, validation distance/speed, manual-assist timeout, obstacle margin, and hazard avoid radius.
- Setup-time fallback policy:
  The apply/merge CLI exposes boolean overrides for degraded-reference and safe-motion-validation skip policy.
- Setup-time no-go entry:
  The apply/merge CLI exposes flags for appending circular and rectangular no-go objects without hand-editing JSON.
- Safe-validation skip boundary:
  `skipSafeMotionValidation` is a bench/debug escape hatch and is explicitly documented as unsuitable for cluttered rooms.
- Pre-run constraint validation:
  The validator checks config shapes and rejects `safeProbeCenter` values outside the calibration zone or inside no-go zones, including configured no-go margins.
- Probe bound validation:
  The validator rejects min probe half-width/height values that exceed their configured max values.
- Optional pre-run cable-sweep validation:
  When `lineEndpoints` are supplied, the validator rejects cable sweeps from endpoints to `safeProbeCenter` that cross margin-expanded no-go zones.
- Pre-run endpoint derivation:
  The apply/merge CLI can populate `lineEndpoints` from existing anchor and Arpeggio eyelet positions in the config for setup validation.
- Explicit derivation failure:
  `--derive-line-endpoints` fails when no endpoints are found unless the operator supplies `--allow-empty-derived-line-endpoints`.
- Derived endpoint safety gate:
  Derived endpoints are validated immediately, so a derived cable sweep through a no-go zone fails before config write.
- Runtime cable-sweep source of truth:
  `lineEndpoints` are setup hints only; runtime calibration uses active solved line endpoints from the controller geometry.
- Raw safety block preservation:
  `calibrationSafety`, `calibration_safety`, `roomSafety`, and `room_safety` keys are preserved when calibration writes the protobuf-backed config.

## Repo artifacts

- Controller orchestration:
  `src/nf_robot/host/observer.py`
- Calibration artifact model:
  `src/nf_robot/host/calibration_artifacts.py`
- Installable config validator:
  `src/nf_robot/host/calibration_safety_cli.py`
- Installable config apply/merge tool:
  `src/nf_robot/host/calibration_safety_apply_cli.py`
- Installable artifact summarizer:
  `src/nf_robot/host/calibration_artifact_cli.py`
- Repo-local wrappers:
  `scripts/apply_calibration_safety.py`
  `scripts/check_calibration_safety.py`
  `scripts/summarize_calibration_artifact.py`
- Console entry points:
  `calibration-safety-apply`
  `calibration-safety-check`
  `calibration-artifact-summary`
- Human docs:
  `docs/calibration_safety.md`
- Machine-readable schema:
  `docs/calibration_safety.schema.json`
- Machine-readable artifact summary schema:
  `docs/calibration_artifact_summary.schema.json`
- Copyable cluttered-room example:
  `docs/calibration_safety.example.json`
- Non-hardware tests:
  `tests/test_calibration_safety_tools.py`
- Source distribution inclusion:
  `MANIFEST.in`

## Runtime activation

The running `stringman-headless` process must be restarted before controller changes take effect.

Recommended operator flow:

```bash
calibration-safety-check bedroom.conf
stringman-headless --config=bedroom.conf
calibration-artifact-summary logs/calibration/<session>.json
```

## Deliberate boundaries

- The hosted UI at `neufangled.com/playroom` is not part of this checkout. This repo now exposes the controller/config/schema contract for a UI to edit or validate constraints.
- The controller does not automatically move to alternate probe centers. In cluttered rooms, `manual_assisted` mode prompts the operator to move the gantry/target into the configured safe zone before calibration continues.
- Failed or cancelled calibration restores the previous active pose state. New calibration geometry is written to config only after the final health gate passes.
## Dynamic-room recovery implementation map

The calibration hardening work is split across runtime controller behavior, pre-run room planning, artifact diagnosis, and guarded apply tooling.

| Requirement | Implementation surface |
| --- | --- |
| Adaptive calibration probe sizing for different room sizes | `Observer` adaptive diamond search and `calibration-room-plan` room-grid planner |
| Catch-risk object avoidance | `calibrationSafety.noGoZones`, planner circle/rect/polygon no-go zones, SVG preview |
| Cable sweep avoidance | Runtime safety checks plus planner `lineEndpoints` and config-derived endpoint support |
| Numerous object workflows | `--room-file`, `docs/calibration_room.example.json`, `docs/calibration_room_input.schema.json` |
| Prior hazard reuse | `--add-hazard`, `recentHazards`, `--hazards-from-artifact`, `--hazards-from-artifact-dir` |
| No unsafe retry after tension/catch hazard | Runtime calibration hazard abort behavior and artifact hazard recording |
| Fresh visual reference reacquire | Runtime reference reset wait/degraded-mode handling and artifact visual summaries |
| Degraded/reference health scoring | Runtime health gate plus artifact `recommended_actions` |
| Safe-motion validation | Runtime tiny validation probes and artifact summary reporting |
| Operator pre-run planning | `scripts/plan_calibration_room.py` |
| Operator visual review | `--svg-output` |
| Operator pre-apply validation | `scripts/check_calibration_safety.py` |
| Apply protection | `scripts/apply_calibration_safety.py` dry-run validation and merge/write controls |
| Failure triage | `scripts/summarize_calibration_artifact.py` on a chosen artifact |
| UI/automation contract | `docs/calibration_safety.schema.json`, `docs/calibration_room_input.schema.json`, and `docs/calibration_room_plan.schema.json` |

Recommended post-failure sequence:

```bash
python scripts/summarize_calibration_artifact.py logs/calibration/failed_artifact.json

python scripts/plan_calibration_room.py \
  --room-file docs/calibration_room.example.json \
  --derive-line-endpoints-from-config path/to/config.json \
  --overwrite-line-endpoints \
  --hazards-from-artifact logs/calibration/failed_artifact.json \
  --require-plan-quality usable \
  --svg-output room_plan.svg \
  --output calibration_safety.generated.json

python scripts/apply_calibration_safety.py path/to/config.json \
  --safety calibration_safety.generated.json \
  --write
```

The apply step only writes the config. The controller must be restarted after a config update before the next calibration run uses the new safety block.
