from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _latest(items: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for item in reversed(items):
        if item.get("kind") == kind:
            return item
    return None


def _recommended_actions(summary: dict[str, Any]) -> list[str]:
    actions: list[str] = []

    latest_line_health = summary.get("latest_line_health") or {}
    high_tension_lines = latest_line_health.get("high_tension_lines") or []
    if high_tension_lines:
        actions.append(f"Inspect and clear high-tension lines {high_tension_lines} before retrying.")
    line_profiles = latest_line_health.get("line_tension_profiles") or []
    nonresponsive = [
        profile.get("line")
        for profile in line_profiles
        if profile.get("status") == "nonresponsive"
    ]
    low_responsive = [
        profile.get("line")
        for profile in line_profiles
        if profile.get("status") == "low_tension_but_responsive"
    ]
    high_friction = [
        profile.get("line")
        for profile in line_profiles
        if profile.get("status") == "high_friction_healthy"
    ]
    if nonresponsive:
        actions.append(f"Diagnose nonresponsive line tension paths {nonresponsive} before retrying calibration.")
    if low_responsive:
        actions.append(f"Lines {low_responsive} read low tension but responded to reel-in; calibration may proceed with profile-aware gates.")
    if high_friction:
        actions.append(f"Lines {high_friction} show higher stable friction/tension and are accepted below the safety ceiling.")
    visual_reference = latest_line_health.get("gantry_visual_reference") or {}
    stale_or_missing = visual_reference.get("stale_or_missing_anchors") or []
    if stale_or_missing:
        actions.append(f"Improve gantry marker visibility for stale/missing anchor cameras {stale_or_missing}.")
    if latest_line_health.get("degraded_reference"):
        actions.append("Reference reset used degraded one-anchor visual evidence; rerun in full mode after both anchor cameras see the gantry marker.")

    artifact_summary = summary.get("summary") or {}
    if artifact_summary.get("stale_or_invalid_line_health"):
        actions.append("Wait for fresh line telemetry from every anchor, then retry calibration.")

    rejected = summary.get("adaptive_diamond_rejections") or []
    if rejected:
        last_reason = rejected[-1].get("reason")
        if last_reason:
            actions.append(f"Adjust calibrationSafety constraints or move the gantry; last rejected diamond reason: {last_reason}.")
            if "outside" in last_reason or "no-go" in last_reason or "cable sweep" in last_reason:
                actions.append("Use manual_assisted mode to reposition the gantry/target inside the safe zone before retrying.")

    validation_rejected = summary.get("safe_validation_rejected")
    if validation_rejected:
        actions.append("Move the gantry/target deeper into the safe zone or relax no-go constraints before safe-motion validation.")
    if summary.get("safe_validation_candidates") == []:
        actions.append("No safe validation directions were available; widen the calibration zone or move the start point away from no-go objects.")

    failed_optimizers = summary.get("failed_optimizers") or []
    if failed_optimizers:
        actions.append(f"Check visual marker coverage and rerun calibration; failed optimizers: {failed_optimizers}.")

    hazards = [hazard for hazard in summary.get("hazards", []) if isinstance(hazard, dict)]
    fatal_hazards = [hazard for hazard in hazards if hazard.get("fatal", True)]
    if fatal_hazards:
        actions.append("Resolve fatal calibration hazards before retrying; do not repeat the same motion path.")

    safety_constraints = summary.get("safety_constraints") or {}
    safety = safety_constraints.get("safety") if isinstance(safety_constraints, dict) else None
    if isinstance(safety, dict) and safety.get("safe_motion_validation_skipped"):
        actions.append("Safe-motion validation was skipped; rerun without skip before trusting calibration in a cluttered room.")

    latest_failure = summary.get("latest_failure") or {}
    health = latest_failure.get("health") if isinstance(latest_failure, dict) else None
    health_reasons = health.get("reasons", []) if isinstance(health, dict) else []
    if any("line deltas" in str(reason) for reason in health_reasons):
        actions.append("Check that eyelet probe moves completed and anchor line-length telemetry is fresh before retrying.")
    if any("visual coverage" in str(reason) for reason in health_reasons):
        actions.append("Improve origin/gantry marker visibility from both anchor cameras, then rerun calibration.")
    if any("probe coverage near minimum" in str(reason) for reason in health_reasons):
        actions.append("If the room is clear, widen probe limits; otherwise keep constrained mode and review calibration confidence.")
    if latest_failure and not actions:
        actions.append(f"Review calibration artifact details for failure: {latest_failure.get('message', 'unknown failure')}.")

    if not actions and summary.get("status") == "completed":
        actions.append("Calibration completed; sanity check anchor positions before normal operation.")

    return actions


def build_summary(
    artifact: dict[str, Any],
    artifact_path: str | None = None,
) -> dict[str, Any]:
    observations = artifact.get("observations") or []
    line_health = artifact.get("line_health_samples") or []
    optimizer_reports = artifact.get("optimizer_reports") or []
    failures = artifact.get("failures") or []
    warnings = artifact.get("warnings") or []

    adaptive_plan = _latest(observations, "adaptive_diamond_plan")
    validation_plan = _latest(observations, "safe_motion_validation_plan")
    validation_result = _latest(observations, "safe_motion_validation_result")
    safety_constraints = _latest(observations, "calibration_safety_constraints")
    hazards = [
        observation.get("hazard")
        for observation in observations
        if observation.get("kind") == "calibration_hazard"
    ]

    summary = {
        "session_id": artifact.get("session_id"),
        "artifact_path": artifact_path,
        "status": artifact.get("status"),
        "phase": artifact.get("phase"),
        "created_at": artifact.get("created_at"),
        "updated_at": artifact.get("updated_at"),
        "summary": artifact.get("summary"),
        "latest_failure": failures[-1] if failures else None,
        "latest_warning": warnings[-1] if warnings else None,
        "failed_optimizers": [
            report.get("name")
            for report in optimizer_reports
            if report.get("success") is False
        ],
        "latest_line_health": line_health[-1] if line_health else None,
        "safety_constraints": safety_constraints,
        "adaptive_diamond_selected": (
            adaptive_plan.get("search", {}).get("selected")
            if adaptive_plan is not None
            else None
        ),
        "adaptive_diamond_rejections": (
            [
                candidate
                for candidate in adaptive_plan.get("search", {}).get("candidates", [])
                if not candidate.get("safe")
            ]
            if adaptive_plan is not None
            else []
        ),
        "safe_validation_candidates": (
            validation_plan.get("safe_candidates")
            if validation_plan is not None
            else None
        ),
        "safe_validation_rejected": (
            validation_plan.get("rejected")
            if validation_plan is not None
            else None
        ),
        "safe_validation_result": validation_result,
        "hazards": hazards,
    }
    summary["recommended_actions"] = _recommended_actions(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact summary of a calibration artifact JSON file.")
    parser.add_argument("artifact", type=Path, help="logs/calibration/<session>.json")
    parser.add_argument("--json", action="store_true", help="Print full machine-readable summary")
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise SystemExit("artifact must be a JSON object")
    summary = build_summary(artifact, str(args.artifact))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"artifact: {args.artifact}")
        print(f"session: {summary['session_id']}")
        print(f"status: {summary['status']} phase: {summary['phase']}")
        if summary["latest_failure"]:
            print(f"latest failure: {summary['latest_failure'].get('message')}")
        if summary["failed_optimizers"]:
            print(f"failed optimizers: {summary['failed_optimizers']}")
        artifact_summary = summary.get("summary") or {}
        if artifact_summary:
            print(f"fatal hazards: {artifact_summary.get('fatal_hazard_count')}")
            print(f"recoverable hazards: {artifact_summary.get('recoverable_hazard_count')}")
        latest_line = summary["latest_line_health"] or {}
        if latest_line:
            print(f"latest line health: {latest_line.get('kind')}")
            print(f"high tension lines: {latest_line.get('high_tension_lines')}")
        selected = summary["adaptive_diamond_selected"]
        if selected:
            print(
                "adaptive diamond: "
                f"half_h={selected.get('half_height_m')} half_w={selected.get('half_width_m')}"
            )
        rejected = summary["adaptive_diamond_rejections"]
        if rejected:
            print(f"adaptive rejected candidates: {len(rejected)}")
            print(f"last rejection: {rejected[-1].get('reason')}")
        if summary["safe_validation_candidates"] is not None:
            print(f"safe validation candidates: {summary['safe_validation_candidates']}")
            print(f"safe validation rejected: {summary['safe_validation_rejected']}")
        if summary["recommended_actions"]:
            print("recommended actions:")
            for action in summary["recommended_actions"]:
                print(f"  - {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
