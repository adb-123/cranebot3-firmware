import json
import os
from pathlib import Path

from nf_robot.host import calibration_artifacts
from nf_robot.host.calibration_artifacts import CalibrationArtifactSession


def _clock():
    return "2026-06-04T00:00:00Z"


def test_calibration_artifact_writes_finite_json(tmp_path):
    session = CalibrationArtifactSession(
        session_id="finite-json",
        artifact_dir=tmp_path,
        now=_clock,
    )
    session.set_phase("marker_capture")
    session.record_observation(
        anchor_num=1,
        marker="origin",
        pose=[1.0, float("nan"), float("inf"), -float("inf")],
    )
    session.record_line_health(
        line_id="anchor-1-spool-0",
        tension=float("nan"),
        speed=0.0,
    )

    path = session.write(status="completed")

    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text

    data = json.loads(text)
    assert data["session_id"] == "finite-json"
    assert data["status"] == "completed"
    assert data["observations"][0]["pose"] == [1.0, None, None, None]
    assert data["line_health_samples"][0]["tension"] is None
    assert any(warning["kind"] == "non_finite_value" for warning in data["warnings"])


def test_calibration_artifact_write_is_atomic(tmp_path, monkeypatch):
    session = CalibrationArtifactSession(
        session_id="atomic-write",
        artifact_dir=tmp_path,
        now=_clock,
    )
    calls = []
    real_replace = os.replace

    def replace_spy(src, dst):
        src_path = Path(src)
        dst_path = Path(dst)
        calls.append((src_path, dst_path))
        assert src_path.parent == tmp_path
        assert src_path.name.startswith(f".{dst_path.name}.")
        assert src_path.suffix == ".tmp"
        assert src_path.exists()
        assert not dst_path.exists()
        real_replace(src, dst)

    monkeypatch.setattr(calibration_artifacts.os, "replace", replace_spy)

    path = session.write(status="completed")

    assert path.exists()
    assert len(calls) == 1
    assert calls[0][1] == path
    assert [item for item in tmp_path.iterdir() if item.suffix == ".tmp"] == []
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "completed"


def test_calibration_artifact_records_warnings_and_failures(tmp_path):
    session = CalibrationArtifactSession(
        session_id="warning failure/session",
        artifact_dir=tmp_path,
        metadata={"robot": "stringman"},
        now=_clock,
    )

    session.set_phase("tension")
    session.warn("line tension is low", line_id="anchor-2-spool-1")
    session.fail("tension timeout", line_id="anchor-2-spool-1")
    path = session.write()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "warning-failure-session.json"
    assert data["metadata"] == {"robot": "stringman"}
    assert data["phase"] == "tension"
    assert data["status"] == "failed"
    assert data["warnings"] == [
        {
            "timestamp": "2026-06-04T00:00:00Z",
            "phase": "tension",
            "message": "line tension is low",
            "line_id": "anchor-2-spool-1",
        }
    ]
    assert data["failures"] == [
        {
            "timestamp": "2026-06-04T00:00:00Z",
            "phase": "tension",
            "message": "tension timeout",
            "line_id": "anchor-2-spool-1",
        }
    ]
    assert data["status_history"][-1]["status"] == "failed"


def test_calibration_artifact_records_optimizer_report(tmp_path):
    session = CalibrationArtifactSession(
        session_id="optimizer-report",
        artifact_dir=tmp_path,
        now=_clock,
    )

    session.set_phase("solve")
    session.record_optimizer_report(
        name="anchor_pose",
        success=True,
        cost=0.012,
        residuals_by_group={"origin": {"rms": 0.01, "max": 0.02}},
    )
    path = session.write(status="completed")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["optimizer_reports"] == [
        {
            "timestamp": "2026-06-04T00:00:00Z",
            "phase": "solve",
            "name": "anchor_pose",
            "success": True,
            "cost": 0.012,
            "residuals_by_group": {"origin": {"rms": 0.01, "max": 0.02}},
        }
    ]
