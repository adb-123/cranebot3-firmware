#!/usr/bin/env python3
"""Repo-local wrapper for nf_robot.host.calibration_room_cli."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nf_robot.host.calibration_room_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
