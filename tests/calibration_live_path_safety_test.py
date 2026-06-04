import asyncio
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from nf_robot.host import observer as observer_module
from nf_robot.host.observer import AsyncObserver


class _Anchor:
    def __init__(self, anchor_num, origin_count):
        self.anchor_num = anchor_num
        self.origin_poses = {'origin': [object()] * origin_count}


class _RectPe:
    def __init__(self, bounds):
        self.work_area = np.array(bounds, dtype=float)
        self.visual_pos = np.array([0.0, 0.0, 1.0])

    def point_inside_work_area_2d(self, point):
        point = np.asarray(point, dtype=float)
        min_xy = np.min(self.work_area, axis=0)
        max_xy = np.max(self.work_area, axis=0)
        return bool(np.all(point[:2] >= min_xy) and np.all(point[:2] <= max_xy))


class _LineRecord:
    def __init__(self, speed):
        self.speed = speed

    def getLast(self):
        return [0.0, 1.0, self.speed, 1.0]


def test_origin_visibility_uses_physical_anchor_numbers_for_shuffled_dict():
    observer = AsyncObserver.__new__(AsyncObserver)
    observer.anchors = {
        10: _Anchor(3, 0),
        20: _Anchor(1, 2),
        30: _Anchor(2, 1),
        40: _Anchor(0, 3),
    }

    counts = observer._origin_detection_counts()

    assert counts == {3: 0, 1: 2, 2: 1, 0: 3}
    assert observer._origin_visible_anchor_nums(counts) == [0, 1, 2]


def test_adaptive_diamond_size_shrinks_to_fit_configured_work_area():
    observer = AsyncObserver.__new__(AsyncObserver)
    observer.pe = _RectPe([
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
    ])

    half_h, half_w = observer._adaptive_diamond_size(default_size=(0.1, 1.0))

    assert half_h <= 0.1
    assert half_w < 1.0
    for point in observer._diamond_probe_points(np.zeros(2), half_h, half_w):
        assert observer.pe.point_inside_work_area_2d(point)


def test_adaptive_diamond_size_fails_when_no_safe_probe_fits():
    observer = AsyncObserver.__new__(AsyncObserver)
    observer.pe = _RectPe([
        [-0.01, -0.01],
        [0.01, -0.01],
        [0.01, 0.01],
        [-0.01, 0.01],
    ])

    with pytest.raises(RuntimeError, match='No safe Arpeggio eyelet calibration diamond'):
        observer._adaptive_diamond_size(default_size=(0.1, 1.0))


def test_require_gantry_observations_rejects_empty_snapshots():
    observer = AsyncObserver.__new__(AsyncObserver)
    observer.snapshot_tag_observations = lambda: {'gantry': [[], []]}

    with pytest.raises(RuntimeError, match='No usable gantry observations'):
        observer._require_gantry_observations('bottom')


def test_diamond_line_settle_timeout_stops_lines_and_raises(monkeypatch):
    async def run():
        observer = AsyncObserver.__new__(AsyncObserver)
        observer.datastore = SimpleNamespace(anchor_line_record=[
            _LineRecord(0.0),
            _LineRecord(0.2),
            _LineRecord(0.0),
            _LineRecord(0.2),
        ])
        commands = []

        async def send_line_speed(line, speed, jog=False):
            commands.append((line, speed, jog))

        observer.send_line_speed = send_line_speed

        original_sleep = observer_module.asyncio.sleep

        async def fast_sleep(_duration):
            await original_sleep(0)

        monkeypatch.setattr(observer_module.asyncio, 'sleep', fast_sleep)

        with pytest.raises(RuntimeError, match='Diamond lines did not settle'):
            await observer._wait_for_diamond_lines_to_stop(timeout=0.01)

        assert (1, 0, False) in commands
        assert (3, 0, False) in commands

    asyncio.run(run())


def test_full_calibration_writes_artifacts_and_passes_arpeggio_tilts_positionally():
    source = inspect.getsource(AsyncObserver.full_auto_calibration)

    assert "_new_calibration_artifact('full_auto_calibration')" in source
    assert "_write_calibration_artifact(calibration_artifact" in source
    assert "(raw_obs, None, None, None, None, tilts)" in source
