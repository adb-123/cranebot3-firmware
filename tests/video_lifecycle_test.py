import time
import threading
import asyncio
from unittest.mock import MagicMock

from nf_robot.common.config_loader import create_default_config
from nf_robot.host.anchor_client import RaspiAnchorClient
from nf_robot.host.video_streamer import MjpegStreamer, NfVideoStreamer


def _wait_until(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_component_video_reconnect_stops_previous_receiver_thread():
    ob = MagicMock()
    ob.config = create_default_config()
    client = RaspiAnchorClient(
        "127.0.0.1",
        8765,
        1,
        datastore=MagicMock(),
        ob=ob,
        pool=MagicMock(),
        stat=MagicMock(),
        telemetry_env=None,
    )
    client.connected = True
    client._connection_generation = 1
    sessions = []

    def fake_receive_video(port, generation, stop_event):
        sessions.append((port, generation, stop_event))
        while not stop_event.wait(0.01):
            pass

    client.receive_video = fake_receive_video

    client._start_video_session(8888, generation=1)
    assert _wait_until(lambda: len(sessions) == 1)
    first_thread = client._video_thread
    first_stop_event = sessions[0][2]

    client._connection_generation = 2
    client._start_video_session(8888, generation=2)

    assert first_stop_event.is_set()
    assert _wait_until(lambda: not first_thread.is_alive())
    assert _wait_until(lambda: len(sessions) == 2)
    assert client._video_thread is not first_thread

    second_thread = client._video_thread
    second_stop_event = sessions[1][2]
    client._stop_video_session()

    assert second_stop_event.is_set()
    assert _wait_until(lambda: not second_thread.is_alive())
    assert client._video_thread is None


def test_component_video_reconnect_refuses_to_replace_stuck_receiver():
    ob = MagicMock()
    ob.config = create_default_config()
    client = RaspiAnchorClient(
        "127.0.0.1",
        8765,
        1,
        datastore=MagicMock(),
        ob=ob,
        pool=MagicMock(),
        stat=MagicMock(),
        telemetry_env=None,
    )
    client.connected = True
    client._connection_generation = 1
    sessions = []
    release = threading.Event()

    def stuck_receive_video(port, generation, stop_event):
        sessions.append((port, generation, stop_event))
        while not release.wait(0.01):
            pass

    client.receive_video = stuck_receive_video

    assert client._start_video_session(8888, generation=1) is True
    assert _wait_until(lambda: len(sessions) == 1)
    first_thread = client._video_thread

    client._connection_generation = 2
    assert client._start_video_session(8888, generation=2) is False

    assert len(sessions) == 1
    assert client._video_thread is first_thread
    assert first_thread.is_alive()
    release.set()
    assert _wait_until(lambda: not first_thread.is_alive())
    assert client._stop_video_session() is True


def test_component_video_retry_starts_replacement_after_stuck_receiver_exits():
    ob = MagicMock()
    ob.config = create_default_config()
    client = RaspiAnchorClient(
        "127.0.0.1",
        8765,
        1,
        datastore=MagicMock(),
        ob=ob,
        pool=MagicMock(),
        stat=MagicMock(),
        telemetry_env=None,
    )
    client.connected = True
    client._connection_generation = 1
    sessions = []
    release = threading.Event()

    def receive_video(port, generation, stop_event):
        sessions.append((port, generation, stop_event))
        while not release.wait(0.01):
            pass

    client.receive_video = receive_video

    assert client._start_video_session(8888, generation=1) is True
    assert _wait_until(lambda: len(sessions) == 1)
    client._connection_generation = 2
    assert client._start_video_session(8888, generation=2) is False

    release.set()
    started = asyncio.run(
        client._retry_video_session_start(
            8888,
            generation=2,
            attempts=2,
            retry_delay=0.01,
        )
    )

    assert started is True
    assert _wait_until(lambda: len(sessions) == 2)
    assert sessions[1][1] == 2
    client._stop_video_session()


def test_recent_video_age_uses_host_receipt_time_not_capture_time():
    ob = MagicMock()
    ob.config = create_default_config()
    client = RaspiAnchorClient(
        "127.0.0.1",
        8765,
        1,
        datastore=MagicMock(),
        ob=ob,
        pool=MagicMock(),
        stat=MagicMock(),
        telemetry_env=None,
    )
    client.last_frame_cap_time = time.time() - 1000
    client._last_frame_host_time = time.time() - 0.25

    assert client._recent_video_age() < 1.0


def test_mjpeg_streamer_stop_is_idempotent_and_stops_server_thread():
    streamer = MjpegStreamer(width=16, height=16, port=0)
    streamer.start()
    thread = streamer._server_thread

    assert thread is not None
    assert thread.is_alive()

    streamer.stop()
    streamer.stop()

    assert streamer.http_server is None
    assert not thread.is_alive()


def test_nf_video_streamer_stop_is_idempotent():
    streamer = NfVideoStreamer(
        width=16,
        height=16,
        fps=10,
        mjpeg_port=0,
        stream_path="test/path",
        telemetry_env=None,
    )
    streamer.start()
    streamer.stop()
    streamer.stop()

    assert streamer._stopped is True
