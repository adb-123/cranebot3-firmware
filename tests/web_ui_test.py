import asyncio
from io import BytesIO
import json
from urllib.error import HTTPError

import pytest
import websockets

from nf_robot.generated.nf import common, control, telemetry
from nf_robot.host.web_ui import (
    StringmanLanWebUI,
    static_response_for_path,
    telemetry_batch_to_browser_message,
)


def test_static_index_is_served_from_packaged_assets():
    response = static_response_for_path("/")

    assert response.status_code == 200
    assert b"Stringman" in response.body
    assert response.headers["Content-Type"].startswith("text/html")


def test_simple_fallback_page_is_still_served():
    response = static_response_for_path("/simple")

    assert response.status_code == 200
    assert b"Stringman Control" in response.body


def test_static_path_traversal_is_rejected():
    response = static_response_for_path("/../pyproject.toml")

    assert response.status_code == 404


def test_telemetry_batch_converts_to_browser_json_shape():
    batch = telemetry.TelemetryBatchUpdate(
        robot_id="robot-a",
        updates=[
            telemetry.TelemetryItem(
                pos_estimate=telemetry.PositionEstimate(
                    gantry_position=common.Vec3(x=1.0, y=2.0, z=3.0)
                )
            )
        ],
    )

    message = telemetry_batch_to_browser_message(batch)

    assert message["type"] == "telemetry"
    assert message["robotId"] == "robot-a"
    assert message["updates"][0]["posEstimate"]["gantryPosition"]["x"] == 1.0


def test_browser_control_payload_serializes_to_controller_protobuf():
    asyncio.run(_exercise_control_send())


async def _exercise_control_send():
    received = []
    received_event = asyncio.Event()

    async def fake_controller(websocket):
        message = await asyncio.wait_for(websocket.recv(), timeout=2)
        received.append(control.ControlBatchUpdate().parse(message))
        received_event.set()

    server = await websockets.serve(fake_controller, "127.0.0.1", 0)
    app = None
    try:
        port = server.sockets[0].getsockname()[1]
        app = StringmanLanWebUI(
            host="127.0.0.1",
            port=0,
            robot_uri=f"ws://127.0.0.1:{port}",
            robot_id="lan-test",
            reconnect_delay=0.05,
        )
        await app.start()

        for _ in range(40):
            if app.robot_ws is not None:
                break
            await asyncio.sleep(0.05)
        assert app.robot_ws is not None

        await app.send_control_payload({"command": "park"})
        await asyncio.wait_for(received_event.wait(), timeout=2)
    finally:
        if app is not None:
            await app.stop()
        server.close()
        await server.wait_closed()

    assert received[0].robot_id == "lan-test"
    assert received[0].updates[0].command.name == control.Command.PARK


def test_bridge_status_message_is_json_serializable():
    app = StringmanLanWebUI(host="127.0.0.1", port=8080)

    json.dumps(app._bridge_status_message())


def test_bridge_status_reports_external_camera_proxy_when_configured():
    app = StringmanLanWebUI(
        host="127.0.0.1",
        port=8080,
        external_camera_bridge_uri="http://127.0.0.1:8091/",
    )

    status = app._bridge_status_message()

    assert status["externalCameras"]["configured"] is True
    assert status["externalCameras"]["uri"] == "http://127.0.0.1:8091"
    assert status["externalCameras"]["proxiedBase"] == "/external-cameras"


def test_external_camera_proxy_returns_503_when_unconfigured():
    app = StringmanLanWebUI(host="127.0.0.1", port=8080)

    response = app._proxy_external_camera_request("/external-cameras")

    assert response.status_code == 503
    assert b"external camera bridge is not configured" in response.body


def test_external_camera_proxy_preserves_upstream_http_errors(monkeypatch):
    def failing_urlopen(_request, timeout):
        raise HTTPError(
            url="http://127.0.0.1:8091/cameras/cam/overlay.jpg",
            code=404,
            msg="Not Found",
            hdrs={"Content-Type": "text/plain; charset=utf-8"},
            fp=BytesIO(b"image unavailable\n"),
        )

    monkeypatch.setattr("nf_robot.host.web_ui.urlopen", failing_urlopen)
    app = StringmanLanWebUI(
        host="127.0.0.1",
        port=8080,
        external_camera_bridge_uri="http://127.0.0.1:8091",
    )

    response = app._proxy_external_camera_request("/external-cameras/cameras/cam/overlay.jpg")

    assert response.status_code == 404
    assert response.body == b"image unavailable\n"
    assert response.headers["Content-Type"] == "text/plain; charset=utf-8"
