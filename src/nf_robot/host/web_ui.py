from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import mimetypes
import socket
import time
from importlib import resources
from pathlib import PurePosixPath
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request as UrlRequest, urlopen

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from nf_robot.generated.nf import control, telemetry
from nf_robot.ros2_bridge import control_item_from_json

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_ROBOT_URI = "ws://127.0.0.1:4245"
DEFAULT_ROBOT_ID = "lan"
WS_PATH = "/ws"
EXTERNAL_CAMERA_PROXY_PATH = "/external-cameras"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _json_safe(payload),
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _response(
    status_code: int,
    reason: str,
    body: bytes,
    content_type: str,
    extra_headers: list[tuple[str, str]] | None = None,
) -> Response:
    headers = Headers(
        [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
            ("X-Content-Type-Options", "nosniff"),
            *(extra_headers or []),
        ]
    )
    return Response(status_code, reason, headers, body)


def _asset_root():
    return resources.files("nf_robot.ui").joinpath("assets")


def static_response_for_path(path: str) -> Response:
    clean_path = path.split("?", 1)[0]
    if clean_path == "/healthz":
        return _response(200, "OK", b'{"ok":true}\n', "application/json")
    if clean_path in ("", "/"):
        clean_path = "/playroom.html"
    elif clean_path == "/simple":
        clean_path = "/index.html"

    posix_path = PurePosixPath(clean_path.lstrip("/"))
    if any(part in ("..", "") for part in posix_path.parts):
        return _response(404, "Not Found", b"Not found\n", "text/plain; charset=utf-8")

    asset = _asset_root().joinpath(*posix_path.parts)
    if not asset.is_file():
        return _response(404, "Not Found", b"Not found\n", "text/plain; charset=utf-8")

    body = asset.read_bytes()
    content_type = mimetypes.guess_type(str(posix_path))[0] or "application/octet-stream"
    if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
        content_type = f"{content_type}; charset=utf-8"
    return _response(200, "OK", body, content_type)


def telemetry_batch_to_browser_message(batch: telemetry.TelemetryBatchUpdate) -> dict[str, Any]:
    data = batch.to_dict()
    return {
        "type": "telemetry",
        "receivedAt": time.time(),
        "robotId": data.get("robotId", batch.robot_id),
        "updates": data.get("updates", []),
    }


def local_lan_urls(host: str, port: int) -> list[str]:
    if host not in {"", "0.0.0.0", "::"}:
        return [f"http://{host}:{port}"]

    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with sock:
            sock.connect(("8.8.8.8", 80))
            addresses.add(sock.getsockname()[0])
    except OSError:
        pass

    usable = sorted(addr for addr in addresses if not addr.startswith("127."))
    if not usable:
        usable = ["127.0.0.1"]
    return [f"http://{addr}:{port}" for addr in usable]


class StringmanLanWebUI:
    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        robot_uri: str = DEFAULT_ROBOT_URI,
        robot_id: str = DEFAULT_ROBOT_ID,
        external_camera_bridge_uri: str | None = None,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.host = host
        self.port = port
        self.robot_uri = robot_uri
        self.robot_id = robot_id
        self.external_camera_bridge_uri = external_camera_bridge_uri.rstrip("/") if external_camera_bridge_uri else None
        self.reconnect_delay = reconnect_delay
        self.browser_clients: set[ServerConnection] = set()
        self.robot_ws: Any | None = None
        self.robot_send_lock = asyncio.Lock()
        self.robot_task: asyncio.Task | None = None
        self.server: Any | None = None
        self.robot_status = {
            "connected": False,
            "detail": "not started",
            "robotUri": self.robot_uri,
            "robotId": self.robot_id,
            "updatedAt": time.time(),
        }

    async def start(self) -> Any:
        self.robot_task = asyncio.create_task(self._robot_loop())
        self.server = await serve(
            self._handle_browser,
            self.host,
            self.port,
            process_request=self._process_request,
            max_size=None,
        )
        return self.server

    async def stop(self) -> None:
        if self.robot_task is not None:
            self.robot_task.cancel()
            await asyncio.gather(self.robot_task, return_exceptions=True)
            self.robot_task = None

        clients = list(self.browser_clients)
        for client in clients:
            await client.close()

        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def serve_forever(self) -> None:
        await self.start()
        self._print_startup()
        try:
            await asyncio.Future()
        finally:
            await self.stop()

    def _print_startup(self) -> None:
        urls = local_lan_urls(self.host, self.port)
        print("Stringman LAN web UI")
        print(f"  Robot websocket: {self.robot_uri}")
        for url in urls:
            print(f"  Browser URL: {url}")

    def _process_request(self, _connection: ServerConnection, request: Request) -> Response | None:
        path = request.path.split("?", 1)[0]
        if path in (WS_PATH, "/robot-ws"):
            return None
        if path == EXTERNAL_CAMERA_PROXY_PATH or path.startswith(f"{EXTERNAL_CAMERA_PROXY_PATH}/"):
            return self._proxy_external_camera_request(request.path)
        return static_response_for_path(request.path)

    def _proxy_external_camera_request(self, request_path: str) -> Response:
        if not self.external_camera_bridge_uri:
            return _response(
                503,
                "Service Unavailable",
                b'{"ok":false,"error":"external camera bridge is not configured"}\n',
                "application/json",
            )

        suffix = request_path[len(EXTERNAL_CAMERA_PROXY_PATH):] or "/healthz"
        if suffix == "/":
            suffix = "/healthz"
        target = urljoin(f"{self.external_camera_bridge_uri}/", suffix.lstrip("/"))
        try:
            with urlopen(UrlRequest(target, headers={"Accept": "*/*"}), timeout=2.0) as upstream:
                body = upstream.read()
                content_type = upstream.headers.get("Content-Type", "application/octet-stream")
                return _response(upstream.status, upstream.reason or "OK", body, content_type)
        except HTTPError as exc:
            body = exc.read()
            content_type = exc.headers.get("Content-Type", "application/octet-stream")
            reason = getattr(exc, "reason", None) or getattr(exc, "msg", None) or "Upstream Error"
            return _response(exc.code, str(reason), body, content_type)
        except Exception as exc:
            payload = _json_bytes({"ok": False, "error": str(exc), "target": target})
            return _response(502, "Bad Gateway", payload, "application/json")

    async def _handle_browser(self, websocket: ServerConnection) -> None:
        path = websocket.request.path.split("?", 1)[0]
        if path == "/robot-ws":
            await self._proxy_raw_robot_websocket(websocket)
            return

        if path != WS_PATH:
            await websocket.close(1008, "unsupported path")
            return

        self.browser_clients.add(websocket)
        await self._send_to_browser(websocket, self._bridge_status_message())
        try:
            async for raw_message in websocket:
                await self._handle_browser_message(websocket, raw_message)
        finally:
            self.browser_clients.discard(websocket)

    async def _proxy_raw_robot_websocket(self, browser_ws: ServerConnection) -> None:
        try:
            async with websockets.connect(self.robot_uri, max_size=None, open_timeout=5) as robot_ws:
                await asyncio.gather(
                    self._pipe_websocket(browser_ws, robot_ws),
                    self._pipe_websocket(robot_ws, browser_ws),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Raw robot websocket proxy closed: %s", exc)
            await browser_ws.close(1011, "robot proxy unavailable")

    async def _pipe_websocket(self, source: Any, sink: Any) -> None:
        async for message in source:
            await sink.send(message)

    async def _handle_browser_message(self, websocket: ServerConnection, raw_message: Any) -> None:
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            message = json.loads(raw_message)
            message_type = message.get("type", "control")
            if message_type == "ping":
                await self._send_to_browser(websocket, {"type": "pong", "time": time.time()})
                return
            if message_type != "control":
                raise ValueError(f"Unsupported browser message type {message_type!r}")
            payload = message.get("payload")
            if payload is None:
                raise ValueError("Missing control payload")
            await self.send_control_payload(payload, robot_id=message.get("robotId"))
            await self._send_to_browser(
                websocket,
                {
                    "type": "commandAck",
                    "sentAt": time.time(),
                    "payload": payload,
                },
            )
        except Exception as exc:
            logger.warning("Browser command rejected: %s", exc)
            await self._send_to_browser(
                websocket,
                {
                    "type": "commandError",
                    "error": str(exc),
                    "at": time.time(),
                },
            )

    async def send_control_payload(self, payload: dict[str, Any] | str, *, robot_id: str | None = None) -> None:
        item = control_item_from_json(payload)
        batch = control.ControlBatchUpdate(robot_id=robot_id or self.robot_id, updates=[item])

        if self.robot_ws is None:
            raise RuntimeError("Stringman controller websocket is not connected")

        async with self.robot_send_lock:
            if self.robot_ws is None:
                raise RuntimeError("Stringman controller websocket is not connected")
            await self.robot_ws.send(bytes(batch))

    async def _robot_loop(self) -> None:
        while True:
            try:
                await self._set_robot_status(False, f"connecting {self.robot_uri}")
                async with websockets.connect(self.robot_uri, max_size=None, open_timeout=5) as websocket:
                    self.robot_ws = websocket
                    await self._set_robot_status(True, f"connected {self.robot_uri}")
                    async for raw_message in websocket:
                        if isinstance(raw_message, str):
                            raw_message = raw_message.encode("utf-8")
                        batch = telemetry.TelemetryBatchUpdate().parse(raw_message)
                        await self._broadcast(telemetry_batch_to_browser_message(batch))
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as exc:
                await self._set_robot_status(False, f"disconnected {exc.code} {exc.reason}".strip())
            except Exception as exc:
                logger.info("Robot websocket unavailable: %s", exc)
                await self._set_robot_status(False, f"error {type(exc).__name__}: {exc}")
            finally:
                self.robot_ws = None
            await asyncio.sleep(self.reconnect_delay)

    async def _set_robot_status(self, connected: bool, detail: str) -> None:
        self.robot_status = {
            "connected": connected,
            "detail": detail,
            "robotUri": self.robot_uri,
            "robotId": self.robot_id,
            "updatedAt": time.time(),
        }
        await self._broadcast(self._bridge_status_message())

    def _bridge_status_message(self) -> dict[str, Any]:
        return {
            "type": "bridgeStatus",
            "bridge": {
                "host": self.host,
                "port": self.port,
                "urls": local_lan_urls(self.host, self.port),
            },
            "robot": self.robot_status,
            "externalCameras": {
                "configured": self.external_camera_bridge_uri is not None,
                "uri": self.external_camera_bridge_uri,
                "proxiedBase": EXTERNAL_CAMERA_PROXY_PATH,
            },
        }

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        if not self.browser_clients:
            return
        encoded = _json_bytes(payload).decode("utf-8")
        clients = list(self.browser_clients)
        results = await asyncio.gather(
            *(client.send(encoded) for client in clients),
            return_exceptions=True,
        )
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                self.browser_clients.discard(client)

    async def _send_to_browser(self, websocket: ServerConnection, payload: dict[str, Any]) -> None:
        await websocket.send(_json_bytes(payload).decode("utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the Stringman LAN browser UI")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Web bind host, default 0.0.0.0")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="Web bind port, default 8080")
    parser.add_argument(
        "--robot-uri",
        default=DEFAULT_ROBOT_URI,
        help="Stringman controller websocket URI, default ws://127.0.0.1:4245",
    )
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="Robot id sent in control batches")
    parser.add_argument(
        "--external-camera-bridge-uri",
        default=None,
        help="Optional external room camera bridge URI to proxy at /external-cameras",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


async def async_main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    if not args.debug:
        logging.getLogger("websockets.server").setLevel(logging.WARNING)
    app = StringmanLanWebUI(
        host=args.host,
        port=args.port,
        robot_uri=args.robot_uri,
        robot_id=args.robot_id,
        external_camera_bridge_uri=args.external_camera_bridge_uri,
    )
    await app.serve_forever()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
