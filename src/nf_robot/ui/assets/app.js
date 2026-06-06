const state = {
  ws: null,
  bridge: null,
  robotConnected: false,
  robotId: "lan",
  components: new Map(),
  anchors: [],
  pos: null,
  grip: null,
  vidStats: null,
  videos: new Map(),
  targets: [],
  named: new Map(),
  operation: null,
  events: [],
  telemetryTicks: [],
  lastTelemetryAt: 0,
  moveTimer: null,
  gripTimer: null,
  roomBounds: null,
  selectedFeed: null,
  erroredFeeds: new Set()
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

function initElements() {
  [
    "bridge-status",
    "controller-status",
    "telemetry-rate",
    "latency",
    "robot-id",
    "gantry-position",
    "gripper-position",
    "line-tension",
    "target-count",
    "component-count",
    "component-list",
    "operation-percent",
    "operation-card",
    "room-scale",
    "room-canvas",
    "feed-tabs",
    "primary-feed-title",
    "primary-feed-source",
    "primary-feed-img",
    "primary-feed-empty",
    "video-list",
    "event-list",
    "move-speed",
    "speed-output",
    "goto-form",
    "target-list",
    "grip-readout",
    "swing-toggle",
    "swing-state",
    "identify-gripper",
    "identify-anchors",
    "json-form",
    "json-command",
    "clear-events",
    "fit-room"
  ].forEach((id) => {
    els[id] = $(id);
  });
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(digits);
}

function vecText(vec) {
  if (!vec) return "--";
  return `${fmt(vec.x)} ${fmt(vec.y)} ${fmt(vec.z)}`;
}

function setStatusPill(el, label, status) {
  const dot = el.querySelector(".dot");
  const text = el.querySelector("span:last-child");
  dot.className = `dot ${status}`;
  text.textContent = label;
}

function pushEvent(text) {
  const stamp = new Date().toLocaleTimeString();
  state.events.unshift(`${stamp} ${text}`);
  state.events = state.events.slice(0, 80);
  renderEvents();
}

function send(payload) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
    pushEvent("bridge not connected");
    return;
  }
  state.ws.send(JSON.stringify({ type: "control", payload }));
}

function command(name) {
  send({ command: name });
}

function connect() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${window.location.host}/ws`);
  state.ws = ws;

  ws.addEventListener("open", () => {
    setStatusPill(els["bridge-status"], "Bridge", "on");
    pushEvent("bridge connected");
  });

  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    handleMessage(message);
  });

  ws.addEventListener("close", () => {
    setStatusPill(els["bridge-status"], "Bridge", "off");
    state.robotConnected = false;
    renderAll();
    window.setTimeout(connect, 1000);
  });

  ws.addEventListener("error", () => {
    pushEvent("websocket error");
  });
}

function handleMessage(message) {
  if (message.type === "bridgeStatus") {
    state.bridge = message.bridge;
    state.robotConnected = Boolean(message.robot && message.robot.connected);
    if (message.robot && message.robot.robotId) {
      state.robotId = message.robot.robotId;
    }
    renderConnection();
    return;
  }

  if (message.type === "telemetry") {
    applyTelemetry(message);
    renderAll();
    return;
  }

  if (message.type === "commandAck") {
    pushEvent("command sent");
    return;
  }

  if (message.type === "commandError") {
    pushEvent(message.error || "command error");
  }
}

function applyTelemetry(message) {
  state.robotId = message.robotId || state.robotId;
  state.lastTelemetryAt = Date.now();
  state.telemetryTicks.push(state.lastTelemetryAt);
  const cutoff = state.lastTelemetryAt - 3000;
  state.telemetryTicks = state.telemetryTicks.filter((tick) => tick >= cutoff);

  for (const item of message.updates || []) {
    if (item.componentConnStatus) applyComponent(item.componentConnStatus);
    if (item.posEstimate) state.pos = item.posEstimate;
    if (item.newAnchorPoses) state.anchors = item.newAnchorPoses.poses || [];
    if (item.vidStats) state.vidStats = item.vidStats;
    if (item.gripSensors) state.grip = item.gripSensors;
    if (item.targetList) state.targets = item.targetList.targets || [];
    if (item.videoReady) applyVideoReady(item.videoReady);
    if (item.operationProgress) state.operation = item.operationProgress;
    if (item.swingCancellationState) {
      els["swing-toggle"].checked = Boolean(item.swingCancellationState.enabled);
      els["swing-state"].textContent = item.swingCancellationState.enabled ? "on" : "off";
    }
    if (item.namedPosition) {
      if (item.namedPosition.position) {
        state.named.set(item.namedPosition.name, item.namedPosition.position);
      } else {
        state.named.delete(item.namedPosition.name);
      }
    }
    if (item.popMessage) pushEvent(item.popMessage.message);
  }
}

function applyComponent(component) {
  const key = component.isGripper ? "gripper" : `anchor-${component.anchorNum ?? 0}`;
  state.components.set(key, component);
}

function normalizeFeedNumber(video) {
  if (video.feedNumber !== undefined && video.feedNumber !== null) {
    return Number(video.feedNumber);
  }
  if (video.isGripper) return 0;
  if (video.anchorNum !== undefined && video.anchorNum !== null) return Number(video.anchorNum) + 1;
  return 0;
}

function feedTitle(video) {
  const feed = normalizeFeedNumber(video);
  if (video.isGripper || feed === 0) return "Gripper";
  if (feed === 3) return "Floor";
  if (feed === 4) return "Heatmap";
  if (video.anchorNum !== undefined && video.anchorNum !== null) return `Anchor ${video.anchorNum}`;
  return `Feed ${feed}`;
}

function feedUrl(video) {
  const raw = video.localUri || "";
  if (!raw) return "";
  try {
    const url = new URL(raw, window.location.href);
    if (["localhost", "127.0.0.1", "0.0.0.0", "::1"].includes(url.hostname)) {
      url.hostname = window.location.hostname;
    }
    return url.toString();
  } catch {
    return raw;
  }
}

function applyVideoReady(video) {
  const feed = normalizeFeedNumber(video);
  const normalized = { ...video, feedNumber: feed, displayTitle: feedTitle(video), webUrl: feedUrl(video) };
  state.videos.set(feed, normalized);
  state.erroredFeeds.delete(feed);
  if (state.selectedFeed === null || !state.videos.has(state.selectedFeed) || feed === 3) {
    state.selectedFeed = feed;
  }
}

function renderConnection() {
  setStatusPill(els["controller-status"], "Controller", state.robotConnected ? "on" : "off");
  els["robot-id"].textContent = state.robotId || "lan";
}

function renderAll() {
  renderConnection();
  renderSummary();
  renderComponents();
  renderOperation();
  renderVideos();
  renderTargets();
  drawRoom();
}

function renderSummary() {
  els["gantry-position"].textContent = vecText(state.pos && state.pos.gantryPosition);
  els["gripper-position"].textContent = state.pos && state.pos.gripperPose
    ? vecText(state.pos.gripperPose.position)
    : "--";
  const tension = state.pos && state.pos.tension ? state.pos.tension.map((n) => fmt(n, 1)).join(" ") : "--";
  els["line-tension"].textContent = tension;
  els["target-count"].textContent = String(state.targets.length);
  const rate = state.telemetryTicks.length > 1 ? (state.telemetryTicks.length - 1) / 3 : 0;
  els["telemetry-rate"].textContent = `${fmt(rate, 1)} Hz`;
  const latency = state.vidStats && state.vidStats.videoLatency !== undefined
    ? `${fmt(state.vidStats.videoLatency * 1000, 0)} ms`
    : "--";
  els["latency"].textContent = latency;
  els["grip-readout"].textContent = state.grip
    ? `angle ${fmt(state.grip.angle, 0)} pressure ${fmt(state.grip.pressure, 2)}`
    : "--";
}

function renderComponents() {
  const list = els["component-list"];
  const components = [...state.components.entries()].sort(([a], [b]) => a.localeCompare(b));
  const online = components.filter(([, component]) => String(component.websocketStatus).toLowerCase().includes("connected")).length;
  els["component-count"].textContent = `${online} online`;

  if (!components.length) {
    list.innerHTML = '<div class="empty-state">No components</div>';
    return;
  }

  list.innerHTML = components.map(([key, component]) => {
    const connected = String(component.websocketStatus).toLowerCase().includes("connected");
    const name = component.isGripper ? "Gripper" : `Anchor ${component.anchorNum}`;
    const details = [
      component.ipAddress,
      component.temp !== undefined ? `${fmt(component.temp, 1)} C` : null,
      component.motorEnabled ? `motor ${component.motorEnabled}` : null
    ].filter(Boolean).join(" | ");
    return `
      <div class="component-row">
        <span class="dot ${connected ? "on" : "off"}"></span>
        <div>
          <div class="component-name">${name}</div>
          <div class="component-meta">${details || key}</div>
        </div>
        <button class="component-action" data-identify="${key}">ID</button>
      </div>
    `;
  }).join("");

  list.querySelectorAll("[data-identify]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.getAttribute("data-identify");
      identifyComponent(key);
    });
  });
}

function renderOperation() {
  const card = els["operation-card"];
  if (!state.operation) {
    els["operation-percent"].textContent = "--";
    card.innerHTML = '<div class="empty-state">Idle</div>';
    return;
  }
  const percent = Math.max(0, Math.min(100, Number(state.operation.percentComplete || 0)));
  els["operation-percent"].textContent = `${fmt(percent, 0)}%`;
  card.innerHTML = `
    <strong>${state.operation.name || "Operation"}</strong>
    <div class="component-meta">${state.operation.currentAction || ""}</div>
    <div class="progress-shell"><div class="progress-bar" style="width:${percent}%"></div></div>
  `;
}

function renderVideos() {
  const videos = [...state.videos.values()].sort((a, b) => Number(a.feedNumber) - Number(b.feedNumber));
  const list = els["video-list"];
  const tabs = els["feed-tabs"];
  const primaryImg = els["primary-feed-img"];
  const empty = els["primary-feed-empty"];

  if (!videos.length) {
    list.innerHTML = '<div class="empty-state">No feeds</div>';
    tabs.innerHTML = "";
    primaryImg.removeAttribute("src");
    primaryImg.style.display = "none";
    empty.style.display = "grid";
    els["primary-feed-title"].textContent = "Camera";
    els["primary-feed-source"].textContent = "waiting for video";
    return;
  }

  if (state.selectedFeed === null || !state.videos.has(state.selectedFeed)) {
    state.selectedFeed = videos[0].feedNumber;
  }

  tabs.innerHTML = videos.map((video) => `
    <button class="feed-tab ${video.feedNumber === state.selectedFeed ? "active" : ""}" data-select-feed="${video.feedNumber}">
      ${video.displayTitle}
    </button>
  `).join("");

  tabs.querySelectorAll("[data-select-feed]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedFeed = Number(button.getAttribute("data-select-feed"));
      renderVideos();
    });
  });

  const selected = state.videos.get(state.selectedFeed) || videos[0];
  const selectedUrl = selected.webUrl || "";
  els["primary-feed-title"].textContent = selected.displayTitle || "Camera";
  els["primary-feed-source"].textContent = selectedUrl || selected.streamPath || "stream ready";
  if (selectedUrl) {
    if (primaryImg.getAttribute("src") !== selectedUrl) {
      primaryImg.src = selectedUrl;
    }
    primaryImg.style.display = "block";
    empty.style.display = "none";
  } else {
    primaryImg.removeAttribute("src");
    primaryImg.style.display = "none";
    empty.style.display = "grid";
  }

  primaryImg.onerror = () => {
    state.erroredFeeds.add(selected.feedNumber);
    empty.textContent = "Camera stream unavailable from this browser";
    empty.style.display = "grid";
  };
  primaryImg.onload = () => {
    state.erroredFeeds.delete(selected.feedNumber);
    empty.textContent = "Waiting for gripper or anchor camera";
    empty.style.display = "none";
  };

  list.innerHTML = videos.map((video) => {
    const uri = video.webUrl || video.streamPath || "";
    const errored = state.erroredFeeds.has(video.feedNumber);
    return `
      <button class="camera-tile ${video.feedNumber === state.selectedFeed ? "active" : ""}" data-select-feed="${video.feedNumber}">
        ${video.webUrl ? `<img class="camera-thumb" src="${escapeHtml(video.webUrl)}" alt="${escapeHtml(video.displayTitle)} feed">` : `<div class="camera-thumb"></div>`}
        <div class="camera-meta">
          <span>${video.displayTitle}</span>
          <span>${errored ? "unavailable" : escapeHtml(uri)}</span>
        </div>
      </button>
    `;
  }).join("");

  list.querySelectorAll("[data-select-feed]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedFeed = Number(button.getAttribute("data-select-feed"));
      renderVideos();
    });
  });
}

function renderTargets() {
  const list = els["target-list"];
  if (!state.targets.length) {
    list.innerHTML = '<div class="empty-state">No targets</div>';
    return;
  }
  list.innerHTML = state.targets.map((target) => `
    <div class="target-row">
      <span>${target.id || "target"} ${target.status || ""}</span>
      <button data-target="${target.id}">Go</button>
    </div>
  `).join("");
  list.querySelectorAll("[data-target]").forEach((button) => {
    button.addEventListener("click", () => {
      send({ move_gripper_to: { target_id: button.getAttribute("data-target") } });
    });
  });
}

function renderEvents() {
  els["event-list"].innerHTML = state.events.map((event) => `<li>${escapeHtml(event)}</li>`).join("");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function pointXY(point) {
  return point ? { x: Number(point.x || 0), y: Number(point.y || 0), z: Number(point.z || 0) } : null;
}

function collectRoomPoints() {
  const points = [];
  for (const pose of state.anchors) {
    const point = pointXY(pose.position);
    if (point) points.push(point);
  }
  if (state.pos && state.pos.gantryPosition) points.push(pointXY(state.pos.gantryPosition));
  if (state.pos && state.pos.gripperPose && state.pos.gripperPose.position) points.push(pointXY(state.pos.gripperPose.position));
  for (const target of state.targets) {
    const point = pointXY(target.position);
    if (point) points.push(point);
  }
  for (const point of state.named.values()) {
    points.push(pointXY(point));
  }
  return points.filter(Boolean);
}

function computeBounds() {
  const points = collectRoomPoints();
  if (!points.length) {
    return { minX: -2, maxX: 2, minY: -2, maxY: 2 };
  }
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);
  const pad = Math.max(0.5, (Math.max(maxX - minX, maxY - minY) || 1) * 0.18);
  minX -= pad;
  maxX += pad;
  minY -= pad;
  maxY += pad;
  return { minX, maxX, minY, maxY };
}

function drawRoom() {
  const canvas = els["room-canvas"];
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const nextWidth = Math.max(320, Math.floor(rect.width * dpr));
  const nextHeight = Math.max(320, Math.floor(rect.height * dpr));
  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth;
    canvas.height = nextHeight;
  }

  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const bounds = state.roomBounds || computeBounds();
  const sx = width / (bounds.maxX - bounds.minX || 1);
  const sy = height / (bounds.maxY - bounds.minY || 1);
  const scale = Math.min(sx, sy) * 0.84;
  const offsetX = width / 2 - ((bounds.minX + bounds.maxX) / 2) * scale;
  const offsetY = height / 2 + ((bounds.minY + bounds.maxY) / 2) * scale;
  const toCanvas = (point) => ({
    x: offsetX + point.x * scale,
    y: offsetY - point.y * scale
  });

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fdfefe";
  ctx.fillRect(0, 0, width, height);

  drawGrid(ctx, width, height, scale, offsetX, offsetY);

  const anchors = state.anchors.map((pose) => pointXY(pose.position)).filter(Boolean);
  if (anchors.length > 1) {
    ctx.strokeStyle = "#d3dce5";
    ctx.lineWidth = 2;
    ctx.beginPath();
    anchors.forEach((point, index) => {
      const c = toCanvas(point);
      if (index === 0) ctx.moveTo(c.x, c.y);
      else ctx.lineTo(c.x, c.y);
    });
    ctx.closePath();
    ctx.stroke();
  }

  anchors.forEach((point, index) => {
    const c = toCanvas(point);
    ctx.fillStyle = "#17202a";
    ctx.fillRect(c.x - 7, c.y - 7, 14, 14);
    drawLabel(ctx, `A${index}`, c.x + 10, c.y - 10);
  });

  const gantry = state.pos && pointXY(state.pos.gantryPosition);
  const gripper = state.pos && state.pos.gripperPose && pointXY(state.pos.gripperPose.position);
  if (gantry && gripper) {
    const g = toCanvas(gantry);
    const p = toCanvas(gripper);
    ctx.strokeStyle = "#7b8998";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(g.x, g.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }
  if (gantry) drawMarker(ctx, toCanvas(gantry), "#345fba", "Gantry");
  if (gripper) drawMarker(ctx, toCanvas(gripper), "#168a5f", "Gripper");

  state.targets.forEach((target) => {
    const point = pointXY(target.position);
    if (!point) return;
    drawMarker(ctx, toCanvas(point), "#b66a00", target.id || "Target", 5);
  });

  for (const [name, point] of state.named.entries()) {
    drawMarker(ctx, toCanvas(pointXY(point)), "#8b5fbf", name, 4);
  }

  els["room-scale"].textContent = `${fmt(bounds.maxX - bounds.minX, 1)} m x ${fmt(bounds.maxY - bounds.minY, 1)} m`;
}

function drawGrid(ctx, width, height, scale, offsetX, offsetY) {
  const step = scale;
  ctx.strokeStyle = "#edf1f5";
  ctx.lineWidth = 1;
  for (let x = offsetX % step; x < width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = offsetY % step; y < height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#cbd5df";
  ctx.beginPath();
  ctx.moveTo(0, offsetY);
  ctx.lineTo(width, offsetY);
  ctx.moveTo(offsetX, 0);
  ctx.lineTo(offsetX, height);
  ctx.stroke();
}

function drawMarker(ctx, point, color, label, radius = 7) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  drawLabel(ctx, label, point.x + radius + 4, point.y - radius - 2);
}

function drawLabel(ctx, label, x, y) {
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillStyle = "#17202a";
  ctx.fillText(String(label), x, y);
}

function startMove(vector, button) {
  stopMove();
  button.classList.add("active");
  const speed = Number(els["move-speed"].value);
  const payload = { move: { direction: { x: vector[0], y: vector[1], z: vector[2] }, speed } };
  send(payload);
  state.moveTimer = window.setInterval(() => send(payload), 120);
}

function stopMove() {
  const hadActiveMove = Boolean(state.moveTimer) || Boolean(document.querySelector("[data-move].active"));
  if (!hadActiveMove) return;
  document.querySelectorAll("[data-move].active").forEach((button) => button.classList.remove("active"));
  if (state.moveTimer) {
    window.clearInterval(state.moveTimer);
    state.moveTimer = null;
  }
  send({ move: { direction: { x: 0, y: 0, z: 0 }, speed: 0 } });
}

function startGrip(payload, button) {
  stopGrip();
  button.classList.add("active");
  send({ gripper_cmd: payload });
  state.gripTimer = window.setInterval(() => send({ gripper_cmd: payload }), 120);
}

function stopGrip() {
  const hadActiveGrip = Boolean(state.gripTimer) || Boolean(document.querySelector("[data-grip].active"));
  if (!hadActiveGrip) return;
  document.querySelectorAll("[data-grip].active").forEach((button) => button.classList.remove("active"));
  if (state.gripTimer) {
    window.clearInterval(state.gripTimer);
    state.gripTimer = null;
  }
  send({ gripper_cmd: { finger_speed: 0, wrist_speed: 0, winch: 0 } });
}

function identifyComponent(key) {
  if (key === "gripper") {
    send({ single_component_action: { is_gripper: true, action: "identify" } });
    return;
  }
  const component = state.components.get(key);
  const anchorNum = component ? component.anchorNum : Number(String(key).replace("anchor-", ""));
  send({ single_component_action: { is_gripper: false, anchor_num: anchorNum, action: "identify" } });
}

function bindControls() {
  document.querySelectorAll("[data-command]").forEach((button) => {
    button.addEventListener("click", () => command(button.getAttribute("data-command")));
  });

  document.querySelectorAll("[data-move]").forEach((button) => {
    const vector = button.getAttribute("data-move").split(",").map(Number);
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      startMove(vector, button);
    });
  });
  window.addEventListener("pointerup", stopMove);
  window.addEventListener("pointercancel", stopMove);

  document.querySelectorAll("[data-grip]").forEach((button) => {
    const payload = JSON.parse(button.getAttribute("data-grip"));
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      startGrip(payload, button);
    });
  });
  window.addEventListener("pointerup", stopGrip);
  window.addEventListener("pointercancel", stopGrip);

  els["move-speed"].addEventListener("input", () => {
    els["speed-output"].textContent = `${Number(els["move-speed"].value).toFixed(2)} m/s`;
  });

  els["goto-form"].addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    send({
      move_gripper_to: {
        x: Number(data.get("x")),
        y: Number(data.get("y")),
        z: Number(data.get("z"))
      }
    });
  });

  els["swing-toggle"].addEventListener("change", () => {
    send({ set_swing_cancellation: els["swing-toggle"].checked });
  });

  els["identify-gripper"].addEventListener("click", () => identifyComponent("gripper"));
  els["identify-anchors"].addEventListener("click", () => {
    for (const key of state.components.keys()) {
      if (key !== "gripper") identifyComponent(key);
    }
  });

  els["json-form"].addEventListener("submit", (event) => {
    event.preventDefault();
    try {
      send(JSON.parse(els["json-command"].value));
    } catch (error) {
      pushEvent(`invalid JSON: ${error.message}`);
    }
  });

  els["clear-events"].addEventListener("click", () => {
    state.events = [];
    renderEvents();
  });

  els["fit-room"].addEventListener("click", () => {
    state.roomBounds = null;
    drawRoom();
  });

  window.addEventListener("resize", drawRoom);

  window.addEventListener("keydown", (event) => {
    if (event.repeat || event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;
    const map = {
      ArrowUp: [0, 1, 0],
      ArrowDown: [0, -1, 0],
      ArrowLeft: [-1, 0, 0],
      ArrowRight: [1, 0, 0],
      PageUp: [0, 0, 1],
      PageDown: [0, 0, -1]
    };
    if (event.key === " ") {
      event.preventDefault();
      command("stop_all");
      return;
    }
    if (map[event.key]) {
      event.preventDefault();
      startMove(map[event.key], document.createElement("button"));
    }
  });
  window.addEventListener("keyup", (event) => {
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown"].includes(event.key)) {
      stopMove();
    }
  });
}

function boot() {
  initElements();
  bindControls();
  renderAll();
  connect();
  window.setInterval(renderSummary, 1000);
}

document.addEventListener("DOMContentLoaded", boot);
