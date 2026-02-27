const $ = (id) => document.getElementById(id);

const ui = {
  wsEndpoint: $("wsEndpoint"),
  userId: $("userId"),
  prompt: $("prompt"),
  language: $("language"),
  wsToggle: $("wsToggle"),
  wsState: $("wsState"),
  planBtn: $("planBtn"),
  staticBtn: $("staticBtn"),
  animateBtn: $("animateBtn"),
  latestBtn: $("latestBtn"),
  imageInput: $("imageInput"),
  matrixWidth: $("matrixWidth"),
  matrixHeight: $("matrixHeight"),
  downsampleBtn: $("downsampleBtn"),
  fps: $("fps"),
  duration: $("duration"),
  animWidth: $("animWidth"),
  animHeight: $("animHeight"),
  animMeta: $("animMeta"),
  matrixBlurToggle: $("matrixBlurToggle"),
  matrixBlurStrength: $("matrixBlurStrength"),
  stripMode: $("stripMode"),
  brightness: $("brightness"),
  speed: $("speed"),
  stripColor: $("stripColor"),
  stripLedCount: $("stripLedCount"),
  stripBtn: $("stripBtn"),
  stripPreview: $("stripPreview"),
  stripCanvas: $("stripCanvas"),
  stripMeta: $("stripMeta"),
  reasonMeta: $("reasonMeta"),
  jsonOut: $("jsonOut"),
  eventLog: $("eventLog"),
  matrixMeta: $("matrixMeta"),
  canvas: $("matrixCanvas"),
  blurCanvas: $("matrixBlurCanvas"),
};

const matrixCtx = ui.canvas.getContext("2d");
const matrixBlurCtx = ui.blurCanvas.getContext("2d");
const stripCtx = ui.stripCanvas.getContext("2d");

const state = {
  ws: null,
  manualClose: false,
  matrixTimer: null,
  latestMatrix: null,
  matrixBlur: {
    enabled: localStorage.getItem("debug.matrixBlurEnabled") === "1",
    strength: clamp(parseNum(localStorage.getItem("debug.matrixBlurStrength"), 6), 1, 20),
  },
  stripRAF: 0,
  strip: {
    mode: "flow",
    speed: 0.6,
    brightness: 0.5,
    ledCount: 24,
    direction: "clockwise",
    palette: [{ r: 90, g: 120, b: 150 }, { r: 110, g: 160, b: 220 }, { r: 70, g: 95, b: 130 }],
    reason: "",
  },
};

function stopStripAnimation() {
  if (state.stripRAF) {
    cancelAnimationFrame(state.stripRAF);
    state.stripRAF = 0;
  }
}

function parseNum(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function clampByte(value, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.round(clamp(n, 0, 255));
}

function userId() {
  return ui.userId.value.trim() || "anon";
}

function setJSON(data) {
  ui.jsonOut.textContent = JSON.stringify(data, null, 2);
}

function appendEvent(line, cls = "") {
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.textContent = line;
  ui.eventLog.appendChild(div);
  ui.eventLog.scrollTop = ui.eventLog.scrollHeight;
}

function toText(value) {
  return typeof value === "string" ? value.trim() : "";
}

function firstReason(data) {
  if (!data || typeof data !== "object") return "";

  const direct = [
    toText(data.reason),
    toText(data.broadcast_copy),
    toText(data.scene_suggestion),
    toText(data.speakable_reason),
  ].find(Boolean);
  if (direct) return direct;

  const suggestedStrip = data.suggested_strip;
  if (suggestedStrip && typeof suggestedStrip === "object") {
    const reason = toText(suggestedStrip.reason);
    if (reason) return reason;
  }

  const intentPlan = data.intent_plan;
  if (intentPlan && typeof intentPlan === "object") {
    const reason = [
      toText(intentPlan.broadcast_copy),
      toText(intentPlan.scene_suggestion),
      toText(intentPlan.reasoning_snapshot),
      toText(intentPlan.suggested_strip?.reason),
    ].find(Boolean);
    if (reason) return reason;
  }

  const zones = data.vehicle_plan?.zones;
  if (Array.isArray(zones)) {
    for (const zone of zones) {
      const reason = toText(zone?.reason);
      if (reason) return reason;
    }
  }

  return "";
}

function updateReasonPanel(data) {
  const reason = firstReason(data);
  if (!reason) return;
  ui.reasonMeta.textContent = reason;
}

function hexToRgb(hex) {
  const value = String(hex || "").trim();
  const matched = /^#?([0-9a-f]{6})$/i.exec(value);
  if (!matched) return { r: 90, g: 120, b: 150 };
  const raw = matched[1];
  return {
    r: parseInt(raw.slice(0, 2), 16),
    g: parseInt(raw.slice(2, 4), 16),
    b: parseInt(raw.slice(4, 6), 16),
  };
}

function rgbToHex(color) {
  const toHex = (n) => clampByte(n, 0).toString(16).padStart(2, "0");
  return `#${toHex(color.r)}${toHex(color.g)}${toHex(color.b)}`;
}

function blend(a, b, t) {
  const ratio = clamp(t, 0, 1);
  return {
    r: clampByte(a.r + (b.r - a.r) * ratio, 90),
    g: clampByte(a.g + (b.g - a.g) * ratio, 120),
    b: clampByte(a.b + (b.b - a.b) * ratio, 150),
  };
}

function samplePalette(palette, t) {
  if (!Array.isArray(palette) || palette.length === 0) return { r: 90, g: 120, b: 150 };
  if (palette.length === 1) return palette[0];
  const x = ((t % 1) + 1) % 1;
  const segment = x * (palette.length - 1);
  const idx = Math.floor(segment);
  if (idx >= palette.length - 1) return palette[palette.length - 1];
  return blend(palette[idx], palette[idx + 1], segment - idx);
}

function withBrightness(color, brightness) {
  const b = clamp(brightness, 0, 1);
  return {
    r: clampByte(color.r * b),
    g: clampByte(color.g * b),
    b: clampByte(color.b * b),
  };
}

function derivePaletteFromBase(color) {
  const base = {
    r: clampByte(color?.r, 90),
    g: clampByte(color?.g, 120),
    b: clampByte(color?.b, 150),
  };
  return [
    base,
    blend(base, { r: 255, g: 180, b: 90 }, 0.28),
    blend(base, { r: 35, g: 120, b: 220 }, 0.25),
  ];
}

async function api(path, method, body, isForm = false) {
  const headers = { "X-User-ID": userId() };
  if (!isForm) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {
    method,
    headers,
    body: isForm ? body : JSON.stringify(body),
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  setJSON(data);
  updateReasonPanel(data);
  return data;
}

function drawMatrix(matrix) {
  if (!matrix || !Array.isArray(matrix.pixels) || !matrix.width || !matrix.height) return;
  state.latestMatrix = matrix;
  drawMatrixToCanvas(matrixCtx, ui.canvas, matrix);
  drawMatrixToCanvas(matrixBlurCtx, ui.blurCanvas, matrix);
  renderMatrixBlur();
  ui.matrixMeta.textContent = `矩阵 ${matrix.width}x${matrix.height} 来源=${matrix.source || "n/a"} 帧=${matrix.frame_index ?? 0}`;
}

function drawMatrixToCanvas(ctx, canvas, matrix) {
  const cellW = canvas.width / matrix.width;
  const cellH = canvas.height / matrix.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < matrix.height; y += 1) {
    for (let x = 0; x < matrix.width; x += 1) {
      const idx = y * matrix.width + x;
      const pixel = matrix.pixels[idx];
      if (!pixel) continue;
      ctx.fillStyle = `rgb(${pixel.r}, ${pixel.g}, ${pixel.b})`;
      ctx.fillRect(Math.floor(x * cellW), Math.floor(y * cellH), Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}

function renderMatrixBlur() {
  if (!state.latestMatrix) {
    matrixBlurCtx.clearRect(0, 0, ui.blurCanvas.width, ui.blurCanvas.height);
    return;
  }

  drawMatrixToCanvas(matrixBlurCtx, ui.blurCanvas, state.latestMatrix);
  const s = clamp(parseNum(state.matrixBlur.strength, 6), 1, 20);
  ui.blurCanvas.style.filter = state.matrixBlur.enabled ? `blur(${s}px)` : "blur(0px)";
}

function playMatrixFrames(frames, fps) {
  if (state.matrixTimer) {
    clearInterval(state.matrixTimer);
    state.matrixTimer = null;
  }
  if (!Array.isArray(frames) || frames.length === 0) return;
  let idx = 0;
  const interval = Math.max(30, Math.floor(1000 / Math.max(1, fps || 20)));
  state.matrixTimer = setInterval(() => {
    const frame = frames[idx % frames.length];
    drawMatrix({
      width: frame.meta.width,
      height: frame.meta.height,
      pixels: frame.pixels,
      source: "animation-playback",
      frame_index: frame.meta.frame_index,
    });
    idx += 1;
  }, interval);
}

function normalizeMode(mode) {
  const value = String(mode || "flow").trim().toLowerCase();
  return value || "flow";
}

function normalizeDirection(direction) {
  return String(direction || "clockwise").trim().toLowerCase() === "counterclockwise" ? "counterclockwise" : "clockwise";
}

function setWsState(online) {
  ui.wsState.textContent = online ? "WS 已连接" : "WS 未连接";
  ui.wsState.classList.toggle("online", online);
  ui.wsToggle.textContent = online ? "断开 WS" : "连接 WS";
}

function speedToMotionRate(mode, speed) {
  const s = clamp(parseNum(speed, 0.6), 0, 4);
  if (mode === "static") {
    return 0;
  }
  return s * 0.28;
}

function applyStripState(partial) {
  state.strip = {
    ...state.strip,
    ...partial,
    mode: normalizeMode(partial.mode ?? state.strip.mode),
    direction: normalizeDirection(partial.direction ?? state.strip.direction),
    speed: clamp(parseNum(partial.speed ?? state.strip.speed, 0.6), 0, 4),
    brightness: clamp(parseNum(partial.brightness ?? state.strip.brightness, 0.5), 0, 1),
    ledCount: Math.max(1, Math.floor(parseNum(partial.ledCount ?? state.strip.ledCount, 24))),
  };
  if (toText(partial.reason)) {
    state.strip.reason = toText(partial.reason);
  }
  if (Array.isArray(partial.palette) && partial.palette.length > 0) {
    state.strip.palette = partial.palette.map((c) => ({
      r: clampByte(c.r, 90),
      g: clampByte(c.g, 120),
      b: clampByte(c.b, 150),
    }));
  }
  ui.stripMode.value = state.strip.mode;
  ui.brightness.value = String(Number(state.strip.brightness.toFixed(2)));
  ui.speed.value = String(Number(state.strip.speed.toFixed(2)));
  ui.stripLedCount.value = String(state.strip.ledCount);
  ui.stripColor.value = rgbToHex(state.strip.palette[0] || { r: 90, g: 120, b: 150 });
}

function applyStripFromCmd(cmd) {
  if (!cmd || typeof cmd !== "object") return false;
  if (!cmd.mode || !cmd.color) return false;
  applyStripState({
    mode: cmd.mode,
    speed: cmd.speed,
    brightness: cmd.brightness,
    ledCount: cmd.led_count,
    palette: derivePaletteFromBase(cmd.color),
    reason: cmd.reason,
  });
  return true;
}

function applyStripFromZone(zone) {
  if (!zone || typeof zone !== "object") return false;
  if (!zone.mode) return false;
  const palette = Array.isArray(zone.colors) && zone.colors.length > 0 ? zone.colors : derivePaletteFromBase({ r: 90, g: 120, b: 150 });
  applyStripState({
    mode: zone.mode,
    speed: zone.speed,
    brightness: zone.brightness,
    palette,
    reason: zone.reason,
  });
  return true;
}

function applyStripFromFrame(frame) {
  if (!frame || typeof frame !== "object") return false;
  if (!Array.isArray(frame.pixels) || frame.pixels.length === 0) return false;

  const step = Math.max(1, Math.floor(frame.pixels.length / 3));
  const palette = [
    frame.pixels[0],
    frame.pixels[Math.min(frame.pixels.length - 1, step)],
    frame.pixels[Math.min(frame.pixels.length - 1, step * 2)],
  ];
  applyStripState({
    mode: frame.mode,
    speed: frame.speed,
    brightness: frame.brightness,
    ledCount: frame.led_count,
    direction: frame.direction,
    palette,
    reason: frame.reason,
  });
  return true;
}

function applyStripFromAnyPayload(data) {
  if (!data || typeof data !== "object") return;
  if (applyStripFromFrame(data)) return;
  if (applyStripFromCmd(data)) return;
  if (applyStripFromCmd(data.suggested_strip)) return;
  if (applyStripFromCmd(data.intent_plan?.suggested_strip)) return;
  const zones = data.vehicle_plan?.zones;
  if (Array.isArray(zones) && zones.length > 0) {
    applyStripFromZone(zones[0]);
  }
}

function stripPixelsAt(timeSec) {
  const ledCount = Math.max(1, state.strip.ledCount);
  const mode = state.strip.mode;
  const speed = Math.max(0, state.strip.speed);
  const brightness = state.strip.brightness;
  const direction = state.strip.direction;
  const palette = state.strip.palette;
  const motionRate = speedToMotionRate(mode, speed);

  const pixels = new Array(ledCount);
  for (let i = 0; i < ledCount; i += 1) {
    const pos = i / Math.max(1, ledCount - 1);
    const phaseBase = timeSec * motionRate;
    const dirPos = direction === "counterclockwise" ? -pos : pos;
    let colorPos = pos;
    let factor = 1;

    if (mode === "static") {
      colorPos = pos;
      factor = 1;
    } else if (mode === "breath" || mode === "pulse") {
      const breath = 0.5 + 0.5 * Math.sin((phaseBase * 0.8) * Math.PI * 2);
      colorPos = pos;
      factor = 0.35 + 0.65 * breath;
    } else if (mode === "chase") {
      const head = 0.5 + 0.5 * Math.sin((phaseBase * 2.2 + dirPos) * Math.PI * 2);
      colorPos = pos + phaseBase * 0.4;
      factor = 0.14 + 0.86 * Math.pow(head, 2.4);
    } else if (mode === "wave") {
      const w1 = 0.5 + 0.5 * Math.sin((phaseBase + dirPos) * Math.PI * 2);
      const w2 = 0.5 + 0.5 * Math.sin((phaseBase * 0.66 + pos * 1.8) * Math.PI * 2);
      colorPos = pos + phaseBase * 0.22;
      factor = 0.3 + 0.7 * (w1 * 0.7 + w2 * 0.3);
    } else if (mode === "sparkle") {
      const sparkle = 0.5 + 0.5 * Math.sin((phaseBase * 8 + i * 0.87) * Math.PI * 2);
      colorPos = pos + phaseBase * 0.3;
      factor = 0.22 + 0.78 * sparkle;
    } else if (mode === "surround_flow") {
      const center = Math.abs(pos - 0.5);
      const shell = 1 - clamp(center * 1.9, 0, 1);
      const flow = 0.5 + 0.5 * Math.sin((phaseBase * 1.3 + dirPos) * Math.PI * 2);
      colorPos = pos + phaseBase * 0.34;
      factor = 0.24 + 0.76 * (flow * 0.55 + shell * 0.45);
    } else {
      const flow = 0.5 + 0.5 * Math.sin((phaseBase * 1.3 + dirPos) * Math.PI * 2);
      colorPos = pos + phaseBase * 0.32;
      factor = 0.28 + 0.72 * flow;
    }

    const sampled = samplePalette(palette, colorPos);
    pixels[i] = withBrightness(sampled, brightness * factor);
  }
  return pixels;
}

function buildStripRailGradient(pixels) {
  if (!Array.isArray(pixels) || pixels.length === 0) {
    return "linear-gradient(90deg, rgb(90,120,150), rgb(110,160,220), rgb(70,95,130))";
  }
  const stops = 7;
  const parts = [];
  for (let i = 0; i < stops; i += 1) {
    const pos = i / Math.max(1, stops - 1);
    const idx = Math.min(pixels.length - 1, Math.round(pos * (pixels.length - 1)));
    const p = pixels[idx] || { r: 90, g: 120, b: 150 };
    parts.push(`rgb(${p.r},${p.g},${p.b}) ${Math.round(pos * 100)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

function drawStripPixels(pixels) {
  if (!Array.isArray(pixels) || pixels.length === 0) return;

  const width = ui.stripCanvas.width;
  const height = ui.stripCanvas.height;
  const radius = 12;
  stripCtx.clearRect(0, 0, width, height);

  stripCtx.save();
  stripCtx.beginPath();
  stripCtx.moveTo(radius, 0);
  stripCtx.lineTo(width - radius, 0);
  stripCtx.arcTo(width, 0, width, radius, radius);
  stripCtx.arcTo(width, height, width - radius, height, radius);
  stripCtx.lineTo(radius, height);
  stripCtx.arcTo(0, height, 0, radius, radius);
  stripCtx.arcTo(0, 0, radius, 0, radius);
  stripCtx.closePath();
  stripCtx.clip();

  stripCtx.fillStyle = "#eff5fb";
  stripCtx.fillRect(0, 0, width, height);

  const n = pixels.length;
  const gap = 1;
  const cell = width / n;
  for (let i = 0; i < n; i += 1) {
    const p = pixels[i];
    const x = i * cell;
    stripCtx.fillStyle = `rgb(${p.r},${p.g},${p.b})`;
    stripCtx.fillRect(x + gap * 0.5, 0, Math.max(1, cell - gap), height);
  }

  const glow = stripCtx.createRadialGradient(width * 0.5, height * 0.5, height * 0.08, width * 0.5, height * 0.5, width * 0.56);
  glow.addColorStop(0, "rgba(255,255,255,0.22)");
  glow.addColorStop(0.5, "rgba(255,255,255,0.11)");
  glow.addColorStop(1, "rgba(255,255,255,0)");
  stripCtx.fillStyle = glow;
  stripCtx.fillRect(0, 0, width, height);

  stripCtx.restore();

  const first = pixels[0] || { r: 90, g: 120, b: 150 };
  const mid = pixels[Math.floor(pixels.length / 2)] || first;
  ui.stripPreview.style.background = buildStripRailGradient(pixels);
  ui.stripPreview.style.boxShadow = `inset 0 0 0 1px rgba(255,255,255,0.1), 0 0 10px rgba(${mid.r},${mid.g},${mid.b},0.28)`;
  ui.stripPreview.style.borderColor = `rgba(${mid.r},${mid.g},${mid.b},0.4)`;

  const speedRule = state.strip.mode === "static" ? "静态" : "越大越快";
  ui.stripMeta.textContent = `模式=${state.strip.mode} 灯珠=${state.strip.ledCount} 亮度=${state.strip.brightness.toFixed(2)} 速度=${state.strip.speed.toFixed(2)}(${speedRule}) 理由=${state.strip.reason || "暂无"}`;
}

function stripLoop(ts) {
  const seconds = ts / 1000;
  const pixels = stripPixelsAt(seconds);
  drawStripPixels(pixels);
  state.stripRAF = requestAnimationFrame(stripLoop);
}

function startStripAnimation() {
  if (state.stripRAF) return;
  if (!wsConnected()) return;
  state.stripRAF = requestAnimationFrame(stripLoop);
}

function wsURL() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${location.host}/v1/ws`;
}

function normalizeWsURL(raw) {
  const value = String(raw || "").trim();
  if (!value) return wsURL();
  if (value.startsWith("ws://") || value.startsWith("wss://")) return value;
  if (value.startsWith("/")) {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    return `${protocol}://${location.host}${value}`;
  }
  return value;
}

function wsConnected() {
  return state.ws && state.ws.readyState === WebSocket.OPEN;
}

function connectWS() {
  if (wsConnected()) {
    state.manualClose = true;
    state.ws.close();
    state.ws = null;
    setWsState(false);
    stopStripAnimation();
    ui.stripMeta.textContent = "WS 未连接，灯带实时画面已暂停";
    appendEvent("[ws] 已手动断开");
    return;
  }

  const endpoint = normalizeWsURL(ui.wsEndpoint.value);
  localStorage.setItem("debug.wsEndpoint", endpoint);
  state.manualClose = false;
  state.ws = new WebSocket(endpoint);
  ui.wsToggle.textContent = "连接中...";
  appendEvent(`[ws] 正在连接 ${endpoint}`);

  state.ws.onopen = () => {
    setWsState(true);
    appendEvent(`[ws] 已连接 ${new Date().toLocaleTimeString()}`);
  };

  state.ws.onclose = (evt) => {
    setWsState(false);
    state.ws = null;
    stopStripAnimation();
    ui.stripMeta.textContent = "WS 未连接，灯带实时画面已暂停";
    if (!state.manualClose) {
      appendEvent(`[ws] 已断开 code=${evt.code} reason=${evt.reason || "n/a"}`, "error");
    }
  };

  state.ws.onerror = () => {
    appendEvent("[ws] 连接异常，请检查反向代理 Upgrade/Connection 头", "error");
  };

  state.ws.onmessage = (message) => {
    try {
      const evt = JSON.parse(message.data);
      appendEvent(`[${evt.type}] @ ${new Date(evt.created_at_unix_ms).toLocaleTimeString()}`);
      if (evt.type === "matrix.updated") drawMatrix(evt.payload);
      if (evt.type === "strip.updated") {
        applyStripFromAnyPayload(evt.payload);
        startStripAnimation();
      }
      if (evt.type === "strip.frame.updated") {
        applyStripFromAnyPayload(evt.payload);
        startStripAnimation();
      }
      if (evt.payload) updateReasonPanel(evt.payload);
      setJSON(evt);
    } catch {
      appendEvent(`[ws] ${message.data}`);
    }
  };
}

ui.wsToggle.addEventListener("click", connectWS);

ui.planBtn.addEventListener("click", async () => {
  try {
    if (!wsConnected()) appendEvent("[场景] 已提交，但 WS 未连接，实时预览可能延迟", "error");
    const data = await api("/v1/voice/command", "POST", {
      user_id: userId(),
      prompt: ui.prompt.value,
      language: ui.language.value,
    });
    appendEvent(`[场景] 已应用，区域数=${data?.vehicle_plan?.zones?.length || 0}`);
  } catch (error) {
    appendEvent(`[场景] ${error.message}`, "error");
  }
});

ui.staticBtn.addEventListener("click", async () => {
  try {
    const data = await api("/v1/app/command", "POST", {
      operation: "matrix_static",
      user_id: userId(),
      prompt: ui.prompt.value,
      width: parseNum(ui.matrixWidth.value, 16),
      height: parseNum(ui.matrixHeight.value, 16),
    });
    drawMatrix(data.matrix);
    appendEvent("[矩阵] 已生成静态图");
  } catch (error) {
    appendEvent(`[矩阵静态] ${error.message}`, "error");
  }
});

ui.animateBtn.addEventListener("click", async () => {
  try {
    const data = await api("/v1/app/command", "POST", {
      operation: "matrix_animate",
      user_id: userId(),
      prompt: ui.prompt.value,
      fps: parseNum(ui.fps.value, 20),
      duration_sec: parseNum(ui.duration.value, 0),
      strict: true,
      width: parseNum(ui.animWidth.value, 16),
      height: parseNum(ui.animHeight.value, 16),
    });
    const frames = Array.isArray(data.frames) ? data.frames : [];
    if (frames.length > 0) {
      playMatrixFrames(frames, data.script?.fps || parseNum(ui.fps.value, 20));
    }
    const infinite = Boolean(data.infinite_duration || (data.script && Number(data.script.duration_sec) <= 0));
    const previewSec = Number(data.preview_duration_sec || data.script?.duration_sec || 0);
    ui.animMeta.textContent = infinite
      ? `无限时长动画已生成（预览 ${previewSec || 8} 秒，共 ${frames.length} 帧），脚本=${data.script?.id || "n/a"}`
      : `已生成 ${frames.length} 帧（时长 ${previewSec} 秒），脚本=${data.script?.id || "n/a"}`;
    appendEvent(`[矩阵动画] 帧数=${frames.length}`);
  } catch (error) {
    appendEvent(`[矩阵动画] ${error.message}`, "error");
  }
});

ui.latestBtn.addEventListener("click", async () => {
  try {
    const data = await api("/v1/app/command", "POST", { operation: "matrix_latest" });
    drawMatrix(data.matrix);
    appendEvent("[矩阵] 已读取最新");
  } catch (error) {
    appendEvent(`[矩阵最新] ${error.message}`, "error");
  }
});

ui.downsampleBtn.addEventListener("click", async () => {
  try {
    const file = ui.imageInput.files?.[0];
    if (!file) throw new Error("请先选择图片");
    const form = new FormData();
    form.append("operation", "matrix_upload");
    form.append("image", file);
    form.append("user_id", userId());
    form.append("width", String(parseNum(ui.matrixWidth.value, 16)));
    form.append("height", String(parseNum(ui.matrixHeight.value, 16)));
    const data = await api("/v1/app/command", "POST", form, true);
    drawMatrix(data.matrix);
    appendEvent("[图片下采样] 处理完成");
  } catch (error) {
    appendEvent(`[图片下采样] ${error.message}`, "error");
  }
});

ui.stripBtn.addEventListener("click", async () => {
  try {
    if (!wsConnected()) {
      appendEvent("[灯带] WS 未连接，不渲染本地灯带画面；请先连接 WS", "error");
    }
    const manualColor = hexToRgb(ui.stripColor.value);
    await api("/v1/app/command", "POST", {
      operation: "manual_strip",
      user_id: userId(),
      mode: ui.stripMode.value,
      brightness: parseNum(ui.brightness.value, 0.6),
      speed: parseNum(ui.speed.value, 0.6),
      led_count: parseNum(ui.stripLedCount.value, 24),
      color: manualColor,
    });
    appendEvent("[灯带] 手动参数已下发");
  } catch (error) {
    appendEvent(`[灯带] ${error.message}`, "error");
  }
});

const savedWs = localStorage.getItem("debug.wsEndpoint");
ui.wsEndpoint.value = savedWs || wsURL();
ui.matrixBlurToggle.checked = state.matrixBlur.enabled;
ui.matrixBlurStrength.value = String(state.matrixBlur.strength);

ui.matrixBlurToggle.addEventListener("change", () => {
  state.matrixBlur.enabled = ui.matrixBlurToggle.checked;
  localStorage.setItem("debug.matrixBlurEnabled", state.matrixBlur.enabled ? "1" : "0");
  renderMatrixBlur();
});

ui.matrixBlurStrength.addEventListener("input", () => {
  state.matrixBlur.strength = clamp(parseNum(ui.matrixBlurStrength.value, 6), 1, 20);
  localStorage.setItem("debug.matrixBlurStrength", String(state.matrixBlur.strength));
  if (state.matrixBlur.enabled) {
    renderMatrixBlur();
  }
});

setWsState(false);
ui.stripMeta.textContent = "WS 未连接，等待实时灯带事件...";
renderMatrixBlur();
appendEvent("调试台已就绪");
