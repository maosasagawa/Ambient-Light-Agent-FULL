# Auto-extracted from main.py for readability.

PROMPT_UI_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>提示词管理台</title>
  <style>
    * { box-sizing: border-box; }
    :root {
      --bg: #0b1020;
      --panel: rgba(255,255,255,0.06);
      --panel2: rgba(255,255,255,0.09);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --border: rgba(255,255,255,0.14);
      --accent: #6ee7ff;
      --accent2: #a78bfa;
      --ok: #34d399;
      --danger: #fb7185;
      --radius: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial;
    }

    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(1200px 800px at 15% 10%, rgba(167, 139, 250, 0.22), transparent 55%),
        radial-gradient(1200px 800px at 85% 30%, rgba(110, 231, 255, 0.18), transparent 60%),
        var(--bg);
      min-height: 100vh;
    }

    .wrap {
      max-width: 1180px;
      margin: 28px auto;
      padding: 0 18px 40px;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 16px;
    }

    h1 {
      margin: 0;
      font-size: 22px;
    }

    .sub {
      font-size: 13px;
      color: var(--muted);
    }

    .pill {
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
      background: rgba(0,0,0,0.2);
    }

    .pill a { color: var(--text); text-decoration: none; }

    .grid {
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 14px;
    }

    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px;
    }

    label {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }

    input, select, textarea {
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(0,0,0,0.2);
      color: var(--text);
      padding: 8px 10px;
      outline: none;
      font-family: var(--sans);
    }

    textarea { min-height: 220px; resize: vertical; font-family: var(--mono); font-size: 12px; }

    button {
      cursor: pointer;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(0,0,0,0.2);
      color: var(--text);
      padding: 8px 10px;
      font-weight: 600;
    }

    button.primary {
      background: linear-gradient(135deg, rgba(110,231,255,0.22), rgba(167,139,250,0.22));
      border-color: rgba(110,231,255,0.3);
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .stack { display: grid; gap: 10px; }

    .status {
      font-size: 12px;
      color: var(--muted);
    }

    .status.ok { color: var(--ok); }
    .status.bad { color: var(--danger); }

    pre {
      margin: 0;
      padding: 12px;
      border-radius: 12px;
      background: rgba(0,0,0,0.28);
      border: 1px solid rgba(255,255,255,0.12);
      font-family: var(--mono);
      font-size: 12px;
      color: rgba(255,255,255,0.85);
      white-space: pre-wrap;
    }

    .mini { font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>提示词管理台</h1>
        <div class="sub">集中维护提示词模板、版本切换与 A/B 测试。</div>
      </div>
      <div class="pill">
        <a href="/ui" target="_blank" rel="noreferrer">调试台</a>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="stack">
          <label>
            提示词 Key
            <select id="promptKey"></select>
          </label>
          <button id="addPromptBtn">新增提示词</button>
          <label>
            版本
            <select id="variantSelect"></select>
          </label>
          <label>
            权重
            <input id="variantWeight" type="number" min="0" step="0.1" />
          </label>
          <div class="row">
            <button id="addVariantBtn">新增版本</button>
            <button class="primary" id="saveVariantBtn">保存模板</button>
          </div>
          <label>
            激活版本
            <input id="activeVariant" placeholder="例如 v1" />
          </label>
          <label style="display:flex; align-items:center; gap:8px;">
            <input type="checkbox" id="abTestToggle" /> 启用 A/B 分流
          </label>
          <button id="saveStateBtn">保存版本配置</button>
          <div class="status" id="statusText">就绪</div>
        </div>
      </div>

      <div class="card">
        <div class="stack">
          <label>
            模板内容
            <textarea id="variantTemplate" placeholder="在这里编辑模板内容"></textarea>
          </label>
          <div class="row">
            <label>
              预览变量 (JSON)
              <textarea id="previewVars" style="min-height:120px;">{}</textarea>
            </label>
            <div class="stack">
              <label>
                A/B Seed
                <input id="previewSeed" placeholder="可选" />
              </label>
              <button class="primary" id="previewBtn">预览提示词</button>
              <div class="mini" id="previewMeta">-</div>
              <pre id="previewOutput">-</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
  const $ = (id) => document.getElementById(id);
  const els = {
    promptKey: $("promptKey"),
    addPromptBtn: $("addPromptBtn"),
    variantSelect: $("variantSelect"),
    variantWeight: $("variantWeight"),
    variantTemplate: $("variantTemplate"),
    addVariantBtn: $("addVariantBtn"),
    saveVariantBtn: $("saveVariantBtn"),
    activeVariant: $("activeVariant"),
    abTestToggle: $("abTestToggle"),
    saveStateBtn: $("saveStateBtn"),
    statusText: $("statusText"),
    previewVars: $("previewVars"),
    previewSeed: $("previewSeed"),
    previewBtn: $("previewBtn"),
    previewOutput: $("previewOutput"),
    previewMeta: $("previewMeta"),
  };

  let store = { prompts: {} };
  let state = { variants: {}, ab_test: false };

  function setStatus(text, ok = null) {
    els.statusText.textContent = text;
    els.statusText.classList.remove("ok", "bad");
    if (ok === true) els.statusText.classList.add("ok");
    if (ok === false) els.statusText.classList.add("bad");
  }

  async function getJson(url) {
    const r = await fetch(url);
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = {}; }
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return data;
  }

  async function postJson(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = {}; }
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return data;
  }

  function getCurrentEntry() {
    const key = els.promptKey.value;
    const entry = store.prompts[key] || { variants: [] };
    if (!Array.isArray(entry.variants)) entry.variants = [];
    return { key, entry };
  }

  function syncVariantForm() {
    const { entry } = getCurrentEntry();
    const variantId = els.variantSelect.value;
    const variant = entry.variants.find((v) => String(v.id) === String(variantId));
    if (!variant) return;
    els.variantTemplate.value = variant.template || "";
    els.variantWeight.value = variant.weight ?? 1;
  }

  function refreshVariantList() {
    const { entry, key } = getCurrentEntry();
    els.variantSelect.innerHTML = "";
    entry.variants.forEach((variant) => {
      const option = document.createElement("option");
      option.value = variant.id;
      option.textContent = `${variant.id}`;
      els.variantSelect.appendChild(option);
    });

    if (!entry.variants.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "(无版本)";
      els.variantSelect.appendChild(option);
    }

    const active = state.variants[key] || entry.variants[0]?.id || "";
    els.variantSelect.value = active || els.variantSelect.value;
    els.activeVariant.value = state.variants[key] || "";
    syncVariantForm();
  }

  function refreshPromptKeys() {
    const keys = Object.keys(store.prompts || {}).sort();
    els.promptKey.innerHTML = "";
    keys.forEach((key) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = key;
      els.promptKey.appendChild(option);
    });
    if (!keys.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "(无提示词)";
      els.promptKey.appendChild(option);
    }
    refreshVariantList();
  }

  async function loadAll() {
    try {
      store = await getJson("/api/prompts/store");
      state = await getJson("/api/prompts/state");
      if (!store.prompts) store.prompts = {};
      if (!state.variants) state.variants = {};
      els.abTestToggle.checked = !!state.ab_test;
      refreshPromptKeys();
      setStatus("已加载", true);
    } catch (e) {
      setStatus(`加载失败：${e.message}`, false);
    }
  }

  async function saveVariant() {
    const { key, entry } = getCurrentEntry();
    const variantId = els.variantSelect.value;
    const variant = entry.variants.find((v) => String(v.id) === String(variantId));
    if (!variant) {
      setStatus("请选择版本", false);
      return;
    }
    variant.template = els.variantTemplate.value;
    variant.weight = Number(els.variantWeight.value || 1);
    store.prompts[key] = entry;

    try {
      await postJson("/api/prompts/store", store);
      setStatus("模板已保存", true);
    } catch (e) {
      setStatus(`保存失败：${e.message}`, false);
    }
  }

  async function addVariant() {
    const { key, entry } = getCurrentEntry();
    const newId = prompt("新版本 id：");
    if (!newId) return;
    entry.variants.push({ id: newId, weight: 1, template: "" });
    store.prompts[key] = entry;
    refreshVariantList();
    els.variantSelect.value = newId;
    syncVariantForm();
  }

  async function addPromptKey() {
    const key = prompt("新提示词 key：");
    if (!key) return;
    if (!store.prompts) store.prompts = {};
    if (!store.prompts[key]) {
      store.prompts[key] = { variants: [{ id: "v1", weight: 1, template: "" }] };
    }
    refreshPromptKeys();
    els.promptKey.value = key;
    refreshVariantList();
  }

  async function saveState() {
    const { key } = getCurrentEntry();
    state.variants = state.variants || {};
    if (els.activeVariant.value.trim()) {
      state.variants[key] = els.activeVariant.value.trim();
    } else {
      delete state.variants[key];
    }
    state.ab_test = !!els.abTestToggle.checked;

    try {
      const res = await postJson("/api/prompts/state", state);
      state = res;
      setStatus("版本配置已保存", true);
    } catch (e) {
      setStatus(`保存失败：${e.message}`, false);
    }
  }

  async function previewPrompt() {
    const { key } = getCurrentEntry();
    if (!key) {
      setStatus("请选择提示词", false);
      return;
    }
    let variables = {};
    try {
      variables = JSON.parse(els.previewVars.value || "{}");
    } catch (e) {
      setStatus("预览变量 JSON 无效", false);
      return;
    }
    try {
      const res = await postJson("/api/prompts/preview", {
        key,
        variables,
        seed: els.previewSeed.value || null,
      });
      els.previewOutput.textContent = res.prompt || "-";
      els.previewMeta.textContent = `版本：${res.variant_id}`;
      setStatus("预览完成", true);
    } catch (e) {
      setStatus(`预览失败：${e.message}`, false);
    }
  }

  els.promptKey.addEventListener("change", refreshVariantList);
  els.variantSelect.addEventListener("change", syncVariantForm);
  els.saveVariantBtn.addEventListener("click", saveVariant);
  els.addVariantBtn.addEventListener("click", addVariant);
  els.addPromptBtn.addEventListener("click", addPromptKey);
  els.saveStateBtn.addEventListener("click", saveState);
  els.previewBtn.addEventListener("click", previewPrompt);

  loadAll();
</script>
</body>
</html>
"""

DEBUG_UI_HTML = r"""

<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>氛围灯调试台</title>
  <!-- Three.js removed (using CSS gradients for strip preview) -->
  <style>
    * { box-sizing: border-box; }
    code { font-family: var(--mono); font-size: 12px; }
    :root {
      --bg: #0b1020;
      --panel: rgba(255,255,255,0.06);
      --panel2: rgba(255,255,255,0.09);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --border: rgba(255,255,255,0.14);
      --accent: #6ee7ff;
      --accent2: #a78bfa;
      --danger: #fb7185;
      --ok: #34d399;
      --shadow: 0 20px 50px rgba(0,0,0,0.35);
      --radius: 16px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }

    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(1200px 800px at 15% 10%, rgba(167, 139, 250, 0.22), transparent 55%),
        radial-gradient(1200px 800px at 85% 30%, rgba(110, 231, 255, 0.18), transparent 60%),
        radial-gradient(900px 700px at 50% 90%, rgba(52, 211, 153, 0.10), transparent 60%),
        var(--bg);
      min-height: 100vh;
    }

    .wrap {
      max-width: 1180px;
      margin: 28px auto;
      padding: 0 18px 38px;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }

    .title {
      display: grid;
      gap: 6px;
    }

    h1 {
      font-size: 22px;
      margin: 0;
      letter-spacing: 0.2px;
    }

    .sub {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.4;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.18);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
      box-shadow: var(--shadow);
    }

    .pill a { color: var(--text); text-decoration: none; border-bottom: 1px dashed rgba(255,255,255,0.3); }

    .grid {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 14px;
    }

    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .card .hd {
      padding: 14px 14px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
    }

    .card .hd .t {
      font-weight: 650;
      font-size: 14px;
    }

    .card .hd .hint {
      font-size: 12px;
      color: var(--muted);
    }

    /* New styles for speakable reason */
    .speakable-box {
        margin: 12px 0 6px;
        padding: 12px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(110,231,255,0.10));
        border: 1px solid rgba(167,139,250,0.25);
        color: #e0e7ff;
        font-size: 15px;
        line-height: 1.5;
        position: relative;
    }
    .speakable-label {
        font-size: 11px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.5);
        margin-bottom: 4px;
        letter-spacing: 0.5px;
        font-weight: 700;
    }

    .card .bd { padding: 14px; }

    textarea {
      width: 100%;
      height: 110px;
      resize: vertical;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(0,0,0,0.20);
      color: var(--text);
      padding: 12px;
      font-family: var(--sans);
      line-height: 1.5;
      outline: none;
    }

    textarea:focus { border-color: rgba(110,231,255,0.55); box-shadow: 0 0 0 3px rgba(110,231,255,0.12); }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }

    @media (max-width: 520px) {
      .row { grid-template-columns: 1fr; }
    }

    label {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }

    input, select {
      height: 38px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(0,0,0,0.20);
      color: var(--text);
      padding: 0 10px;
      outline: none;
    }

    input:focus, select:focus { border-color: rgba(167,139,250,0.6); box-shadow: 0 0 0 3px rgba(167,139,250,0.13); }

    .btns {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
      align-items: center;
    }

    button {
      cursor: pointer;
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(0,0,0,0.22);
      color: var(--text);
      font-weight: 600;
      letter-spacing: 0.2px;
      transition: transform 0.05s ease, background 0.2s ease, border-color 0.2s ease;
    }

    button:hover { background: rgba(255,255,255,0.10); border-color: rgba(255,255,255,0.22); }
    button:active { transform: translateY(1px); }

    .primary {
      background: linear-gradient(135deg, rgba(110,231,255,0.22), rgba(167,139,250,0.22));
      border-color: rgba(110,231,255,0.25);
    }

    .danger { border-color: rgba(251,113,133,0.35); }

    .meta {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }

    .statusDot {
      width: 9px;
      height: 9px;
      border-radius: 99px;
      background: rgba(255,255,255,0.35);
      display: inline-block;
      margin-right: 6px;
    }

    .ok { background: rgba(52,211,153,0.85); }
    .bad { background: rgba(251,113,133,0.85); }

    .split {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 12px;
    }

    @media (max-width: 980px) {
      .split { grid-template-columns: 1fr; }
    }

    .previewBox {
      background: var(--panel2);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 14px;
      padding: 12px;
    }

    #stripPreview {
      margin-top: 10px;
      width: 100%;
      height: 36px;
      border-radius: 18px; /* Pill shape for strip */
      border: 1px solid rgba(255,255,255,0.1);
      background: #050505;
      box-shadow: inset 0 2px 8px rgba(0,0,0,0.8); /* Inner shadow for depth */
      position: relative;
      overflow: hidden;
    }
    
    #stripCanvas {
      width: 100%;
      height: 100%;
      border-radius: 18px;
      image-rendering: auto; /* Smooth rendering for gradients */
    }
    
    /* "Diffuser" overlay for realistic LED strip effect */
    #stripPreview::after {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: linear-gradient(to bottom, 
        rgba(255,255,255,0.15) 0%, 
        rgba(255,255,255,0) 40%, 
        rgba(0,0,0,0.1) 70%,
        rgba(0,0,0,0.3) 100%);
      pointer-events: none;
      border-radius: 18px;
    }

    #matrixCanvas {
      width: 100%;
      max-width: 320px;
      height: auto;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: #000;
      display: block;
      image-rendering: pixelated;
      --matrix-blur: 8px;
    }

    #matrixCanvas.matrix-blur {
      filter: blur(var(--matrix-blur));
      transform: scale(0.92);
    }

    .swatches {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
      align-items: stretch;
    }

    .swatch {
      display: grid;
      gap: 6px;
      padding: 10px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(0,0,0,0.18);
      flex: 1 1 140px;
      max-width: 220px;
      min-width: 0;
      overflow: hidden;
    }

    .swatch span {
      overflow-wrap: anywhere;
    }

    .chip {
      height: 26px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.10);
      background: #222;
    }

    .kv {
      display: grid;
      gap: 6px;
      margin-top: 10px;
      font-size: 13px;
    }

    .kv b { font-weight: 650; }

    pre {
      margin: 0;
      padding: 12px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(0,0,0,0.24);
      overflow: auto;
      max-height: 320px;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.4;
      color: rgba(255,255,255,0.85);
    }

    .mini {
      font-size: 12px;
      color: var(--muted);
    }

    .section {
      margin-top: 16px;
      padding-top: 12px;
      border-top: 1px dashed rgba(255,255,255,0.12);
    }

    .section-title {
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 8px;
    }

    .inline {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .input-sm {
      height: 32px;
      border-radius: 10px;
    }

    .log {
      margin-top: 8px;
      padding: 10px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(0,0,0,0.22);
      max-height: 160px;
      overflow: auto;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.4;
      color: rgba(255,255,255,0.85);
    }

    .kb-list {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }

    .kb-item {
      margin: 0;
      max-height: 180px;
    }

    .saved-anim-panel {
      margin-top: 10px;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      overflow: hidden;
    }
    .saved-anim-panel summary {
      cursor: pointer;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 650;
      color: var(--muted);
      background: rgba(0,0,0,0.18);
      user-select: none;
      list-style: none;
    }
    .saved-anim-panel summary::-webkit-details-marker { display: none; }
    .saved-anim-panel[open] summary { border-bottom: 1px solid rgba(255,255,255,0.08); }
    .saved-anim-list {
      max-height: 200px;
      overflow-y: auto;
      padding: 6px;
      display: grid;
      gap: 6px;
    }
    .saved-anim-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(0,0,0,0.15);
      font-size: 12px;
    }
    .saved-anim-item .anim-label {
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--text);
    }
    .saved-anim-item .anim-ts {
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
    }
    .saved-anim-item button { padding: 4px 8px; font-size: 11px; border-radius: 8px; }
    .saved-anim-empty { padding: 10px 12px; font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="title">
        <h1>氛围灯调试台</h1>
        <div class="sub">用于联调 / 预览：调用 <code>/api/voice/submit</code>（先返回口播文案，后台执行生图+落盘），并通过 <code>/api/data/*</code> 预览矩阵与灯带数据。</div>
      </div>
      <div class="pill">
        <span>快捷：</span>
        <a href="/docs" target="_blank" rel="noreferrer">OpenAPI 文档</a>
        <span>·</span>
        <a href="/api/data/matrix/json" target="_blank" rel="noreferrer">当前矩阵</a>
        <span>·</span>
        <a href="/api/data/strip" target="_blank" rel="noreferrer">当前灯带</a>
        <span>·</span>
        <a href="/ui/prompts" target="_blank" rel="noreferrer">提示词管理</a>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="hd">
          <div class="t">请求</div>
          <div class="hint">提交后会触发生图+落盘（可能较慢）</div>
        </div>
        <div class="bd">
          <label>
            指令（自然语言）
            <textarea id="instruction" placeholder="例如：营造一个温暖放松的氛围；矩阵显示落日海边的像素风场景"></textarea>
          </label>


          <div class="btns">
            <button class="primary" id="runBtn">运行</button>
            <button id="loadCurrentBtn">读取当前硬件数据</button>
            <button class="danger" id="clearBtn">清空</button>
          </div>

          <div class="meta" id="meta">
            <span><span class="statusDot" id="dot"></span><span id="statusText">就绪</span></span>
            <span>耗时：<span id="elapsed">-</span></span>
            <span class="mini">提示：真实生成可能较慢（生图/网络请求）</span>
          </div>

          <div class="section">
            <div class="section-title">图片下采样</div>
            <label>
              选择图片
              <input id="imageFile" type="file" accept="image/*" />
            </label>
            <div class="row">
              <label>
                宽度
                <input class="input-sm" id="matrixWidth" type="number" min="1" max="64" value="16" />
              </label>
              <label>
                高度
                <input class="input-sm" id="matrixHeight" type="number" min="1" max="64" value="16" />
              </label>
            </div>
            <div class="inline" style="margin-top:8px;">
              <label style="display:flex; align-items:center; gap:8px; font-size:12px; color: var(--muted);">
                <input type="checkbox" id="includeRaw" checked /> 包含 raw_base64
              </label>
            </div>
            <div class="btns">
              <button id="downsampleBtn">上传并下采样</button>
            </div>
            <div class="mini" id="downsampleHint">支持 PNG/JPG/WEBP，最大 10MB（可配）。</div>
          </div>

          <div class="section">
            <div class="section-title">矩阵动画</div>
            <label>
              动画指令（可独立）
              <input id="matrixAnimInstruction" placeholder="例如：像素风霓虹波纹" />
            </label>
            <div class="row">
              <label>
                FPS
                <input class="input-sm" id="matrixFps" type="number" min="1" max="60" step="1" value="12" />
              </label>
              <label>
                持续时间 (秒)
                <input class="input-sm" id="matrixDuration" type="number" min="0" max="300" step="0.5" value="0" />
              </label>
            </div>
            <div class="inline" style="margin-top:8px;">
              <label style="display:flex; align-items:center; gap:8px; font-size:12px; color: var(--muted);">
                <input type="checkbox" id="matrixStoreFrames" checked /> 落盘完整帧序列
              </label>
            </div>
            <div class="btns">
              <button class="primary" id="matrixAnimateBtn">生成动画</button>
              <button id="matrixStopBtn">停止动画</button>
              <button id="matrixSaveBtn">保存动画</button>
              <input id="matrixAnimSaveName" type="text" placeholder="动画名称（可选）" maxlength="100" style="flex:1; min-width:80px; font-size:12px; padding:4px 8px; border-radius:8px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12); color:inherit;" />
            </div>
            <div class="mini" id="matrixAnimHint">使用当前矩阵宽高；持续时间填 0 可循环播放（需手动停止）。</div>
            <div class="mini" id="matrixAnimError" style="color: var(--danger);">-</div>
            <details class="saved-anim-panel" id="savedAnimPanel">
              <summary>▸ 已保存动画 <span id="savedAnimCount" style="margin-left:4px; font-weight:400;"></span></summary>
              <div class="saved-anim-list" id="savedAnimList">
                <div class="saved-anim-empty">展开后自动加载列表</div>
              </div>
              <div class="btns" style="padding: 6px 10px 10px;">
                <button id="savedAnimRefreshBtn" style="font-size:11px; padding:4px 10px;">刷新列表</button>
              </div>
            </details>
          </div>

          <div class="section">
            <div class="section-title">灯带指令</div>
            <div class="row">
              <label>
                模式
                <select id="stripMode">
                  <option value="static">static (静态)</option>
                  <option value="breath">breath (呼吸)</option>
                  <option value="flow">flow (流动)</option>
                  <option value="chase">chase (流星/追逐)</option>
                  <option value="pulse">pulse (脉冲)</option>
                  <option value="wave">wave (波浪)</option>
                  <option value="sparkle">sparkle (闪烁)</option>
                </select>
              </label>
              <label>
                LED 数量
                <input class="input-sm" id="stripLedCount" type="number" min="1" max="2000" value="60" />
              </label>
            </div>
            <div class="row">
              <label>
                亮度
                <input class="input-sm" id="stripBrightness" type="number" min="0" max="1" step="0.05" value="1" />
              </label>
              <label>
                速度
                <input class="input-sm" id="stripSpeed" type="number" min="0.1" step="0.1" value="2" />
              </label>
            </div>
            <label>
              颜色 (rgb; 分号分隔)
              <input id="stripColors" placeholder="255,140,60;255,160,190" />
            </label>
            <div class="btns">
              <button class="primary" id="stripApplyBtn">下发指令</button>
              <button id="stripLoadBtn">读取当前指令</button>
              <button id="stripPreviewStartBtn">开始预览</button>
              <button id="stripPreviewStopBtn">停止预览</button>
            </div>
            <div class="mini" id="stripCmdHint">颜色为空时保持当前颜色；chase (流星/追逐) 模式支持多色渐变尾迹。</div>
          </div>

          <div class="section">
            <div class="section-title">硬件帧</div>
            <div class="inline">
              <label>
                LED 数量
                <input class="input-sm" id="frameLedCount" type="number" min="1" max="2000" value="60" />
              </label>
              <button id="fetchFrameJsonBtn">读取 JSON 帧</button>
              <button id="fetchFrameRawBtn">读取 RAW 帧</button>
            </div>
            <div class="mini" id="frameInfo">-</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="hd">
          <div class="t">预览</div>
          <div class="hint">矩阵：Canvas 自适应 · 灯带：色块</div>
        </div>
        <div class="bd">
          
          <!-- Speakable Reason Display -->
          <div class="speakable-box" id="speakableBox" style="display:none">
              <div class="speakable-label">AI 口语反馈 (TTS)</div>
              <div id="speakableText"></div>
          </div>

          <div class="split" style="margin-top:12px; display:block;"> <!-- Remove split grid, stack them -->
            <div class="previewBox" style="margin-bottom: 12px;">
              <div class="kv"><b>矩阵预览</b><span class="mini" id="matrixMeta">-</span></div>
              <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                  <div>
                    <canvas id="matrixCanvas" class="matrix-blur" width="16" height="16"></canvas>
                    <div class="mini" style="margin-top:6px; display:flex; align-items:center; gap:10px;">
                        <label style="display:inline-flex; align-items:center; gap:6px;">
                        <input type="checkbox" id="matrixBlurToggle" checked />
                        高斯模糊预览
                        </label>
                        <label style="display:inline-flex; align-items:center; gap:6px;">
                        强度
                        <input type="range" id="matrixBlurAmount" min="0" max="16" step="1" value="8" />
                        </label>
                    </div>
                  </div>
                  <div style="flex: 1; min-width: 200px;">
                    <div class="kv">
                        <div><b>技术原理</b>：<span id="matrixScene">-</span></div>
                        <div><b>理由</b>：<span id="matrixReason">-</span></div>
                    </div>
                  </div>
              </div>
            </div>

            <div class="previewBox">
              <div class="kv"><b>灯带预览</b> <span class="mini" id="stripMeta">-</span></div>
              <div id="stripPreview" style="margin-top:10px;">
                <canvas id="stripCanvas"></canvas>
              </div>
              
              <div class="kv" style="margin-top:10px;">
                <div><b>主题</b>：<span id="stripTheme">-</span></div>
                <div><b>理由</b>：<span id="stripReason">-</span></div>
              </div>
              <div class="swatches" id="swatches"></div>
              <div class="mini" id="stripHint" style="margin-top:10px;">-</div>
            </div>

            <div class="previewBox" style="margin-top: 12px;">
              <div class="kv"><b>本次知识库参考</b><span class="mini" id="kbMeta">尚未检索</span></div>
              <div class="kb-list" id="kbReferences"></div>
            </div>
          </div>

          <div style="margin-top:12px;" class="previewBox">
            <div class="kv"><b>原始响应（调试）</b><span class="mini">可用于粘贴给后端定位问题</span></div>
            <pre id="raw">{}</pre>
          </div>

          <div style="margin-top:12px;" class="previewBox">
            <div class="kv"><b>WebSocket 监控</b><span class="mini" id="wsStatus">未连接</span></div>
            <div class="log" id="wsLog">-</div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
  const $ = (id) => document.getElementById(id);

  const els = {
    instruction: $("instruction"),
    runBtn: $("runBtn"),
    clearBtn: $("clearBtn"),
    loadCurrentBtn: $("loadCurrentBtn"),
    dot: $("dot"),
    statusText: $("statusText"),
    elapsed: $("elapsed"),
    raw: $("raw"),
    matrixCanvas: $("matrixCanvas"),
    matrixBlurToggle: $("matrixBlurToggle"),
    matrixBlurAmount: $("matrixBlurAmount"),
    matrixMeta: $("matrixMeta"),
    matrixScene: $("matrixScene"),
    matrixReason: $("matrixReason"),
    stripTheme: $("stripTheme"),
    stripReason: $("stripReason"),
    swatches: $("swatches"),
    stripHint: $("stripHint"),
    stripMeta: $("stripMeta"),
    kbMeta: $("kbMeta"),
    kbReferences: $("kbReferences"),
    speakableBox: $("speakableBox"),
    speakableText: $("speakableText"),
    imageFile: $("imageFile"),
    matrixWidth: $("matrixWidth"),
    matrixHeight: $("matrixHeight"),
    includeRaw: $("includeRaw"),
    downsampleBtn: $("downsampleBtn"),
    downsampleHint: $("downsampleHint"),
    matrixFps: $("matrixFps"),
    matrixDuration: $("matrixDuration"),
    matrixStoreFrames: $("matrixStoreFrames"),
    matrixAnimInstruction: $("matrixAnimInstruction"),
    matrixAnimateBtn: $("matrixAnimateBtn"),
    matrixStopBtn: $("matrixStopBtn"),
    matrixSaveBtn: $("matrixSaveBtn"),
    matrixAnimSaveName: $("matrixAnimSaveName"),
    matrixAnimHint: $("matrixAnimHint"),
    matrixAnimError: $("matrixAnimError"),
    savedAnimPanel: $("savedAnimPanel"),
    savedAnimList: $("savedAnimList"),
    savedAnimCount: $("savedAnimCount"),
    savedAnimRefreshBtn: $("savedAnimRefreshBtn"),
    stripMode: $("stripMode"),
    stripLedCount: $("stripLedCount"),
    stripBrightness: $("stripBrightness"),
    stripSpeed: $("stripSpeed"),
    stripColors: $("stripColors"),
    stripApplyBtn: $("stripApplyBtn"),
    stripLoadBtn: $("stripLoadBtn"),
    stripPreviewStartBtn: $("stripPreviewStartBtn"),
    stripPreviewStopBtn: $("stripPreviewStopBtn"),
    stripCmdHint: $("stripCmdHint"),
    stripPreview: $("stripPreview"),
    frameLedCount: $("frameLedCount"),
    fetchFrameJsonBtn: $("fetchFrameJsonBtn"),
    fetchFrameRawBtn: $("fetchFrameRawBtn"),
    frameInfo: $("frameInfo"),
    wsStatus: $("wsStatus"),
    wsLog: $("wsLog"),
  };

  const wsLogEntries = [];

  function setMatrixBlur(enabled) {
    if (!els.matrixCanvas) return;
    els.matrixCanvas.classList.toggle("matrix-blur", !!enabled);
  }

  function setMatrixBlurAmount(value) {
    if (!els.matrixCanvas) return;
    const amount = Number(value);
    const clamped = Number.isFinite(amount) ? Math.max(0, Math.min(16, amount)) : 0;
    els.matrixCanvas.style.setProperty("--matrix-blur", `${clamped}px`);
  }

  if (els.matrixBlurToggle) {
    setMatrixBlur(els.matrixBlurToggle.checked);
    els.matrixBlurToggle.addEventListener("change", () => {
      setMatrixBlur(els.matrixBlurToggle.checked);
    });
  }

  if (els.matrixBlurAmount) {
    setMatrixBlurAmount(els.matrixBlurAmount.value);
    els.matrixBlurAmount.addEventListener("input", () => {
      setMatrixBlurAmount(els.matrixBlurAmount.value);
    });
  }

  function setStatus(text, ok=null) {
    els.statusText.textContent = text;
    els.dot.classList.remove("ok", "bad");
    if (ok === true) els.dot.classList.add("ok");
    if (ok === false) els.dot.classList.add("bad");
  }

  function setMatrixCanvasAspect(width, height) {
    if (!els.matrixCanvas) return;
    const w = Math.max(1, Number(width) || 1);
    const h = Math.max(1, Number(height) || 1);
    els.matrixCanvas.style.aspectRatio = `${w} / ${h}`;
  }

  function drawMatrix(matrixJson) {
    const canvas = els.matrixCanvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!matrixJson || !matrixJson.pixels) {
      setMatrixCanvasAspect(1, 1);
      els.matrixMeta.textContent = "无数据";
      return;
    }

    const w = matrixJson.width || 16;
    const h = matrixJson.height || 16;
    const pixels = matrixJson.pixels;

    canvas.width = w;
    canvas.height = h;
    setMatrixCanvasAspect(w, h);

    // Render directly: data row 0 maps to canvas row 0
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const rgb = (pixels[y] && pixels[y][x]) ? pixels[y][x] : [0,0,0];
        ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
        ctx.fillRect(x, y, 1, 1);
      }
    }

    els.matrixMeta.textContent = `${w}×${h}`;
  }

  function drawMatrixFromRaw(rawBase64, width, height) {
    if (!rawBase64) return;
    const w = Number(width || 16);
    const h = Number(height || 16);
    const canvas = els.matrixCanvas;
    const ctx = canvas.getContext("2d");

    canvas.width = w;
    canvas.height = h;
    setMatrixCanvasAspect(w, h);

    const bytes = Uint8Array.from(atob(rawBase64), (c) => c.charCodeAt(0));
    const imageData = ctx.createImageData(w, h);

    // Render directly: data row 0 maps to canvas row 0
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const srcIdx = (y * w + x) * 3;
        const dstIdx = (y * w + x) * 4;
        imageData.data[dstIdx] = bytes[srcIdx] || 0;
        imageData.data[dstIdx + 1] = bytes[srcIdx + 1] || 0;
        imageData.data[dstIdx + 2] = bytes[srcIdx + 2] || 0;
        imageData.data[dstIdx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
    els.matrixMeta.textContent = `${w}×${h}`;
  }

  function renderStripFromSelection(finalSelection) {
    els.swatches.innerHTML = "";

    if (!Array.isArray(finalSelection) || finalSelection.length === 0) {
      els.stripHint.textContent = "无灯带颜色数据";
      return;
    }

    for (const c of finalSelection) {
      const rgb = c.rgb || [0,0,0];
      const name = c.name || "(unnamed)";
      const div = document.createElement("div");
      div.className = "swatch";
      div.innerHTML = `
        <div class="chip" style="background: rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})"></div>
        <div style="display:flex; justify-content:space-between; gap:10px;">
          <span style="font-weight:650;">${name}</span>
          <span class="mini">rgb(${rgb.join(",")})</span>
        </div>
      `;
      els.swatches.appendChild(div);
    }

    els.stripHint.textContent = `共 ${finalSelection.length} 个色块（此处用于调试预览）`;
  }

  function renderStripFromRgbList(rgbList) {
    const selection = (Array.isArray(rgbList) ? rgbList : []).map((rgb, idx) => ({
      name: `Color ${idx+1}`,
      rgb: rgb,
    }));
    renderStripFromSelection(selection);
  }

  function renderKbReferences(items) {
    if (!els.kbReferences || !els.kbMeta) return;
    els.kbReferences.innerHTML = "";

    const list = Array.isArray(items) ? items.filter((item) => typeof item === "string" && item.trim()) : [];
    if (!list.length) {
      els.kbMeta.textContent = "本次未命中知识库";
      const empty = document.createElement("div");
      empty.className = "mini";
      empty.textContent = "-";
      els.kbReferences.appendChild(empty);
      return;
    }

    els.kbMeta.textContent = `命中 ${list.length} 条`;
    for (const item of list) {
      const pre = document.createElement("pre");
      pre.className = "kb-item";
      pre.textContent = item;
      els.kbReferences.appendChild(pre);
    }
  }

  const stripPreviewState = {
    timer: null,
    loading: false,
    ledCount: 60,
    ws: null,
    fallbackTimer: null,
    pendingFrame: null,
    raf: null,
    canvasWidth: 0,
    canvasHeight: 0,
    canvasDpr: 1,
  };

  function ensureStripPreview(ledCount) {
    const raw = Number(ledCount);
    const normalized = Number.isFinite(raw) ? Math.max(1, Math.min(2000, Math.round(raw))) : 60;
    stripPreviewState.ledCount = normalized;
    if (els.stripLedCount && String(els.stripLedCount.value || "") !== String(normalized)) {
      els.stripLedCount.value = String(normalized);
    }
    if (els.frameLedCount && String(els.frameLedCount.value || "") !== String(normalized)) {
      els.frameLedCount.value = String(normalized);
    }
    if (els.stripMeta) {
      els.stripMeta.textContent = `${normalized} LEDs`;
    }
  }

  function decodeStripBytes(bytes) {
      const count = Math.floor(bytes.length / 3);
      const frame = new Array(count);
      for (let i = 0; i < count; i++) {
          const idx = i * 3;
          frame[i] = [bytes[idx], bytes[idx + 1], bytes[idx + 2]];
      }
      return frame;
  }

  function updateStripPreviewFrame(frame) {
      if (!els.stripPreview) return;
      if (frame instanceof ArrayBuffer) {
          frame = decodeStripBytes(new Uint8Array(frame));
      } else if (frame instanceof Uint8Array) {
          frame = decodeStripBytes(frame);
      }
      if (!Array.isArray(frame)) return;
      if (frame.length === 0) return;

      stripPreviewState.pendingFrame = frame;
      if (stripPreviewState.raf) return;
      stripPreviewState.raf = requestAnimationFrame(drawStripPreviewFrame);
  }

  function drawStripPreviewFrame() {
      stripPreviewState.raf = null;
      const frame = stripPreviewState.pendingFrame;
      if (!frame || !els.stripPreview) return;

      const count = frame.length;
      const canvas = document.getElementById('stripCanvas');
      if (!canvas) return;

      const ctx = canvas.getContext('2d', { alpha: false });
      if (!ctx) return;

      const rect = els.stripPreview.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      if (w <= 0 || h <= 0) return;

      const dpr = window.devicePixelRatio || 1;
      if (w !== stripPreviewState.canvasWidth || h !== stripPreviewState.canvasHeight || dpr !== stripPreviewState.canvasDpr) {
          stripPreviewState.canvasWidth = w;
          stripPreviewState.canvasHeight = h;
          stripPreviewState.canvasDpr = dpr;
          canvas.width = Math.round(w * dpr);
          canvas.height = Math.round(h * dpr);
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // Natural Color Fusion: Use a linear gradient across the entire strip
      const gradient = ctx.createLinearGradient(0, 0, w, 0);

      if (count === 1) {
          const rgb = frame[0] || [0, 0, 0];
          ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
          ctx.fillRect(0, 0, w, h);
      } else {
          const maxStops = 200;
          const step = Math.max(1, Math.ceil(count / maxStops));
          for (let i = 0; i < count; i += step) {
              const rgb = frame[i] || [0, 0, 0];
              const stop = i / (count - 1);
              gradient.addColorStop(stop, `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`);
          }
          if ((count - 1) % step !== 0) {
              const rgb = frame[count - 1] || [0, 0, 0];
              gradient.addColorStop(1, `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`);
          }
          ctx.fillStyle = gradient;
          ctx.fillRect(0, 0, w, h);
      }

      if (els.stripMeta) {
          els.stripMeta.textContent = `${count} LEDs`;
      }
  }


  async function fetchStripPreviewFrame() {
    if (stripPreviewState.loading) return;
    const ledCount = Number(els.stripLedCount.value || 60);

    stripPreviewState.loading = true;
    try {
      const frame = await getJson(`/api/data/strip/frame/json?led_count=${ledCount}`);
      updateStripPreviewFrame(frame);
    } catch (e) {
      // Silent fail; preview is best-effort.
    } finally {
      stripPreviewState.loading = false;
    }
  }

  function startStripPreviewPolling() {
    if (stripPreviewState.timer) return;
    fetchStripPreviewFrame();
    stripPreviewState.timer = setInterval(fetchStripPreviewFrame, 250);
  }

  function stopStripPreviewPolling() {
    if (!stripPreviewState.timer) return;
    if (typeof stripPreviewState.timer === "number") {
      if (stripPreviewState.timer > 1000) {
        cancelAnimationFrame(stripPreviewState.timer);
      } else {
        clearInterval(stripPreviewState.timer);
      }
    }
    stripPreviewState.timer = null;
  }

  function startStripPreviewWs() {
    const ledCount = Number(els.stripLedCount.value || 60);
    const proto = (location.protocol === "https:") ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/strip/raw?fps=30&led_count=${ledCount}&encoding=rgb24`);
    ws.binaryType = "arraybuffer";
    stripPreviewState.ws = ws;

    if (stripPreviewState.fallbackTimer) {
      clearTimeout(stripPreviewState.fallbackTimer);
    }
    stripPreviewState.fallbackTimer = setTimeout(() => {
      stripPreviewState.fallbackTimer = null;
      if (!stripPreviewState.ws || stripPreviewState.ws.readyState !== WebSocket.OPEN) {
        startStripPreviewPolling();
      }
    }, 1200);

    ws.onmessage = (evt) => {
      if (stripPreviewState.fallbackTimer) {
        clearTimeout(stripPreviewState.fallbackTimer);
        stripPreviewState.fallbackTimer = null;
      }
      stopStripPreviewPolling();
      updateStripPreviewFrame(evt.data);
    };

    ws.onclose = () => {
      if (stripPreviewState.ws === ws) {
        stripPreviewState.ws = null;
      }
      startStripPreviewPolling();
    };

    ws.onerror = () => {
      try { ws.close(); } catch (_) {}
    };
  }

  function startStripPreview() {
    stopStripPreview();
    startStripPreviewWs();
  }

  function stopStripPreview() {
    stopStripPreviewPolling();
    if (stripPreviewState.fallbackTimer) {
      clearTimeout(stripPreviewState.fallbackTimer);
      stripPreviewState.fallbackTimer = null;
    }
    if (stripPreviewState.ws) {
      try { stripPreviewState.ws.close(); } catch (_) {}
      stripPreviewState.ws = null;
    }
    if (stripPreviewState.raf) {
      cancelAnimationFrame(stripPreviewState.raf);
      stripPreviewState.raf = null;
    }
    stripPreviewState.pendingFrame = null;
  }

  function safeStr(v) {
    if (v === null || v === undefined) return "-";
    if (typeof v === "string" && v.trim() === "") return "-";
    return String(v);
  }

  function setWsStatus(text) {
    if (els.wsStatus) {
      els.wsStatus.textContent = text;
    }
  }

  function appendWsLog(type, payload) {
    if (!els.wsLog) return;
    const ts = new Date().toLocaleTimeString();
    let summary = "";
    if (payload && typeof payload === "object") {
      summary = JSON.stringify(payload);
      if (summary.length > 400) summary = summary.slice(0, 400) + "…";
    }
    const line = summary ? `[${ts}] ${type} ${summary}` : `[${ts}] ${type}`;
    wsLogEntries.unshift(line);
    wsLogEntries.splice(8);
    els.wsLog.textContent = wsLogEntries.join("\n");
  }

  function parseRgbInput(value) {
    const entries = (value || "").split(";").map((item) => item.trim()).filter(Boolean);
    const colors = [];

    for (const entry of entries) {
      const parts = entry.split(",").map((item) => item.trim()).filter(Boolean);
      if (parts.length !== 3) continue;
      const nums = parts.map((item) => Number(item));
      if (nums.some((n) => Number.isNaN(n))) continue;
      const rgb = nums.map((n) => Math.min(255, Math.max(0, Math.round(n))));
      colors.push({ rgb });
    }

    return colors;
  }

  function formatColorsInput(colors) {
    if (!Array.isArray(colors)) return "";
    return colors
      .map((c) => (Array.isArray(c) ? c : (c.rgb || [])).join(","))
      .filter(Boolean)
      .join(";");
  }

  function updateStripCommandForm(envelope) {
    if (!envelope || !envelope.command) return;
    const cmd = envelope.command || {};
    if (cmd.mode) els.stripMode.value = cmd.mode;
    if (cmd.led_count !== undefined) {
      els.stripLedCount.value = cmd.led_count;
      els.frameLedCount.value = cmd.led_count;
    }
    if (cmd.brightness !== undefined) els.stripBrightness.value = cmd.brightness;
    if (cmd.speed !== undefined) els.stripSpeed.value = cmd.speed;
    if (cmd.colors) els.stripColors.value = formatColorsInput(cmd.colors);
    ensureStripPreview(cmd.led_count || 60);
  }

  async function postJson(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${text}`);
    return data;
  }

  async function getJson(url) {
    const r = await fetch(url);
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${text}`);
    return data;
  }

  async function downsampleImage() {
    const file = els.imageFile.files && els.imageFile.files[0];
    if (!file) {
      setStatus("请选择图片", false);
      return;
    }

    const width = Number(els.matrixWidth.value || 16);
    const height = Number(els.matrixHeight.value || 16);
    const includeRaw = els.includeRaw.checked;

    const form = new FormData();
    form.append("file", file);

    setStatus("上传中…", null);
    els.elapsed.textContent = "-";

    const t0 = performance.now();
    try {
      const url = `/api/matrix/downsample?width=${width}&height=${height}&include_raw=${includeRaw ? "true" : "false"}`;
      const r = await fetch(url, { method: "POST", body: form });
      const text = await r.text();
      let data;
      try { data = JSON.parse(text); } catch { data = { raw: text }; }
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${text}`);

      els.raw.textContent = JSON.stringify(data, null, 2);
      drawMatrix(data.json || null);
      els.matrixScene.textContent = "(上传图片下采样)";
      els.matrixReason.textContent = "-";
      setStatus("下采样完成", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function animateMatrix() {
    const customInstruction = els.matrixAnimInstruction.value.trim();
    const instruction = customInstruction || els.instruction.value.trim();
    if (!instruction) {
      setStatus("请输入指令", false);
      return;
    }

    const width = Number(els.matrixWidth.value || 16);
    const height = Number(els.matrixHeight.value || 16);
    const fps = Number(els.matrixFps.value || 12);
    const durationRaw = els.matrixDuration.value;
    const duration = durationRaw === "" ? 0 : Number(durationRaw);
    const storeFrames = !!els.matrixStoreFrames.checked;

    const payload = {
      instruction,
      width,
      height,
      fps,
      duration_s: duration,
      store_frames: storeFrames,
    };

    setStatus("生成动画脚本…", null);
    els.elapsed.textContent = "-";
    if (els.matrixAnimError) {
      els.matrixAnimError.textContent = "-";
    }
    const t0 = performance.now();

    try {
      const res = await postJson("/api/matrix/animate", payload);
      els.raw.textContent = JSON.stringify(res, null, 2);
      els.matrixScene.textContent = safeStr(res.summary || instruction);
      els.matrixReason.textContent = "动画运行中…";
      if (res.job_id) {
        els.matrixAnimHint.textContent = "动画任务已提交，等待启动";
      } else {
        els.matrixAnimHint.textContent = "动画已启动，等待帧推送";
      }
      setStatus("动画已启动", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
      els.matrixAnimHint.textContent = "动画启动失败";
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function stopMatrixAnimation() {
    setStatus("停止动画…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();

    try {
      const res = await postJson("/api/matrix/animate/stop", {});
      els.raw.textContent = JSON.stringify(res, null, 2);
      els.matrixReason.textContent = "动画已停止";
      els.matrixAnimHint.textContent = "动画已停止";
      setStatus("动画已停止", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function saveMatrixAnimation() {
    setStatus("保存动画…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();
    const name = els.matrixAnimSaveName ? els.matrixAnimSaveName.value.trim() : "";

    try {
      const res = await postJson("/api/matrix/animate/save", name ? { name } : {});
      els.raw.textContent = JSON.stringify(res, null, 2);
      const savedName = res.saved && (res.saved.name || res.saved.id) ? (res.saved.name || res.saved.id) : "-";
      els.matrixAnimHint.textContent = `动画已保存：${savedName}`;
      setStatus("动画已保存", true);
      await loadSavedAnimations();
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
      els.matrixAnimHint.textContent = "动画保存失败";
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  function escHtml(str) {
    return String(str).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
  }

  async function loadSavedAnimations() {
    if (!els.savedAnimList) return;
    try {
      const res = await getJson("/api/matrix/animate/saved");
      const items = (res.items || []).slice().reverse();
      if (els.savedAnimCount) {
        els.savedAnimCount.textContent = items.length ? `(${items.length})` : "";
      }
      if (!items.length) {
        els.savedAnimList.innerHTML = '<div class="saved-anim-empty">暂无已保存动画</div>';
        return;
      }
      els.savedAnimList.innerHTML = items.map(item => {
        const label = item.name || item.instruction || item.id;
        const shortLabel = label.length > 40 ? label.slice(0, 40) + "…" : label;
        const ts = new Date(item.created_at_ms).toLocaleString("zh-CN", { month:"2-digit", day:"2-digit", hour:"2-digit", minute:"2-digit" });
        return `<div class="saved-anim-item">
          <span class="anim-label" title="${escHtml(label)}">${escHtml(shortLabel)}</span>
          <span class="anim-ts">${ts}</span>
          <button onclick="runSavedAnimation('${escHtml(item.id)}')">▶ 运行</button>
          <button onclick="deleteSavedAnimation('${escHtml(item.id)}')" style="border-color:rgba(251,113,133,0.3);">✕</button>
        </div>`;
      }).join("");
    } catch (e) {
      els.savedAnimList.innerHTML = `<div class="saved-anim-empty" style="color:var(--danger)">加载失败：${escHtml(e.message)}</div>`;
    }
  }

  async function runSavedAnimation(id) {
    setStatus("加载已保存动画…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();
    try {
      const fps = Number(els.matrixFps ? els.matrixFps.value : 12) || 12;
      const duration_s = Number(els.matrixDuration ? els.matrixDuration.value : 0) || 0;
      const res = await postJson(`/api/matrix/animate/saved/${encodeURIComponent(id)}/run`, { fps, duration_s });
      els.raw.textContent = JSON.stringify(res, null, 2);
      els.matrixAnimHint.textContent = `已启动保存的动画（job: ${res.job_id || "-"}）`;
      setStatus("已加载并启动", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function deleteSavedAnimation(id) {
    if (!confirm("确认删除这个保存的动画？")) return;
    try {
      const resp = await fetch(`/api/matrix/animate/saved/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || resp.statusText);
      }
      await loadSavedAnimations();
      setStatus("已删除", true);
    } catch (e) {
      setStatus(`删除失败：${e.message}`, false);
    }
  }

  async function applyStripCommand() {
    const colorInput = els.stripColors.value;
    const colors = parseRgbInput(colorInput);
    if (colorInput.trim() && colors.length === 0) {
      setStatus("颜色格式不正确", false);
      els.stripCmdHint.textContent = "颜色格式示例：255,140,60;255,160,190";
      return;
    }

    const payload = {
      mode: els.stripMode.value || "static",
      colors: colors,
      brightness: Number(els.stripBrightness.value || 1),
      speed: Number(els.stripSpeed.value || 2),
      led_count: Number(els.stripLedCount.value || 60),
    };

    setStatus("下发灯带指令…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();

    try {
      const res = await postJson("/api/app/strip/command", payload);
      els.raw.textContent = JSON.stringify(res, null, 2);
      updateStripCommandForm(res);
      const previewColors = res.command && res.command.colors ? res.command.colors : colors;
      if (previewColors.length) {
        renderStripFromSelection(previewColors);
        els.stripHint.textContent = `指令已更新（${previewColors.length} 色）`;
      } else {
        els.stripHint.textContent = "指令已更新";
      }
      els.stripCmdHint.textContent = "指令已写入";
      ensureStripPreview(res.command ? res.command.led_count : payload.led_count);
      startStripPreview();
      setStatus("灯带指令已更新", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function loadStripCommand() {
    setStatus("读取灯带指令…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();

    try {
      const res = await getJson("/api/data/strip/command");
      updateStripCommandForm(res);
      if (res.command && res.command.colors) {
        renderStripFromSelection(res.command.colors);
        els.stripHint.textContent = `当前指令（${res.command.colors.length} 色）`;
      }
      ensureStripPreview(res.command ? res.command.led_count : 60);
      startStripPreview();
      els.stripCmdHint.textContent = "已读取当前指令";
      els.raw.textContent = JSON.stringify(res, null, 2);
      setStatus("灯带指令已加载", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function fetchFrameJson() {
    const ledCount = Number(els.frameLedCount.value || 60);
    setStatus("读取 JSON 帧…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();

    try {
      const frame = await getJson(`/api/data/strip/frame/json?led_count=${ledCount}`);
      const count = Array.isArray(frame) ? frame.length : 0;
      const preview = Array.isArray(frame) ? frame.slice(0, 24) : [];
      if (preview.length) {
        renderStripFromRgbList(preview);
        els.stripHint.textContent = `帧预览前 ${preview.length} 颗 LED`;
      }

      updateStripPreviewFrame(frame);
      els.frameInfo.textContent = `JSON 帧：${count} 颗 LED`;

      els.raw.textContent = JSON.stringify({ led_count: ledCount, preview: preview, total: count }, null, 2);
      setStatus("JSON 帧已读取", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function fetchFrameRaw() {
    const ledCount = Number(els.frameLedCount.value || 60);
    setStatus("读取 RAW 帧…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();

    try {
      const r = await fetch(`/api/data/strip/frame/raw?led_count=${ledCount}`);
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const buffer = await r.arrayBuffer();
      els.frameInfo.textContent = `RAW 帧：${buffer.byteLength} bytes`;
      els.raw.textContent = JSON.stringify({ led_count: ledCount, raw_bytes: buffer.byteLength }, null, 2);
      setStatus("RAW 帧已读取", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }


  function renderGenerate(res) {
    els.raw.textContent = JSON.stringify(res, null, 2);

    // Prefer top-level speakable_reason, fall back to nested
    let speak = res.speakable_reason;
    if (!speak && res.matrix) speak = res.matrix.speakable_reason;
    if (!speak && res.strip) speak = res.strip.speakable_reason;
    
    if (speak) {
        els.speakableBox.style.display = "block";
        els.speakableText.textContent = speak;
    } else {
        els.speakableBox.style.display = "none";
    }

    // Adapt to different response structures (VoiceSubmit vs Generate)
    // Structure 1 (VoiceSubmit): res.matrix / res.strip (direct plans)
    // Structure 2 (Generate/WS): res.data.matrix / res.data.strip
    
    const m = res.matrix || (res.data ? res.data.matrix : {}) || {};
    const s = res.strip || (res.data ? res.data.strip : {}) || {};

    // Matrix
    els.matrixScene.textContent = safeStr(m.scene_prompt || m.prompt_used);
    els.matrixReason.textContent = safeStr(m.reason);
    
    // Explicitly handle "Generating" state for matrix
    // If we have a prompt but NO json data, it means it's still generating in background.
    if (m.scene_prompt && !m.json) {
        els.matrixMeta.textContent = "生成中…";
        // Clear canvas or show placeholder
        const canvas = els.matrixCanvas;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#222";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else {
        drawMatrix(m.json || null);
    }

    // Strip
    els.stripTheme.textContent = safeStr(s.theme);
    els.stripReason.textContent = safeStr(s.reason);
    
    // Support both 'colors' (Plan) and 'final_selection' (Exec)
    const colors = s.colors || s.final_selection || [];
    renderStripFromSelection(colors);
  }

  async function run() {
    const instruction = els.instruction.value.trim();

    if (!instruction) {
      setStatus("请输入指令", false);
      return;
    }

    setStatus("请求中…", null);
    els.elapsed.textContent = "-";

    const t0 = performance.now();
    try {
      const [res, kb] = await Promise.all([
        postJson("/api/voice/submit", { instruction }),
        postJson("/api/strip/kb/retrieve", { instruction }),
      ]);
      renderGenerate(res);
      renderKbReferences(kb.items);
      setStatus("已规划 (后台执行中)", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
      renderKbReferences([]);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function loadCurrent() {
    setStatus("读取当前数据…", null);
    els.elapsed.textContent = "-";

    const t0 = performance.now();
    try {
      const [matrix, strip, stripCommand] = await Promise.all([
        getJson("/api/data/matrix/json"),
        getJson("/api/data/strip"),
        getJson("/api/data/strip/command"),
      ]);

      els.raw.textContent = JSON.stringify({ matrix, strip, stripCommand }, null, 2);

      drawMatrix(matrix);
      els.matrixScene.textContent = "(当前落盘数据)";
      els.matrixReason.textContent = "-";

      if (Array.isArray(strip)) {
        els.stripTheme.textContent = "(当前落盘数据)";
        els.stripReason.textContent = "-";
        renderStripFromRgbList(strip);
      } else {
        els.stripTheme.textContent = "-";
        els.stripReason.textContent = "-";
        els.swatches.innerHTML = "";
      }

      renderKbReferences([]);

      if (stripCommand) {
        updateStripCommandForm(stripCommand);
        els.stripCmdHint.textContent = "已同步当前指令";
      }

      startStripPreview();
      setStatus("完成", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  function clearAll() {
    els.instruction.value = "";
    els.raw.textContent = "{}";
    els.matrixScene.textContent = "-";
    els.matrixReason.textContent = "-";
    els.matrixMeta.textContent = "-";
    els.stripTheme.textContent = "-";
    els.stripReason.textContent = "-";
    els.swatches.innerHTML = "";
    els.stripHint.textContent = "-";
    renderKbReferences([]);
    els.speakableBox.style.display = "none";
    els.imageFile.value = "";
    els.matrixWidth.value = "16";
    els.matrixHeight.value = "16";
    els.includeRaw.checked = true;
    els.matrixFps.value = "12";
    els.matrixDuration.value = "0";
    els.matrixStoreFrames.checked = true;
    els.matrixAnimInstruction.value = "";
    els.matrixAnimHint.textContent = "使用当前矩阵宽高；持续时间填 0 可循环播放（需手动停止）。";
    if (els.matrixAnimError) {
      els.matrixAnimError.textContent = "-";
    }
    if (els.matrixBlurToggle) {
      els.matrixBlurToggle.checked = true;
      setMatrixBlur(true);
    }
    if (els.matrixBlurAmount) {
      els.matrixBlurAmount.value = "8";
      setMatrixBlurAmount(els.matrixBlurAmount.value);
    }
    els.stripMode.value = "static";
    els.stripLedCount.value = "60";
    els.stripBrightness.value = "1";
    els.stripSpeed.value = "2";
    els.stripColors.value = "";
    els.frameLedCount.value = "60";
    els.frameInfo.textContent = "-";
    if (els.wsLog) {
      wsLogEntries.length = 0;
      els.wsLog.textContent = "-";
    }
    stopStripPreview();
    drawMatrix(null);
    setStatus("就绪", null);
    els.elapsed.textContent = "-";
  }

  els.runBtn.addEventListener("click", run);
  els.clearBtn.addEventListener("click", clearAll);
  els.loadCurrentBtn.addEventListener("click", loadCurrent);
  els.downsampleBtn.addEventListener("click", downsampleImage);
  els.matrixAnimateBtn.addEventListener("click", animateMatrix);
  els.matrixStopBtn.addEventListener("click", stopMatrixAnimation);
  if (els.matrixSaveBtn) {
    els.matrixSaveBtn.addEventListener("click", saveMatrixAnimation);
  }
  if (els.savedAnimRefreshBtn) {
    els.savedAnimRefreshBtn.addEventListener("click", loadSavedAnimations);
  }
  if (els.savedAnimPanel) {
    els.savedAnimPanel.addEventListener("toggle", () => {
      if (els.savedAnimPanel.open) loadSavedAnimations();
    });
  }
  els.stripApplyBtn.addEventListener("click", applyStripCommand);
  els.stripLoadBtn.addEventListener("click", loadStripCommand);
  els.stripPreviewStartBtn.addEventListener("click", startStripPreview);
  els.stripPreviewStopBtn.addEventListener("click", stopStripPreview);
  if (els.stripLedCount) {
    els.stripLedCount.addEventListener("change", () => {
      ensureStripPreview(Number(els.stripLedCount.value || 60));
      startStripPreview();
    });
  }
  els.fetchFrameJsonBtn.addEventListener("click", fetchFrameJson);
  els.fetchFrameRawBtn.addEventListener("click", fetchFrameRaw);

  window.addEventListener("resize", () => {
    // No-op for now (CSS handles resizing)
  });

  function connectWs() {
    try {
      const proto = (location.protocol === "https:") ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);

      ws.onopen = () => {
        setStatus("WebSocket 已连接", true);
        setWsStatus("已连接");
        appendWsLog("connected");
        // Keepalive ping every 20s
        setInterval(() => {
          try { ws.send("ping"); } catch (_) {}
        }, 20000);
      };

      ws.onmessage = (evt) => {
        let msg;
        try { msg = JSON.parse(evt.data); } catch { return; }
        if (!msg || !msg.type) return;

        // Live preview: generate results
        if (msg.type === "generate") {
          renderGenerate(msg.payload);
          setStatus("收到推送：generate", true);
          appendWsLog("generate", msg.payload);
        }

        if (msg.type === "matrix_update") {
          const p = msg.payload || {};
          drawMatrix(p.json || null);
          els.matrixScene.textContent = "(矩阵已更新)";
          els.matrixReason.textContent = "-";
          appendWsLog("matrix_update", msg.payload);
        }

        if (msg.type === "matrix_animation_start") {
          const p = msg.payload || {};
          els.matrixScene.textContent = safeStr(p.summary || "矩阵动画");
          els.matrixReason.textContent = "动画运行中…";
          els.matrixAnimHint.textContent = `动画已启动 (${p.width || 16}×${p.height || 16}, ${p.fps || 0} fps)`;
          if (els.matrixAnimError) {
            els.matrixAnimError.textContent = "-";
          }
          appendWsLog("matrix_animation_start", msg.payload);
        }

        if (msg.type === "matrix_animation_queued") {
          const p = msg.payload || {};
          els.matrixScene.textContent = safeStr(p.summary || "矩阵动画");
          els.matrixReason.textContent = "动画排队中…";
          els.matrixAnimHint.textContent = "动画任务已提交，等待启动";
          if (els.matrixAnimError) {
            els.matrixAnimError.textContent = "-";
          }
          appendWsLog("matrix_animation_queued", msg.payload);
        }

        if (msg.type === "matrix_frame") {
          const p = msg.payload || {};
          drawMatrixFromRaw(p.data, p.width, p.height);
          const frameIndex = p.frame_index ?? "-";
          els.matrixReason.textContent = `帧 ${frameIndex}`;
          els.matrixAnimHint.textContent = `动画进行中（帧 ${frameIndex}）`;
          appendWsLog("matrix_frame", { frame_index: p.frame_index });
        }

        if (msg.type === "matrix_animation_fallback") {
          const p = msg.payload || {};
          const reason = p.reason || "未知错误";
          const missing = Array.isArray(p.missing_dependencies) ? p.missing_dependencies : [];
          const failedCode = p.failed_code || "";
          const missingText = missing.length ? `缺少依赖 ${missing.join(", ")}` : reason;
          els.matrixAnimHint.textContent = `已切换兜底动画（原因：${missingText}）`;
          if (els.matrixAnimError) {
            els.matrixAnimError.textContent = `动画错误详情：${reason}`;
          }
          appendWsLog("matrix_animation_fallback", { reason, missing_dependencies: missing });
          if (failedCode) {
            console.error("=== 失败的动画脚本 ===\n" + failedCode);
            appendWsLog("failed_animation_code", { code_preview: failedCode.slice(0, 500) + (failedCode.length > 500 ? "..." : "") });
          }
        }

        if (msg.type === "matrix_animation_complete") {
          const p = msg.payload || {};
          els.matrixReason.textContent = p.status === "completed" ? "动画完成" : "动画结束";
          let hint = "动画已完成";
          const detail = p.error_detail || {};
          const missing = Array.isArray(detail.missing_dependencies) ? detail.missing_dependencies : [];
          const detailMessage = detail.message ? String(detail.message) : "";
          const reason = missing.length
            ? `缺少依赖 ${missing.join(", ")}`
            : (detailMessage || p.error || "未知错误");
          if (p.fallback_used) {
            hint = `动画已完成（已降级兜底，原因：${reason}）`;
          } else if (missing.length) {
            hint = `动画出错：缺少依赖 ${missing.join(", ")}`;
          } else if (detailMessage) {
            hint = `动画出错：${detailMessage}`;
          } else if (p.error) {
            hint = `动画出错：${p.error}`;
          }
          els.matrixAnimHint.textContent = hint;
          if (els.matrixAnimError) {
            if (p.fallback_used || missing.length || detailMessage || p.error) {
              els.matrixAnimError.textContent = `动画错误详情：${reason}`;
            }
          }
          if (p.fallback_used) {
            appendWsLog("matrix_animation_fallback", { reason, missing_dependencies: missing });
          }
          appendWsLog("matrix_animation_complete", msg.payload);
        }

        if (msg.type === "strip_command_update") {

          updateStripCommandForm(msg.payload);
          startStripPreview();
          appendWsLog("strip_command_update", msg.payload);
        }

        if (msg.type !== "generate" && msg.type !== "matrix_update" && msg.type !== "strip_command_update") {
          appendWsLog(msg.type, msg.payload);
        }
      };

      ws.onclose = () => {
        setStatus("WebSocket 已断开（重连中）", false);
        setWsStatus("断开，重连中…");
        appendWsLog("disconnected");
        setTimeout(connectWs, 1200);
      };

      ws.onerror = () => {
        // onclose will handle reconnect
      };
    } catch (_) {
      setStatus("WebSocket 不可用", false);
      setWsStatus("不可用");
      appendWsLog("error");
    }
  }

  setStatus("就绪", null);
  setWsStatus("连接中…");
  drawMatrix(null);
  ensureStripPreview(Number(els.stripLedCount.value || 60));
  startStripPreview();
  connectWs();
</script>
</body>
</html>
"""
