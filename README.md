# Ambient Light Service（统一氛围灯服务）

一个基于 FastAPI 的统一服务：输入自然语言指令，生成 **16×16 LED 矩阵像素图** 与 **LED 灯带配色方案**，并提供硬件友好的数据读取接口与调试 UI。

## Features

- **语音入口**：`POST /api/voice/submit`（先规划快速返回，后台执行生图+落盘）
- **矩阵生图**：支持 FLUX（异步两步端口：`flux-kontext-pro/max`）与 FLUX 通用一步接口（`FLUX.1-Kontext-pro` / `FLUX-1.1-pro`），以及 Google Imagen（`imagen-*`）
- **灯带配色**：LLM 生成配色方案，内置亮度校验与候选筛选
- **硬件兼容读取**：
  - `GET /api/data/matrix/raw`（二进制 RGB，适合 MCU 拉取）
  - `GET /api/data/matrix/json`（前端可读像素矩阵）
  - `GET /api/data/strip`（灯带 RGB 列表）
- **调试 UI**：`/ui` 一键生成并预览矩阵与灯带（含 WebSocket 实时推送预览）
- **MCP 支持**：`mcp_server.py` 提供 `generate_lighting_effect` 工具（会生成并落盘）

## Quickstart

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Run HTTP server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- OpenAPI：`http://localhost:8000/docs`
- 调试台：`http://localhost:8000/ui`

## Hardware

- 面向硬件供应商的对接文档：`HARDWARE_API.md`

## Voice

- 面向语音团队的对接文档：`VOICE_API.md`（推荐接口：`POST /api/voice/submit` 并行下发）

## API

### Voice Submit（语音入口，推荐）

- `POST /api/voice/submit`

说明：接口会先返回“口播文案 + 规划信息”，并在后台执行生图/落盘；执行完成后会通过 WebSocket/MQTT 推送 `generate` 事件（用于调试面板实时刷新）。

Request:

```json
{ "instruction": "营造一个温暖放松的氛围，矩阵显示落日海边像素风" }
```

Response（简化示意）：

```json
{
  "status": "accepted",
  "target": "both",
  "instruction": "...",
  "description": "Planning complete for both",
  "speakable_reason": "...",
  "matrix": { "scene_prompt": "...", "reason": "..." },
  "strip": { "theme": "...", "reason": "...", "colors": [ {"name": "...", "rgb": [0,170,255]} ] },
  "timings": { "planner_llm": 0.532 }
}
```

### Hardware data endpoints（硬件读取）

- `GET /api/data/matrix/raw` → `application/octet-stream`（默认 16×16×3 = 768 bytes）
- `GET /api/data/matrix/json` → 16×16 像素矩阵（RGB 三元组）
- `GET /api/data/strip` → `[[R,G,B], ...]`

### Matrix Downsample（上传图片 → 矩阵像素）

- `POST /api/matrix/downsample`

说明：上传图片后按 `width`×`height` 下采样，**同时更新当前矩阵数据**（写入 `latest_led_data.json`，硬件读取接口会立刻变更），并通过 WebSocket/MQTT 推送 `matrix_update` 事件。

Query params:

- `width`：目标宽度（默认 16，范围 1..64）
- `height`：目标高度（默认 16，范围 1..64）
- `include_raw`：是否返回 `raw_base64`（默认 true）

Request（multipart/form-data）:

- `file`：图片文件（仅允许 `png/jpg/jpeg/webp`）

Example:

```bash
curl -X POST \
  -F "file=@./demo.png" \
  "http://localhost:8000/api/matrix/downsample?width=16&height=16&include_raw=true"
```

Response（简化示意）：

```json
{
  "json": { "width": 16, "height": 16, "pixels": [[[0,0,0]]] },
  "raw_base64": "...",
  "filename": "demo.png",
  "content_type": "image/png"
}
```

### Realtime（WebSocket / MQTT 广播）

服务端会在以下动作后主动广播事件（用于面板实时刷新）：

- 调用 `/api/voice/submit` 后（后台执行完成时）广播 `generate`（其中 `payload.data.matrix` / `payload.data.strip` 分别包含矩阵/灯带数据）
- 调用 `/api/matrix/downsample` 后广播 `matrix_update`（仅矩阵数据）

#### WebSocket

- 连接地址：`ws://<host>:8000/ws`（HTTPS 则使用 `wss://`）
- 心跳：客户端建议每 20s 发送任意文本（例如 `ping`），服务端只用于保活不解析内容

消息格式：

```json
{ "type": "<event_type>", "payload": { } }
```

事件：

- `generate`
  - 来源：`/api/voice/submit`（后台执行完成时推送）
  - `payload`：与 HTTP 返回结构一致（`status/target/description/data`）
  - 说明：灯带与矩阵都通过该事件分发（当 `target` 为 `strip` 或 `both` 时，灯带数据会更新；当 `target` 为 `matrix` 或 `both` 时，矩阵数据会更新）
- `matrix_update`
  - 来源：`/api/matrix/downsample`
  - `payload.json`：`{ width, height, pixels }`
  - `payload.raw_base64`：RGB 原始字节流的 base64（可选用于网关/MCU 侧解码；面板渲染通常只需 `payload.json.pixels`）
  - `payload.filename` / `payload.content_type`：上传文件信息

#### MQTT

MQTT 广播的事件结构与 WebSocket 完全一致（同样是 `{type, payload}` JSON）。

- 启用方式：设置 `MQTT_ENABLED=true`
- 连接信息：`MQTT_HOST` / `MQTT_PORT`
- Topic：`MQTT_TOPIC`（默认 `ambient-light/events`）

## Configuration

通过环境变量控制运行行为（推荐在生产环境使用 env，避免硬编码密钥）：

- `AIHUBMIX_API_KEY`：AIHubMix API Key
- `BFL_API_KEY`：BFL Flux API Key
- `MATRIX_IMAGE_MODEL`：矩阵生图模型（默认 `flux-kontext-pro`）
- `FLUX_POLL_INTERVAL_S`：FLUX 异步轮询间隔（默认 `0.25`）
- `FLUX_POLL_MAX_SECONDS`：FLUX 异步轮询最大等待（默认 `20`）
- `STRIP_KB_FILE`：灯带知识库文本路径（默认 `strip_kb.txt`，每行一条）
- `MAX_UPLOAD_MB`：图片上传最大体积（默认 `10`，用于 `/api/matrix/downsample`）
- `MAX_IMAGE_PIXELS`：图片像素总数上限（默认 `10000000`，用于 `/api/matrix/downsample`）
- `MQTT_ENABLED`：是否启用 MQTT 广播（默认 false）
- `MQTT_HOST`：MQTT Broker 地址（默认 `localhost`）
- `MQTT_PORT`：MQTT Broker 端口（默认 `1883`）
- `MQTT_TOPIC`：广播 Topic（默认 `ambient-light/events`）

## MCP

启动 MCP 服务器：

```bash
python mcp_server.py
```

它暴露 `generate_lighting_effect` 工具，输入 `instruction`，会生成并落盘（行为与服务端后台落地一致）。

## Data persistence

- 矩阵落盘：`latest_led_data.json`
- 灯带落盘：`latest_strip_data.json`

## Notes

- API Key 已改为环境变量注入，生产环境建议配合认证/限流。
