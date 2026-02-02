# 氛围灯服务（Ambient Light Service）

这是一个基于 FastAPI 的统一服务：输入自然语言指令，生成 **LED 矩阵像素图** 与 **LED 灯带配色方案**，并提供硬件友好的数据读取接口与调试 UI。

## 功能

- **语音/文本入口**：`POST /api/voice/submit`
- **矩阵生图**：支持 FLUX（异步两步端口：`flux-kontext-pro/max`）与 FLUX 通用一步接口（`FLUX.1-Kontext-pro` / `FLUX-1.1-pro`），以及 Google Imagen（`imagen-*`）
- **矩阵动画**：`POST /api/matrix/animate` 生成 Python 动画脚本并流式推送帧数据，支持沙盒执行、兜底降级与错误实时通知
- **灯带配色**：LLM 生成配色方案，内置亮度校验与候选筛选，支持多种灯效模式（static/breath/flow/chase/pulse/wave/sparkle）
- **硬件读取接口**：矩阵/灯带 JSON + 原始字节流
- **调试 UI**：`/ui` 可视化预览（支持 WebSocket 实时推送，独立错误显示区）
- **提示词管理台**：`/ui/prompts`（版本切换 / A-B 测试 / 预览）
- **MCP 支持**：`mcp_server.py` 提供 `voice_generate` 工具

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 启动 HTTP 服务

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

- OpenAPI 文档：`http://localhost:8000/docs`
- 调试台：`http://localhost:8000/ui`
- 提示词管理台：`http://localhost:8000/ui/prompts`

## 硬件接口

- 面向以太网网关的对接文档：`HARDWARE_API_ETHERNET_10BASET1S.md`
- 面向硬件网关的同步接口：`HARDWARE_API_HW_GATEWAY.md`

## 语音接口

- 面向语音团队的对接文档：`VOICE_API.md`（推荐接口：`POST /api/voice/submit` 并行下发）

## API

### Voice Submit（语音入口，推荐）

- `POST /api/voice/submit`

说明：接口先返回“口播文案 + 规划信息”，后台异步执行生图/落盘；执行完成后通过 WebSocket/MQTT 推送 `generate` 事件（用于调试面板实时刷新）。

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

### 硬件读取接口（Hardware data endpoints）

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

### Matrix Animate（矩阵动画）

- `POST /api/matrix/animate`
- `POST /api/matrix/animate/stop`
- `POST /api/matrix/animate/save`
- `GET /api/matrix/animate/saved`

说明：使用 LLM 生成 Python 动画脚本并在沙盒中执行，按 `fps` 生成帧并实时推送（同时写入 `latest_led_data.json`，可选保存完整帧序列到 `latest_matrix_animation.json`）。可通过 `/api/matrix/animate/save` 暂存当前动画脚本（只保存指令与代码）到 `saved_matrix_animations.json`。

**沙盒特性**：
- 支持 Python 内置函数（`hasattr`/`isinstance`/`print` 等）
- CPU 限制：默认 60 秒（可通过 `MATRIX_ANIMATION_CPU_SECONDS` 配置）
- 内存限制：默认 1024 MB（可通过 `MATRIX_ANIMATION_MAX_MEMORY_MB` 配置）
- 失败时自动降级到默认动画并推送 `matrix_animation_fallback` 事件

Request JSON:

```json
{
  "instruction": "夜空里流动的星河",
  "width": 16,
  "height": 16,
  "fps": 12,
  "duration_s": 30,
  "store_frames": true
}
```

提示：`duration_s=0` 表示持续播放，需调用 `/api/matrix/animate/stop` 手动停止。

Query:

- `include_code`：是否返回生成的 Python 脚本（默认 false）

Response（简化示意）：

```json
{
  "status": "accepted",
  "instruction": "...",
  "summary": "...",
  "width": 16,
  "height": 16,
  "fps": 12,
  "duration_s": 30,
  "model_used": "gemini-3-flash",
  "timings": {"animator_llm": 0.45}
}
```

### Prompt Management（提示词管理）

- `GET /api/prompts/store`：读取提示词 JSON
- `POST /api/prompts/store`：更新提示词 JSON（完整覆盖）
- `GET /api/prompts/state`：读取版本选择与 A/B 状态
- `POST /api/prompts/state`：更新版本选择与 A/B 状态
- `POST /api/prompts/preview`：渲染提示词预览

### Realtime（WebSocket / MQTT 广播）

服务端会在以下动作后主动广播事件（用于面板实时刷新）：

- 调用 `/api/voice/submit` 后（后台执行完成时）广播 `generate`
- 调用 `/api/matrix/downsample` 后广播 `matrix_update`
- 调用 `/api/matrix/animate` 后广播 `matrix_animation_start` / `matrix_frame` / `matrix_animation_complete`

#### WebSocket

- 连接地址：`ws://<host>:8000/ws`（HTTPS 则使用 `wss://`）
- Matrix 二进制流：`ws://<host>:8000/ws/matrix/raw`（rgb24 raw bytes）
- 心跳：客户端建议每 20s 发送任意文本（例如 `ping`），服务端只用于保活不解析内容

消息格式：

```json
{ "type": "<event_type>", "payload": { } }
```

事件：

- `generate`
  - 来源：`/api/voice/submit`
  - `payload`：与 HTTP 返回结构一致（`status/target/description/data`）
- `matrix_update`
  - 来源：`/api/matrix/downsample`
  - `payload.json`：`{ width, height, pixels }`
  - `payload.raw_base64`：RGB 原始字节流的 base64
  - `payload.filename` / `payload.content_type`：上传文件信息
- `matrix_animation_start`
  - 来源：`/api/matrix/animate`
  - `payload`：包含 `summary/width/height/fps/duration_s/model_used`
- `matrix_frame`
  - 来源：`/api/matrix/animate`
  - `payload.data`：单帧 RGB 原始字节流 base64（`encoding=rgb24`）
- `matrix_animation_fallback`
  - 来源：沙盒脚本执行失败时自动降级
  - `payload.reason`：失败原因（如 `sandbox exited`、`缺少依赖 xxx`）
  - `payload.missing_dependencies`：缺失的模块列表
  - `payload.failed_code`：失败的脚本片段（用于调试）
- `matrix_animation_complete`
  - 来源：`/api/matrix/animate`
  - `payload.status` / `payload.frame_count` / `payload.error`
  - `payload.fallback_used`：是否使用了兜底动画
  - `payload.error_detail`：包含 `message` 和 `missing_dependencies`

#### MQTT

MQTT 广播事件结构与 WebSocket 完全一致（同样是 `{type, payload}` JSON）。

- 启用方式：设置 `MQTT_ENABLED=true`
- 连接信息：`MQTT_HOST` / `MQTT_PORT`
- Topic：`MQTT_TOPIC`（默认 `ambient-light/events`）

## 配置（环境变量）

- `AIHUBMIX_API_KEY`：AIHubMix API Key
- `BFL_API_KEY`：BFL Flux API Key
- `MATRIX_IMAGE_MODEL`：矩阵生图模型（默认 `flux-kontext-pro`）
- `MATRIX_ANIMATION_MODEL`：矩阵动画脚本模型（默认 `gemini-3-flash`）
- `MATRIX_ANIMATION_MAX_FRAMES`：单次动画最大帧数（默认 `3600`）
- `MATRIX_ANIMATION_SAVED_FILE`：动画脚本收藏文件（默认 `saved_matrix_animations.json`）
- `GEMINI_TIMEOUT_S`：Gemini 请求超时（默认 `180` 秒）
- `MATRIX_ANIMATION_MAX_CODE_CHARS`：动画脚本最大长度（默认 `8000`）
- `MATRIX_ANIMATION_TIMEOUT_S`：沙盒执行超时（默认 `10` 秒）
- `MATRIX_ANIMATION_CPU_SECONDS`：沙盒 CPU 上限（默认 `60` 秒，提升以支持复杂动画）
- `MATRIX_ANIMATION_MAX_MEMORY_MB`：沙盒内存上限（默认 `1024` MB，提升以支持 numpy/scipy）
- `FLUX_POLL_INTERVAL_S`：FLUX 异步轮询间隔（默认 `0.25`）
- `FLUX_POLL_MAX_SECONDS`：FLUX 异步轮询最大等待（默认 `20`）
- `STRIP_KB_FILE`：灯带知识库文本路径（默认 `strip_kb.txt`，每行一条）
- `PROMPT_STORE_FILE`：提示词 JSON 文件路径（默认 `prompts.json`）
- `PROMPT_STATE_FILE`：提示词状态文件路径（默认 `prompt_state.json`）
- `PROMPT_VARIANT`：全局提示词版本（可选）
- `PROMPT_VARIANT_*`：按 key 覆盖提示词版本（如 `PROMPT_VARIANT_PLANNER`）
- `PROMPT_AB_TEST`：是否启用 A/B 分流
- `MAX_UPLOAD_MB`：图片上传最大体积（默认 `10`，用于 `/api/matrix/downsample`）
- `MAX_IMAGE_PIXELS`：图片像素总数上限（默认 `10000000`，用于 `/api/matrix/downsample`）
- `MQTT_ENABLED`：是否启用 MQTT 广播（默认 false）
- `MQTT_HOST`：MQTT Broker 地址（默认 `localhost`）
- `MQTT_PORT`：MQTT Broker 端口（默认 `1883`）
- `MQTT_TOPIC`：广播 Topic（默认 `ambient-light/events`）
- `MQTT_STRIP_STREAM_ENABLED`：是否启用灯带帧流 MQTT 推送（默认 false）
- `STRIP_STREAM_FPS`：灯带帧推送 FPS（默认 `20`）
- `STRIP_STREAM_ENCODING`：灯带帧编码（`rgb24`/`rgb565`/`rgb111`，默认 `rgb24`）

## MCP

启动 MCP 服务器：

```bash
python mcp_server.py
```

它暴露 `voice_generate` 工具，输入 `instruction`，会生成并落盘（行为与服务端后台落地一致）。

灯带指令新增 `render_target` 字段：`cloud`（默认）/`device`，用于切换云端算帧或端侧算帧。

## 数据落盘

- 矩阵落盘：`latest_led_data.json`
- 矩阵动画落盘：`latest_matrix_animation.json`
- 灯带落盘：`latest_strip_data.json`
- 默认动画脚本：当 LLM 生成脚本执行失败时，系统会自动切换到内置的无依赖火焰动画脚本（仅使用 Python 内置 `math` 模块）。

## UI 说明

- **调试台**（`/ui`）：实时预览矩阵动画与灯带效果，WebSocket 状态监控，独立错误显示区（`matrixAnimError`）用于展示动画执行失败/降级原因
- **提示词管理台**（`/ui/prompts`）：集中管理提示词模板、版本切换与 A/B 测试

## 备注

- API Key 使用环境变量注入，生产环境建议配合认证/限流。
- 动画沙盒支持常用 Python 内置函数（`hasattr`/`isinstance`/`print`/`range` 等），但限制部分危险模块（`os`/`sys`/`subprocess` 等）。
- 矩阵预览已做垂直翻转处理，数据中 y=0 为底部，canvas 渲染时已自动翻转。

