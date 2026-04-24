# Ambient Light Service

基于 FastAPI 的氛围灯服务。

它接收自然语言指令，分别生成：

- LED 矩阵图像或动画
- LED 灯带颜色和模式
- 面向 App、硬件网关、调试面板的读取与控制接口

当前对外实时通道仅保留 WebSocket，不再使用 MQTT。

## 功能概览

- 语音/文本入口：`POST /api/voice/submit`
- App 控制接口：亮度、开关、灯带模式、聚合状态
- Matrix 图片下采样：`POST /api/matrix/downsample`
- Matrix 动画任务：`/api/matrix/animate*`
- 硬件网关接口：`/api/hw/v1/*` 和 `ws://<host>/ws/hw/v1`
- 调试 UI：`/ui`
- 提示词管理台：`/ui/prompts`
- MCP 工具：`voice_generate`

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动 HTTP 服务

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

常用入口：

- OpenAPI：`http://localhost:8080/docs`
- 调试台：`http://localhost:8080/ui`
- 提示词管理台：`http://localhost:8080/ui/prompts`

## 对接文档

- 硬件网关：`hardware/docs/HARDWARE_API_HW_GATEWAY.md`
- 以太网网关：`HARDWARE_API_ETHERNET_10BASET1S.md`

## API 分类

### 1. Voice / Planner

#### `POST /api/voice/submit`

语音入口，推荐给语音侧或智能体侧使用。

行为：

1. 立即返回规划结果和口播文案
2. 后台异步执行矩阵生成、灯带落盘
3. 完成后通过 `/ws` 广播 `generate`

请求：

```json
{
  "instruction": "营造一个温暖放松的氛围，矩阵显示落日海边像素风"
}
```

响应示意：

```json
{
  "status": "accepted",
  "target": "both",
  "instruction": "营造一个温暖放松的氛围，矩阵显示落日海边像素风",
  "description": "both 规划完成",
  "speakable_reason": "已为你点亮温暖橘光，像夕阳般抚平疲惫。",
  "matrix": {
    "scene_prompt": "pixel art sunset beach with a single glowing sun, high contrast, dark background",
    "reason": "选择高对比简洁主体，适合 16x16 像素显示。",
    "image_model": "flux-kontext-pro",
    "note": "dry-run (no image generated)"
  },
  "strip": {
    "theme": "温暖放松",
    "reason": "使用暖色系营造舒缓氛围。",
    "mode": "static",
    "colors": [
      {"name": "Warm Orange", "rgb": [255, 140, 60]},
      {"name": "Soft Pink", "rgb": [255, 160, 190]}
    ]
  },
  "timings": {
    "planner_llm": 0.42
  }
}
```

### 2. App 控制接口

#### `POST /api/app/submit`

与 `POST /api/voice/submit` 等价，适合 App 侧直接提交自然语言指令。

请求：

```json
{
  "instruction": "灯带调成蓝紫渐变，矩阵显示星空"
}
```

#### `POST /api/app/strip/command`

更新灯带控制命令。

支持字段：

- `render_target`: `cloud` | `device`
- `mode`: `static` | `breath` | `chase` | `pulse` | `flow` | `wave` | `sparkle`
- `colors`: 颜色数组，元素格式为 `{name?, rgb}`
- `brightness`: `0..1`
- `speed`: 正数
- `led_count`: `1..2000`
- `mode_options`: 可选扩展参数

说明：

- `colors[].name` 现在会持久化保存，并在读回命令时保留
- 实际渲染仍以 `rgb` 为准，`name` 主要用于 UI 展示和语义回显

请求示例：

```json
{
  "render_target": "cloud",
  "mode": "flow",
  "colors": [
    {"name": "Deep Blue", "rgb": [50, 90, 255]},
    {"name": "Lavender", "rgb": [180, 120, 255]}
  ],
  "brightness": 0.8,
  "speed": 2.5,
  "led_count": 60,
  "mode_options": null
}
```

响应示例：

```json
{
  "command": {
    "render_target": "cloud",
    "mode": "flow",
    "colors": [
      {"name": "Deep Blue", "rgb": [50, 90, 255]},
      {"name": "Lavender", "rgb": [180, 120, 255]}
    ],
    "brightness": 0.8,
    "speed": 2.5,
    "led_count": 60,
    "mode_options": null
  },
  "updated_at_ms": 1730000000000
}
```

#### `GET /api/app/brightness`

读取硬件亮度状态。

#### `POST /api/app/brightness`

更新硬件亮度状态。

请求：

```json
{
  "matrix": 0.7,
  "strip": 0.9
}
```

#### `GET /api/app/power`

读取矩阵与灯带开关状态。

#### `POST /api/app/power`

更新矩阵与灯带开关状态。

请求：

```json
{
  "matrix": true,
  "strip": false
}
```

#### `GET /api/app/state`

返回 App 侧聚合状态快照。

当前返回包含：

- `matrix`: 当前矩阵像素数据
- `strip.colors`: 当前灯带 RGB 数组
- `strip.command`: 当前灯带命令完整结构
- `brightness`: 当前亮度状态
- `power`: 当前开关状态

响应示例：

```json
{
  "matrix": {
    "width": 16,
    "height": 16,
    "pixels": [[[0, 0, 0]]]
  },
  "strip": {
    "colors": [[50, 90, 255], [180, 120, 255]],
    "command": {
      "command": {
        "render_target": "cloud",
        "mode": "flow",
        "colors": [
          {"name": "Deep Blue", "rgb": [50, 90, 255]},
          {"name": "Lavender", "rgb": [180, 120, 255]}
        ],
        "brightness": 0.8,
        "speed": 2.5,
        "led_count": 60,
        "mode_options": null
      },
      "updated_at_ms": 1730000000000
    }
  },
  "brightness": {
    "brightness": {
      "matrix": 0.7,
      "strip": 0.9
    },
    "updated_at_ms": 1730000000001
  },
  "power": {
    "power": {
      "matrix": true,
      "strip": false
    },
    "updated_at_ms": 1730000000002
  }
}
```

### 3. Matrix 接口

#### `POST /api/matrix/downsample`

上传图片并下采样为矩阵像素，同时立即更新当前矩阵数据。

查询参数：

- `width`: 默认 `16`，范围 `1..64`
- `height`: 默认 `16`，范围 `1..64`
- `include_raw`: 默认 `true`

请求方式：`multipart/form-data`

- `file`: `png/jpg/jpeg/webp`

响应示例：

```json
{
  "json": {
    "width": 16,
    "height": 16,
    "pixels": [[[0, 0, 0]]]
  },
  "raw_base64": "...",
  "filename": "demo.png",
  "content_type": "image/png"
}
```

下采样完成后会通过 `/ws` 广播 `matrix_update`。

#### `GET|POST /api/matrix/animate`

创建矩阵动画异步任务。

请求字段：

- `instruction`
- `width`
- `height`
- `fps`
- `duration_s`
- `store_frames`

查询参数：

- `include_code`: 是否把生成代码一并返回

响应字段包含：

- `job_id`
- `status_url`
- `async_mode`

#### 动画附属接口

- `POST /api/matrix/animate/stop`
- `GET /api/matrix/animate/job/{job_id}`
- `GET /api/matrix/animate/saved`
- `GET /api/matrix/animate/saved/{animation_id}`
- `POST /api/matrix/animate/save`
- `POST /api/matrix/animate/saved/{animation_id}/run`
- `DELETE /api/matrix/animate/saved/{animation_id}`

### 4. 数据读取接口

#### Matrix

- `GET|POST /api/data/matrix/json`
- `GET|POST /api/data/matrix/raw`

#### Strip

- `GET /api/data/strip`
- `GET /api/data/strip/frame/json`
- `GET /api/data/strip/frame/raw`

其中：

- `/api/data/strip` 返回当前灯带 RGB 数组
- `/api/data/strip/frame/*` 会按当前命令实时渲染一帧

### 5. 硬件网关接口

#### HTTP

- `GET /api/hw/v1/config`
- `GET /api/hw/v1/commands`
- `GET /api/hw/v1/frame/raw`

#### WebSocket

- `WS /ws/hw/v1`

详细协议见：`hardware/docs/HARDWARE_API_HW_GATEWAY.md`

### 6. Prompt 管理接口

- `GET /api/prompts/store`
- `POST /api/prompts/store`
- `GET /api/prompts/state`
- `POST /api/prompts/state`
- `POST /api/prompts/preview`

## WebSocket

### `WS /ws`

通用业务事件广播。

消息格式：

```json
{
  "type": "<event_type>",
  "payload": {}
}
```

常见事件：

- `generate`
- `matrix_update`
- `matrix_animation_queued`
- `matrix_animation_start`
- `matrix_frame`
- `matrix_animation_fallback`
- `matrix_animation_complete`
- `strip_command_update`
- `brightness_update`
- `power_update`
- `commands`

### `WS /ws/matrix/raw`

矩阵原始二进制帧流。

### `WS /ws/strip/raw`

灯带原始二进制帧流。

查询参数：

- `fps`
- `led_count`
- `encoding`: `rgb24` | `rgb565` | `rgb111`

## MCP

启动：

```bash
python mcp_server.py
```

当前暴露工具：

- `voice_generate`

输入：

```json
{
  "instruction": "把灯光调成适合专注工作的冷色调"
}
```

它会直接执行完整生成并落盘，返回完整结果。

## 数据文件

- `latest_led_data.json`: 当前矩阵数据
- `latest_matrix_animation.json`: 最近一次动画帧数据
- `latest_strip_data.json`: 当前灯带颜色数组
- `latest_strip_command.json`: 当前灯带命令
- `latest_hw_brightness.json`: 当前硬件亮度
- `latest_hw_power.json`: 当前硬件开关状态
- `saved_matrix_animations.json`: 收藏的动画脚本

## 环境变量

完整说明见：`ENV.md`

常用项：

- `AIHUBMIX_API_KEY`
- `BFL_API_KEY`
- `MATRIX_IMAGE_MODEL`
- `MATRIX_ANIMATION_MODEL`
- `GEMINI_TIMEOUT_S`
- `HW_MATRIX_WIDTH`
- `HW_MATRIX_HEIGHT`
- `HW_STRIP_LED_COUNTS`
- `HW_SYNC_FPS`
- `STRIP_KB_FILE`
- `STRIP_STREAM_FPS`
- `STRIP_STREAM_ENCODING`
- `MAX_UPLOAD_MB`
- `MAX_IMAGE_PIXELS`

## 调试 UI

- `/ui`: 调试面板，实时预览矩阵和灯带效果
- `/ui/prompts`: 提示词管理台

## 备注

- `.env` 的优先级高于系统环境变量
- 矩阵动画运行在受限沙盒中，失败时会自动降级
- 服务目前不包含认证、限流和权限控制，生产环境接入前建议放在网关后面
