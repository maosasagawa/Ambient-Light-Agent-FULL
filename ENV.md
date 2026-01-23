# 环境变量与本地配置文件

本项目支持两种配置来源，并且 **配置文件优先级高于系统环境变量**。

## 配置加载顺序

1. 读取项目根目录的 `.env` 文件（默认）
2. 如果 `.env` 中没有该项，再读取系统环境变量

> 可使用 `ENV_FILE` 指定配置文件路径（相对路径以项目根目录为基准）。

## `.env` 文件格式

```
# 示例
AIHUBMIX_API_KEY=sk-xxxx
MATRIX_IMAGE_MODEL=flux-kontext-pro
MQTT_ENABLED=true
```

- 支持 `KEY=VALUE` 形式。
- 支持 `export KEY=VALUE`。
- 行首 `#` 为注释。
- 值可使用单/双引号包裹。

## 通用配置

- `AIHUBMIX_API_KEY`：AIHubMix API Key（必填，语音/矩阵/灯带 LLM 相关）
- `GEMINI_TIMEOUT_S`：Gemini 请求超时（默认 `180` 秒）
- `BFL_API_KEY`：BFL Flux API Key（使用 FLUX 模型时必填）
- `ENV_FILE`：指定 `.env` 文件路径（可选）

## 提示词模板

- `PROMPT_STORE_FILE`：提示词 JSON 文件路径（默认 `prompts.json`）
- `PROMPT_STATE_FILE`：提示词状态文件路径（默认 `prompt_state.json`）
- `PROMPT_VARIANT`：全局选择的提示词版本（可选）
- `PROMPT_VARIANT_PLANNER`：Planner 提示词版本（可选，覆盖全局）
- `PROMPT_VARIANT_STRIP`：Strip 提示词版本（可选，覆盖全局）
- `PROMPT_VARIANT_MATRIX_ANIMATION`：矩阵动画提示词版本（可选，覆盖全局）
- `PROMPT_AB_TEST`：是否启用 A/B 版本加权选择（默认 false，启用后按输入稳定分流）

## 矩阵生图

- `MATRIX_IMAGE_MODEL`：矩阵生图模型（默认 `flux-kontext-pro`）
- `FLUX_POLL_INTERVAL_S`：FLUX 异步轮询间隔（默认 `0.25`）
- `FLUX_POLL_MAX_SECONDS`：FLUX 异步轮询最大等待（默认 `20`）

## 矩阵动画

- `MATRIX_ANIMATION_MODEL`：动画脚本模型（默认 `gemini-3-flash`）
- `MATRIX_ANIMATION_MAX_FRAMES`：单次动画最大帧数（默认 `3600`）
- `MATRIX_ANIMATION_SAVED_FILE`：动画脚本收藏文件（默认 `saved_matrix_animations.json`）
- `MATRIX_ANIMATION_MAX_CODE_CHARS`：动画脚本最大长度（默认 `8000`）
- `MATRIX_ANIMATION_TIMEOUT_S`：沙盒执行超时（默认 `10` 秒）
- `MATRIX_ANIMATION_CPU_SECONDS`：沙盒 CPU 上限（默认 `5` 秒）
- `MATRIX_ANIMATION_MAX_MEMORY_MB`：沙盒内存上限（默认 `256` MB）

## 灯带

- `STRIP_KB_FILE`：灯带知识库文本路径（默认 `strip_kb.txt`）
- `STRIP_STREAM_FPS`：灯带帧推送 FPS（默认 `20`）
- `STRIP_STREAM_ENCODING`：灯带帧编码（`rgb24`/`rgb565`/`rgb111`，默认 `rgb24`）

## 上传与安全

- `MAX_UPLOAD_MB`：图片上传最大体积（默认 `10`）
- `MAX_IMAGE_PIXELS`：图片像素总数上限（默认 `10000000`）
- `VOICE_PROMPT_MAX_CHARS`：语音规划提示上限（默认 `500`）

## MQTT

- `MQTT_ENABLED`：是否启用 MQTT 广播（默认 false）
- `MQTT_HOST`：MQTT Broker 地址（默认 `localhost`）
- `MQTT_PORT`：MQTT Broker 端口（默认 `1883`）
- `MQTT_TOPIC`：广播 Topic（默认 `ambient-light/events`）
- `MQTT_STRIP_STREAM_ENABLED`：是否启用灯带帧流 MQTT 推送（默认 false）
