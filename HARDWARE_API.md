# 氛围灯硬件对接文档（供应商版）

本文档面向 MCU/硬件固件与网关侧开发，描述如何从服务端获取矩阵与灯带数据，并可选接入实时广播。

- Base URL：`https://light.dntc.com.cn`
- OpenAPI（给联调用）：`https://light.dntc.com.cn/docs`

> 注意：当前服务默认不需要鉴权。建议仅在内网/白名单环境使用，或在网关层加鉴权与限流。

---

## 1. 数据模型约定

### 1.1 RGB 字节顺序

- 每个灯珠/像素占 3 字节：`R, G, B`
- 每个分量范围：`0..255`

### 1.2 矩阵像素排列（Row-major）

`raw` 数据按行展开：从左到右、从上到下：

- `index = (y * width + x) * 3`
- `raw[index+0] = R`
- `raw[index+1] = G`
- `raw[index+2] = B`

其中：
- `(x=0,y=0)` 为左上角
- `width/height` 默认 `16×16`

> 若你的灯板走线为“蛇形/折返”（serpentine）或颜色顺序为 `GRB`，需要在 MCU 侧做映射/通道交换。

---

## 2. 拉取接口（推荐给 MCU / 网关）

### 2.1 获取矩阵原始数据（binary，推荐）

- URL：`GET /api/data/matrix/raw`
- 完整地址：`https://light.dntc.com.cn/api/data/matrix/raw`
- Response:
  - `Content-Type: application/octet-stream`
  - Body：连续 RGB 字节流，长度通常为 `16 * 16 * 3 = 768` bytes

**示例（curl）**

```bash
curl -sS -o matrix.bin \
  "https://light.dntc.com.cn/api/data/matrix/raw"
# matrix.bin 即 RGB 原始缓冲区
```

**固件侧建议**
- 拉取周期（轮询）：建议 100ms ~ 500ms（根据网络与功耗调优）
- 使用 HTTP keep-alive（如果网关支持），降低握手开销
- 超时建议：2s~5s

### 2.2 获取矩阵 JSON 数据（调试/网关可用）

- URL：`GET /api/data/matrix/json`
- 完整地址：`https://light.dntc.com.cn/api/data/matrix/json`
- Response：
  - `Content-Type: application/json`
  - Body：

```json
{
  "width": 16,
  "height": 16,
  "pixels": [
    [[0,0,0],[255,0,0]],
    [[0,255,0],[0,0,255]]
  ]
}
```

### 2.3 获取灯带数据（RGB 列表，兼容接口）

- URL：`GET /api/data/strip`
- 完整地址：`https://light.dntc.com.cn/api/data/strip`
- Response：
  - `Content-Type: application/json`
  - Body：`[[R,G,B], ...]`

示例：

```json
[[255,140,60],[255,160,190]]
```

**说明**
- 这是兼容旧固件/简化端侧的接口（通常用于 `static` 常亮或端侧插值）

### 2.4 获取灯带“模式 + 参数”（推荐）

用于让端侧实现灯效引擎（呼吸/流水/渐变等），降低服务端带宽与算力。

- `chase`（流水）：当前服务端/参考实现默认按“多点追逐”（`points=3`）理解；如端侧实现支持，可扩展为可配置点数。

- URL：`GET /api/data/strip/command`
- Response：`application/json`

示例：

```json
{
  "command": {
    "mode": "breath",
    "colors": [{"name": "Ocean Blue", "rgb": [0, 120, 255]}],
    "brightness": 0.8,
    "speed": 2.0,
    "led_count": 60
  },
  "updated_at_ms": 1730000000000
}
```

### 2.5 获取灯带“实时帧数据”（服务端算帧，20fps 场景）

当硬件效果需要以服务端为准、或端侧暂不具备灯效引擎时，可使用“取帧”方式。

- 单帧 raw：`GET /api/data/strip/frame/raw?led_count=60`
  - `Content-Type: application/octet-stream`
  - Body：`led_count * 3` bytes，RGB 顺序
- 单帧 json：`GET /api/data/strip/frame/json?led_count=60`
  - Body：`[[R,G,B], ...]`

> 实时推荐链路：WebSocket 或 MQTT（见第 4 节）。

---

## 3. 上行接口（可选：网关/运营侧上传图片）

> MCU 通常不直接上传图片；如果你们有网关/面板端，可用该接口“上传图片→下采样→立刻同步到矩阵”。

### 3.1 上传图片并下采样（会同步更新矩阵数据）

- URL：`POST /api/matrix/downsample`
- 完整地址：`https://light.dntc.com.cn/api/matrix/downsample`
- Query:
  - `width`：默认 16（1..64）
  - `height`：默认 16（1..64）
  - `include_raw`：默认 true（是否在响应里附带 `raw_base64`）
- Body：`multipart/form-data`
  - `file`：图片文件

**上传限制（安全）**
- 仅允许：PNG/JPEG/WEBP
- 限制文件体积：`MAX_UPLOAD_MB`（默认 10MB）
- 限制像素总数：`MAX_IMAGE_PIXELS`（默认 10,000,000）
- 不支持动图

**示例（curl）**

```bash
curl -X POST \
  -F "file=@./demo.png" \
  "https://light.dntc.com.cn/api/matrix/downsample?width=16&height=16&include_raw=false"
```

**效果**
- 服务端会将结果写入 `latest_led_data.json`
- 随后 MCU/网关 拉取 `/api/data/matrix/raw` 将立即拿到新画面

---

## 4. 实时广播（可选：面板/网关订阅更新）

如果你们不想轮询，可通过 WebSocket 或 MQTT 接收更新事件。

### 4.1 WebSocket

- 事件订阅（状态变更推送）：`wss://light.dntc.com.cn/ws`
- 灯带取帧（20fps，二进制 raw）：`wss://light.dntc.com.cn/ws/strip/raw?fps=20&led_count=60`
  - 每条消息为 binary：`led_count * 3` bytes（RGB）

- 事件消息结构（仅 `/ws`）：

```json
{ "type": "<event_type>", "payload": { } }
```

事件：
- `matrix_update`：当调用 `/api/matrix/downsample` 成功后推送
  - `payload.json`：`{width,height,pixels}`
  - `payload.raw_base64`：RGB bytes 的 base64（便于网关解码）
- `generate`：当调用 `/api/voice/submit` 后后台落地完成时推送（可能包含矩阵与/或灯带）
  - `payload.data.matrix`：矩阵数据（如有）
  - `payload.data.strip`：灯带数据（如有）

心跳：客户端建议每 20s 发一次任意文本（例如 `ping`）。

### 4.2 MQTT

- 事件 JSON 与 WebSocket 完全一致（同样 `{type,payload}`）
- 启用需服务端配置：
  - `MQTT_ENABLED=true`
  - `MQTT_HOST` / `MQTT_PORT`
  - `MQTT_TOPIC`（默认 `ambient-light/events`）

#### 4.2.1 灯带取帧（可选，20fps）

- 启用：`MQTT_STRIP_STREAM_ENABLED=true`
- 可调：`STRIP_STREAM_FPS`（默认 20）
- 推送事件：
  - `type=strip_frame`
  - `payload.format=raw_base64`
  - `payload.data` 为 base64 编码的 `led_count*3` bytes（RGB）

> MQTT Broker 地址/账号由项目方另行提供。

---

## 5. 常见问题

### Q1：MCU 是否需要处理 `raw_base64`？
不需要。MCU/网关最推荐直接拉取 `GET /api/data/matrix/raw`（二进制，最省事/最小）。

### Q2：如果灯板是蛇形走线怎么办？
服务端输出是标准 row-major。蛇形/折返映射请在固件侧做坐标→索引变换。

### Q3：颜色顺序是 GRB 怎么办？
服务端输出为 RGB。固件侧发送到灯珠前做通道交换即可（例如 `RGB -> GRB`）。
