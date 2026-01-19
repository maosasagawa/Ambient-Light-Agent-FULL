# 氛围灯语音团队对接文档（Voice API）

本文档面向**语音识别/语义理解/NLU/TTS**团队，描述如何将用户的语音指令对接到「统一氛围灯服务」，实现：

- 语音侧快速拿到**可口播的一句话反馈**（`speakable_reason`）
- 后端根据指令控制：
  - **16×16 LED 矩阵**（像素画面）
  - **LED 灯带**（环境配色）

> 推荐对接接口：`POST /api/voice/submit`（先规划、再后台执行）。
>
> 以服务端 `GET /docs`（OpenAPI）为准；如遇到接口缺失或权限问题，请联系后端/运维确认部署版本与鉴权策略。

---

## 0. 关键概念

- **instruction**：语音识别后的用户原始文本（建议尽量保留口语表达，不要“过度规范化”）。
- **target**：本次指令要控制的目标。
  - `matrix`：只控制矩阵
  - `strip`：只控制灯带
  - `both`：两者都控制
- **speakable_reason**：返回给语音侧用于 TTS 的一句话（尽量短、可直接播报）。

---

## 1. 接入方式概览（推荐时序）

语音侧推荐走“快速应答 + 后台落地”的异步链路：

1. 语音侧识别文本后调用 `POST /api/voice/submit`
2. 服务端返回：
   - `target`（这次会影响哪些硬件）
   - `speakable_reason`（立刻可用于口播）
   - `matrix/strip` 的规划信息（可用于 debug 或二次确认）
3. 服务端在后台执行真实落地（例如矩阵生图、写入落盘数据）
4. 硬件侧通过拉取接口或订阅推送拿到新数据：
   - 拉取：`GET /api/data/matrix/raw`、`GET /api/data/strip`
   - 推送：`ws(s)://<host>/ws`（`type=generate`）

> 语音侧通常只需要第 1~2 步即可完成用户体验闭环（快速播报）；硬件同步可以由网关/面板/设备侧处理。

---

## 2. Base URL 与环境

- Base URL（示例，按环境替换）：`https://light.dntc.com.cn`
- OpenAPI（联调用）：`https://light.dntc.com.cn/docs`

### 2.1 鉴权

- 当前默认**不需要鉴权**。
- 如后续加网关鉴权/白名单，语音侧通常无需变化（由网关层代理处理）。

### 2.2 通用 Header

- `Content-Type: application/json`

---

## 3. 推荐接口：语音提交（规划 + 后台执行）

### 3.1 `POST /api/voice/submit`

- 目标：
  - **尽快**返回 `speakable_reason`，用于语音侧 TTS
  - 同时触发后台任务去“真正更新硬件数据”（矩阵可能耗时较长）

#### Request

```json
{
  "instruction": "我有点困，灯带调亮一点，矩阵显示警示图标"
}
```

字段说明：
- `instruction`：必填，用户原始语音转写文本

#### Response（成功）

```json
{
  "status": "accepted",
  "target": "both",
  "instruction": "我有点困，灯带调亮一点，矩阵显示警示图标",
  "description": "Planning complete for both",
  "speakable_reason": "你说有点困，我把灯带调亮，用黄红色更醒目。",
  "matrix": {
    "scene_prompt": "English prompt for image generation...",
    "reason": "...",
    "speakable_reason": "你说有点困，我把灯带调亮，用黄红色更醒目。",
    "image_model": "flux-kontext-pro",
    "note": "dry-run (no image generated)",
    "current": {
      "width": 16,
      "height": 16,
      "pixels": [[[0,0,0]]]
    }
  },
  "strip": {
    "theme": "警示/提神",
    "reason": "用户表达困倦，优先提升可感知亮度并使用警示色（黄/红）。",
    "speakable_reason": "你说有点困，我把灯带调亮，用黄红色更醒目。",
    "colors": [
      {"name": "Alert Yellow", "rgb": [255, 220, 40]},
      {"name": "Warm Red", "rgb": [255, 60, 40]}
    ]
  },
  "timings": {
    "planner_llm": 0.532
  }
}
```

说明：
- `status=accepted` 表示**已接受并完成规划**（语音侧可以立刻播报）。
- `matrix.note` 可能显示 `dry-run`：表示这是规划信息，不代表矩阵已完成生图。
- 真实落地完成后，服务端可通过 WebSocket/MQTT 推送 `type=generate`（见第 5 节）。

#### Response（失败）

- 典型 HTTP 状态：`4xx/5xx`
- Body 一般为 JSON，语音侧建议：
  - 兜底口播：例如“我这边暂时没连上灯光服务，稍后再试试。”
  - 对用户体验：宁可给出兜底，不要长时间沉默

---

## 4. 同步生成接口

当前版本仅提供 `POST /api/voice/submit` 作为入口：先快速返回口播与规划信息，再后台落地生成与写盘。

---

## 5. 状态通知（可选，但强烈推荐给联调/面板）

语音侧通常不需要订阅；但在联调阶段或有“灯效已生效”的播报需求时，可用推送。

### 5.1 WebSocket

- 地址：`ws://<host>:8000/ws`（HTTPS 则 `wss://`）
- 心跳：客户端建议每 20s 发送任意文本（例如 `ping`），服务端仅用于保活。

消息格式：

```json
{ "type": "generate", "payload": { "status": "success", "target": "both", "data": { } } }
```

事件：
- `type=generate`：当生成/后台执行完成后推送
  - `payload` 的结构与后台生成结果一致（`status/target/description/data`）

### 5.2 MQTT（可选）

- 与 WebSocket 的事件结构完全一致：`{type, payload}`
- 需要服务端配置启用：
  - `MQTT_ENABLED=true`
  - `MQTT_HOST` / `MQTT_PORT`
  - `MQTT_TOPIC`（默认 `ambient-light/events`）

---

## 6. 语音侧接入建议（很重要）

### 6.1 超时与重试

- `POST /api/voice/submit` 建议超时：`2s~4s`（以“快速应答”为目标）
- 可重试：1 次（网络抖动/偶发失败），避免多次重复触发

### 6.2 口播策略（TTS）

- 优先播报 `speakable_reason`
- 如果接口失败：
  - 兜底句尽量短：例如“我这边灯光服务暂时不可用，我们稍后再试。”
- 不建议把 `reason`（内部解释）直接口播给用户

### 6.3 指令拼接建议

- 如果语音 NLU 已拆出结构化意图（比如“亮一点”“冷色”），也可以保留在原句中，让后端统一理解：
  - 示例：`"有点困（需要提神），灯带亮一点偏黄红，矩阵显示警示符号"`

---

## 7. 常见问题（FAQ）

### Q1：语音侧要不要等待矩阵生图完成？
不需要。推荐走 `POST /api/voice/submit`：先播报 `speakable_reason`，矩阵/灯带在后台生效。

### Q2：如果用户只说“氛围灯舒服一点”，target 是什么？
服务端会尽量判定为 `both`（矩阵 + 灯带）。语音侧无需自己强行拆分。

### Q3：语音侧需要关心硬件数据格式吗？
通常不需要。硬件/网关侧拉取：
- `GET /api/data/matrix/raw`
- `GET /api/data/strip`
详见 `HARDWARE_API.md`。

---

## 8. 变更与兼容性

- 建议语音侧只依赖：`status`、`target`、`speakable_reason` 三个字段。
- `matrix/strip` 内部字段允许逐步演进（新增字段向后兼容）。
