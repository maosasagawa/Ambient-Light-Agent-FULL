# 硬件网关对接文档（WebSocket Only, V1）

本版本只定义 WebSocket 对接，不使用 HTTP 轮询。

## 1. 对接目标

- 连接云端 `wss://light.dntc.com.cn/ws/hw/v1`
- 接收四类服务端消息：
  - 文本 JSON：`hello_ack`、`commands`、`brightness_update`、`power_update`
  - 二进制：`FrameHeaderV1(32B) + RGB Payload`
- 可主动向服务端发送：
  - `set_brightness`：上报设备端亮度（如物理旋钮调节）
- 设备按通道执行：
  - `color_mode`：端侧算法渲染（含 `enabled` 开关字段）
  - `raw_stream`：消费云端二进制帧（含 `enabled` 开关字段）

通道约定：

- `matrix:0`：矩阵
- `strip:1`：仪表台长灯带（主氛围灯，通常最长）
- `strip:2`：左前门板灯带
- `strip:3`：左后门板灯带
- `strip:4`：右前门板灯带
- `strip:5`：右后门板灯带
- `strip:6..N`：预留扩展灯带（如后排、脚窝、顶棚等，需项目另行约定）

> **供应商接线建议**：请按上述 `channel_id` 固定接线与固件映射，不要按物理发现顺序动态重排。云端会按该顺序下发 `commands` 与二进制帧。

## 2. 最小接入流程

1. 建立 WS：`wss://light.dntc.com.cn/ws/hw/v1`
2. 发送 `hello`（订阅通道 + 编码偏好）
3. 收到 `hello_ack` 后进入接收循环（可用 `hw_sdk_parse_hello_ack()` 解析）
4. 收到 `commands`：更新每个通道状态机（注意 `enabled` 字段控制通道开关）
5. 收到 `brightness_update`：更新矩阵/灯带输出亮度（可用 `hw_sdk_parse_brightness_update()` 解析）
6. 收到 `power_update`：更新矩阵/灯带开关状态（可用 `hw_sdk_parse_power_update()` 解析）
7. 收到二进制帧：解析头部后按 `channel_id` 输出
8. （可选）设备端亮度变化时，发送 `set_brightness` 通知云端（可用 `hw_sdk_ws_send_brightness()` 发送）

## 3. 握手协议

客户端发送：

```json
{
  "type": "hello",
  "subscribe": ["matrix:0", "strip:1", "strip:2", "strip:3", "strip:4", "strip:5"],
  "prefer_encoding": "rgb565"
}
```

服务端返回：

```json
{
  "type": "hello_ack",
  "payload": {
    "sync_fps": 20.0,
    "encoding": "rgb565",
    "channels": ["matrix:0", "strip:1", "strip:2", "strip:3", "strip:4", "strip:5"]
  }
}
```

说明：

- `subscribe` 可选；不传则默认订阅全部可用通道
- `prefer_encoding` 推荐 `rgb565`；当前硬件网关实际支持 `rgb565` / `rgb24`

## 4. 命令消息（Text JSON，服务端 → 设备）

服务端推送：

```json
{
  "type": "commands",
  "payload": {
    "updated_at_ms": 1730000000000,
    "commands": [
      {
        "channel": "strip:1",
        "kind": "color_mode",
        "enabled": true,
        "mode_code": "chase",
        "params": {
          "brightness": 1.0,
          "speed": 3.0,
          "colors": [[200, 30, 30], [80, 10, 10]],
          "mode_options": null
        }
      },
      {
        "channel": "matrix:0",
        "kind": "raw_stream",
        "enabled": true,
        "fps": 20,
        "encoding": "rgb565"
      }
    ]
  }
}
```

`mode_code` 取值（与云端 planner 一致）：`static` / `breath` / `flow` / `chase` / `pulse` / `wave` / `sparkle`

设备处理规则：

- `enabled=false`：**关闭该通道输出**，停止发光（`color_mode` 和 `raw_stream` 均适用）
- `enabled=true`（或字段缺省）：正常输出
- `kind=color_mode`：本地接管该通道，不再消费该通道云帧
- `kind=raw_stream`：开始/继续消费该通道二进制帧

## 4.1 亮度推送（Text JSON，服务端 → 设备）

服务端会在连接建立后先下发一次当前亮度，后续仅在亮度变更时再次推送：

```json
{
  "type": "brightness_update",
  "payload": {
    "brightness": {
      "matrix": 0.8,
      "strip": 0.5
    },
    "updated_at_ms": 1730000000000
  }
}
```

设备处理规则：

- `payload.brightness.matrix`：矩阵输出亮度（0~1）
- `payload.brightness.strip`：灯带输出亮度（0~1）
- 建议设备将其作为最终输出增益，支持实时生效

## 4.2 亮度上报（Text JSON，设备 → 服务端）

设备端亮度发生变化时（如物理旋钮、按钮），向服务端主动上报：

```json
{
  "type": "set_brightness",
  "payload": {
    "matrix": 0.7,
    "strip": 0.9
  }
}
```

说明：

- `matrix` / `strip` 均为 0.0~1.0 的浮点数
- 服务端收到后立即持久化并向所有 App 客户端广播 `brightness_update`
- 若只需调节灯带，可将 `matrix` 传当前值保持不变；也可省略未变更字段，服务端会保留原值

## 4.3 开关推送（Text JSON，服务端 → 设备）

服务端在连接建立后先下发一次当前开关状态，后续仅在状态变更时再次推送：

```json
{
  "type": "power_update",
  "payload": {
    "power": {
      "matrix": true,
      "strip": false
    },
    "updated_at_ms": 1730000000000
  }
}
```

设备处理规则：

- `payload.power.matrix`：矩阵是否开启（`true`=开, `false`=关）
- `payload.power.strip`：**全部**灯带是否开启（`true`=开, `false`=关）
- 收到后立即更新本地开关状态，与 `brightness_update` 独立维护
- 关闭时建议直接关断对应通道输出，而非将亮度归零

> **注意**：`power_update` 和 `commands`（`enabled` 字段）均可控制通道输出。设备端建议以**两者任一为 false 即关闭**为准，即取逻辑与：`on = power.strip && command.enabled`

## 5. 帧协议（Binary）

二进制消息结构：

`FrameHeaderV1 (32 bytes, Big-Endian) + Payload`

### 5.1 FrameHeaderV1

| 字段 | 类型 | 说明 |
|---|---|---|
| magic | u32 | 固定 `ALHW` (`0x414C4857`) |
| version | u8 | 固定 `1` |
| target | u8 | `1`=matrix, `2`=strip |
| encoding | u8 | `1`=rgb24, `2`=rgb565；`3`=rgb111 为保留编码，当前硬件网关不主动下发 |
| flags | u8 | 预留，当前 `0` |
| channel_id | u16 | matrix=`0`; strip=`1` 仪表台，`2` 左前门板，`3` 左后门板，`4` 右前门板，`5` 右后门板，`6..N` 预留扩展 |
| sync_seq | u32 | 全局同步序号 |
| ts_ms | u64 | 时间戳毫秒 |
| param1 | u16 | matrix=width; strip=led_count |
| param2 | u16 | matrix=height; strip=`0` |
| reserved | u16 | 预留 |
| payload_len | u32 | Payload 长度（字节） |

### 5.2 Payload

- Matrix：row-major，`width * height * bytes_per_pixel`
- Strip：按 LED 顺序，`led_count * bytes_per_pixel`
- `rgb24`：`R,G,B` 三字节
- `rgb565`：16-bit 大端两字节
- `rgb111`：每像素 1 字节（保留编码；当前 `/ws/hw/v1` 不主动下发）

## 6. 同步与刷新策略

- `sync_seq` 相同代表同一拍（tick）
- 多通道同步建议：
  - 以 `sync_seq` 聚合各通道帧
  - 等待窗口 10~30ms 收齐后同时刷新
  - 超时缺帧时可用上一帧补齐

## 7. 设备端状态机建议

每通道维护以下状态：

| 状态变量 | 来源 | 说明 |
|---|---|---|
| `mode` | `commands.kind` | `COLOR_MODE` / `RAW_STREAM` |
| `cmd_enabled` | `commands.enabled` | 命令层开关（默认 true） |
| `power_on` | `power_update.power.*` | 全局开关层（矩阵/灯带各一路） |
| `brightness` | `brightness_update.brightness.*` | 亮度增益（0~1） |

**最终输出决策**：

```
output_active = cmd_enabled && power_on
```

两者任一为 false 则关断该通道。

**事件处理**：

- 收到 `commands`：更新 `mode` 和 `cmd_enabled`，切换通道逻辑
- 收到 `brightness_update`：实时更新亮度增益，立即生效
- 收到 `power_update`：更新 `power_on`，重新计算 `output_active`
- 收到 binary frame：
  - 校验 `magic/version/payload_len`
  - 按 `channel_id` 路由到实际端口
  - `output_active=false` 或通道当前为 `COLOR_MODE` 时可丢弃该帧

## 8. C SDK（WebSocket）

仓库已提供 WebSocket 版关键代码：

- `hardware/sdk/hw_gateway_sdk.h`
- `hardware/sdk/hw_gateway_sdk.c`
- `hardware/sdk/hw_gateway_sdk_self_test.c`（含自测用例，编译即可验证）

### SDK 提供的功能

| 函数 | 说明 |
|---|---|
| `hw_sdk_init()` | 初始化客户端（URL + 传输适配器） |
| `hw_sdk_ws_connect()` | 建立 WebSocket 连接 |
| `hw_sdk_ws_send_hello()` | 发送 hello 订阅消息 |
| `hw_sdk_ws_recv()` | 接收一条 WebSocket 消息 |
| `hw_sdk_ws_close()` | 关闭连接 |
| `hw_sdk_detect_text_type()` | 识别文本消息类型（hello_ack / commands / brightness_update / power_update） |
| `hw_sdk_parse_hello_ack()` | 解析 hello_ack，提取 sync_fps 和 encoding |
| `hw_sdk_parse_brightness_update()` | 解析 brightness_update，提取 matrix/strip 亮度值（自动 clamp 0~1） |
| `hw_sdk_parse_power_update()` | 解析 power_update，提取 matrix/strip 开关状态（bool → 0/1） |
| `hw_sdk_ws_send_brightness()` | 向服务端发送 set_brightness（设备 → 云端） |
| `hw_sdk_parse_frame()` | 解析二进制帧头，校验 magic/version，返回 payload 指针 |
| `hw_sdk_result_str()` | 错误码转字符串 |

### SDK 使用的数据结构

```c
/* 亮度值（矩阵 / 全局灯带） */
typedef struct {
    float matrix;  /* 0.0 ~ 1.0 */
    float strip;   /* 0.0 ~ 1.0 */
} hw_sdk_brightness_t;

/* 开关状态（矩阵 / 全局灯带） */
typedef struct {
    uint8_t matrix;  /* 0 = off, 1 = on */
    uint8_t strip;   /* 0 = off, 1 = on */
} hw_sdk_power_t;

/* hello_ack 解析结果 */
typedef struct {
    float sync_fps;
    char  encoding[24];  /* "rgb24" / "rgb565" / "rgb111" */
} hw_sdk_hello_ack_t;
```

### 你需要实现

- 平台 WS 适配层（实现 `hw_sdk_ws_transport_t` 的 connect/send_text/recv/close 四个函数），适配 ESP-IDF / lwIP / mbedTLS / RTOS 网络栈
- `commands` JSON 解析，驱动本地模式引擎（color_mode 时）

### 编译自测

```bash
gcc hardware/sdk/hw_gateway_sdk.c hardware/sdk/hw_gateway_sdk_self_test.c -o hw_self_test && ./hw_self_test
# 输出: hw_gateway_sdk_self_test: PASS
```

## 9. 联调建议

- 先只订阅一个通道（如 `strip:1`）确认链路稳定
- 再开启 `matrix:0 + strip:N` 验证 `sync_seq` 同步刷新
- 验证 `brightness_update` 解析后输出是否实时响应
- 验证 `power_update`：通过 `POST /api/app/power {"matrix":false,"strip":true}` 关闭矩阵，确认设备矩阵熄灭、灯带仍亮
- 验证 `commands.enabled=false`：通过 `POST /api/app/power {"matrix":true,"strip":false}` 关闭灯带，确认 `commands` 中 `strip:N` 携带 `enabled:false`
- 测试 `set_brightness` 上报：通过宝塔/App 查看云端亮度是否同步更新
- 统计并上报：断连次数、解析失败次数、帧丢失率

---

这份规范即为 WebSocket 单通道/多通道对接基线，实现后可直接进入联调。
