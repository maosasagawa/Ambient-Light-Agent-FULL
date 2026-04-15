# 硬件网关对接文档（WebSocket Only, V1）

本版本只定义 WebSocket 对接，不使用 HTTP 轮询。

## 1. 对接目标

- 连接云端 `wss://light.dntc.com.cn/ws/hw/v1`
- 接收三类服务端消息：
  - 文本 JSON：`hello_ack`、`commands`、`brightness_update`
  - 二进制：`FrameHeaderV1(32B) + RGB Payload`
- 可主动向服务端发送：
  - `set_brightness`：上报设备端亮度（如物理旋钮调节）
- 设备按通道执行：
  - `color_mode`：端侧算法渲染
  - `raw_stream`：消费云端二进制帧

通道约定：

- `matrix:0`：矩阵
- `strip:1..N`：灯带

## 2. 最小接入流程

1. 建立 WS：`wss://light.dntc.com.cn/ws/hw/v1`
2. 发送 `hello`（订阅通道 + 编码偏好）
3. 收到 `hello_ack` 后进入接收循环（可用 `hw_sdk_parse_hello_ack()` 解析）
4. 收到 `commands`：更新每个通道状态机
5. 收到 `brightness_update`：更新矩阵/灯带输出亮度（可用 `hw_sdk_parse_brightness_update()` 解析）
6. 收到二进制帧：解析头部后按 `channel_id` 输出
7. （可选）设备端亮度变化时，发送 `set_brightness` 通知云端（可用 `hw_sdk_ws_send_brightness()` 发送）

## 3. 握手协议

客户端发送：

```json
{
  "type": "hello",
  "subscribe": ["matrix:0", "strip:1", "strip:2"],
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
    "channels": ["matrix:0", "strip:1", "strip:2"]
  }
}
```

说明：

- `subscribe` 可选；不传则默认订阅全部可用通道
- `prefer_encoding` 推荐 `rgb565`

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
- 若只需调节灯带，可将 `matrix` 传当前值保持不变（或省略，服务端保留原值）

## 5. 帧协议（Binary）

二进制消息结构：

`FrameHeaderV1 (32 bytes, Big-Endian) + Payload`

### 5.1 FrameHeaderV1

| 字段 | 类型 | 说明 |
|---|---|---|
| magic | u32 | 固定 `ALHW` (`0x414C4857`) |
| version | u8 | 固定 `1` |
| target | u8 | `1`=matrix, `2`=strip |
| encoding | u8 | `1`=rgb24, `2`=rgb565, `3`=rgb111 |
| flags | u8 | 预留，当前 `0` |
| channel_id | u16 | matrix=`0`; strip=`1..N` |
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
- `rgb111`：每像素 1 字节（位压缩）

## 6. 同步与刷新策略

- `sync_seq` 相同代表同一拍（tick）
- 多通道同步建议：
  - 以 `sync_seq` 聚合各通道帧
  - 等待窗口 10~30ms 收齐后同时刷新
  - 超时缺帧时可用上一帧补齐

## 7. 设备端状态机建议

- 每通道维护 `COLOR_MODE` / `RAW_STREAM`
- 接收 `commands` 时按通道切换状态
- 维护独立亮度状态：`matrix_brightness` / `strip_brightness`
- 接收 `brightness_update` 时实时更新两路亮度增益
- 接收 binary frame 时：
  - 校验 `magic/version/payload_len`
  - 按 `channel_id` 路由到实际端口
  - `channel` 当前为 `COLOR_MODE` 时可丢弃该帧

## 8. C SDK（WebSocket）

仓库已提供 WebSocket 版关键代码：

- `hw_gateway_sdk.h`
- `hw_gateway_sdk.c`
- `hw_gateway_sdk_self_test.c`（含自测用例，编译即可验证）

### SDK 提供的功能

| 函数 | 说明 |
|---|---|
| `hw_sdk_init()` | 初始化客户端（URL + 传输适配器） |
| `hw_sdk_ws_connect()` | 建立 WebSocket 连接 |
| `hw_sdk_ws_send_hello()` | 发送 hello 订阅消息 |
| `hw_sdk_ws_recv()` | 接收一条 WebSocket 消息 |
| `hw_sdk_ws_close()` | 关闭连接 |
| `hw_sdk_detect_text_type()` | 识别文本消息类型（hello_ack / commands / brightness_update） |
| `hw_sdk_parse_hello_ack()` | 解析 hello_ack，提取 sync_fps 和 encoding |
| `hw_sdk_parse_brightness_update()` | 解析 brightness_update，提取 matrix/strip 亮度值（自动 clamp 0~1） |
| `hw_sdk_ws_send_brightness()` | 向服务端发送 set_brightness（设备 → 云端） |
| `hw_sdk_parse_frame()` | 解析二进制帧头，校验 magic/version，返回 payload 指针 |
| `hw_sdk_result_str()` | 错误码转字符串 |

### SDK 使用的数据结构

```c
/* 亮度值（双路） */
typedef struct {
    float matrix;  /* 0.0 ~ 1.0 */
    float strip;   /* 0.0 ~ 1.0 */
} hw_sdk_brightness_t;

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
gcc hw_gateway_sdk.c hw_gateway_sdk_self_test.c -o hw_self_test && ./hw_self_test
# 输出: hw_gateway_sdk_self_test: PASS
```

## 9. 联调建议

- 先只订阅一个通道（如 `strip:1`）确认链路稳定
- 再开启 `matrix:0 + strip:N` 验证 `sync_seq` 同步刷新
- 验证 `brightness_update` 解析后输出是否实时响应
- 测试 `set_brightness` 上报：通过宝塔/App 查看云端亮度是否同步更新
- 统计并上报：断连次数、解析失败次数、帧丢失率

---

这份规范即为 WebSocket 单通道/多通道对接基线，实现后可直接进入联调。
