# Hardware Gateway C SDK

This SDK is the C-side integration baseline for the ambient-light hardware gateway WebSocket protocol.

## Files

| File | Purpose |
|---|---|
| `hw_gateway_sdk.h` | Public C API, protocol types, transport hook definitions |
| `hw_gateway_sdk.c` | SDK implementation: hello, receive wrapper, text type detection, brightness/power parsing, frame parsing |
| `hw_gateway_sdk_self_test.c` | Host-side self-test with mock WebSocket transport |
| `hw_gateway_ws_example.c` | Integration skeleton showing where platform WebSocket and device output code plug in |
| `../docs/HARDWARE_API_HW_GATEWAY.md` | Full WebSocket protocol, JSON messages, frame header, state-machine guidance |

## Quick self-test

On a PC with `gcc`:

```bash
make test
```

Expected output:

```text
hw_gateway_sdk_self_test: PASS
```

Equivalent raw command:

```bash
gcc -std=c99 -Wall -Wextra -pedantic hw_gateway_sdk.c hw_gateway_sdk_self_test.c -o hw_self_test
./hw_self_test
```

## Integration steps

1. Implement `hw_sdk_ws_transport_t` for the target platform:
   - `connect(ctx, url)`
   - `send_text(ctx, data, len)`
   - `recv(ctx, out_buf, out_cap, out_len, out_type, timeout_ms)`
   - `close(ctx)`
2. Initialize the client with `hw_sdk_init()` and connect to `wss://light.dntc.com.cn/ws/hw/v1`.
3. Send `hello` using `hw_sdk_ws_send_hello()` with subscribed channels such as `matrix:0`, `strip:1`, `strip:2`, `strip:3`, `strip:4`, `strip:5`.
4. In the receive loop:
   - For text frames, call `hw_sdk_detect_text_type()`.
   - Parse `hello_ack`, `brightness_update`, and `power_update` with the SDK helpers.
   - Parse `commands` according to `HARDWARE_API_HW_GATEWAY.md` and update the per-channel state machine.
   - For binary frames, call `hw_sdk_parse_frame()` and route by `target` + `channel_id`.
5. Gate final output with both command and power state:
   - `output_active = command.enabled && power_on`
6. If device-side brightness changes, report it with `hw_sdk_ws_send_brightness()`.

## Vehicle channel mapping

The first three strip channels are reserved for fixed cabin positions:

| Channel | Area | Notes |
|---|---|---|
| `strip:1` | Dashboard long strip | Main ambient strip, usually the longest strip |
| `strip:2` | Left front door panel | Left-front door ambient strip |
| `strip:3` | Left rear door panel | Left-rear door ambient strip |
| `strip:4` | Right front door panel | Right-front door ambient strip |
| `strip:5` | Right rear door panel | Right-rear door ambient strip |
| `strip:6..N` | Reserved expansion | Rear row, footwell, roof, or other areas; define per project |

Firmware should keep this mapping stable and must not reorder channels by physical discovery order.

## Platform responsibilities

The SDK deliberately does not include TLS, TCP, RTOS, JSON library, or LED-driver dependencies. The product firmware must provide:

- WebSocket transport implementation for ESP-IDF/lwIP/mbedTLS or the target network stack.
- Reconnect and backoff policy after disconnect or transport errors.
- Device-side `commands` JSON parsing for `color_mode` and `raw_stream` switching.
- LED output code for `rgb24`, `rgb565`, or `rgb111` payloads.
- Optional frame synchronization by `sync_seq` when multiple channels are subscribed.

## Acceptance checklist

- `make test` passes on host.
- Device receives `hello_ack` after sending `hello`.
- Device receives initial `commands`, `brightness_update`, and `power_update` after connection.
- `POST /api/app/power` changes are reflected by `power_update` and `commands.enabled`.
- `POST /api/app/brightness` changes are reflected by `brightness_update` and visible output gain.
- Binary frames pass `hw_sdk_parse_frame()` and route correctly by channel.
- Reconnect recovers after network interruption without requiring device reboot.
