#ifndef HW_GATEWAY_SDK_H
#define HW_GATEWAY_SDK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HW_SDK_MAX_WS_URL 192
#define HW_SDK_FRAME_HEADER_SIZE 32
#define HW_SDK_FRAME_MAGIC_U32 0x414C4857u /* "ALHW" */

typedef enum {
    HW_SDK_OK = 0,
    HW_SDK_ERR_ARG = -1,
    HW_SDK_ERR_URL_TOO_LONG = -2,
    HW_SDK_ERR_TRANSPORT = -3,
    HW_SDK_ERR_PARSE = -4,
    HW_SDK_ERR_PROTOCOL = -5,
    HW_SDK_ERR_BUF_SMALL = -6
} hw_sdk_result_t;

typedef enum {
    HW_SDK_WS_MSG_NONE = 0,
    HW_SDK_WS_MSG_TEXT = 1,
    HW_SDK_WS_MSG_BINARY = 2,
    HW_SDK_WS_MSG_TIMEOUT = 3,
    HW_SDK_WS_MSG_CLOSED = 4
} hw_sdk_ws_msg_type_t;

typedef enum {
    HW_SDK_TEXT_UNKNOWN = 0,
    HW_SDK_TEXT_HELLO_ACK = 1,
    HW_SDK_TEXT_COMMANDS = 2,
    HW_SDK_TEXT_BRIGHTNESS_UPDATE = 3
} hw_sdk_text_type_t;

typedef enum {
    HW_SDK_TARGET_MATRIX = 1,
    HW_SDK_TARGET_STRIP = 2
} hw_sdk_target_t;

typedef struct {
    float matrix; /* 0.0 ~ 1.0 */
    float strip;  /* 0.0 ~ 1.0 */
} hw_sdk_brightness_t;

typedef struct {
    float sync_fps;
    char  encoding[24]; /* "rgb24" / "rgb565" / "rgb111" */
} hw_sdk_hello_ack_t;

typedef enum {
    HW_SDK_ENCODING_RGB24 = 1,
    HW_SDK_ENCODING_RGB565 = 2,
    HW_SDK_ENCODING_RGB111 = 3
} hw_sdk_encoding_t;

typedef struct {
    uint32_t magic;
    uint8_t version;
    uint8_t target;
    uint8_t encoding;
    uint8_t flags;
    uint16_t channel_id;
    uint32_t sync_seq;
    uint64_t ts_ms;
    uint16_t param1;
    uint16_t param2;
    uint16_t reserved;
    uint32_t payload_len;
} hw_sdk_frame_header_t;

/*
 * WebSocket platform hooks.
 *
 * Convention:
 * - connect/send/close: return 0 when success.
 * - recv: return 0 when a frame is received, non-zero for transport error.
 * - recv should map status to out_type:
 *   TEXT/BINARY/TIMEOUT/CLOSED.
 */
typedef int (*hw_sdk_ws_connect_fn)(void *ctx, const char *url);
typedef int (*hw_sdk_ws_send_text_fn)(void *ctx, const uint8_t *data, size_t len);
typedef int (*hw_sdk_ws_recv_fn)(
    void *ctx,
    uint8_t *out_buf,
    size_t out_cap,
    size_t *out_len,
    hw_sdk_ws_msg_type_t *out_type,
    uint32_t timeout_ms
);
typedef int (*hw_sdk_ws_close_fn)(void *ctx);

typedef struct {
    hw_sdk_ws_connect_fn connect;
    hw_sdk_ws_send_text_fn send_text;
    hw_sdk_ws_recv_fn recv;
    hw_sdk_ws_close_fn close;
    void *ctx;
} hw_sdk_ws_transport_t;

typedef struct {
    char ws_url[HW_SDK_MAX_WS_URL];
    hw_sdk_ws_transport_t transport;
} hw_sdk_client_t;

hw_sdk_result_t hw_sdk_init(
    hw_sdk_client_t *client,
    const char *ws_url,
    const hw_sdk_ws_transport_t *transport
);

hw_sdk_result_t hw_sdk_ws_connect(hw_sdk_client_t *client);

hw_sdk_result_t hw_sdk_ws_send_hello(
    hw_sdk_client_t *client,
    const char **channels,
    size_t channel_count,
    const char *prefer_encoding
);

hw_sdk_result_t hw_sdk_ws_recv(
    hw_sdk_client_t *client,
    uint8_t *out_buf,
    size_t out_cap,
    size_t *out_len,
    hw_sdk_ws_msg_type_t *out_type,
    uint32_t timeout_ms
);

hw_sdk_result_t hw_sdk_ws_close(hw_sdk_client_t *client);

hw_sdk_text_type_t hw_sdk_detect_text_type(const char *json_text);

/*
 * Parse a brightness_update JSON message from the server.
 * Extracts payload.brightness.matrix and payload.brightness.strip (clamped 0..1).
 * Example input:
 *   {"type":"brightness_update","payload":{"brightness":{"matrix":0.8,"strip":0.5},"updated_at_ms":1700000000000}}
 */
hw_sdk_result_t hw_sdk_parse_brightness_update(
    const char *json_text,
    hw_sdk_brightness_t *out
);

/*
 * Parse a hello_ack JSON message from the server.
 * Extracts payload.sync_fps and payload.encoding.
 * Example input:
 *   {"type":"hello_ack","payload":{"sync_fps":30.0,"encoding":"rgb565","channels":["strip:1"]}}
 */
hw_sdk_result_t hw_sdk_parse_hello_ack(
    const char *json_text,
    hw_sdk_hello_ack_t *out
);

/*
 * Send a set_brightness message to the server (hardware -> cloud).
 * Sends: {"type":"set_brightness","payload":{"matrix":<matrix>,"strip":<strip>}}
 * Values are clamped to [0.0, 1.0] before sending.
 */
hw_sdk_result_t hw_sdk_ws_send_brightness(
    hw_sdk_client_t *client,
    float matrix,
    float strip
);

hw_sdk_result_t hw_sdk_parse_frame(
    const uint8_t *frame,
    size_t frame_len,
    hw_sdk_frame_header_t *out_header,
    const uint8_t **out_payload
);

const char *hw_sdk_result_str(hw_sdk_result_t code);

#ifdef __cplusplus
}
#endif

#endif
