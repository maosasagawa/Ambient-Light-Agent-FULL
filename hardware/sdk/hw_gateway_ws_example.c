#include "hw_gateway_sdk.h"

#include <stdio.h>
#include <string.h>

/* Replace with your platform websocket context. */
typedef struct {
    int placeholder;
} platform_ws_ctx_t;

/* ===== Platform adapter stubs (must implement) ===== */
static int platform_ws_connect(void *ctx, const char *url) {
    (void)ctx;
    (void)url;
    return -1;
}

static int platform_ws_send_text(void *ctx, const uint8_t *data, size_t len) {
    (void)ctx;
    (void)data;
    (void)len;
    return -1;
}

static int platform_ws_recv(
    void *ctx,
    uint8_t *out_buf,
    size_t out_cap,
    size_t *out_len,
    hw_sdk_ws_msg_type_t *out_type,
    uint32_t timeout_ms
) {
    (void)ctx;
    (void)out_buf;
    (void)out_cap;
    (void)out_len;
    (void)out_type;
    (void)timeout_ms;
    return -1;
}

static int platform_ws_close(void *ctx) {
    (void)ctx;
    return 0;
}

/* ===== Device handlers ===== */
static void on_commands_json(const char *json_text, size_t len) {
    printf("commands json len=%u\n", (unsigned)len);
    (void)json_text;
    /*
     * Parse and update per-channel mode:
     * - color_mode: switch to device rendering
     * - raw_stream: consume binary frames for this channel
     */
}

static void on_brightness_update_json(const char *json_text, size_t len) {
    printf("brightness_update json len=%u\n", (unsigned)len);
    (void)json_text;
    /*
     * Parse payload.brightness.matrix / payload.brightness.strip,
     * then update the hardware output gain for matrix and strip.
     */
}

static void on_power_update_json(const char *json_text, size_t len) {
    printf("power_update json len=%u\n", (unsigned)len);
    (void)json_text;
    /*
     * Parse payload.power.matrix / payload.power.strip,
     * then gate the final output state for matrix and strip.
     */
}

static void output_frame_to_hardware(const hw_sdk_frame_header_t *h, const uint8_t *payload) {
    if (h->target == HW_SDK_TARGET_MATRIX) {
        printf("matrix id=%u seq=%u w=%u h=%u enc=%u len=%u\n",
               h->channel_id, h->sync_seq, h->param1, h->param2, h->encoding, h->payload_len);
        (void)payload;
        return;
    }

    if (h->target == HW_SDK_TARGET_STRIP) {
        printf("strip id=%u seq=%u leds=%u enc=%u len=%u\n",
               h->channel_id, h->sync_seq, h->param1, h->encoding, h->payload_len);
        (void)payload;
    }
}

int main(void) {
    hw_sdk_client_t client;
    hw_sdk_ws_transport_t transport;
    platform_ws_ctx_t ws_ctx;
    hw_sdk_result_t rc;

    uint8_t rx_buf[65536];
    size_t rx_len = 0;
    hw_sdk_ws_msg_type_t msg_type = HW_SDK_WS_MSG_NONE;

    const char *channels[] = {"matrix:0", "strip:1", "strip:2", "strip:3", "strip:4", "strip:5"};

    memset(&ws_ctx, 0, sizeof(ws_ctx));
    memset(&transport, 0, sizeof(transport));
    transport.connect = platform_ws_connect;
    transport.send_text = platform_ws_send_text;
    transport.recv = platform_ws_recv;
    transport.close = platform_ws_close;
    transport.ctx = &ws_ctx;

    rc = hw_sdk_init(&client, "wss://light.dntc.com.cn/ws/hw/v1", &transport);
    if (rc != HW_SDK_OK) {
        printf("init failed: %s\n", hw_sdk_result_str(rc));
        return 1;
    }

    rc = hw_sdk_ws_connect(&client);
    if (rc != HW_SDK_OK) {
        printf("connect failed: %s\n", hw_sdk_result_str(rc));
        return 1;
    }

    rc = hw_sdk_ws_send_hello(&client, channels, 6, "rgb565");
    if (rc != HW_SDK_OK) {
        printf("send hello failed: %s\n", hw_sdk_result_str(rc));
        hw_sdk_ws_close(&client);
        return 1;
    }

    for (;;) {
        rc = hw_sdk_ws_recv(&client, rx_buf, sizeof(rx_buf), &rx_len, &msg_type, 3000);
        if (rc != HW_SDK_OK) {
            printf("recv error: %s\n", hw_sdk_result_str(rc));
            break;
        }

        if (msg_type == HW_SDK_WS_MSG_TIMEOUT) {
            continue;
        }
        if (msg_type == HW_SDK_WS_MSG_CLOSED) {
            printf("server closed\n");
            break;
        }

        if (msg_type == HW_SDK_WS_MSG_TEXT) {
            hw_sdk_text_type_t t;
            if (rx_len >= sizeof(rx_buf)) {
                continue;
            }
            rx_buf[rx_len] = '\0';
            t = hw_sdk_detect_text_type((const char *)rx_buf);
            if (t == HW_SDK_TEXT_HELLO_ACK) {
                printf("hello_ack received\n");
            } else if (t == HW_SDK_TEXT_COMMANDS) {
                on_commands_json((const char *)rx_buf, rx_len);
            } else if (t == HW_SDK_TEXT_BRIGHTNESS_UPDATE) {
                on_brightness_update_json((const char *)rx_buf, rx_len);
            } else if (t == HW_SDK_TEXT_POWER_UPDATE) {
                on_power_update_json((const char *)rx_buf, rx_len);
            }
            continue;
        }

        if (msg_type == HW_SDK_WS_MSG_BINARY) {
            hw_sdk_frame_header_t h;
            const uint8_t *payload = NULL;
            rc = hw_sdk_parse_frame(rx_buf, rx_len, &h, &payload);
            if (rc != HW_SDK_OK) {
                printf("frame parse failed: %s\n", hw_sdk_result_str(rc));
                continue;
            }
            output_frame_to_hardware(&h, payload);
        }
    }

    hw_sdk_ws_close(&client);
    return 0;
}
