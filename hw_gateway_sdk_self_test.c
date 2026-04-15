#include "hw_gateway_sdk.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    char last_url[256];
    char last_sent[512];
    size_t last_sent_len;
} mock_ws_ctx_t;

static int mock_connect(void *ctx, const char *url) {
    mock_ws_ctx_t *m = (mock_ws_ctx_t *)ctx;
    size_t n = strlen(url);
    if (n >= sizeof(m->last_url)) {
        return -1;
    }
    memcpy(m->last_url, url, n + 1);
    return 0;
}

static int mock_send_text(void *ctx, const uint8_t *data, size_t len) {
    mock_ws_ctx_t *m = (mock_ws_ctx_t *)ctx;
    if (len >= sizeof(m->last_sent)) {
        return -1;
    }
    memcpy(m->last_sent, data, len);
    m->last_sent[len] = '\0';
    m->last_sent_len = len;
    return 0;
}

static int mock_recv(
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
    (void)timeout_ms;
    *out_type = HW_SDK_WS_MSG_TIMEOUT;
    return 0;
}

static int mock_close(void *ctx) {
    (void)ctx;
    return 0;
}

static int test_init_connect_hello(void) {
    hw_sdk_client_t client;
    hw_sdk_ws_transport_t t;
    mock_ws_ctx_t ctx;
    const char *channels[] = {"matrix:0", "strip:1"};
    hw_sdk_result_t rc;

    memset(&ctx, 0, sizeof(ctx));
    memset(&t, 0, sizeof(t));
    t.connect = mock_connect;
    t.send_text = mock_send_text;
    t.recv = mock_recv;
    t.close = mock_close;
    t.ctx = &ctx;

    rc = hw_sdk_init(&client, "wss://light.dntc.com.cn/ws/hw/v1", &t);
    if (rc != HW_SDK_OK) {
        return 1;
    }
    rc = hw_sdk_ws_connect(&client);
    if (rc != HW_SDK_OK) {
        return 2;
    }
    if (strcmp(ctx.last_url, "wss://light.dntc.com.cn/ws/hw/v1") != 0) {
        return 3;
    }

    rc = hw_sdk_ws_send_hello(&client, channels, 2, "rgb565");
    if (rc != HW_SDK_OK) {
        return 4;
    }
    if (strstr(ctx.last_sent, "\"type\":\"hello\"") == NULL) {
        return 5;
    }
    if (strstr(ctx.last_sent, "\"matrix:0\"") == NULL) {
        return 6;
    }
    if (strstr(ctx.last_sent, "\"strip:1\"") == NULL) {
        return 7;
    }
    if (strstr(ctx.last_sent, "\"prefer_encoding\":\"rgb565\"") == NULL) {
        return 8;
    }

    return 0;
}

/* Actual server format includes updated_at_ms inside payload */
#define BRIGHTNESS_UPDATE_JSON \
    "{\"type\":\"brightness_update\",\"payload\":{\"brightness\":{\"matrix\":0.8,\"strip\":0.5},\"updated_at_ms\":1700000000000}}"

#define HELLO_ACK_JSON \
    "{\"type\":\"hello_ack\",\"payload\":{\"sync_fps\":30.0,\"encoding\":\"rgb565\",\"channels\":[\"strip:1\"]}}"

static int test_detect_text_type(void) {
    const char *hello_ack = HELLO_ACK_JSON;
    const char *commands = "{\"type\":\"commands\",\"payload\":{}}";
    const char *brightness_update = BRIGHTNESS_UPDATE_JSON;

    if (hw_sdk_detect_text_type(hello_ack) != HW_SDK_TEXT_HELLO_ACK) {
        return 1;
    }
    if (hw_sdk_detect_text_type(commands) != HW_SDK_TEXT_COMMANDS) {
        return 2;
    }
    if (hw_sdk_detect_text_type(brightness_update) != HW_SDK_TEXT_BRIGHTNESS_UPDATE) {
        return 3;
    }
    return 0;
}

static int test_parse_brightness_update(void) {
    hw_sdk_brightness_t b;
    hw_sdk_result_t rc;

    rc = hw_sdk_parse_brightness_update(BRIGHTNESS_UPDATE_JSON, &b);
    if (rc != HW_SDK_OK) {
        return 1;
    }
    /* Allow small floating-point tolerance */
    if (b.matrix < 0.79f || b.matrix > 0.81f) {
        return 2;
    }
    if (b.strip < 0.49f || b.strip > 0.51f) {
        return 3;
    }

    /* Clamping: values above 1.0 should clamp to 1.0 */
    rc = hw_sdk_parse_brightness_update(
        "{\"type\":\"brightness_update\",\"payload\":{\"brightness\":{\"matrix\":1.5,\"strip\":-0.2},"
        "\"updated_at_ms\":1}}",
        &b
    );
    if (rc != HW_SDK_OK) {
        return 4;
    }
    if (b.matrix != 1.0f) {
        return 5;
    }
    if (b.strip != 0.0f) {
        return 6;
    }
    return 0;
}

static int test_parse_hello_ack(void) {
    hw_sdk_hello_ack_t ack;
    hw_sdk_result_t rc;

    rc = hw_sdk_parse_hello_ack(HELLO_ACK_JSON, &ack);
    if (rc != HW_SDK_OK) {
        return 1;
    }
    if (ack.sync_fps < 29.9f || ack.sync_fps > 30.1f) {
        return 2;
    }
    if (strcmp(ack.encoding, "rgb565") != 0) {
        return 3;
    }
    return 0;
}

static int test_send_brightness(void) {
    hw_sdk_client_t client;
    hw_sdk_ws_transport_t t;
    mock_ws_ctx_t ctx;
    hw_sdk_result_t rc;

    memset(&ctx, 0, sizeof(ctx));
    memset(&t, 0, sizeof(t));
    t.connect = mock_connect;
    t.send_text = mock_send_text;
    t.recv = mock_recv;
    t.close = mock_close;
    t.ctx = &ctx;

    rc = hw_sdk_init(&client, "wss://light.dntc.com.cn/ws/hw/v1", &t);
    if (rc != HW_SDK_OK) {
        return 1;
    }

    rc = hw_sdk_ws_send_brightness(&client, 0.6f, 0.8f);
    if (rc != HW_SDK_OK) {
        return 2;
    }
    if (strstr(ctx.last_sent, "\"type\":\"set_brightness\"") == NULL) {
        return 3;
    }
    if (strstr(ctx.last_sent, "\"matrix\"") == NULL) {
        return 4;
    }
    if (strstr(ctx.last_sent, "\"strip\"") == NULL) {
        return 5;
    }

    /* Clamping: values out of range should still produce valid JSON */
    rc = hw_sdk_ws_send_brightness(&client, -0.5f, 1.5f);
    if (rc != HW_SDK_OK) {
        return 6;
    }
    /* After clamping, matrix should be 0.0000 and strip should be 1.0000 */
    if (strstr(ctx.last_sent, "\"matrix\":0.0000") == NULL) {
        return 7;
    }
    if (strstr(ctx.last_sent, "\"strip\":1.0000") == NULL) {
        return 8;
    }
    return 0;
}

static int test_parse_frame(void) {
    uint8_t frame[32 + 4];
    hw_sdk_frame_header_t h;
    const uint8_t *payload = NULL;
    hw_sdk_result_t rc;

    memset(frame, 0, sizeof(frame));
    frame[0] = 0x41;
    frame[1] = 0x4C;
    frame[2] = 0x48;
    frame[3] = 0x57;
    frame[4] = 1;
    frame[5] = 2;
    frame[6] = 2;
    frame[7] = 0;
    frame[8] = 0;
    frame[9] = 1;
    frame[10] = 0;
    frame[11] = 0;
    frame[12] = 0;
    frame[13] = 7;
    frame[22] = 0;
    frame[23] = 60;
    frame[24] = 0;
    frame[25] = 0;
    frame[28] = 0;
    frame[29] = 0;
    frame[30] = 0;
    frame[31] = 4;
    frame[32] = 0x11;
    frame[33] = 0x22;
    frame[34] = 0x33;
    frame[35] = 0x44;

    rc = hw_sdk_parse_frame(frame, sizeof(frame), &h, &payload);
    if (rc != HW_SDK_OK) {
        return 1;
    }
    if (h.target != HW_SDK_TARGET_STRIP || h.channel_id != 1 || h.payload_len != 4 || h.sync_seq != 7) {
        return 2;
    }
    if (payload == NULL || payload[0] != 0x11 || payload[3] != 0x44) {
        return 3;
    }

    return 0;
}

int main(void) {
    int rc;

    rc = test_init_connect_hello();
    if (rc != 0) {
        printf("test_init_connect_hello failed: %d\n", rc);
        return 1;
    }

    rc = test_detect_text_type();
    if (rc != 0) {
        printf("test_detect_text_type failed: %d\n", rc);
        return 1;
    }

    rc = test_parse_frame();
    if (rc != 0) {
        printf("test_parse_frame failed: %d\n", rc);
        return 1;
    }

    rc = test_parse_brightness_update();
    if (rc != 0) {
        printf("test_parse_brightness_update failed: %d\n", rc);
        return 1;
    }

    rc = test_parse_hello_ack();
    if (rc != 0) {
        printf("test_parse_hello_ack failed: %d\n", rc);
        return 1;
    }

    rc = test_send_brightness();
    if (rc != 0) {
        printf("test_send_brightness failed: %d\n", rc);
        return 1;
    }

    printf("hw_gateway_sdk_self_test: PASS\n");
    return 0;
}
