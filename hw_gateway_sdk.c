#include "hw_gateway_sdk.h"

#include <stdio.h>
#include <string.h>

static uint16_t be16(const uint8_t *p) {
    return (uint16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

static uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) |
           (uint32_t)p[3];
}

static uint64_t be64(const uint8_t *p) {
    return ((uint64_t)p[0] << 56) |
           ((uint64_t)p[1] << 48) |
           ((uint64_t)p[2] << 40) |
           ((uint64_t)p[3] << 32) |
           ((uint64_t)p[4] << 24) |
           ((uint64_t)p[5] << 16) |
           ((uint64_t)p[6] << 8) |
           (uint64_t)p[7];
}

static int contains_any(const char *s, const char *a, const char *b) {
    if (s == NULL) {
        return 0;
    }
    if (a != NULL && strstr(s, a) != NULL) {
        return 1;
    }
    if (b != NULL && strstr(s, b) != NULL) {
        return 1;
    }
    return 0;
}

hw_sdk_result_t hw_sdk_init(
    hw_sdk_client_t *client,
    const char *ws_url,
    const hw_sdk_ws_transport_t *transport
) {
    size_t len;

    if (client == NULL || ws_url == NULL || transport == NULL) {
        return HW_SDK_ERR_ARG;
    }
    if (transport->connect == NULL || transport->send_text == NULL ||
        transport->recv == NULL || transport->close == NULL) {
        return HW_SDK_ERR_ARG;
    }

    len = strlen(ws_url);
    if (len == 0 || len >= HW_SDK_MAX_WS_URL) {
        return HW_SDK_ERR_URL_TOO_LONG;
    }

    memset(client, 0, sizeof(*client));
    memcpy(client->ws_url, ws_url, len);
    client->ws_url[len] = '\0';
    client->transport = *transport;
    return HW_SDK_OK;
}

hw_sdk_result_t hw_sdk_ws_connect(hw_sdk_client_t *client) {
    if (client == NULL) {
        return HW_SDK_ERR_ARG;
    }
    if (client->transport.connect(client->transport.ctx, client->ws_url) != 0) {
        return HW_SDK_ERR_TRANSPORT;
    }
    return HW_SDK_OK;
}

hw_sdk_result_t hw_sdk_ws_send_hello(
    hw_sdk_client_t *client,
    const char **channels,
    size_t channel_count,
    const char *prefer_encoding
) {
    uint8_t payload[384];
    size_t pos = 0;
    size_t i;
    int n;

    if (client == NULL) {
        return HW_SDK_ERR_ARG;
    }
    if (prefer_encoding == NULL || prefer_encoding[0] == '\0') {
        prefer_encoding = "rgb565";
    }

    n = snprintf((char *)(payload + pos), sizeof(payload) - pos,
                 "{\"type\":\"hello\",\"subscribe\":[");
    if (n < 0 || (size_t)n >= sizeof(payload) - pos) {
        return HW_SDK_ERR_BUF_SMALL;
    }
    pos += (size_t)n;

    for (i = 0; i < channel_count; i++) {
        const char *ch = channels != NULL ? channels[i] : NULL;
        if (ch == NULL || ch[0] == '\0') {
            continue;
        }
        n = snprintf((char *)(payload + pos), sizeof(payload) - pos,
                     "%s\"%s\"", (pos > 0 && payload[pos - 1] != '[') ? "," : "", ch);
        if (n < 0 || (size_t)n >= sizeof(payload) - pos) {
            return HW_SDK_ERR_BUF_SMALL;
        }
        pos += (size_t)n;
    }

    n = snprintf((char *)(payload + pos), sizeof(payload) - pos,
                 "],\"prefer_encoding\":\"%s\"}", prefer_encoding);
    if (n < 0 || (size_t)n >= sizeof(payload) - pos) {
        return HW_SDK_ERR_BUF_SMALL;
    }
    pos += (size_t)n;

    if (client->transport.send_text(client->transport.ctx, payload, pos) != 0) {
        return HW_SDK_ERR_TRANSPORT;
    }
    return HW_SDK_OK;
}

hw_sdk_result_t hw_sdk_ws_recv(
    hw_sdk_client_t *client,
    uint8_t *out_buf,
    size_t out_cap,
    size_t *out_len,
    hw_sdk_ws_msg_type_t *out_type,
    uint32_t timeout_ms
) {
    if (client == NULL || out_buf == NULL || out_len == NULL || out_type == NULL) {
        return HW_SDK_ERR_ARG;
    }
    if (client->transport.recv(client->transport.ctx, out_buf, out_cap, out_len, out_type, timeout_ms) != 0) {
        return HW_SDK_ERR_TRANSPORT;
    }
    return HW_SDK_OK;
}

hw_sdk_result_t hw_sdk_ws_close(hw_sdk_client_t *client) {
    if (client == NULL) {
        return HW_SDK_ERR_ARG;
    }
    if (client->transport.close(client->transport.ctx) != 0) {
        return HW_SDK_ERR_TRANSPORT;
    }
    return HW_SDK_OK;
}

hw_sdk_text_type_t hw_sdk_detect_text_type(const char *json_text) {
    if (contains_any(json_text, "\"type\":\"hello_ack\"", "\"type\": \"hello_ack\"")) {
        return HW_SDK_TEXT_HELLO_ACK;
    }
    if (contains_any(json_text, "\"type\":\"commands\"", "\"type\": \"commands\"")) {
        return HW_SDK_TEXT_COMMANDS;
    }
    if (contains_any(json_text, "\"type\":\"brightness_update\"", "\"type\": \"brightness_update\"")) {
        return HW_SDK_TEXT_BRIGHTNESS_UPDATE;
    }
    return HW_SDK_TEXT_UNKNOWN;
}

hw_sdk_result_t hw_sdk_parse_frame(
    const uint8_t *frame,
    size_t frame_len,
    hw_sdk_frame_header_t *out_header,
    const uint8_t **out_payload
) {
    const uint8_t *p;

    if (frame == NULL || out_header == NULL || out_payload == NULL) {
        return HW_SDK_ERR_ARG;
    }
    if (frame_len < HW_SDK_FRAME_HEADER_SIZE) {
        return HW_SDK_ERR_PARSE;
    }

    p = frame;
    out_header->magic = be32(p + 0);
    out_header->version = p[4];
    out_header->target = p[5];
    out_header->encoding = p[6];
    out_header->flags = p[7];
    out_header->channel_id = be16(p + 8);
    out_header->sync_seq = be32(p + 10);
    out_header->ts_ms = be64(p + 14);
    out_header->param1 = be16(p + 22);
    out_header->param2 = be16(p + 24);
    out_header->reserved = be16(p + 26);
    out_header->payload_len = be32(p + 28);

    if (out_header->magic != HW_SDK_FRAME_MAGIC_U32) {
        return HW_SDK_ERR_PROTOCOL;
    }
    if (out_header->version != 1) {
        return HW_SDK_ERR_PROTOCOL;
    }
    if (frame_len < HW_SDK_FRAME_HEADER_SIZE + (size_t)out_header->payload_len) {
        return HW_SDK_ERR_PARSE;
    }

    *out_payload = frame + HW_SDK_FRAME_HEADER_SIZE;
    return HW_SDK_OK;
}

const char *hw_sdk_result_str(hw_sdk_result_t code) {
    switch (code) {
        case HW_SDK_OK:
            return "ok";
        case HW_SDK_ERR_ARG:
            return "invalid argument";
        case HW_SDK_ERR_URL_TOO_LONG:
            return "url too long";
        case HW_SDK_ERR_TRANSPORT:
            return "transport error";
        case HW_SDK_ERR_PARSE:
            return "parse error";
        case HW_SDK_ERR_PROTOCOL:
            return "protocol error";
        case HW_SDK_ERR_BUF_SMALL:
            return "buffer too small";
        default:
            return "unknown";
    }
}
