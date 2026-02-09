#include <WiFi.h>
#include <WebSocketsClient.h> // 请在库管理器搜索并安装 "WebSockets" (by Markus Sattler)
#include <Adafruit_NeoPixel.h>

// --- 配置区域 ---
const char* ssid = "Oneplus";
const char* password = "12345677";

#define LED_PIN     13
#define NUM_LEDS    60

// WebSocket 接口配置
// 域名: light.dntc.com.cn -> 对应 WSS (443端口)
const char* ws_host = "light.dntc.com.cn";
const int ws_port = 443;
const char* ws_url = "/ws/hw/v1";

// --- 初始化 ---
const uint8_t PROGMEM gamma8[] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
    1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
    2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,
    5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10,
   10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16,
   17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25,
   25, 26, 27, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36,
   37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 50,
   51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68,
   69, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89,
   90, 92, 93, 95, 96, 98, 99,101,102,104,105,107,109,110,112,114,
  115,117,119,120,122,124,126,127,129,131,133,135,137,138,140,142,
  144,146,148,150,152,154,156,158,160,162,164,167,169,171,173,175,
  177,180,182,184,186,189,191,193,196,198,200,203,205,208,210,213,
  215,218,220,223,225,228,231,233,236,239,241,244,247,249,252,255 };

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);
WebSocketsClient webSocket;

// 处理 WebSocket 事件
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] 状态: 已断开连接");
      break;
    case WStype_CONNECTED:
      Serial.printf("[WS] 状态: 已连接到 %s\n", payload);
      // 根据项目协议，连接后发送 hello 进行订阅
      // 订阅 strip:1 通道，编码使用 rgb24
      webSocket.sendTXT("{\"type\":\"hello\",\"subscribe\":[\"strip:1\"],\"prefer_encoding\":\"rgb24\"}");
      break;
    case WStype_TEXT:
      // 服务器发来的 JSON 命令（如模式切换、亮度调节等）
      Serial.printf("[WS] 收到指令: %s\n", payload);
      break;
    case WStype_BIN:
      // 收到二进制流：FrameHeaderV1 (32字节) + RGB数据
      if (length >= 32) {
        // 验证协议 Magic: "ALHW"
        if (payload[0] == 'A' && payload[1] == 'L' && payload[2] == 'H' && payload[3] == 'W') {
          // payload[2] 为 target (2=strip), payload[11] 为 channel_id
          
          uint8_t* rgbData = payload + 32; // 跳过 32 字节帧头
          size_t pixelBytes = length - 32;
          int ledsToUpdate = pixelBytes / 3;
          
          // 如果数据长度超过硬件定义的灯珠数，取最小值
          int count = (ledsToUpdate < NUM_LEDS) ? ledsToUpdate : NUM_LEDS;
          
          for (int i = 0; i < count; i++) {
            uint8_t r = pgm_read_byte(&gamma8[rgbData[i*3]]);
            uint8_t g = pgm_read_byte(&gamma8[rgbData[i*3+1]]);
            uint8_t b = pgm_read_byte(&gamma8[rgbData[i*3+2]]);
            strip.setPixelColor(i, strip.Color(r, g, b));
          }
          strip.show();
        }
      }
      break;
    case WStype_ERROR:
      Serial.println("[WS] 错误!");
      break;
  }
}

void setup() {
  Serial.begin(115200);
  strip.begin();
  strip.show(); 

  Serial.print("正在连接 WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi 已连接");

  // 初始化 WebSocket 连接 (SSL加密)
  webSocket.beginSSL(ws_host, ws_port, ws_url);
  webSocket.onEvent(webSocketEvent);
  
  // 设置自动重连
  webSocket.setReconnectInterval(5000);
}

void loop() {
  webSocket.loop();
}
