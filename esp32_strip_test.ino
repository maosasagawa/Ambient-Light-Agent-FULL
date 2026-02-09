#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <Adafruit_NeoPixel.h>

// --- 配置区域 ---
const char* ssid = "您的WiFi名称";
const char* password = "您的WiFi密码";

// 硬件引脚和灯珠数量
#define LED_PIN     13    // ESP32 连接灯带的 GPIO 引脚
#define NUM_LEDS    60    // 您的灯带灯珠数量

// 接口地址
const String host = "https://light.dntc.com.cn";
const String url = host + "/api/hw/v1/frame/raw?channel=strip:1&encoding=rgb24";

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
WiFiClientSecure client;
HTTPClient http;

void setup() {
  Serial.begin(115200);
  strip.begin();
  strip.show(); 

  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");

  client.setInsecure(); // 跳过证书验证
  http.setReuse(true);  // 关键：启用连接复用 (Keep-Alive)
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // 仅在初始或连接断开时调用 begin
    if (!http.connected()) {
      Serial.println("Connecting to server (New TLS Handshake)...");
      http.begin(client, url);
      http.addHeader("Connection", "keep-alive");
    }

    int httpCode = http.GET();

    if (httpCode == HTTP_CODE_OK) {
      WiFiClient* stream = http.getStreamPtr();
      
      uint8_t header[32];
      // 使用 timedRead 保证在流不稳定时不会永久阻塞
      if (stream->readBytes(header, 32) == 32 && header[0] == 'A') {
        uint8_t rgb[3];
        for (int i = 0; i < NUM_LEDS; i++) {
          if (stream->readBytes(rgb, 3) == 3) {
            uint8_t r = pgm_read_byte(&gamma8[rgb[0]]);
            uint8_t g = pgm_read_byte(&gamma8[rgb[1]]);
            uint8_t b = pgm_read_byte(&gamma8[rgb[2]]);
            strip.setPixelColor(i, strip.Color(r, g, b));
          }
        }
        strip.show();
      }
    } else if (httpCode == 204) {
      // 保持沉默，无更新
    } else {
      Serial.printf("HTTP Error: %d\n", httpCode);
      http.end(); // 出错时断开重连
    }
  }
  delay(20); // 启用 Keep-Alive 后，尝试更快的刷新率
}
