# 氛围灯 APP — 语音侧对接文档

## 概述

氛围灯 APP 通过 Android ContentProvider 暴露两个接口：

| 接口 | 用途 |
|------|------|
| 查询接管状态 | 判断氛围灯 APP 当前是否要求接管语音结果 |
| 投递 STT 文字 | 将语音识别结果推送给氛围灯 APP 处理 |

当接管状态为**开启**时，语音侧不再将 STT 结果直发后端，改为通过 ContentProvider 投递给氛围灯 APP，由其负责 AI 解析和灯光控制。

---

## 前置配置

在你的 APP 的 `AndroidManifest.xml` 中声明读写权限：

```xml
<!-- 查询接管状态需要读权限 -->
<uses-permission android:name="com.light.agent.permission.READ_STATE" />

<!-- 投递 STT 文字需要写权限 -->
<uses-permission android:name="com.light.agent.permission.WRITE_STATE" />
```

> 这两个权限的 `protectionLevel` 均为 `normal`，安装时自动授予，无需运行时弹窗。

---

## 接口一：查询接管状态

**URI：** `content://com.light.agent.provider/takeover`

**调用方式：** `contentResolver.query()`

**返回列：**

| 列名 | 类型 | 说明 |
|------|------|------|
| `is_active` | Int | `1` = 接管开启，`0` = 关闭 |

**示例代码（Kotlin）：**

```kotlin
fun isLightAppTakeover(): Boolean {
    val uri = Uri.parse("content://com.light.agent.provider/takeover")
    return contentResolver.query(uri, null, null, null, null)?.use { cursor ->
        cursor.moveToFirst() && cursor.getInt(cursor.getColumnIndexOrThrow("is_active")) == 1
    } ?: false
}
```

**实时监听接管状态变化（推荐）：**

```kotlin
// 在 Activity/Service 中注册 ContentObserver，避免每次说话前轮询
private val takeoverObserver = object : ContentObserver(Handler(Looper.getMainLooper())) {
    override fun onChange(selfChange: Boolean) {
        isTakeoverActive = isLightAppTakeover()
    }
}

override fun onStart() {
    super.onStart()
    contentResolver.registerContentObserver(
        Uri.parse("content://com.light.agent.provider/takeover"),
        false,
        takeoverObserver
    )
    isTakeoverActive = isLightAppTakeover() // 初始化一次
}

override fun onStop() {
    super.onStop()
    contentResolver.unregisterContentObserver(takeoverObserver)
}
```

---

## 接口二：投递 STT 文字

**URI：** `content://com.light.agent.provider/voice_input`

**调用方式：** `contentResolver.insert()`

**ContentValues 字段：**

| 字段名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `text` | String | 是 | STT 识别出的完整文字 |

**示例代码（Kotlin）：**

```kotlin
fun deliverSttResult(text: String) {
    val uri = Uri.parse("content://com.light.agent.provider/voice_input")
    val values = ContentValues().apply {
        put("text", text)
    }
    contentResolver.insert(uri, values)
}
```

---

## 完整调用流程

```kotlin
// 在你的语音识别结果回调中

fun onSttResult(recognizedText: String) {
    if (isTakeoverActive) {
        // 接管模式：投递给氛围灯 APP，不直发后端
        deliverSttResult(recognizedText)
    } else {
        // 正常模式：走原有后端逻辑
        sendToOriginalBackend(recognizedText)
    }
}
```

---

## 时序图

```
语音助手 APP                         氛围灯 APP
     │                                   │
     │  用户按下语音按钮                    │
     │                                   │
     │── query(takeover) ───────────────>│
     │<── is_active = 1 ────────────────│
     │                                   │
     │  STT 识别完成："把灯变成蓝色呼吸"     │
     │                                   │
     │── insert(voice_input, text) ─────>│
     │                                   │── ContentObserver.onChange()
     │                                   │── submitAiInstruction("把灯变成蓝色呼吸")
     │                                   │── POST /api/app/submit
     │                                   │── 灯光响应
```

---

## 边界情况说明

**`is_active = 0` 时：** 直接走你们原有流程，无需调用 `voice_input` 接口。

**氛围灯 APP 未安装/未启动：** `query()` 返回 `null`，视为接管关闭，走原有流程即可。安全处理示例：

```kotlin
fun isLightAppTakeover(): Boolean {
    return try {
        val uri = Uri.parse("content://com.light.agent.provider/takeover")
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            cursor.moveToFirst() && cursor.getInt(cursor.getColumnIndexOrThrow("is_active")) == 1
        } ?: false
    } catch (e: Exception) {
        false // APP 未安装或 Provider 不可用
    }
}
```

**STT 结果为空字符串：** 氛围灯 APP 会自动忽略，无需在你侧过滤。

**并发投递：** 每次 `insert` 覆盖上一次，氛围灯 APP 读取后立即清除，不存在队列积压。

---

## 联调验证

可用 `adb shell` 直接模拟投递，无需修改代码即可验证氛围灯 APP 响应：

```bash
# 1. 检查接管状态
adb shell content query \
  --uri content://com.light.agent.provider/takeover

# 2. 模拟投递 STT 结果
adb shell content insert \
  --uri content://com.light.agent.provider/voice_input \
  --bind text:s:"把灯变成蓝色呼吸效果"
```

第二条命令执行后，氛围灯 APP 应在 1 秒内响应（可观察灯光变化或 APP 界面 AI 响应气泡）。
