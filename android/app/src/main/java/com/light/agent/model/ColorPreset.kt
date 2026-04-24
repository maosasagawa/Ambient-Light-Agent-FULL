package com.light.agent.model

import androidx.compose.ui.graphics.Color

data class RgbColor(val r: Int, val g: Int, val b: Int)

data class ColorPreset(
    val id: String,
    val name: String,
    val effectLabel: String,           // shown as subtitle on card e.g. "流光", "呼吸"
    val mode: String,                  // backend mode string
    val colors: List<RgbColor>,
    val gradientStart: Color,
    val gradientEnd: Color
)

val DefaultPresets = listOf(
    ColorPreset(
        id = "warm_sun",
        name = "暖阳",
        effectLabel = "流光",
        mode = "flow",
        colors = listOf(RgbColor(255, 140, 60), RgbColor(255, 200, 80)),
        gradientStart = Color(0xFFFF8C3C),
        gradientEnd = Color(0xFFFFC850)
    ),
    ColorPreset(
        id = "daylight",
        name = "白昼",
        effectLabel = "常亮",
        mode = "static",
        colors = listOf(RgbColor(255, 240, 200)),
        gradientStart = Color(0xFFFFF0C8),
        gradientEnd = Color(0xFFFFD580)
    ),
    ColorPreset(
        id = "starlight",
        name = "星光",
        effectLabel = "闪烁",
        mode = "sparkle",
        colors = listOf(RgbColor(160, 80, 255), RgbColor(80, 160, 255)),
        gradientStart = Color(0xFFA050FF),
        gradientEnd = Color(0xFF50A0FF)
    ),
    ColorPreset(
        id = "aurora",
        name = "极光",
        effectLabel = "波浪",
        mode = "wave",
        colors = listOf(RgbColor(60, 255, 160), RgbColor(60, 200, 255)),
        gradientStart = Color(0xFF3CFFA0),
        gradientEnd = Color(0xFF3CC8FF)
    ),
    ColorPreset(
        id = "ocean",
        name = "海洋",
        effectLabel = "呼吸",
        mode = "breath",
        colors = listOf(RgbColor(30, 100, 255), RgbColor(60, 200, 255)),
        gradientStart = Color(0xFF1E64FF),
        gradientEnd = Color(0xFF3CC8FF)
    ),
    ColorPreset(
        id = "neon",
        name = "霓虹",
        effectLabel = "追逐",
        mode = "chase",
        colors = listOf(RgbColor(255, 80, 200), RgbColor(160, 40, 255)),
        gradientStart = Color(0xFFFF50C8),
        gradientEnd = Color(0xFFA028FF)
    ),
    ColorPreset(
        id = "pulse",
        name = "心跳",
        effectLabel = "脉冲",
        mode = "pulse",
        colors = listOf(RgbColor(255, 60, 80), RgbColor(255, 140, 60)),
        gradientStart = Color(0xFFFF3C50),
        gradientEnd = Color(0xFFFF8C3C)
    ),
    ColorPreset(
        id = "forest",
        name = "森林",
        effectLabel = "呼吸",
        mode = "breath",
        colors = listOf(RgbColor(40, 200, 80), RgbColor(80, 255, 120)),
        gradientStart = Color(0xFF28C850),
        gradientEnd = Color(0xFF50FF78)
    )
)
