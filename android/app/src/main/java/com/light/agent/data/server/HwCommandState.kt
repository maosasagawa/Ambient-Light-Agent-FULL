package com.light.agent.data.server

data class HwCommandState(
    val stripMode: String = "static",
    val stripColors: List<List<Int>> = listOf(listOf(255, 200, 100)),
    val stripBrightness: Float = 0.7f,
    val stripSpeed: Float = 2.0f,
    val matrixBrightness: Float = 0.7f,
    val isPoweredOn: Boolean = true
)
