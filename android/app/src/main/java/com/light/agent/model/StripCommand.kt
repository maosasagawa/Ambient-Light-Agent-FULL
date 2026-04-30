package com.light.agent.model

import com.google.gson.annotations.SerializedName

data class StripColorEntry(
    @SerializedName("name") val name: String? = null,
    @SerializedName("rgb") val rgb: List<Int>
)

data class StripCommandBody(
    @SerializedName("render_target") val renderTarget: String = "cloud",
    @SerializedName("mode") val mode: String,
    @SerializedName("colors") val colors: List<StripColorEntry>,
    @SerializedName("brightness") val brightness: Float = 0.7f,
    @SerializedName("speed") val speed: Float = 2.0f,
    @SerializedName("led_count") val ledCount: Int = 60,
    @SerializedName("mode_options") val modeOptions: Any? = null
)

data class HwPowerState(
    @SerializedName("matrix") val matrix: Boolean,
    @SerializedName("strip") val strip: Boolean
)

data class HwBrightnessState(
    @SerializedName("matrix") val matrix: Float,
    @SerializedName("strip") val strip: Float
)

data class InstructionRequest(
    @SerializedName("instruction") val instruction: String
)

// Envelopes for API responses
data class HwPowerEnvelope(
    @SerializedName("power") val power: HwPowerState? = null,
    @SerializedName("matrix") val matrix: Boolean? = null,
    @SerializedName("strip") val strip: Boolean? = null
)

data class HwBrightnessEnvelope(
    @SerializedName("brightness") val brightness: HwBrightnessState? = null,
    @SerializedName("matrix") val matrix: Float? = null,
    @SerializedName("strip") val strip: Float? = null
)

data class StripCommandEnvelope(
    @SerializedName("command") val command: StripCommandBody? = null,
    @SerializedName("updated_at_ms") val updatedAtMs: Long? = null
)

data class VoiceAcceptResponse(
    @SerializedName("status") val status: String? = null,
    @SerializedName("speakable_reason") val speakableReason: String? = null,
    @SerializedName("plan") val plan: Any? = null
)

data class MatrixPixelData(
    @SerializedName("width") val width: Int,
    @SerializedName("height") val height: Int,
    @SerializedName("pixels") val pixels: List<List<List<Int>>>? = null
)

data class MatrixDownsampleResponse(
    @SerializedName("json") val jsonPayload: MatrixPixelData? = null,
    @SerializedName("raw_base64") val rawBase64: String? = null,
    @SerializedName("filename") val filename: String? = null,
    @SerializedName("content_type") val contentType: String? = null
)

data class StripState(
    @SerializedName("colors") val colors: List<List<Int>>? = null,
    @SerializedName("command") val command: StripCommandEnvelope? = null
)

data class AppState(
    @SerializedName("matrix") val matrix: MatrixPixelData? = null,
    @SerializedName("strip") val strip: StripState? = null,
    @SerializedName("power") val power: HwPowerEnvelope? = null,
    @SerializedName("brightness") val brightness: HwBrightnessEnvelope? = null
)
