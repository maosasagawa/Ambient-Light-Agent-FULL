package com.light.agent.data.repository

import android.graphics.BitmapFactory
import android.util.Base64
import com.google.gson.Gson
import com.light.agent.data.api.ApiClient
import com.light.agent.data.api.LightApiService
import com.light.agent.data.python.PythonLightBridge
import com.light.agent.data.websocket.LightWebSocketClient
import com.light.agent.data.websocket.WsEvent
import com.light.agent.model.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

class LightRepository(
    private val wsClient: LightWebSocketClient,
    private val pythonBridge: PythonLightBridge? = null
) {

    private val _state = MutableStateFlow(LightUiState())
    val state: StateFlow<LightUiState> = _state.asStateFlow()

    private var api: LightApiService? = null
    private var backendMode: BackendMode = BackendMode.AUTO
    private var aiHubMixApiKey: String = ""
    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.IO)

    init {
        scope.launch {
            wsClient.events.collect { event ->
                when (event) {
                    is WsEvent.Connected -> _state.update {
                        it.copy(
                            isConnected = true,
                            effectiveBackend = BackendRuntime.ONLINE_BACKEND,
                            onlineHealth = OnlineHealth.CONNECTED
                        )
                    }
                    is WsEvent.Disconnected -> _state.update {
                        it.copy(
                            isConnected = false,
                            onlineHealth = OnlineHealth.OFFLINE,
                            effectiveBackend = if (backendMode == BackendMode.AUTO) {
                                BackendRuntime.NATIVE_FALLBACK
                            } else {
                                it.effectiveBackend
                            }
                        )
                    }
                    is WsEvent.PowerUpdate -> _state.update {
                        it.copy(isPoweredOn = event.matrix || event.strip)
                    }
                    is WsEvent.BrightnessUpdate -> _state.update {
                        it.copy(brightness = event.strip)
                    }
                    is WsEvent.StripCommandUpdate -> {
                        val matched = DefaultPresets.find { p -> p.mode == event.mode }
                        if (matched != null) _state.update { it.copy(selectedPreset = matched) }
                    }
                    else -> Unit
                }
            }
        }
    }

    fun configure(serverUrl: String) {
        configure(BackendMode.ONLINE_BACKEND, serverUrl)
    }

    fun configure(mode: BackendMode, serverUrl: String, apiKey: String = aiHubMixApiKey) {
        backendMode = mode
        aiHubMixApiKey = apiKey
        _state.update { it.copy(backendMode = mode) }
        when (mode) {
            BackendMode.LOCAL_FULL -> configureLocal(BackendRuntime.LOCAL_FULL)
            BackendMode.ONLINE_BACKEND -> configureOnline(serverUrl)
            BackendMode.AUTO -> {
                if (serverUrl.isBlank()) {
                    configureLocal(BackendRuntime.NATIVE_FALLBACK)
                } else {
                    configureOnline(serverUrl)
                }
            }
        }
    }

    private fun configureOnline(serverUrl: String) {
        api = ApiClient.getService(serverUrl)
        wsClient.connect(serverUrl)
        _state.update {
            it.copy(
                effectiveBackend = BackendRuntime.ONLINE_BACKEND,
                onlineHealth = OnlineHealth.UNKNOWN
            )
        }
    }

    private fun configureLocal(runtime: BackendRuntime) {
        api = null
        wsClient.disconnect()
        _state.update {
            it.copy(
                isConnected = true,
                effectiveBackend = runtime,
                onlineHealth = if (backendMode == BackendMode.LOCAL_FULL) OnlineHealth.UNKNOWN else OnlineHealth.OFFLINE
            )
        }
        if (runtime == BackendRuntime.LOCAL_FULL) {
            scope.launch {
                pythonBridge?.boot(aiHubMixApiKey)?.onFailure {
                    _state.update { state ->
                        state.copy(
                            effectiveBackend = BackendRuntime.NATIVE_FALLBACK,
                            onlineHealth = OnlineHealth.DEGRADED
                        )
                    }
                }
            }
        }
    }

    fun setBackendMode(mode: BackendMode, serverUrl: String) {
        configure(mode, serverUrl)
    }

    fun setAiHubMixApiKey(apiKey: String) {
        aiHubMixApiKey = apiKey
        if (backendMode == BackendMode.LOCAL_FULL) {
            scope.launch { pythonBridge?.boot(apiKey) }
        }
    }

    suspend fun fetchInitialState(): Result<Unit> {
        val svc = api ?: return Result.success(Unit)
        return runCatching {
            val appState = svc.getState()
            val command = appState.strip?.command?.command
            val matched = DefaultPresets.find { preset -> preset.mode == command?.mode }
            _state.update { ui ->
                ui.copy(
                    isPoweredOn = appState.power?.power?.strip ?: false,
                    brightness = appState.brightness?.brightness?.strip ?: 0.7f,
                    selectedPreset = matched
                )
            }
        }.onFailure {
            _state.update { state ->
                state.copy(
                    onlineHealth = OnlineHealth.DEGRADED,
                    effectiveBackend = if (backendMode == BackendMode.AUTO) BackendRuntime.NATIVE_FALLBACK else state.effectiveBackend,
                    isConnected = backendMode == BackendMode.AUTO
                )
            }
        }
    }

    suspend fun setPower(on: Boolean): Result<Unit> = runCatching {
        api?.setPower(HwPowerState(matrix = on, strip = on))
        _state.update { it.copy(isPoweredOn = on) }
    }

    suspend fun setBrightness(value: Float): Result<Unit> = runCatching {
        api?.setBrightness(HwBrightnessState(matrix = value, strip = value))
        _state.update { it.copy(brightness = value) }
    }

    suspend fun applyPreset(preset: ColorPreset, brightness: Float): Result<Unit> = runCatching {
        val cmd = StripCommandBody(
            mode = preset.mode,
            colors = preset.colors.map { StripColorEntry(rgb = listOf(it.r, it.g, it.b)) },
            brightness = brightness
        )
        api?.setStripCommand(cmd)
        _state.update { it.copy(selectedPreset = preset) }
    }

    suspend fun applyCustomCommand(
        mode: String,
        colors: List<RgbColor>,
        speed: Float,
        brightness: Float
    ): Result<Unit> = runCatching {
        val cmd = StripCommandBody(
            mode = mode,
            colors = colors.map { StripColorEntry(rgb = listOf(it.r, it.g, it.b)) },
            brightness = brightness,
            speed = speed
        )
        api?.setStripCommand(cmd)
        _state.update { it.copy(selectedPreset = null) }
    }

    suspend fun submitInstruction(text: String): Result<String?> = runCatching {
        val svc = api
        if (svc != null) {
            svc.submitInstruction(InstructionRequest(instruction = text)).speakableReason
        } else if (backendMode == BackendMode.LOCAL_FULL && pythonBridge != null) {
            val acceptJson = pythonBridge.acceptInstruction(text).getOrThrow()
            scope.launch { pythonBridge.generateLightingEffect(text) }
            gson.fromJson(acceptJson, VoiceAcceptResponse::class.java).speakableReason
        } else {
            "本地兜底模式已接管，基础灯控可继续使用。"
        }
    }

    suspend fun uploadMatrixImage(
        fileName: String,
        mediaType: String,
        bytes: ByteArray
    ): Result<MatrixDownsampleResponse> = runCatching {
        val svc = api
        if (svc != null) {
            val requestBody = bytes.toRequestBody(mediaType.toMediaTypeOrNull())
            val filePart = MultipartBody.Part.createFormData("file", fileName, requestBody)
            svc.uploadMatrixImage(filePart)
        } else if (backendMode == BackendMode.LOCAL_FULL && pythonBridge != null) {
            val bridge = pythonBridge
            runCatching {
                val json = bridge.downsampleImage(fileName, mediaType, bytes).getOrThrow()
                gson.fromJson(json, MatrixDownsampleResponse::class.java)
            }.getOrElse {
                _state.update { state -> state.copy(effectiveBackend = BackendRuntime.NATIVE_FALLBACK) }
                downsampleMatrixImageLocally(fileName, mediaType, bytes)
            }
        } else {
            downsampleMatrixImageLocally(fileName, mediaType, bytes)
        }
    }

    private fun downsampleMatrixImageLocally(
        fileName: String,
        mediaType: String,
        bytes: ByteArray
    ): MatrixDownsampleResponse {
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: error("无法解析图片")
        val scaled = android.graphics.Bitmap.createScaledBitmap(bitmap, 16, 16, true)
        val pixels = List(16) { y ->
            List(16) { x ->
                val color = scaled.getPixel(x, y)
                listOf(
                    android.graphics.Color.red(color),
                    android.graphics.Color.green(color),
                    android.graphics.Color.blue(color)
                )
            }
        }
        val raw = ByteArray(16 * 16 * 3)
        pixels.forEachIndexed { y, row ->
            row.forEachIndexed { x, rgb ->
                val idx = (y * 16 + x) * 3
                raw[idx] = rgb[0].toByte()
                raw[idx + 1] = rgb[1].toByte()
                raw[idx + 2] = rgb[2].toByte()
            }
        }
        if (scaled != bitmap) scaled.recycle()
        bitmap.recycle()
        return MatrixDownsampleResponse(
            jsonPayload = MatrixPixelData(width = 16, height = 16, pixels = pixels),
            rawBase64 = Base64.encodeToString(raw, Base64.NO_WRAP),
            filename = fileName,
            contentType = mediaType
        )
    }

    fun disconnect() = wsClient.disconnect()
}
