package com.light.agent.data.repository

import com.light.agent.data.api.ApiClient
import com.light.agent.data.api.LightApiService
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

class LightRepository(private val wsClient: LightWebSocketClient) {

    private val _state = MutableStateFlow(LightUiState())
    val state: StateFlow<LightUiState> = _state.asStateFlow()

    private var api: LightApiService? = null
    private val scope = CoroutineScope(Dispatchers.IO)

    init {
        scope.launch {
            wsClient.events.collect { event ->
                when (event) {
                    is WsEvent.Connected -> _state.update { it.copy(isConnected = true) }
                    is WsEvent.Disconnected -> _state.update { it.copy(isConnected = false) }
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
        api = ApiClient.getService(serverUrl)
        wsClient.connect(serverUrl)
    }

    suspend fun fetchInitialState() {
        val svc = api ?: return
        runCatching {
            val appState = svc.getState()
            _state.update { ui ->
                ui.copy(
                    isPoweredOn = appState.power?.strip ?: false,
                    brightness = appState.brightness?.strip ?: 0.7f
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
        val resp = api?.submitInstruction(InstructionRequest(instruction = text))
        resp?.speakableReason
    }

    fun disconnect() = wsClient.disconnect()
}
