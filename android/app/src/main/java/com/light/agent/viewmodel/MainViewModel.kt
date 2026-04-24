package com.light.agent.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.light.agent.data.api.ApiClient
import com.light.agent.data.prefs.AppPreferences
import com.light.agent.data.repository.LightRepository
import com.light.agent.data.websocket.LightWebSocketClient
import com.light.agent.model.ColorPreset
import com.light.agent.model.LightUiState
import com.light.agent.model.RgbColor
import com.light.agent.provider.TakeoverStateProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val prefs = AppPreferences(application)
    private val wsClient = LightWebSocketClient(ApiClient.getOkHttpClient())
    private val repo = LightRepository(wsClient)

    private val _uiState = MutableStateFlow(
        LightUiState(
            serverUrl = prefs.serverUrl,
            isVoiceTakeover = prefs.isTakeoverActive,
            showServerDialog = prefs.serverUrl.isEmpty()
        )
    )
    val uiState: StateFlow<LightUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            repo.state.collect { repoState ->
                _uiState.update { current ->
                    current.copy(
                        isPoweredOn = repoState.isPoweredOn,
                        brightness = repoState.brightness,
                        selectedPreset = repoState.selectedPreset,
                        isConnected = repoState.isConnected
                    )
                }
            }
        }
        if (prefs.serverUrl.isNotEmpty()) {
            connect(prefs.serverUrl)
        }
    }

    fun connect(url: String) {
        prefs.serverUrl = url
        _uiState.update { it.copy(serverUrl = url, showServerDialog = false) }
        repo.configure(url)
        viewModelScope.launch(Dispatchers.IO) {
            repo.fetchInitialState()
        }
    }

    fun togglePower() {
        val newValue = !_uiState.value.isPoweredOn
        viewModelScope.launch(Dispatchers.IO) {
            repo.setPower(newValue).onFailure { showError(it.message) }
        }
    }

    fun toggleVoiceTakeover() {
        val newValue = !_uiState.value.isVoiceTakeover
        _uiState.update { it.copy(isVoiceTakeover = newValue) }
        prefs.isTakeoverActive = newValue
        TakeoverStateProvider.setActive(getApplication(), newValue)
    }

    fun setBrightness(value: Float) {
        _uiState.update { it.copy(brightness = value) }
    }

    fun commitBrightness(value: Float) {
        viewModelScope.launch(Dispatchers.IO) {
            repo.setBrightness(value).onFailure { showError(it.message) }
        }
    }

    fun selectPreset(preset: ColorPreset) {
        viewModelScope.launch(Dispatchers.IO) {
            repo.applyPreset(preset, _uiState.value.brightness)
                .onFailure { showError(it.message) }
        }
    }

    fun applyCustomCommand(mode: String, colors: List<RgbColor>, speed: Float) {
        viewModelScope.launch(Dispatchers.IO) {
            repo.applyCustomCommand(mode, colors, speed, _uiState.value.brightness)
                .onFailure { showError(it.message) }
        }
    }

    fun updateAiInput(text: String) {
        _uiState.update { it.copy(aiInputText = text) }
    }

    fun submitAiInstruction() {
        val text = _uiState.value.aiInputText.trim()
        if (text.isEmpty()) return
        _uiState.update { it.copy(isLoading = true, aiInputText = "") }
        viewModelScope.launch(Dispatchers.IO) {
            repo.submitInstruction(text)
                .onSuccess { reason ->
                    _uiState.update { it.copy(isLoading = false, aiResponse = reason) }
                }
                .onFailure { err ->
                    _uiState.update { it.copy(isLoading = false) }
                    showError(err.message)
                }
        }
    }

    fun dismissAiResponse() = _uiState.update { it.copy(aiResponse = null) }
    fun dismissError() = _uiState.update { it.copy(errorMessage = null) }
    fun showServerDialog() = _uiState.update { it.copy(showServerDialog = true) }

    private fun showError(msg: String?) {
        _uiState.update { it.copy(errorMessage = msg ?: "未知错误") }
    }

    override fun onCleared() {
        super.onCleared()
        repo.disconnect()
    }
}
