package com.light.agent.viewmodel

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.light.agent.data.api.ApiClient
import com.light.agent.data.prefs.AppPreferences
import com.light.agent.data.python.PythonLightBridge
import com.light.agent.data.repository.LightRepository
import com.light.agent.data.websocket.LightWebSocketClient
import com.light.agent.model.BackendMode
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
    private val pythonBridge = PythonLightBridge(application)
    private val repo = LightRepository(wsClient, pythonBridge)

    private val _uiState = MutableStateFlow(
        LightUiState(
            serverUrl = prefs.serverUrl,
            isVoiceTakeover = prefs.isTakeoverActive,
            backendMode = prefs.backendMode,
            isDeveloperUnlocked = prefs.developerUnlocked,
            aiHubMixApiKey = prefs.aiHubMixApiKey,
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
                        isConnected = repoState.isConnected,
                        backendMode = repoState.backendMode,
                        effectiveBackend = repoState.effectiveBackend,
                        onlineHealth = repoState.onlineHealth
                    )
                }
            }
        }
        if (prefs.serverUrl.isNotEmpty() || prefs.backendMode != BackendMode.ONLINE_BACKEND) {
            configureBackend(prefs.backendMode, prefs.serverUrl)
        }
    }

    fun connect(url: String) {
        prefs.serverUrl = url
        _uiState.update { it.copy(serverUrl = url, showServerDialog = false) }
        repo.configure(prefs.backendMode, url, prefs.aiHubMixApiKey)
        viewModelScope.launch(Dispatchers.IO) {
            repo.fetchInitialState().onFailure { showError(it.message) }
        }
    }

    fun configureBackend(mode: BackendMode, url: String = _uiState.value.serverUrl) {
        prefs.backendMode = mode
        if (url.isNotBlank()) prefs.serverUrl = url
        _uiState.update {
            it.copy(
                backendMode = mode,
                serverUrl = url,
                showServerDialog = false
            )
        }
        repo.configure(mode, url, prefs.aiHubMixApiKey)
        viewModelScope.launch(Dispatchers.IO) {
            repo.fetchInitialState().onFailure { showError(it.message) }
        }
    }

    fun updateAiHubMixApiKey(apiKey: String) {
        prefs.aiHubMixApiKey = apiKey
        _uiState.update { it.copy(aiHubMixApiKey = apiKey) }
        repo.setAiHubMixApiKey(apiKey)
    }

    fun unlockDeveloperOptions() {
        prefs.developerUnlocked = true
        _uiState.update { it.copy(isDeveloperUnlocked = true) }
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

    fun uploadMatrixImage(uri: Uri) {
        if (_uiState.value.serverUrl.isBlank() && _uiState.value.backendMode == BackendMode.ONLINE_BACKEND) {
            showError("请先连接服务")
            return
        }
        _uiState.update { it.copy(isUploadingMatrix = true, matrixUploadSummary = null) }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val resolver = getApplication<Application>().contentResolver
                val bytes = resolver.openInputStream(uri)?.use { it.readBytes() }
                    ?: error("无法读取图片")
                val fileName = queryDisplayName(uri) ?: "matrix-upload.png"
                val mediaType = resolver.getType(uri) ?: "image/png"
                repo.uploadMatrixImage(fileName, mediaType, bytes).getOrThrow()
            }.onSuccess { response ->
                val summary = response.jsonPayload?.let {
                    "已上传到 ${it.width}×${it.height} 矩阵"
                } ?: "矩阵图片上传成功"
                _uiState.update {
                    it.copy(
                        isUploadingMatrix = false,
                        matrixUploadSummary = summary
                    )
                }
            }.onFailure { err ->
                _uiState.update { it.copy(isUploadingMatrix = false) }
                showError(err.message)
            }
        }
    }

    fun dismissAiResponse() = _uiState.update { it.copy(aiResponse = null) }
    fun dismissError() = _uiState.update { it.copy(errorMessage = null) }
    fun showServerDialog() = _uiState.update { it.copy(showServerDialog = true) }
    fun dismissServerDialog() = _uiState.update { it.copy(showServerDialog = false) }

    private fun showError(msg: String?) {
        _uiState.update { it.copy(errorMessage = msg ?: "未知错误") }
    }

    private fun queryDisplayName(uri: Uri): String? {
        val resolver = getApplication<Application>().contentResolver
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (!cursor.moveToFirst()) return null
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index < 0) return null
            return cursor.getString(index)
        }
        return null
    }

    override fun onCleared() {
        super.onCleared()
        repo.disconnect()
    }
}
