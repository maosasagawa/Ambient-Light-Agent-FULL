package com.light.agent.model

data class LightUiState(
    val isPoweredOn: Boolean = false,
    val isVoiceTakeover: Boolean = false,
    val brightness: Float = 0.7f,
    val selectedPreset: ColorPreset? = null,
    val backendMode: BackendMode = BackendMode.AUTO,
    val effectiveBackend: BackendRuntime = BackendRuntime.NATIVE_FALLBACK,
    val onlineHealth: OnlineHealth = OnlineHealth.UNKNOWN,
    val isDeveloperUnlocked: Boolean = false,
    val aiHubMixApiKey: String = "",
    val aiInputText: String = "",
    val isLoading: Boolean = false,
    val isUploadingMatrix: Boolean = false,
    val serverUrl: String = "",
    val showServerDialog: Boolean = false,
    val aiResponse: String? = null,
    val matrixUploadSummary: String? = null,
    val errorMessage: String? = null,
    val isConnected: Boolean = false,
    val localServerAddress: String? = null
)
