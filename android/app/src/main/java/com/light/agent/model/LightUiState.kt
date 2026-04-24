package com.light.agent.model

data class LightUiState(
    val isPoweredOn: Boolean = false,
    val isVoiceTakeover: Boolean = false,
    val brightness: Float = 0.7f,
    val selectedPreset: ColorPreset? = null,
    val aiInputText: String = "",
    val isLoading: Boolean = false,
    val serverUrl: String = "",
    val showServerDialog: Boolean = false,
    val aiResponse: String? = null,
    val errorMessage: String? = null,
    val isConnected: Boolean = false
)
