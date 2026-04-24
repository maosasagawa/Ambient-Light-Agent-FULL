package com.light.agent

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import com.light.agent.theme.LightAgentTheme
import com.light.agent.ui.MainScreen
import com.light.agent.viewmodel.MainViewModel

class MainActivity : ComponentActivity() {

    private val vm: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            LightAgentTheme {
                val state by vm.uiState.collectAsState()
                MainScreen(
                    state = state,
                    onTogglePower = vm::togglePower,
                    onToggleTakeover = vm::toggleVoiceTakeover,
                    onBrightnessChange = vm::setBrightness,
                    onBrightnessCommit = vm::commitBrightness,
                    onPresetSelect = vm::selectPreset,
                    onApplyCustom = vm::applyCustomCommand,
                    onAiInputChange = vm::updateAiInput,
                    onAiSend = vm::submitAiInstruction,
                    onServerDialogConfirm = vm::connect,
                    onShowServerDialog = vm::showServerDialog,
                    onDismissError = vm::dismissError
                )
            }
        }
    }
}
