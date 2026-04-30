package com.light.agent.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Snackbar
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.light.agent.model.BackendMode
import com.light.agent.model.ColorPreset
import com.light.agent.model.LightUiState
import com.light.agent.model.RgbColor
import com.light.agent.theme.Accent
import com.light.agent.theme.BgPrimary

@Composable
fun MainScreen(
    state: LightUiState,
    onTogglePower: () -> Unit,
    onToggleTakeover: () -> Unit,
    onBrightnessChange: (Float) -> Unit,
    onBrightnessCommit: (Float) -> Unit,
    onPresetSelect: (ColorPreset) -> Unit,
    onApplyCustom: (mode: String, colors: List<RgbColor>, speed: Float) -> Unit,
    onAiInputChange: (String) -> Unit,
    onAiSend: () -> Unit,
    onPickMatrixImage: () -> Unit,
    onServerDialogConfirm: (String) -> Unit,
    onBackendModeChange: (BackendMode, String) -> Unit,
    onDeveloperUnlock: () -> Unit,
    onAiHubMixKeyChange: (String) -> Unit,
    onShowServerDialog: () -> Unit,
    onDismissServerDialog: () -> Unit,
    onDismissError: () -> Unit
) {
    // Ambient background that subtly reflects the current preset colour
    val ambientTint by animateColorAsState(
        targetValue = state.selectedPreset?.gradientStart?.copy(alpha = 0.10f)
            ?: Accent.copy(alpha = 0.05f),
        animationSpec = tween(800),
        label = "ambient"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BgPrimary)
    ) {
        // Ambient radial glow from right-panel area
        Box(
            modifier = Modifier
                .fillMaxSize()
                .drawBehind {
                    drawRect(
                        brush = Brush.radialGradient(
                            colors = listOf(ambientTint, Color.Transparent),
                            center = Offset(size.width * 0.68f, size.height * 0.78f),
                            radius = size.width * 0.55f
                        )
                    )
                }
        )
        // Secondary left-panel accent wash
        Box(
            modifier = Modifier
                .fillMaxSize()
                .drawBehind {
                    drawRect(
                        brush = Brush.radialGradient(
                            colors = listOf(Accent.copy(alpha = 0.06f), Color.Transparent),
                            center = Offset(size.width * 0.12f, size.height * 0.90f),
                            radius = size.width * 0.28f
                        )
                    )
                }
        )

        Row(modifier = Modifier.fillMaxSize()) {
            LeftPanel(
                state = state,
                onTogglePower = onTogglePower,
                onAiInputChange = onAiInputChange,
                onAiSend = onAiSend,
                onPickMatrixImage = onPickMatrixImage,
                onSettingsClick = onShowServerDialog,
                onDeveloperUnlock = onDeveloperUnlock,
                modifier = Modifier.weight(0.38f)
            )
            // Thin vertical separator
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .width(1.dp)
                    .background(Color.White.copy(alpha = 0.06f))
            )
            RightPanel(
                state = state,
                onBrightnessChange = onBrightnessChange,
                onBrightnessCommit = onBrightnessCommit,
                onPresetSelect = onPresetSelect,
                onApplyCustom = onApplyCustom,
                modifier = Modifier.weight(0.62f)
            )
        }

        if (state.showServerDialog) {
            ServerSetupDialog(
                initialUrl = state.serverUrl,
                isVoiceTakeover = state.isVoiceTakeover,
                backendMode = state.backendMode,
                isDeveloperUnlocked = state.isDeveloperUnlocked,
                aiHubMixApiKey = state.aiHubMixApiKey,
                onToggleVoiceTakeover = onToggleTakeover,
                onBackendModeChange = onBackendModeChange,
                onAiHubMixKeyChange = onAiHubMixKeyChange,
                onConfirm = onServerDialogConfirm,
                onDismiss = if (state.serverUrl.isNotEmpty()) {
                    onDismissServerDialog
                } else null
            )
        }

        state.errorMessage?.let { msg ->
            Snackbar(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp),
                containerColor = MaterialTheme.colorScheme.surface,
                contentColor = MaterialTheme.colorScheme.onSurface,
                action = {
                    TextButton(onClick = onDismissError) { Text("关闭", color = Accent) }
                }
            ) { Text(msg) }
        }
    }
}
