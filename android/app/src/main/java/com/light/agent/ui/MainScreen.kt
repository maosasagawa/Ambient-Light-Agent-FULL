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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
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
    onServerDialogConfirm: (String) -> Unit,
    onShowServerDialog: () -> Unit,
    onDismissError: () -> Unit
) {
    // Ambient background that subtly reflects the current preset colour
    val ambientTint by animateColorAsState(
        targetValue = state.selectedPreset?.gradientStart?.copy(alpha = 0.10f)
            ?: Accent.copy(alpha = 0.06f),
        animationSpec = tween(600),
        label = "ambient"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BgPrimary)
    ) {
        // Ambient radial wash from bottom
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(Color.Transparent, ambientTint),
                        startY = 0f,
                        endY = Float.POSITIVE_INFINITY
                    )
                )
        )

        Row(modifier = Modifier.fillMaxSize()) {
            LeftPanel(
                state = state,
                onTogglePower = onTogglePower,
                onToggleTakeover = onToggleTakeover,
                onAiInputChange = onAiInputChange,
                onAiSend = onAiSend,
                onSettingsClick = onShowServerDialog,
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
                onConfirm = onServerDialogConfirm,
                onDismiss = if (state.serverUrl.isNotEmpty()) {
                    { onDismissError() }
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
