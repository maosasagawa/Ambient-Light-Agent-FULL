package com.light.agent.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.light.agent.model.LightUiState
import com.light.agent.ui.components.AiInputBar
import com.light.agent.ui.components.AppHeader
import com.light.agent.ui.components.MatrixUploadCard
import com.light.agent.ui.components.PowerButton

@Composable
fun LeftPanel(
    state: LightUiState,
    onTogglePower: () -> Unit,
    onAiInputChange: (String) -> Unit,
    onAiSend: () -> Unit,
    onPickMatrixImage: () -> Unit,
    onSettingsClick: () -> Unit,
    onDeveloperUnlock: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxHeight()
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        AppHeader(
            isConnected = state.isConnected,
            onSettingsClick = onSettingsClick,
            onDeveloperUnlock = onDeveloperUnlock
        )

        Spacer(Modifier.height(2.dp))

        PowerButton(
            isPoweredOn = state.isPoweredOn,
            onToggle = onTogglePower
        )

        MatrixUploadCard(
            isUploading = state.isUploadingMatrix,
            summary = state.matrixUploadSummary,
            onClick = onPickMatrixImage
        )

        Spacer(modifier = Modifier.weight(1f))

        AnimatedVisibility(visible = state.aiResponse != null) {
            state.aiResponse?.let { AiResponseBubble(response = it) }
        }

        AiInputBar(
            text = state.aiInputText,
            onTextChange = onAiInputChange,
            onSend = onAiSend,
            isLoading = state.isLoading
        )
    }
}
