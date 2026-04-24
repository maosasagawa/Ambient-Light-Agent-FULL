package com.light.agent.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.light.agent.R
import com.light.agent.model.LightUiState
import com.light.agent.ui.components.AiInputBar
import com.light.agent.ui.components.AppHeader
import com.light.agent.ui.components.LargeToggle

@Composable
fun LeftPanel(
    state: LightUiState,
    onTogglePower: () -> Unit,
    onToggleTakeover: () -> Unit,
    onAiInputChange: (String) -> Unit,
    onAiSend: () -> Unit,
    onSettingsClick: () -> Unit,
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
            onSettingsClick = onSettingsClick
        )

        Spacer(Modifier.height(2.dp))

        LargeToggle(
            label = "氛围灯",
            subtitle = if (state.isPoweredOn) "已点亮" else "已关闭",
            checked = state.isPoweredOn,
            onToggle = onTogglePower,
            icon = painterResource(R.drawable.ic_bulb)
        )

        LargeToggle(
            label = "接管语音",
            subtitle = if (state.isVoiceTakeover) "语音将路由到 Agent" else "使用默认语音助手",
            checked = state.isVoiceTakeover,
            onToggle = onToggleTakeover,
            icon = painterResource(R.drawable.ic_mic)
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
