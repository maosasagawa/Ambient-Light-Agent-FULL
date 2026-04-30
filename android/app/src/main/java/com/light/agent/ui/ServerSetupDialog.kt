package com.light.agent.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.light.agent.R
import com.light.agent.model.BackendMode
import com.light.agent.theme.Accent
import com.light.agent.theme.AccentDark
import com.light.agent.theme.BgSurface
import com.light.agent.theme.BgSurfaceHi
import com.light.agent.theme.StrokeSoft

@Composable
fun ServerSetupDialog(
    initialUrl: String,
    isVoiceTakeover: Boolean,
    backendMode: BackendMode,
    isDeveloperUnlocked: Boolean,
    aiHubMixApiKey: String,
    localServerAddress: String? = null,
    onToggleVoiceTakeover: () -> Unit,
    onBackendModeChange: (BackendMode, String) -> Unit,
    onAiHubMixKeyChange: (String) -> Unit,
    onConfirm: (String) -> Unit,
    onDismiss: (() -> Unit)? = null
) {
    var url by remember { mutableStateOf(initialUrl.ifEmpty { "http://192.168.1." }) }
    var keyText by remember(aiHubMixApiKey) { mutableStateOf(aiHubMixApiKey) }

    Dialog(onDismissRequest = { onDismiss?.invoke() }) {
        Column(
            modifier = Modifier
                .width(480.dp)
                .clip(RoundedCornerShape(22.dp))
                .background(BgSurface)
                .border(1.dp, StrokeSoft, RoundedCornerShape(22.dp))
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Brush.linearGradient(listOf(Accent, AccentDark))),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        painter = painterResource(R.drawable.ic_link),
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(22.dp)
                    )
                }
                Column {
                    Text("连接后端服务", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurface)
                    Text("FastAPI 氛围灯服务地址", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }

            OutlinedTextField(
                value = url,
                onValueChange = { url = it },
                label = { Text("服务地址", color = MaterialTheme.colorScheme.onSurfaceVariant) },
                placeholder = { Text("http://192.168.1.100:8000", color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.5f)) },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Accent,
                    focusedLabelColor = Accent,
                    unfocusedBorderColor = StrokeSoft,
                    unfocusedLabelColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    cursorColor = Accent,
                    focusedContainerColor = BgSurfaceHi,
                    unfocusedContainerColor = BgSurfaceHi,
                    focusedTextColor = MaterialTheme.colorScheme.onSurface,
                    unfocusedTextColor = MaterialTheme.colorScheme.onSurface
                )
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(BgSurfaceHi)
                    .border(1.dp, StrokeSoft, RoundedCornerShape(16.dp))
                    .padding(horizontal = 16.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    Text(
                        text = "接管语音",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = if (isVoiceTakeover) "语音将路由到 Agent" else "使用默认语音助手",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = isVoiceTakeover,
                    onCheckedChange = { onToggleVoiceTakeover() },
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.White,
                        checkedTrackColor = Accent,
                        uncheckedThumbColor = Color.White,
                        uncheckedTrackColor = Accent.copy(alpha = 0.28f)
                    )
                )
            }

            if (isDeveloperUnlocked) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(18.dp))
                        .background(Color.White.copy(alpha = 0.035f))
                        .border(1.dp, Accent.copy(alpha = 0.20f), RoundedCornerShape(18.dp))
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(
                            text = "开发者后端",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                        Text(
                            text = "本地完整移植 / 在线后端 / 自动兜底",
                            fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        BackendMode.entries.forEach { mode ->
                            val selected = backendMode == mode
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .height(38.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(if (selected) Accent.copy(alpha = 0.22f) else BgSurfaceHi)
                                    .border(
                                        1.dp,
                                        if (selected) Accent.copy(alpha = 0.75f) else StrokeSoft,
                                        RoundedCornerShape(12.dp)
                                    )
                                    .clickable { onBackendModeChange(mode, url.trim()) },
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = when (mode) {
                                        BackendMode.LOCAL_FULL -> "本地"
                                        BackendMode.ONLINE_BACKEND -> "在线"
                                        BackendMode.AUTO -> "自动"
                                    },
                                    fontSize = 12.sp,
                                    fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium,
                                    color = if (selected) Accent else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }

                    OutlinedTextField(
                        value = keyText,
                        onValueChange = {
                            keyText = it
                            onAiHubMixKeyChange(it)
                        },
                        label = { Text("AIHubMix API Key", color = MaterialTheme.colorScheme.onSurfaceVariant) },
                        placeholder = { Text("sk-...", color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.45f)) },
                        singleLine = true,
                        visualTransformation = PasswordVisualTransformation(),
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Accent,
                            focusedLabelColor = Accent,
                            unfocusedBorderColor = StrokeSoft,
                            unfocusedLabelColor = MaterialTheme.colorScheme.onSurfaceVariant,
                            cursorColor = Accent,
                            focusedContainerColor = BgSurfaceHi,
                            unfocusedContainerColor = BgSurfaceHi,
                            focusedTextColor = MaterialTheme.colorScheme.onSurface,
                            unfocusedTextColor = MaterialTheme.colorScheme.onSurface
                        )
                    )

                    if (localServerAddress != null) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(BgSurfaceHi)
                                .border(1.dp, StrokeSoft, RoundedCornerShape(12.dp))
                                .padding(horizontal = 14.dp, vertical = 10.dp),
                            verticalArrangement = Arrangement.spacedBy(2.dp)
                        ) {
                            Text(
                                text = "本机硬件出口",
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = localServerAddress,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = Accent
                            )
                        }
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                if (onDismiss != null) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f).height(48.dp),
                        shape = RoundedCornerShape(14.dp),
                        border = androidx.compose.foundation.BorderStroke(1.dp, StrokeSoft)
                    ) {
                        Text("取消", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .height(48.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .background(Brush.linearGradient(listOf(Accent, AccentDark)))
                        .clickable { if (url.trim().isNotEmpty()) onConfirm(url.trim()) },
                    contentAlignment = Alignment.Center
                ) {
                    Text("连接", color = Color.White, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
                }
            }
        }
    }
}
