package com.light.agent.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.light.agent.R
import com.light.agent.theme.Accent
import com.light.agent.theme.AccentDark
import com.light.agent.theme.StrokeSoft

@Composable
fun AiInputBar(
    text: String,
    onTextChange: (String) -> Unit,
    onSend: () -> Unit,
    isLoading: Boolean,
    modifier: Modifier = Modifier
) {
    val canSend = text.trim().isNotEmpty() && !isLoading

    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(22.dp))
            .background(MaterialTheme.colorScheme.surface)
            .border(1.dp, StrokeSoft, RoundedCornerShape(22.dp))
            .padding(horizontal = 6.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        BasicInput(
            text = text,
            onTextChange = onTextChange,
            onSend = { if (canSend) onSend() },
            enabled = !isLoading,
            modifier = Modifier.weight(1f).padding(start = 14.dp)
        )

        SendFab(
            active = canSend,
            loading = isLoading,
            onClick = { if (canSend) onSend() }
        )
    }
}

@Composable
private fun BasicInput(
    text: String,
    onTextChange: (String) -> Unit,
    onSend: () -> Unit,
    enabled: Boolean,
    modifier: Modifier = Modifier
) {
    TextField(
        value = text,
        onValueChange = onTextChange,
        modifier = modifier,
        placeholder = {
            Text(
                text = "告诉氛围灯你想要什么…",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 15.sp
            )
        },
        colors = TextFieldDefaults.colors(
            focusedContainerColor = Color.Transparent,
            unfocusedContainerColor = Color.Transparent,
            disabledContainerColor = Color.Transparent,
            focusedIndicatorColor = Color.Transparent,
            unfocusedIndicatorColor = Color.Transparent,
            disabledIndicatorColor = Color.Transparent,
            focusedTextColor = MaterialTheme.colorScheme.onSurface,
            unfocusedTextColor = MaterialTheme.colorScheme.onSurface,
            cursorColor = Accent
        ),
        textStyle = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Normal),
        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
        keyboardActions = KeyboardActions(onSend = { onSend() }),
        maxLines = 2,
        enabled = enabled
    )
}

@Composable
private fun SendFab(active: Boolean, loading: Boolean, onClick: () -> Unit) {
    val size = 48.dp
    val brush = if (active) Brush.linearGradient(listOf(Accent, AccentDark))
                else Brush.linearGradient(listOf(Color(0xFF3A3D45), Color(0xFF2E3039)))

    Box(
        modifier = Modifier
            .size(size)
            .shadow(
                elevation = if (active) 8.dp else 0.dp,
                shape = CircleShape,
                spotColor = Accent,
                ambientColor = Accent
            )
            .clip(CircleShape)
            .background(brush)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        if (loading) {
            CircularProgressIndicator(
                modifier = Modifier.size(22.dp),
                color = Color.White,
                strokeWidth = 2.5.dp
            )
        } else {
            Icon(
                painter = painterResource(R.drawable.ic_send),
                contentDescription = "发送",
                tint = Color.White.copy(alpha = if (active) 1f else 0.4f),
                modifier = Modifier.size(18.dp)
            )
        }
    }
}
