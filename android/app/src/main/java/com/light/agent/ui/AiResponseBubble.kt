package com.light.agent.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.light.agent.theme.Accent
import com.light.agent.theme.AccentContainer
import com.light.agent.theme.AccentSoft

@Composable
fun AiResponseBubble(response: String, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(AccentContainer)
            .border(1.dp, Accent.copy(alpha = 0.35f), RoundedCornerShape(16.dp))
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.Top,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .offset(y = 8.dp)
                .background(Accent, RoundedCornerShape(50))
        )
        Column {
            Text(
                text = "AI 反馈",
                fontSize = 10.sp,
                fontWeight = FontWeight.SemiBold,
                color = AccentSoft
            )
            Spacer(Modifier.height(2.dp))
            Text(
                text = response,
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White.copy(alpha = 0.92f),
                fontSize = 13.sp
            )
        }
    }
}
