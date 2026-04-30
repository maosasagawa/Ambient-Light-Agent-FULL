package com.light.agent.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.painter.Painter
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.light.agent.theme.Accent
import com.light.agent.theme.AccentDark
import com.light.agent.theme.BgSurfaceHi
import com.light.agent.theme.StrokeSoft
import com.light.agent.theme.SwitchTrackOff
import com.light.agent.theme.TextMuted

@Composable
fun LargeToggle(
    label: String,
    subtitle: String,
    checked: Boolean,
    onToggle: () -> Unit,
    icon: Painter,
    modifier: Modifier = Modifier
) {
    val iconBgColor by animateColorAsState(
        targetValue = if (checked) Accent else BgSurfaceHi,
        animationSpec = tween(220),
        label = "iconBg"
    )
    val iconTint by animateColorAsState(
        targetValue = if (checked) Color.White else TextMuted,
        animationSpec = tween(220),
        label = "iconTint"
    )

    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surface)
            .border(1.dp, StrokeSoft, RoundedCornerShape(20.dp))
            .clickable(onClick = onToggle)
    ) {
        // Subtle top divider line for card edge definition on light bg
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(Color.Black.copy(alpha = 0.03f))
        )
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 18.dp, vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        // Icon badge
        Box(
            modifier = Modifier
                .size(46.dp)
                .clip(RoundedCornerShape(14.dp))
                .background(
                    if (checked) Brush.linearGradient(listOf(Accent, AccentDark))
                    else Brush.linearGradient(listOf(BgSurfaceHi, BgSurfaceHi))
                ),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                painter = icon,
                contentDescription = null,
                tint = iconTint,
                modifier = Modifier.size(22.dp)
            )
        }

        // Label + subtitle
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = label,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        PillSwitch(checked = checked)
    }
    } // close outer Box
}

@Composable
private fun PillSwitch(checked: Boolean) {
    val trackColor by animateColorAsState(
        targetValue = if (checked) Accent else SwitchTrackOff,
        animationSpec = tween(220), label = "track"
    )
    val thumbOffset by animateDpAsState(
        targetValue = if (checked) 28.dp else 4.dp,
        animationSpec = spring(stiffness = Spring.StiffnessMediumLow), label = "thumb"
    )
    Box(
        modifier = Modifier
            .width(58.dp)
            .height(32.dp)
            .clip(RoundedCornerShape(50))
            .background(trackColor)
    ) {
        Box(
            modifier = Modifier
                .offset(x = thumbOffset, y = 4.dp)
                .size(24.dp)
                .clip(CircleShape)
                .background(Color.White)
        )
    }
}
