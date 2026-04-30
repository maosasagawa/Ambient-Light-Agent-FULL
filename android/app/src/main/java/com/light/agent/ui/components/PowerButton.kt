package com.light.agent.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
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
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.light.agent.R
import com.light.agent.theme.Accent
import com.light.agent.theme.AccentDark
import com.light.agent.theme.BgSurface
import com.light.agent.theme.BgSurfaceHi
import com.light.agent.theme.StrokeSoft
import com.light.agent.theme.TextMuted

@Composable
fun PowerButton(
    isPoweredOn: Boolean,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier
) {
    val shape = RoundedCornerShape(20.dp)

    val infiniteTransition = rememberInfiniteTransition(label = "powerPulse")

    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = if (isPoweredOn) 0.55f else 0f,
        targetValue = 0f,
        animationSpec = infiniteRepeatable(
            animation = tween(1400, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "pulseAlpha"
    )
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = if (isPoweredOn) 1.05f else 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1400, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "pulseScale"
    )

    val iconTint by animateColorAsState(
        targetValue = if (isPoweredOn) Color.White else TextMuted,
        animationSpec = tween(300),
        label = "iconTint"
    )

    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(96.dp)
    ) {
        // Pulse ring drawn behind the main card
        if (isPoweredOn) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .scale(pulseScale)
                    .clip(shape)
                    .border(2.dp, Accent.copy(alpha = pulseAlpha), shape)
            )
        }

        // Main card
        Box(
            modifier = Modifier
                .fillMaxSize()
                .shadow(
                    elevation = if (isPoweredOn) 20.dp else 4.dp,
                    shape = shape,
                    spotColor = if (isPoweredOn) Accent else Color.Transparent,
                    ambientColor = if (isPoweredOn) Accent.copy(0.4f) else Color.Transparent
                )
                .clip(shape)
                .background(
                    if (isPoweredOn) {
                        Brush.radialGradient(
                            colors = listOf(Accent.copy(alpha = 0.85f), AccentDark.copy(alpha = 0.95f))
                        )
                    } else {
                        Brush.linearGradient(listOf(BgSurface, BgSurfaceHi))
                    }
                )
                .border(
                    1.dp,
                    if (isPoweredOn) Accent.copy(0.45f) else StrokeSoft,
                    shape
                )
                .clickable(onClick = onToggle)
        ) {
            // Inner top highlight: only on the orange ON state
            if (isPoweredOn) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .fillMaxHeight()
                        .background(
                            Brush.verticalGradient(
                                listOf(Color.White.copy(alpha = 0.18f), Color.Transparent),
                                endY = 80f
                            )
                        )
                )
            }

            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 22.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Icon circle with glow
                Box(
                    modifier = Modifier
                        .size(46.dp)
                        .shadow(
                            elevation = if (isPoweredOn) 14.dp else 0.dp,
                            shape = CircleShape,
                            spotColor = Accent,
                            ambientColor = Accent.copy(0.5f)
                        )
                        .clip(CircleShape)
                        .background(
                            if (isPoweredOn) Color.White.copy(alpha = 0.22f)
                            else Color.White.copy(alpha = 0.06f)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        painter = painterResource(R.drawable.ic_bulb),
                        contentDescription = null,
                        tint = iconTint,
                        modifier = Modifier.size(22.dp)
                    )
                }

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "氛围灯",
                        fontSize = 17.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = if (isPoweredOn) Color.White else MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = if (isPoweredOn) "已点亮" else "已关闭",
                        fontSize = 12.sp,
                        color = if (isPoweredOn) Color.White.copy(0.75f)
                               else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                // ON / OFF pill
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(50))
                        .background(
                            if (isPoweredOn) Color.White.copy(alpha = 0.20f) else BgSurfaceHi
                        )
                        .padding(horizontal = 16.dp, vertical = 7.dp)
                ) {
                    Text(
                        text = if (isPoweredOn) "ON" else "OFF",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (isPoweredOn) Color.White
                               else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}
