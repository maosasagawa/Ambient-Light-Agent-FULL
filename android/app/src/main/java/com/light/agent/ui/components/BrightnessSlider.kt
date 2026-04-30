package com.light.agent.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.light.agent.R
import com.light.agent.theme.Accent
import com.light.agent.theme.AccentSoft
import com.light.agent.theme.BgSurfaceHi
import com.light.agent.theme.StrokeSoft
import kotlin.math.roundToInt

@Composable
fun BrightnessSlider(
    brightness: Float,
    onValueChange: (Float) -> Unit,
    onValueChangeFinished: (Float) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surface)
            .border(1.dp, StrokeSoft, RoundedCornerShape(20.dp))
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Label row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Bottom
            ) {
                Text(
                    text = "亮度",
                    style = MaterialTheme.typography.labelMedium
                )
                // Large impactful number
                Row(verticalAlignment = Alignment.Bottom) {
                    Text(
                        text = "${(brightness * 100).roundToInt()}",
                        fontSize = 40.sp,
                        fontWeight = FontWeight.Bold,
                        color = Accent,
                        lineHeight = 40.sp
                    )
                    Text(
                        text = "%",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Medium,
                        color = Accent.copy(alpha = 0.55f),
                        modifier = Modifier.padding(bottom = 6.dp, start = 2.dp)
                    )
                }
            }

            // Gradient track with glow trail
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(44.dp),
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .padding(horizontal = 10.dp)
                        .fillMaxWidth()
                        .height(12.dp)
                        .clip(RoundedCornerShape(6.dp))
                ) {
                    // Base gradient track
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                Brush.horizontalGradient(listOf(BgSurfaceHi, AccentSoft, Accent))
                            )
                    )
                    // Dark mask after thumb — creates "glow trail" on filled portion
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                Brush.horizontalGradient(
                                    colorStops = arrayOf(
                                        0f to Color.Transparent,
                                        (brightness * 0.92f).coerceAtMost(0.97f) to Color.Transparent,
                                        brightness.coerceAtLeast(0.03f) to Color.Black.copy(0.58f),
                                        1f to Color.Black.copy(0.58f)
                                    )
                                )
                            )
                    )
                }

                // Transparent interactive slider (on top)
                Slider(
                    value = brightness,
                    onValueChange = onValueChange,
                    onValueChangeFinished = { onValueChangeFinished(brightness) },
                    modifier = Modifier.fillMaxWidth(),
                    colors = SliderDefaults.colors(
                        thumbColor = Color.White,
                        activeTrackColor = Color.Transparent,
                        inactiveTrackColor = Color.Transparent,
                        activeTickColor = Color.Transparent,
                        inactiveTickColor = Color.Transparent
                    )
                )
            }

            // Sun icons row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    painter = painterResource(R.drawable.ic_sun),
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
                    modifier = Modifier.size(14.dp)
                )
                Icon(
                    painter = painterResource(R.drawable.ic_sun),
                    contentDescription = null,
                    tint = Accent,
                    modifier = Modifier.size(20.dp)
                )
            }
        }
    }
}
