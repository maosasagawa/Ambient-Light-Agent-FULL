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
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.light.agent.R
import com.light.agent.theme.BgSurface
import com.light.agent.theme.StrokeSoft
import com.light.agent.theme.TextPrimary
import com.light.agent.theme.TextSecondary
import kotlin.math.roundToInt

// Warm neutral for inactive portion of the track
private val TrackInactive = Color(0xFFE6E3DE)

// Palette for the "light intensity" metaphor: dim warm → golden → bright warm cream
private fun trackBrush(brightness: Float) = Brush.horizontalGradient(
    colorStops = arrayOf(
        0f                                              to Color(0xFF8C7A60),
        (brightness * 0.50f).coerceAtLeast(0.001f)     to Color(0xFFFFBF2E),
        brightness.coerceIn(0.02f, 0.96f)              to Color(0xFFFFF3B0),
        (brightness + 0.025f).coerceAtMost(1f)         to TrackInactive,
        1f                                              to TrackInactive
    )
)

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
            .background(BgSurface)
            .border(1.dp, StrokeSoft, RoundedCornerShape(20.dp))
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // ── Label + value row ────────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Bottom
            ) {
                Text(
                    text = "亮度",
                    style = MaterialTheme.typography.labelMedium
                )
                Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(1.dp)) {
                    Text(
                        text = "${(brightness * 100).roundToInt()}",
                        fontSize = 36.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary,
                        lineHeight = 36.sp
                    )
                    Text(
                        text = "%",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium,
                        color = TextSecondary,
                        modifier = Modifier.padding(bottom = 5.dp, start = 1.dp)
                    )
                }
            }

            // ── Track ────────────────────────────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(44.dp),
                contentAlignment = Alignment.Center
            ) {
                // Track with warm glow that follows brightness
                val glowAlpha = (brightness * 0.55f).coerceIn(0f, 0.55f)
                Box(
                    modifier = Modifier
                        .padding(horizontal = 10.dp)
                        .fillMaxWidth()
                        .height(12.dp)
                        .shadow(
                            elevation = (brightness * 10f).dp,
                            shape = RoundedCornerShape(6.dp),
                            spotColor = Color(0xFFFFD060).copy(alpha = glowAlpha),
                            ambientColor = Color(0xFFFFD060).copy(alpha = glowAlpha * 0.5f)
                        )
                        .clip(RoundedCornerShape(6.dp))
                        .background(trackBrush(brightness))
                )

                // Transparent interactive slider on top
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

            // ── Sun icons ────────────────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    painter = painterResource(R.drawable.ic_sun),
                    contentDescription = null,
                    tint = TextSecondary.copy(alpha = 0.35f),
                    modifier = Modifier.size(13.dp)
                )
                Icon(
                    painter = painterResource(R.drawable.ic_sun),
                    contentDescription = null,
                    tint = Color(0xFFFFBF2E).copy(alpha = (0.4f + brightness * 0.6f)),
                    modifier = Modifier.size(19.dp)
                )
            }
        }
    }
}
