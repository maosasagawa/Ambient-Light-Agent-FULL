package com.light.agent.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
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
import com.light.agent.theme.BgSurfaceHi
import com.light.agent.theme.TextPrimary
import com.light.agent.theme.TextSecondary
import kotlin.math.roundToInt

private val TrackInactive = Color(0xFFE6E3DE)

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
    val glowAlpha = (brightness * 0.55f).coerceIn(0f, 0.55f)
    Row(
        modifier = modifier
            .fillMaxWidth()
            .height(56.dp)
            .clip(RoundedCornerShape(28.dp))
            .background(BgSurfaceHi)
            .padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Icon(
            painter = painterResource(R.drawable.ic_sun),
            contentDescription = null,
            tint = Color(0xFFFFBF2E).copy(alpha = (0.45f + brightness * 0.55f)),
            modifier = Modifier.size(20.dp)
        )

        // Track + transparent slider on top
        Box(
            modifier = Modifier
                .weight(1f)
                .height(40.dp),
            contentAlignment = Alignment.Center
        ) {
            Box(
                modifier = Modifier
                    .padding(horizontal = 8.dp)
                    .fillMaxWidth()
                    .height(10.dp)
                    .shadow(
                        elevation = (brightness * 8f).dp,
                        shape = RoundedCornerShape(5.dp),
                        spotColor = Color(0xFFFFD060).copy(alpha = glowAlpha),
                        ambientColor = Color(0xFFFFD060).copy(alpha = glowAlpha * 0.5f)
                    )
                    .clip(RoundedCornerShape(5.dp))
                    .background(trackBrush(brightness))
            )
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

        // Value badge
        Row(
            verticalAlignment = Alignment.Bottom,
            modifier = Modifier.widthIn(min = 48.dp),
            horizontalArrangement = Arrangement.End
        ) {
            Text(
                text = "${(brightness * 100).roundToInt()}",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = TextPrimary,
                lineHeight = 18.sp
            )
            Text(
                text = "%",
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium,
                color = TextSecondary,
                modifier = Modifier.padding(bottom = 2.dp, start = 1.dp)
            )
        }
    }
}
