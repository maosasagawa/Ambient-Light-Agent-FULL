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
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "亮度",
                style = MaterialTheme.typography.labelMedium
            )
            Row(verticalAlignment = Alignment.Bottom) {
                Text(
                    text = "${(brightness * 100).roundToInt()}",
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold,
                    color = Accent
                )
                Text(
                    text = "%",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium,
                    color = Accent.copy(alpha = 0.6f),
                    modifier = Modifier.padding(bottom = 4.dp, start = 2.dp)
                )
            }
        }

        // Gradient-track slider
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(44.dp),
            contentAlignment = Alignment.Center
        ) {
            // Visual track
            Box(
                modifier = Modifier
                    .padding(horizontal = 10.dp)
                    .fillMaxWidth()
                    .height(12.dp)
                    .clip(RoundedCornerShape(6.dp))
                    .background(
                        Brush.horizontalGradient(
                            listOf(BgSurfaceHi, AccentSoft, Accent)
                        )
                    )
            )
            // Transparent interactive slider
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
