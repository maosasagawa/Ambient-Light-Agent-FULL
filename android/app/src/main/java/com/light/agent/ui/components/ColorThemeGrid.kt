package com.light.agent.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.light.agent.model.ColorPreset

private val FlowingModes = setOf("flow", "wave", "chase")

@Composable
fun ColorThemeGrid(
    presets: List<ColorPreset>,
    selectedPreset: ColorPreset?,
    onSelect: (ColorPreset) -> Unit,
    modifier: Modifier = Modifier
) {
    LazyVerticalGrid(
        columns = GridCells.Fixed(3),
        modifier = modifier,
        contentPadding = PaddingValues(4.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        items(presets, key = { it.id }) { preset ->
            PresetCard(
                preset = preset,
                isSelected = preset.id == selectedPreset?.id,
                onClick = { onSelect(preset) }
            )
        }
    }
}

@Composable
private fun PresetCard(
    preset: ColorPreset,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val shape = RoundedCornerShape(18.dp)
    val elevation by animateDpAsState(
        targetValue = if (isSelected) 16.dp else 6.dp,
        animationSpec = tween(220), label = "elev"
    )
    val borderColor by animateColorAsState(
        targetValue = if (isSelected) Color.White.copy(alpha = 0.85f) else Color.Transparent,
        animationSpec = tween(220), label = "border"
    )

    val infiniteTransition = rememberInfiniteTransition(label = "cardAnim")

    // Breathing glow overlay (selected only)
    val breathAlpha by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = if (isSelected) 0.18f else 0f,
        animationSpec = infiniteRepeatable(
            animation = tween(2400, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "breath"
    )

    // Sweep light translation for flowing modes
    val sweepOffset by infiniteTransition.animateFloat(
        initialValue = -1.5f,
        targetValue = 1.5f,
        animationSpec = infiniteRepeatable(
            animation = tween(3600, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "sweep"
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(82.dp)
            .shadow(
                elevation = elevation,
                shape = shape,
                spotColor = preset.gradientStart,
                ambientColor = preset.gradientStart
            )
            .clip(shape)
            .background(
                Brush.linearGradient(listOf(preset.gradientStart, preset.gradientEnd))
            )
            .border(2.dp, borderColor, shape)
            .clickable(onClick = onClick)
    ) {
        // Inner top highlight — glass edge effect
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight()
                .background(
                    Brush.verticalGradient(
                        listOf(Color.White.copy(alpha = 0.10f), Color.Transparent),
                        endY = 60f
                    )
                )
        )

        // Breathing overlay
        if (isSelected) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.White.copy(alpha = breathAlpha))
            )

            // Sweep light for flowing modes
            if (preset.mode in FlowingModes) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .graphicsLayer { translationX = size.width * sweepOffset }
                        .background(
                            Brush.linearGradient(
                                colors = listOf(
                                    Color.Transparent,
                                    Color.White.copy(alpha = 0.30f),
                                    Color.Transparent
                                ),
                                start = Offset(0f, Float.POSITIVE_INFINITY),
                                end = Offset(100f, 0f)
                            )
                        )
                )
            }
        }

        // Labels bottom-left
        Column(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(14.dp)
        ) {
            Text(
                text = preset.name,
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 17.sp
            )
            Text(
                text = preset.effectLabel,
                color = Color.White.copy(alpha = 0.8f),
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium
            )
        }

        // Effect dot top-right
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(12.dp)
                .size(8.dp)
                .background(
                    Color.White.copy(alpha = if (isSelected) 0.95f else 0.35f),
                    RoundedCornerShape(50)
                )
        )
    }
}
