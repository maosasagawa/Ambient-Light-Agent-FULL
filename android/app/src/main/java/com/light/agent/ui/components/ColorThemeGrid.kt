package com.light.agent.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.light.agent.model.ColorPreset

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
        targetValue = if (isSelected) 14.dp else 6.dp,
        animationSpec = tween(220), label = "elev"
    )
    val borderColor by animateColorAsState(
        targetValue = if (isSelected) Color.White.copy(alpha = 0.85f) else Color.Transparent,
        animationSpec = tween(220), label = "border"
    )
    val overlayAlpha by animateFloatAsState(
        targetValue = if (isSelected) 0.25f else 0f,
        animationSpec = tween(220), label = "overlay"
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
        // Subtle inner highlight for selected state (glossy sheen)
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        listOf(
                            Color.White.copy(alpha = overlayAlpha),
                            Color.Transparent
                        )
                    )
                )
        )
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
        // Effect mode indicator dot top-right
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(12.dp)
                .size(8.dp)
                .clip(RectangleShape)
                .background(Color.White.copy(alpha = if (isSelected) 0.95f else 0.35f), RoundedCornerShape(50))
        )
    }
}
