package com.light.agent.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
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
        contentPadding = PaddingValues(2.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
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
    val shape = RoundedCornerShape(14.dp)
    val borderWidth by animateDpAsState(
        targetValue = if (isSelected) 2.dp else 0.dp,
        animationSpec = tween(180), label = "border"
    )
    val borderColor by animateColorAsState(
        targetValue = if (isSelected) Color.White else Color.Transparent,
        animationSpec = tween(180), label = "borderColor"
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(72.dp)
            .clip(shape)
            .background(
                Brush.linearGradient(listOf(preset.gradientStart, preset.gradientEnd))
            )
            .border(borderWidth, borderColor, shape)
            .clickable(onClick = onClick)
    ) {
        // Subtle top gloss
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight()
                .background(
                    Brush.verticalGradient(
                        listOf(Color.White.copy(alpha = 0.08f), Color.Transparent),
                        endY = 50f
                    )
                )
        )

        // Labels
        Column(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(12.dp)
        ) {
            Text(
                text = preset.name,
                color = Color.White,
                fontWeight = FontWeight.SemiBold,
                fontSize = 15.sp
            )
            Text(
                text = preset.effectLabel,
                color = Color.White.copy(alpha = 0.78f),
                fontSize = 10.sp,
                fontWeight = FontWeight.Medium
            )
        }

        // Selected indicator dot
        if (isSelected) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(10.dp)
                    .size(6.dp)
                    .background(Color.White, RoundedCornerShape(50))
            )
        }
    }
}
