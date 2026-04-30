package com.light.agent.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.light.agent.model.ColorPreset
import com.light.agent.model.DefaultPresets
import com.light.agent.model.LightUiState
import com.light.agent.model.RgbColor
import com.light.agent.theme.Accent
import com.light.agent.theme.BgSurfaceHi
import com.light.agent.theme.StrokeSoft
import com.light.agent.ui.components.BrightnessSlider
import com.light.agent.ui.components.ColorThemeGrid
import com.light.agent.ui.components.CustomColorPicker

@Composable
fun RightPanel(
    state: LightUiState,
    onBrightnessChange: (Float) -> Unit,
    onBrightnessCommit: (Float) -> Unit,
    onPresetSelect: (ColorPreset) -> Unit,
    onApplyCustom: (mode: String, colors: List<RgbColor>, speed: Float) -> Unit,
    modifier: Modifier = Modifier
) {
    var selectedTab by remember { mutableIntStateOf(0) }

    Column(
        modifier = modifier
            .fillMaxHeight()
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        BrightnessSlider(
            brightness = state.brightness,
            onValueChange = onBrightnessChange,
            onValueChangeFinished = onBrightnessCommit
        )

        // Color section card
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .clip(RoundedCornerShape(20.dp))
                .background(MaterialTheme.colorScheme.surface)
                .border(1.dp, StrokeSoft, RoundedCornerShape(20.dp))
        ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "颜色",
                    style = MaterialTheme.typography.labelMedium
                )
                PillTabs(
                    selectedIndex = selectedTab,
                    tabs = listOf("主题", "自定义"),
                    onSelect = { selectedTab = it }
                )
            }

            AnimatedContent(
                targetState = selectedTab,
                transitionSpec = { fadeIn() togetherWith fadeOut() },
                label = "tabContent",
                modifier = Modifier.fillMaxWidth().weight(1f)
            ) { idx ->
                when (idx) {
                    0 -> ColorThemeGrid(
                        presets = DefaultPresets,
                        selectedPreset = state.selectedPreset,
                        onSelect = onPresetSelect,
                        modifier = Modifier.fillMaxSize()
                    )
                    else -> CustomColorPicker(
                        onApply = onApplyCustom,
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }
        }
        } // close outer Box
    }
}

@Composable
private fun PillTabs(
    selectedIndex: Int,
    tabs: List<String>,
    onSelect: (Int) -> Unit
) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(50))
            .background(BgSurfaceHi)
            .padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        tabs.forEachIndexed { index, title ->
            val selected = index == selectedIndex
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(50))
                    .background(if (selected) Accent else Color.Transparent)
                    .clickable { onSelect(index) }
                    .padding(horizontal = 18.dp, vertical = 8.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = title,
                    fontSize = 13.sp,
                    fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium,
                    color = if (selected) Color.White else MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}
