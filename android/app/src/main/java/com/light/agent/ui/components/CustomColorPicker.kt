package com.light.agent.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.light.agent.model.RgbColor
import com.light.agent.theme.Accent
import com.light.agent.theme.AccentContainer
import com.light.agent.theme.AccentDark
import com.light.agent.theme.BgSurfaceHi
import com.light.agent.theme.BgTrack
import com.light.agent.theme.StrokeMid
import com.light.agent.theme.StrokeSoft
import kotlin.math.roundToInt

// ── helpers ──────────────────────────────────────────────────────────────────

fun hsvToColor(h: Float, s: Float, v: Float): Color {
    if (s == 0f) { val g = v; return Color(g, g, g) }
    val sector = (h / 60f) % 6f
    val i = sector.toInt()
    val f = sector - i
    val p = v * (1f - s)
    val q = v * (1f - f * s)
    val t = v * (1f - (1f - f) * s)
    return when (i) {
        0 -> Color(v, t, p)
        1 -> Color(q, v, p)
        2 -> Color(p, v, t)
        3 -> Color(p, q, v)
        4 -> Color(t, p, v)
        else -> Color(v, p, q)
    }
}

fun hsvToRgb(h: Float, s: Float, v: Float): RgbColor {
    val c = hsvToColor(h, s, v)
    return RgbColor(
        (c.red * 255).roundToInt(),
        (c.green * 255).roundToInt(),
        (c.blue * 255).roundToInt()
    )
}

private val HueGradient = Brush.horizontalGradient(
    listOf(
        Color.Red, Color(1f, 1f, 0f), Color.Green,
        Color.Cyan, Color.Blue, Color(1f, 0f, 1f), Color.Red
    )
)

val AllModes = listOf(
    "static"  to "常亮",
    "breath"  to "呼吸",
    "flow"    to "流光",
    "chase"   to "追逐",
    "wave"    to "波浪",
    "sparkle" to "闪烁",
    "pulse"   to "脉冲"
)

private val TwoColorModes = setOf("flow", "wave", "chase", "sparkle")

data class HsvColor(val hue: Float = 30f, val sat: Float = 0.9f, val value: Float = 1f) {
    fun toComposeColor() = hsvToColor(hue, sat, value)
    fun toRgb() = hsvToRgb(hue, sat, value)
}

// ── main composable ───────────────────────────────────────────────────────────

@Composable
fun CustomColorPicker(
    onApply: (mode: String, colors: List<RgbColor>, speed: Float) -> Unit,
    modifier: Modifier = Modifier
) {
    var slot1 by remember { mutableStateOf(HsvColor(28f, 0.88f, 1f)) }
    var slot2 by remember { mutableStateOf(HsvColor(215f, 0.8f, 1f)) }
    var activeSlot by remember { mutableIntStateOf(0) }
    var mode by remember { mutableStateOf("breath") }
    var speed by remember { mutableFloatStateOf(3f) }

    val color1 = slot1.toComposeColor()
    val color2 = slot2.toComposeColor()
    val editing = if (activeSlot == 0) slot1 else slot2
    val needsTwo = mode in TwoColorModes

    fun update(c: HsvColor) { if (activeSlot == 0) slot1 = c else slot2 = c }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        // ── Colour slots ─────────────────────────────────────────────────────
        SectionLabel("颜色")
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            ColorSlot(
                label = "主色",
                color = color1,
                selected = activeSlot == 0,
                onClick = { activeSlot = 0 },
                modifier = Modifier.weight(1f)
            )
            ColorSlot(
                label = "副色",
                color = color2,
                selected = activeSlot == 1,
                dimmed = !needsTwo,
                onClick = { activeSlot = 1 },
                modifier = Modifier.weight(1f)
            )
        }

        // ── Live preview bar (mixed gradient) ────────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp)
                .shadow(12.dp, RoundedCornerShape(14.dp), spotColor = color1, ambientColor = color1)
                .clip(RoundedCornerShape(14.dp))
                .background(
                    if (needsTwo) Brush.horizontalGradient(listOf(color1, color2))
                    else Brush.horizontalGradient(listOf(color1, color1))
                )
        )

        // ── HSV sliders card ─────────────────────────────────────────────────
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(BgSurfaceHi)
                .padding(horizontal = 14.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            GradientSliderRow(
                label = "色相",
                value = editing.hue / 360f,
                onValueChange = { update(editing.copy(hue = it * 360f)) },
                gradient = HueGradient
            )
            GradientSliderRow(
                label = "饱和",
                value = editing.sat,
                onValueChange = { update(editing.copy(sat = it)) },
                gradient = Brush.horizontalGradient(
                    listOf(Color.White, hsvToColor(editing.hue, 1f, editing.value))
                )
            )
            GradientSliderRow(
                label = "明度",
                value = editing.value,
                onValueChange = { update(editing.copy(value = it)) },
                gradient = Brush.horizontalGradient(
                    listOf(Color.Black, hsvToColor(editing.hue, editing.sat, 1f))
                )
            )
        }

        // ── Mode selector ────────────────────────────────────────────────────
        SectionLabel("效果模式")
        ModeGrid(selectedMode = mode, onSelect = { mode = it })

        // ── Speed (animated only) ────────────────────────────────────────────
        AnimatedVisibility(
            visible = mode != "static",
            enter = expandVertically(tween(220)),
            exit = shrinkVertically(tween(180))
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    SectionLabel("速度")
                    Text(
                        text = when {
                            speed < 2.5f -> "慢"
                            speed < 6f   -> "中"
                            else         -> "快"
                        },
                        color = Accent,
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 13.sp
                    )
                }
                Slider(
                    value = speed,
                    onValueChange = { speed = it },
                    valueRange = 0.5f..10f,
                    modifier = Modifier.fillMaxWidth().height(40.dp),
                    colors = SliderDefaults.colors(
                        thumbColor = Accent,
                        activeTrackColor = Accent,
                        inactiveTrackColor = AccentContainer
                    )
                )
            }
        }

        // ── Apply ────────────────────────────────────────────────────────────
        ApplyButton(
            onClick = {
                val colors = buildList {
                    add(slot1.toRgb())
                    if (needsTwo) add(slot2.toRgb())
                }
                onApply(mode, colors, speed)
            }
        )
        Spacer(Modifier.height(4.dp))
    }
}

// ── sub-composables ───────────────────────────────────────────────────────────

@Composable
private fun SectionLabel(text: String) {
    Text(text = text, style = MaterialTheme.typography.labelMedium)
}

@Composable
private fun ColorSlot(
    label: String,
    color: Color,
    selected: Boolean,
    dimmed: Boolean = false,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val alpha = if (dimmed && !selected) 0.5f else 1f
    val borderAlpha by animateFloatAsState(
        targetValue = if (selected) 1f else 0f,
        animationSpec = tween(220), label = "slotBorder"
    )
    Row(
        modifier = modifier
            .height(62.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(MaterialTheme.colorScheme.surface)
            .border(
                width = 2.dp,
                color = Accent.copy(alpha = borderAlpha),
                shape = RoundedCornerShape(16.dp)
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Colour dot with soft halo
        Box(
            modifier = Modifier
                .size(36.dp)
                .shadow(8.dp, CircleShape, spotColor = color, ambientColor = color)
                .clip(CircleShape)
                .background(color.copy(alpha = alpha))
                .border(1.5.dp, Color.White.copy(0.2f * alpha), CircleShape)
        )
        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(
                text = label,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = alpha),
                fontSize = 14.sp,
                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium
            )
            Text(
                text = "#%02X%02X%02X".format(
                    (color.red * 255).roundToInt(),
                    (color.green * 255).roundToInt(),
                    (color.blue * 255).roundToInt()
                ),
                fontSize = 11.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha)
            )
        }
    }
}

@Composable
private fun GradientSliderRow(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    gradient: Brush
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text(
            text = label,
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.width(32.dp)
        )
        Box(
            modifier = Modifier.weight(1f).height(44.dp),
            contentAlignment = Alignment.Center
        ) {
            Box(
                modifier = Modifier
                    .padding(horizontal = 10.dp)
                    .fillMaxWidth()
                    .height(10.dp)
                    .clip(RoundedCornerShape(5.dp))
                    .background(gradient)
            )
            Slider(
                value = value,
                onValueChange = onValueChange,
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
    }
}

@Composable
private fun ModeGrid(selectedMode: String, onSelect: (String) -> Unit) {
    val cols = 4
    val rows = AllModes.chunked(cols)
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        rows.forEach { row ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                row.forEach { (id, name) ->
                    ModeChip(
                        label = name,
                        selected = selectedMode == id,
                        onClick = { onSelect(id) },
                        modifier = Modifier.weight(1f)
                    )
                }
                repeat(cols - row.size) { Spacer(Modifier.weight(1f)) }
            }
        }
    }
}

@Composable
private fun ModeChip(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .height(46.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(if (selected) AccentContainer else BgTrack)
            .border(
                width = 1.dp,
                color = if (selected) Accent else StrokeSoft,
                shape = RoundedCornerShape(12.dp)
            )
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = label,
            fontSize = 14.sp,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium,
            color = if (selected) Accent else MaterialTheme.colorScheme.onSurface
        )
    }
}

@Composable
private fun ApplyButton(onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(52.dp)
            .shadow(10.dp, RoundedCornerShape(16.dp), spotColor = Accent, ambientColor = Accent)
            .clip(RoundedCornerShape(16.dp))
            .background(Brush.linearGradient(listOf(Accent, AccentDark)))
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = "应用效果",
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
    }
}
