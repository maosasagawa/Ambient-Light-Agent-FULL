package com.light.agent.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightAgentColors = lightColorScheme(
    primary              = Accent,
    onPrimary            = Color.White,
    primaryContainer     = AccentContainer,
    onPrimaryContainer   = AccentDark,
    secondary            = AccentSoft,
    background           = BgPrimary,
    onBackground         = TextPrimary,
    surface              = BgSurface,
    onSurface            = TextPrimary,
    surfaceVariant       = BgSurfaceHi,
    onSurfaceVariant     = TextSecondary,
    outline              = StrokeSoft,
    outlineVariant       = StrokeMid
)

@Composable
fun LightAgentTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightAgentColors,
        typography = Typography,
        content = content
    )
}
