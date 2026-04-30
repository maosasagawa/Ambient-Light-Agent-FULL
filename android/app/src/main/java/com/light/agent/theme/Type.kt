package com.light.agent.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp

val Typography = Typography(
    displayLarge = TextStyle(
        fontSize = 32.sp, fontWeight = FontWeight.Bold, letterSpacing = (-0.02).em, color = TextPrimary
    ),
    titleLarge = TextStyle(
        fontSize = 22.sp, fontWeight = FontWeight.Bold, letterSpacing = (-0.01).em, color = TextPrimary
    ),
    titleMedium = TextStyle(
        fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = TextPrimary
    ),
    titleSmall = TextStyle(
        fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = TextPrimary
    ),
    bodyLarge = TextStyle(
        fontSize = 16.sp, fontWeight = FontWeight.Normal, color = TextPrimary
    ),
    bodyMedium = TextStyle(
        fontSize = 14.sp, fontWeight = FontWeight.Normal, color = TextSecondary
    ),
    bodySmall = TextStyle(
        fontSize = 12.sp, fontWeight = FontWeight.Normal, color = TextSecondary
    ),
    labelLarge = TextStyle(
        fontSize = 13.sp, fontWeight = FontWeight.Medium, letterSpacing = 0.03.em
    ),
    // Micro caps: "POWER", "COLOR", etc.
    labelMedium = TextStyle(
        fontSize = 11.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.18.em, color = TextMuted
    ),
    labelSmall = TextStyle(
        fontSize = 10.sp, fontWeight = FontWeight.Medium, letterSpacing = 0.08.em, color = TextMuted
    )
)
