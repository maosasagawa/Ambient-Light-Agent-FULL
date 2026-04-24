package com.light.agent.theme

import androidx.compose.ui.graphics.Color

// ── Surfaces (dark premium) ──────────────────────────────────────────────────
val BgPrimary      = Color(0xFF0B0C11)
val BgSurface      = Color(0xFF15171F)
val BgSurfaceHi    = Color(0xFF1F2129)
val BgTrack        = Color(0xFF26282F)

// ── Strokes ─────────────────────────────────────────────────────────────────
val StrokeSoft     = Color(0x14FFFFFF)
val StrokeMid      = Color(0x1FFFFFFF)
val StrokeStrong   = Color(0x33FFFFFF)

// ── Text ────────────────────────────────────────────────────────────────────
val TextPrimary    = Color(0xFFF3F4F7)
val TextSecondary  = Color(0xFF9EA2AB)
val TextMuted      = Color(0xFF5E626C)

// ── Brand ───────────────────────────────────────────────────────────────────
val Accent         = Color(0xFFFF8A3D)
val AccentSoft     = Color(0xFFFFB27A)
val AccentDark     = Color(0xFFE56A15)
val AccentContainer = Color(0x2EFF8A3D) // ~18% orange (for tinted buttons/tracks)
val AccentGlow     = Color(0x55FF8A3D)

// ── Status ──────────────────────────────────────────────────────────────────
val ConnectedGreen = Color(0xFF4ADE80)

// ── Legacy-compat aliases used by existing code ─────────────────────────────
val Background          = BgPrimary
val Surface             = BgSurface
val OnSurface           = TextPrimary
val OnSurfaceVariant    = TextSecondary
val SwitchTrackOff      = Color(0xFF2E3140)
val CardStroke          = StrokeSoft
