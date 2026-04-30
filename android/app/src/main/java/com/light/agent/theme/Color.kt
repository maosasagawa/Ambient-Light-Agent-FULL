package com.light.agent.theme

import androidx.compose.ui.graphics.Color

// ── Surfaces (light) ─────────────────────────────────────────────────────────
val BgPrimary      = Color(0xFFF2F3F7)   // main background: light warm gray
val BgSurface      = Color(0xFFFFFFFF)   // cards: white
val BgSurfaceHi    = Color(0xFFF4F5F8)   // inputs / elevated: very light gray
val BgTrack        = Color(0xFFE4E6ED)   // chip / track backgrounds

// ── Strokes (dark on light) ──────────────────────────────────────────────────
val StrokeSoft     = Color(0x12000000)   //  7% black
val StrokeMid      = Color(0x1F000000)   // 12% black
val StrokeStrong   = Color(0x33000000)   // 20% black

// ── Text ─────────────────────────────────────────────────────────────────────
val TextPrimary    = Color(0xFF1A1B1F)   // near black
val TextSecondary  = Color(0xFF6B6F7A)   // medium gray
val TextMuted      = Color(0xFFA0A4AF)   // light gray

// ── Brand ─────────────────────────────────────────────────────────────────────
val Accent         = Color(0xFFFF8A3D)
val AccentSoft     = Color(0xFFFFB27A)
val AccentDark     = Color(0xFFE56A15)
val AccentContainer = Color(0x18FF8A3D)
val AccentGlow     = Color(0x40FF8A3D)

// ── Status ───────────────────────────────────────────────────────────────────
val ConnectedGreen = Color(0xFF16A34A)   // darker green for light background

// ── Legacy-compat aliases ────────────────────────────────────────────────────
val Background          = BgPrimary
val Surface             = BgSurface
val OnSurface           = TextPrimary
val OnSurfaceVariant    = TextSecondary
val SwitchTrackOff      = Color(0xFFDFE2EA)
val CardStroke          = StrokeSoft
