package com.light.agent.data.prefs

import android.content.Context

class AppPreferences(context: Context) {
    private val prefs = context.getSharedPreferences("light_agent_prefs", Context.MODE_PRIVATE)

    var serverUrl: String
        get() = prefs.getString(KEY_SERVER_URL, "") ?: ""
        set(value) = prefs.edit().putString(KEY_SERVER_URL, value).apply()

    var isTakeoverActive: Boolean
        get() = prefs.getBoolean(KEY_TAKEOVER, false)
        set(value) = prefs.edit().putBoolean(KEY_TAKEOVER, value).apply()

    companion object {
        private const val KEY_SERVER_URL = "server_url"
        private const val KEY_TAKEOVER = "takeover_active"
    }
}
