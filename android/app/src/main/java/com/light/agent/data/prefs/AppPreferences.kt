package com.light.agent.data.prefs

import android.content.Context
import com.light.agent.model.BackendMode

class AppPreferences(context: Context) {
    private val prefs = context.getSharedPreferences("light_agent_prefs", Context.MODE_PRIVATE)

    var serverUrl: String
        get() = prefs.getString(KEY_SERVER_URL, "") ?: ""
        set(value) = prefs.edit().putString(KEY_SERVER_URL, value).apply()

    var isTakeoverActive: Boolean
        get() = prefs.getBoolean(KEY_TAKEOVER, false)
        set(value) = prefs.edit().putBoolean(KEY_TAKEOVER, value).apply()

    var backendMode: BackendMode
        get() = BackendMode.fromStored(prefs.getString(KEY_BACKEND_MODE, BackendMode.AUTO.name))
        set(value) = prefs.edit().putString(KEY_BACKEND_MODE, value.name).apply()

    var developerUnlocked: Boolean
        get() = prefs.getBoolean(KEY_DEVELOPER_UNLOCKED, false)
        set(value) = prefs.edit().putBoolean(KEY_DEVELOPER_UNLOCKED, value).apply()

    var aiHubMixApiKey: String
        get() = prefs.getString(KEY_AIHUBMIX_API_KEY, "") ?: ""
        set(value) = prefs.edit().putString(KEY_AIHUBMIX_API_KEY, value).apply()

    companion object {
        private const val KEY_SERVER_URL = "server_url"
        private const val KEY_TAKEOVER = "takeover_active"
        private const val KEY_BACKEND_MODE = "backend_mode"
        private const val KEY_DEVELOPER_UNLOCKED = "developer_unlocked"
        private const val KEY_AIHUBMIX_API_KEY = "aihubmix_api_key"
    }
}
