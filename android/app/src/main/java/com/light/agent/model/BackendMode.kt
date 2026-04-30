package com.light.agent.model

enum class BackendMode {
    LOCAL_FULL,
    ONLINE_BACKEND,
    AUTO;

    companion object {
        fun fromStored(value: String?): BackendMode = entries.firstOrNull { it.name == value } ?: AUTO
    }
}

enum class BackendRuntime {
    LOCAL_FULL,
    ONLINE_BACKEND,
    NATIVE_FALLBACK
}

enum class OnlineHealth {
    UNKNOWN,
    CONNECTED,
    DEGRADED,
    OFFLINE
}
