package com.light.agent.data.websocket

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener

sealed class WsEvent {
    data class PowerUpdate(val matrix: Boolean, val strip: Boolean) : WsEvent()
    data class BrightnessUpdate(val matrix: Float, val strip: Float) : WsEvent()
    data class StripCommandUpdate(val mode: String?, val colors: List<List<Int>>?) : WsEvent()
    object GenerateComplete : WsEvent()
    data class Connected(val url: String) : WsEvent()
    object Disconnected : WsEvent()
}

class LightWebSocketClient(private val okHttpClient: OkHttpClient) {

    private var webSocket: WebSocket? = null

    private val _events = MutableSharedFlow<WsEvent>(extraBufferCapacity = 64)
    val events: SharedFlow<WsEvent> = _events

    fun connect(baseUrl: String) {
        disconnect()
        val wsUrl = baseUrl.replace("http://", "ws://").replace("https://", "wss://")
            .trimEnd('/') + "/ws"
        val request = Request.Builder().url(wsUrl).build()
        webSocket = okHttpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                _events.tryEmit(WsEvent.Connected(wsUrl))
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                parseAndEmit(text)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                _events.tryEmit(WsEvent.Disconnected)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                _events.tryEmit(WsEvent.Disconnected)
            }
        })
    }

    fun disconnect() {
        webSocket?.close(1000, "Client disconnect")
        webSocket = null
    }

    private fun parseAndEmit(text: String) {
        runCatching {
            val obj: JsonObject = JsonParser.parseString(text).asJsonObject
            when (obj.get("type")?.asString) {
                "power_update" -> {
                    val payload = obj.getAsJsonObject("payload") ?: return
                    val data = payload.getAsJsonObject("power") ?: payload
                    _events.tryEmit(
                        WsEvent.PowerUpdate(
                            matrix = data.get("matrix")?.asBoolean ?: false,
                            strip = data.get("strip")?.asBoolean ?: false
                        )
                    )
                }
                "brightness_update" -> {
                    val payload = obj.getAsJsonObject("payload") ?: return
                    val data = payload.getAsJsonObject("brightness") ?: payload
                    _events.tryEmit(
                        WsEvent.BrightnessUpdate(
                            matrix = data.get("matrix")?.asFloat ?: 0.7f,
                            strip = data.get("strip")?.asFloat ?: 0.7f
                        )
                    )
                }
                "strip_command_update" -> {
                    val payload = obj.getAsJsonObject("payload")
                    val command = payload?.getAsJsonObject("command") ?: payload
                    val mode = command?.get("mode")?.asString
                    _events.tryEmit(WsEvent.StripCommandUpdate(mode = mode, colors = null))
                }
                "generate" -> _events.tryEmit(WsEvent.GenerateComplete)
                else -> Unit
            }
        }
    }
}
