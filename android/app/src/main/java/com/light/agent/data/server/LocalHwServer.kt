package com.light.agent.data.server

import com.google.gson.Gson
import fi.iki.elonen.NanoWSD
import fi.iki.elonen.NanoWSD.WebSocketFrame.CloseCode
import java.io.IOException
import java.util.concurrent.CopyOnWriteArrayList

class LocalHwServer(port: Int = 8765) : NanoWSD(port) {

    private val gson = Gson()
    private val clients = CopyOnWriteArrayList<HwSocket>()

    @Volatile private var state = HwCommandState()

    override fun openWebSocket(handshake: IHTTPSession): WebSocket {
        val ws = HwSocket(handshake)
        clients.add(ws)
        return ws
    }

    fun updateState(newState: HwCommandState) {
        state = newState
        broadcast(commandsJson(newState))
        broadcast(brightnessJson(newState))
        broadcast(powerJson(newState))
    }

    fun updatePower(on: Boolean) {
        state = state.copy(isPoweredOn = on)
        broadcast(powerJson(state))
        broadcast(commandsJson(state))
    }

    fun updateBrightness(matrix: Float, strip: Float) {
        state = state.copy(matrixBrightness = matrix, stripBrightness = strip)
        broadcast(brightnessJson(state))
    }

    private fun broadcast(msg: String) {
        val stale = mutableListOf<HwSocket>()
        for (ws in clients) {
            try {
                ws.send(msg)
            } catch (_: IOException) {
                stale.add(ws)
            }
        }
        clients.removeAll(stale)
    }

    private fun commandsJson(s: HwCommandState) = gson.toJson(
        mapOf(
            "type" to "commands",
            "payload" to mapOf(
                "updated_at_ms" to System.currentTimeMillis(),
                "commands" to listOf(
                    mapOf(
                        "channel" to "strip:1",
                        "kind" to "color_mode",
                        "enabled" to s.isPoweredOn,
                        "mode_code" to s.stripMode,
                        "params" to mapOf(
                            "brightness" to s.stripBrightness,
                            "speed" to s.stripSpeed,
                            "colors" to s.stripColors,
                            "mode_options" to null
                        )
                    )
                )
            )
        )
    )

    private fun brightnessJson(s: HwCommandState) = gson.toJson(
        mapOf(
            "type" to "brightness_update",
            "payload" to mapOf(
                "brightness" to mapOf(
                    "matrix" to s.matrixBrightness,
                    "strip" to s.stripBrightness
                ),
                "updated_at_ms" to System.currentTimeMillis()
            )
        )
    )

    private fun powerJson(s: HwCommandState) = gson.toJson(
        mapOf(
            "type" to "power_update",
            "payload" to mapOf(
                "power" to mapOf(
                    "matrix" to s.isPoweredOn,
                    "strip" to s.isPoweredOn
                ),
                "updated_at_ms" to System.currentTimeMillis()
            )
        )
    )

    inner class HwSocket(handshake: IHTTPSession) : WebSocket(handshake) {

        override fun onOpen() {
            val snap = state
            try {
                send(gson.toJson(mapOf(
                    "type" to "hello_ack",
                    "payload" to mapOf(
                        "sync_fps" to 20.0,
                        "encoding" to "rgb24",
                        "channels" to listOf("strip:1")
                    )
                )))
                send(brightnessJson(snap))
                send(powerJson(snap))
                send(commandsJson(snap))
            } catch (_: IOException) {
                clients.remove(this)
            }
        }

        override fun onMessage(message: WebSocketFrame) {
            val text = message.textPayload ?: return
            runCatching {
                @Suppress("UNCHECKED_CAST")
                val msg = gson.fromJson(text, Map::class.java) as? Map<String, Any> ?: return
                if (msg["type"] == "hello") {
                    onOpen()
                }
            }
        }

        override fun onClose(code: CloseCode?, reason: String?, initiatedByRemote: Boolean) {
            clients.remove(this)
        }

        override fun onPong(pong: WebSocketFrame?) = Unit
        override fun onException(exception: IOException?) {
            clients.remove(this)
        }
    }
}
