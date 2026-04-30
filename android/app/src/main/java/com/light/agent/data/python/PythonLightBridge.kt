package com.light.agent.data.python

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class PythonLightBridge(private val context: Context) {
    private val dataDir = context.filesDir.resolve("python-light-state")

    suspend fun boot(apiKey: String? = null): Result<String> = withContext(Dispatchers.IO) {
        runCatching {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context.applicationContext))
            }
            dataDir.mkdirs()
            module().callAttr("boot", dataDir.absolutePath, apiKey).toString()
        }
    }

    suspend fun acceptInstruction(instruction: String): Result<String> = callJson("accept_instruction", instruction)

    suspend fun generateLightingEffect(instruction: String): Result<String> = callJson("generate_lighting_effect", instruction)

    suspend fun downsampleImage(fileName: String, contentType: String, bytes: ByteArray): Result<String> =
        withContext(Dispatchers.IO) {
            runCatching {
                ensureStarted()
                module().callAttr("downsample_image", fileName, contentType, bytes).toString()
            }
        }

    suspend fun generateMatrixAnimationCode(
        instruction: String,
        width: Int,
        height: Int,
        fps: Float,
        durationSeconds: Float
    ): Result<String> = withContext(Dispatchers.IO) {
        runCatching {
            ensureStarted()
            module().callAttr("generate_matrix_animation_code", instruction, width, height, fps, durationSeconds).toString()
        }
    }

    suspend fun renderAnimationFrames(
        code: String,
        width: Int,
        height: Int,
        fps: Float,
        durationSeconds: Float,
        maxFrames: Int
    ): Result<String> = withContext(Dispatchers.IO) {
        runCatching {
            ensureStarted()
            module().callAttr(
                "render_animation_frames_threaded",
                code,
                width,
                height,
                fps,
                durationSeconds,
                maxFrames
            ).toString()
        }
    }

    private suspend fun callJson(functionName: String, arg: String): Result<String> = withContext(Dispatchers.IO) {
        runCatching {
            ensureStarted()
            module().callAttr(functionName, arg).toString()
        }
    }

    private fun ensureStarted() {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context.applicationContext))
        }
        dataDir.mkdirs()
        module().callAttr("boot", dataDir.absolutePath, null)
    }

    private fun module() = Python.getInstance().getModule("android_bridge")
}
