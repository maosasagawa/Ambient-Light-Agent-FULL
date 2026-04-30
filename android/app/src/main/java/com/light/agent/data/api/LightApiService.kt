package com.light.agent.data.api

import com.light.agent.model.*
import okhttp3.MultipartBody
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface LightApiService {
    @GET("api/app/state")
    suspend fun getState(): AppState

    @GET("api/app/power")
    suspend fun getPower(): HwPowerEnvelope

    @POST("api/app/power")
    suspend fun setPower(@Body body: HwPowerState): HwPowerEnvelope

    @GET("api/app/brightness")
    suspend fun getBrightness(): HwBrightnessEnvelope

    @POST("api/app/brightness")
    suspend fun setBrightness(@Body body: HwBrightnessState): HwBrightnessEnvelope

    @POST("api/app/strip/command")
    suspend fun setStripCommand(@Body cmd: StripCommandBody): StripCommandEnvelope

    @POST("api/app/submit")
    suspend fun submitInstruction(@Body body: InstructionRequest): VoiceAcceptResponse

    @Multipart
    @POST("api/matrix/downsample")
    suspend fun uploadMatrixImage(@Part file: MultipartBody.Part): MatrixDownsampleResponse
}
