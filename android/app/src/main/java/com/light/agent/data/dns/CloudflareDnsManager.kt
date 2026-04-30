package com.light.agent.data.dns

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class CloudflareDnsManager(
    private val apiToken: String,
    private val zoneId: String,
    private val recordName: String,
    private val client: OkHttpClient
) {
    private val base = "https://api.cloudflare.com/client/v4"
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    @Volatile private var cachedRecordId: String? = null

    suspend fun updateRecord(ip: String): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val recordId = cachedRecordId ?: fetchRecordId().getOrThrow()
            val body = JSONObject().apply {
                put("type", "A")
                put("name", recordName)
                put("content", ip)
                put("ttl", 60)
                put("proxied", false)
            }.toString().toRequestBody(jsonMediaType)

            val request = Request.Builder()
                .url("$base/zones/$zoneId/dns_records/$recordId")
                .header("Authorization", "Bearer $apiToken")
                .put(body)
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    error("Cloudflare DNS update failed: ${response.code} ${response.body?.string()}")
                }
                val result = JSONObject(response.body?.string() ?: "{}")
                if (!result.optBoolean("success", false)) {
                    error("Cloudflare API error: ${result.optJSONArray("errors")}")
                }
            }
        }
    }

    private suspend fun fetchRecordId(): Result<String> = withContext(Dispatchers.IO) {
        runCatching {
            val request = Request.Builder()
                .url("$base/zones/$zoneId/dns_records?type=A&name=${recordName.trim()}")
                .header("Authorization", "Bearer $apiToken")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    error("Cloudflare list records failed: ${response.code}")
                }
                val body = JSONObject(response.body?.string() ?: "{}")
                val results = body.getJSONArray("result")
                if (results.length() == 0) {
                    error("No A record found for \"$recordName\" in zone $zoneId")
                }
                val id = results.getJSONObject(0).getString("id")
                cachedRecordId = id
                id
            }
        }
    }

    fun invalidateCache() {
        cachedRecordId = null
    }
}
