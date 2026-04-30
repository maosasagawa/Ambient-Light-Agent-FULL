package com.light.agent.provider

import android.content.ContentProvider
import android.content.ContentValues
import android.content.Context
import android.content.UriMatcher
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri

class TakeoverStateProvider : ContentProvider() {

    companion object {
        const val AUTHORITY = "com.light.agent.provider"
        const val PATH_TAKEOVER = "takeover"
        const val PATH_VOICE_INPUT = "voice_input"
        val TAKEOVER_URI: Uri = Uri.parse("content://$AUTHORITY/$PATH_TAKEOVER")
        val VOICE_INPUT_URI: Uri = Uri.parse("content://$AUTHORITY/$PATH_VOICE_INPUT")

        private const val PREF_FILE = "takeover_provider_prefs"
        private const val KEY_ACTIVE = "is_active"
        private const val KEY_VOICE_TEXT = "voice_input_text"

        private val uriMatcher = UriMatcher(UriMatcher.NO_MATCH).apply {
            addURI(AUTHORITY, PATH_TAKEOVER, 1)
            addURI(AUTHORITY, PATH_VOICE_INPUT, 2)
        }

        fun setActive(context: Context, active: Boolean) {
            context.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
                .edit().putBoolean(KEY_ACTIVE, active).apply()
            context.contentResolver.notifyChange(TAKEOVER_URI, null)
        }

        fun isActive(context: Context): Boolean =
            context.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
                .getBoolean(KEY_ACTIVE, false)

        fun getLastVoiceInput(context: Context): String? =
            context.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
                .getString(KEY_VOICE_TEXT, null)

        fun clearVoiceInput(context: Context) {
            context.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
                .edit().remove(KEY_VOICE_TEXT).apply()
        }
    }

    override fun onCreate(): Boolean = true

    override fun query(
        uri: Uri, projection: Array<out String>?, selection: String?,
        selectionArgs: Array<out String>?, sortOrder: String?
    ): Cursor? {
        if (uriMatcher.match(uri) != 1) return null
        val isActive = context?.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
            ?.getBoolean(KEY_ACTIVE, false) ?: false
        val cursor = MatrixCursor(arrayOf("_id", "is_active"))
        cursor.addRow(arrayOf(1, if (isActive) 1 else 0))
        cursor.setNotificationUri(context?.contentResolver, TAKEOVER_URI)
        return cursor
    }

    override fun insert(uri: Uri, values: ContentValues?): Uri? {
        return when (uriMatcher.match(uri)) {
            1 -> {
                val active = values?.getAsInteger("is_active") == 1
                context?.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
                    ?.edit()?.putBoolean(KEY_ACTIVE, active)?.apply()
                context?.contentResolver?.notifyChange(TAKEOVER_URI, null)
                TAKEOVER_URI
            }
            2 -> {
                val text = values?.getAsString("text") ?: return null
                context?.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
                    ?.edit()?.putString(KEY_VOICE_TEXT, text)?.apply()
                context?.contentResolver?.notifyChange(VOICE_INPUT_URI, null)
                VOICE_INPUT_URI
            }
            else -> null
        }
    }

    override fun update(
        uri: Uri, values: ContentValues?, selection: String?,
        selectionArgs: Array<out String>?
    ): Int {
        if (uriMatcher.match(uri) != 1) return 0
        val active = values?.getAsInteger("is_active") == 1
        context?.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
            ?.edit()?.putBoolean(KEY_ACTIVE, active)?.apply()
        context?.contentResolver?.notifyChange(TAKEOVER_URI, null)
        return 1
    }

    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?) = 0
    override fun getType(uri: Uri): String? = null
}
