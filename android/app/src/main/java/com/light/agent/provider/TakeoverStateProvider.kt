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
        val TAKEOVER_URI: Uri = Uri.parse("content://$AUTHORITY/$PATH_TAKEOVER")

        private const val PREF_FILE = "takeover_provider_prefs"
        private const val KEY_ACTIVE = "is_active"

        private val uriMatcher = UriMatcher(UriMatcher.NO_MATCH).apply {
            addURI(AUTHORITY, PATH_TAKEOVER, 1)
        }

        fun setActive(context: Context, active: Boolean) {
            context.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
                .edit().putBoolean(KEY_ACTIVE, active).apply()
            context.contentResolver.notifyChange(TAKEOVER_URI, null)
        }

        fun isActive(context: Context): Boolean =
            context.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
                .getBoolean(KEY_ACTIVE, false)
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
        if (uriMatcher.match(uri) != 1) return null
        val active = values?.getAsInteger("is_active") == 1
        context?.getSharedPreferences(PREF_FILE, Context.MODE_PRIVATE)
            ?.edit()?.putBoolean(KEY_ACTIVE, active)?.apply()
        context?.contentResolver?.notifyChange(TAKEOVER_URI, null)
        return TAKEOVER_URI
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
