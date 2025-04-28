package net.dhleong.mangaocr.tflite

import android.content.Context
import com.google.android.gms.tflite.java.TfLite
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

object TfLiteHelper {
    suspend fun initialize(context: Context) {
        suspendCoroutine { cont ->
            TfLite
                .initialize(context)
                .addOnSuccessListener { cont.resume(Unit) }
                .addOnFailureListener { cont.resumeWithException(it) }
        }
    }
}
