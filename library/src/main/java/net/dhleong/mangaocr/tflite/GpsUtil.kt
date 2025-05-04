package net.dhleong.mangaocr.tflite

import com.google.android.gms.tasks.Task
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

suspend fun <TResult> Task<TResult>.await() {
    suspendCoroutine { cont ->
        // TODO: Support cancellation?
        addOnSuccessListener { result ->
            cont.resume(result)
        }
        addOnFailureListener { e ->
            cont.resumeWithException(e)
        }
    }
}
