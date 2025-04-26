package net.dhleong.mangaocr.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

suspend inline fun createSession(
    modelFile: File,
    env: OrtEnvironment = OrtEnvironment.getEnvironment(),
    crossinline configureOptions: OrtSession.SessionOptions.() -> Unit,
) = withContext(Dispatchers.IO) {
    val start = System.currentTimeMillis()
    val session =
        env.createSession(
            modelFile.absolutePath,
            OrtSession.SessionOptions().apply(configureOptions),
        )
    Log.v("ORT", "Created session @ $modelFile in ${System.currentTimeMillis() - start}ms")
    session
}
