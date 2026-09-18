package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleCoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class DetectorManager(
    context: Context,
    scope: LifecycleCoroutineScope,
    lifecycle: Lifecycle,
    private val type: Detector.Type = Detector.Type.YoloCoco,
) : BaseManager<Detector>(context, scope, lifecycle),
    Detector {
    override suspend fun initialize(context: Context): Detector = Detector.initialize(context, type = type)

    override suspend fun process(bitmap: Bitmap): List<Detector.Result> {
        val model = awaitModel()
        return withContext(Dispatchers.IO) {
            model.process(bitmap)
        }
    }
}
