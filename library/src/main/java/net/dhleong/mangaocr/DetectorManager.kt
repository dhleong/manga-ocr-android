package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleCoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import net.dhleong.mangaocr.detector.TfliteMangaTextDetector

class DetectorManager(
    context: Context,
    scope: LifecycleCoroutineScope,
    lifecycle: Lifecycle,
    private val processorType: TfliteMangaTextDetector.Processor.Type = TfliteMangaTextDetector.Processor.DEFAULT_TYPE,
    private val forceLegacy: Boolean = false,
) : BaseManager<Detector>(context, scope, lifecycle),
    Detector {
    override suspend fun initialize(context: Context): Detector =
        if (forceLegacy) {
            Detector.initializeLegacy(context)
        } else {
            Detector.initialize(context, processorType = processorType)
        }

    override suspend fun process(bitmap: Bitmap): List<Detector.Result> {
        val model = awaitModel()
        return withContext(Dispatchers.IO) {
            model.process(bitmap)
        }
    }
}
