package net.dhleong.mangaocr

import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class LoggingMangaOcr(
    private val delegate: MangaOcr,
) : MangaOcr by delegate {
    override suspend fun process(bitmap: Bitmap): Flow<MangaOcr.Result> {
        var start = System.currentTimeMillis()
        return flow {
            val flow = delegate.process(bitmap)
            var isFirst = true
            val nonFirstTimings = mutableListOf<Long>()
            flow.collect {
                val now = System.currentTimeMillis()
                val delta = now - start
                start = now

                if (isFirst) {
                    isFirst = false
                    Log.v("OCR", "First emit after ${delta}ms")
                } else {
                    nonFirstTimings += delta
                }
                emit(it)
            }

            Log.v("OCR", "Over ${nonFirstTimings.size} avg = ${nonFirstTimings.average()}")
        }
    }
}
