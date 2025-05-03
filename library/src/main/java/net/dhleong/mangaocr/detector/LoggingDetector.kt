package net.dhleong.mangaocr.detector

import android.graphics.Bitmap
import android.util.Log
import net.dhleong.mangaocr.Detector

class LoggingDetector(
    private val delegate: Detector,
) : Detector by delegate {
    override suspend fun process(bitmap: Bitmap): List<Detector.Result> {
        val start = System.currentTimeMillis()
        val result = delegate.process(bitmap)
        val total = System.currentTimeMillis() - start
        Log.v("Detector", "Detected (${result.size}) in $total ms: $result")
        return result
    }
}
